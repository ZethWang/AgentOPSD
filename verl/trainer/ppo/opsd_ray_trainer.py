"""
OPSD Trainer.

OPSD ("Bayesian Value Recursion for Token-Level Credit Assignment") extends the
standard RayPPOTrainer with a teacher forward pass that uses privileged
information (skills) to construct token-level advantages. The teacher machinery
is identical to RLSD (self-oracle: teacher = student + skill prefix); only the
token-advantage math differs — OPSD interprets the teacher/student log-ratio as
a Bayesian sufficient statistic for the per-token success belief and produces a
telescoping credit decomposition of the sparse episode return.
"""

from pprint import pprint

import json
import os

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    _timer,
    apply_invalid_action_penalty,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.opsd_utils import (
    SkillProvider,
    compute_group_mean_v0,
    compute_opsd_token_advantage,
    compute_opsd_turn_advantage,
)
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.model import compute_position_id_with_mask

from agent_system.multi_turn_rollout import adjust_batch


def build_teacher_batch(
    batch: DataProto,
    skill_provider: SkillProvider,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "error",
    skill_position: str = "legacy",
):
    """
    Build a teacher batch by prepending privileged skill info to each sample's prompt.

    The teacher sees (x, r) where r is the skill/privileged information.
    We prepend the skill text as a system message before the user prompt,
    then re-tokenize to get teacher input_ids/attention_mask/position_ids.
    The responses remain unchanged.

    Args:
        batch: The original student batch with prompts and responses.
        skill_provider: SkillProvider instance for loading skills.
        tokenizer: The tokenizer.
        max_prompt_length: Maximum prompt length.
        truncation: Truncation mode.
        skill_position: Where to inject the skill text. "legacy" prepends it before
            the system prompt (original behavior). "user" inserts it at the start of
            the first user turn (after the system prompt).

    Returns:
        teacher_batch: A DataProto with modified input_ids/attention_mask/position_ids
            but the same responses, suitable for computing teacher log probs.
    """
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    for i in range(bs):
        # Decode the original prompt (student input minus response)
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        # Find the first non-padding token in prompt
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        if len(valid_start) > 0:
            valid_start = valid_start[0].item()
        else:
            valid_start = 0

        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        # Get privileged skill info based on gamefile, data_source, or prompt text
        gamefile = batch.non_tensor_batch.get("gamefile", None)
        data_source = batch.non_tensor_batch.get("data_source", None)
        if gamefile is not None:
            gf = gamefile[i]
            if gf is not None:
                gf = gf if isinstance(gf, str) else str(gf)
                skill_text = skill_provider.get_privileged_info(gf)
            elif data_source is not None:
                ds = data_source[i] if isinstance(data_source[i], str) else str(data_source[i])
                skill_text = skill_provider.get_privileged_info_from_data_source(ds, prompt_text)
            else:
                skill_text = skill_provider.get_privileged_info_from_prompt(prompt_text)
        elif data_source is not None:
            ds = data_source[i] if isinstance(data_source[i], str) else str(data_source[i])
            skill_text = skill_provider.get_privileged_info_from_data_source(ds, prompt_text)
        else:
            skill_text = skill_provider.get_privileged_info_from_prompt(prompt_text)

        # Construct teacher prompt: inject skill text either before the system prompt
        # (legacy) or at the start of the first user turn (user).
        skill_block = f"[Privileged Skill Information]\n{skill_text}\n\n"
        if skill_position == "user":
            # Insert skill right after the first user-turn marker so it lands inside the
            # user message (after the system prompt). "user" placement is also more robust
            # to left-truncation since the skill sits later in the string, closer to the
            # kept tail. Primary marker is the Qwen/ChatML user opener; if it's absent
            # (unknown template) fall back to legacy prepend so nothing breaks.
            user_marker = "<|im_start|>user\n"
            marker_idx = prompt_text.find(user_marker)
            if marker_idx != -1:
                insert_at = marker_idx + len(user_marker)
                teacher_prompt_text = prompt_text[:insert_at] + skill_block + prompt_text[insert_at:]
            else:
                teacher_prompt_text = skill_block + prompt_text
        else:
            teacher_prompt_text = skill_block + prompt_text

        # Tokenize the teacher prompt
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # Truncate if needed (left truncation to keep the end of prompt)
        if len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        # Pad to max_prompt_length (left padding)
        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        # Combine with response
        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    teacher_input_ids = torch.stack(teacher_input_ids_list)
    teacher_attention_mask = torch.stack(teacher_attention_mask_list)
    teacher_position_ids = torch.stack(teacher_position_ids_list)

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "responses": batch.batch["responses"],
        },
    )

    return teacher_batch


class OPSDRayTrainer(RayPPOTrainer):
    """
    OPSD trainer that extends RayPPOTrainer with Bayesian value-recursion
    token-level credit assignment using privileged skill information as the
    teacher signal.
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider
        # OPSD hyperparams from config
        opsd_cfg = self.config.algorithm.get("opsd", {})
        self.opsd_v0_prior = opsd_cfg.get("v0_prior", 0.5)
        self.opsd_skill_position = opsd_cfg.get("skill_position", "legacy")
        # Credit-assignment granularity: "token" (default, baseline) or "turn"
        # (step-level Bayesian belief recursion across an episode's turns).
        self.opsd_granularity = opsd_cfg.get("granularity", "token")
        # Paper Eq.(20)(21) stabilizers as independent ablation toggles.
        self.opsd_use_anchoring = opsd_cfg.get("use_anchoring", False)
        self.opsd_use_group_norm = opsd_cfg.get("use_group_norm", False)
        # Diagnostic fixes for the E[delta] = -KL negative-drift saturation.
        self.opsd_center_delta = opsd_cfg.get("center_delta", False)
        # Geometric decay of accumulated evidence by turn/token distance
        # (recency-weighted belief). 1.0 = plain cumsum (no decay, current SABRE).
        self.opsd_belief_decay_gamma = opsd_cfg.get("belief_decay_gamma", 1.0)
        self.opsd_tether_seq = opsd_cfg.get("tether_seq", False)
        self.opsd_tether_lambda = opsd_cfg.get("tether_lambda", 0.5)
        self.opsd_tether_band = opsd_cfg.get("tether_band", 0.2)
        # belief_mult route: RLSD-style multiplicative credit from the accumulated
        # belief-increment MAGNITUDE, with an RLSD-style linear lambda decay so the
        # extra signal fades to pure GRPO by warmdown_steps (fair-comparison knob).
        self.opsd_belief_mult = opsd_cfg.get("belief_mult", False)
        self.opsd_mult_lambda_init = opsd_cfg.get("mult_lambda", 1.0)
        self.opsd_mult_band = opsd_cfg.get("mult_band", 0.2)
        # warmdown_steps: linearly decay mult_lambda from its init to 0 over this
        # many steps. -1 (default) => no decay (constant lambda), mirroring how the
        # tether route holds a fixed lambda.
        self.opsd_mult_warmdown_steps = opsd_cfg.get("warmdown_steps", -1)
        # belief_mult ablation toggles (default True => the full SABRE behavior):
        #   belief_gate=False => drop the implicit V(1-V) saturation gate (use raw
        #     evidence |delta| for the multiplier instead of |V_k - V_{k-1}|).
        #   mult_signed=False => drop the verifier's signed direction (magnitude only).
        self.opsd_belief_gate = opsd_cfg.get("belief_gate", True)
        self.opsd_mult_signed = opsd_cfg.get("mult_signed", True)
        #   signed=True => (paper Eq.6) build m_k from the z-scored SIGNED belief
        #     increment steered by sign(A^{(i)}) (outcome-consistency), instead of the
        #     magnitude-only |V_k - V_{k-1}| z-score. Orthogonal to mult_signed.
        self.opsd_signed = opsd_cfg.get("signed", False)

    def _get_opsd_mult_lambda(self, step: int) -> float:
        """Linearly decay belief_mult lambda from mult_lambda_init to 0 over
        warmdown_steps. warmdown_steps <= 0 disables decay (constant lambda).
        Mirrors RLSDRayTrainer._get_rlsd_lambda so the two are comparable."""
        w = int(self.opsd_mult_warmdown_steps)
        if w <= 0:
            return float(self.opsd_mult_lambda_init)
        if step >= w:
            return 0.0
        return float(self.opsd_mult_lambda_init) * (1.0 - step / w)

    def fit(self):
        """
        The training loop of OPSD, extending the standard PPO/GRPO loop
        with teacher forward pass and token-level advantage computation.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="OPSD Training")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                        )

                    del batch
                    batch = gen_batch_output

                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Compute student log probs (old_log_probs)
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # ---- OPSD: Teacher forward pass ----
                    with _timer("teacher_forward", timing_raw):
                        teacher_log_probs = self._compute_teacher_log_probs(batch)
                        batch.batch["teacher_log_probs"] = teacher_log_probs

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute standard GRPO sequence-level advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                            step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                            gigpo_mode=self.config.algorithm.gigpo.mode,
                            gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                            gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                        )

                        # ---- OPSD: Replace sequence-level advantage with Bayesian
                        # value-recursion token-level advantage ----
                        seq_advantages = batch.batch["advantages"]
                        student_log_probs = batch.batch["old_log_probs"]
                        teacher_log_probs = batch.batch["teacher_log_probs"]
                        response_mask = batch.batch["response_mask"]
                        token_level_rewards = batch.batch["token_level_rewards"]

                        # Resolve V_0. Prefer the GRPO group mean of R (same grouping
                        # GRPO already uses: uid, dedup steps by traj_uid); fall back to
                        # the config prior if the grouping keys are unavailable.
                        v0_per_traj = None
                        if "uid" in batch.non_tensor_batch and "traj_uid" in batch.non_tensor_batch:
                            v0_per_traj = compute_group_mean_v0(
                                token_level_rewards=token_level_rewards,
                                response_mask=response_mask,
                                index=batch.non_tensor_batch["uid"],
                                traj_index=batch.non_tensor_batch["traj_uid"],
                            )

                        # Per-row scalar GRPO sequence advantage (for direction
                        # anchoring): the broadcast seq advantage is identical across
                        # a row's valid tokens, so take the first valid position.
                        first_valid = response_mask.long().argmax(dim=-1)  # (bs,)
                        row_idx = torch.arange(seq_advantages.size(0), device=seq_advantages.device)
                        seq_advantage_per_row = seq_advantages[row_idx, first_valid]  # (bs,)

                        uid_arr = batch.non_tensor_batch.get("uid", None)

                        # belief_mult lambda for this step (RLSD-style linear decay).
                        current_mult_lambda = self._get_opsd_mult_lambda(self.global_steps)

                        # ---- OPSD: choose credit-assignment granularity ----
                        if self.opsd_granularity == "turn":
                            opsd_advantages = compute_opsd_turn_advantage(
                                token_level_rewards=token_level_rewards,
                                student_log_probs=student_log_probs,
                                teacher_log_probs=teacher_log_probs,
                                response_mask=response_mask,
                                traj_uid=batch.non_tensor_batch["traj_uid"],
                                turn_step=batch.non_tensor_batch["turn_step"],
                                episode_rewards=batch.non_tensor_batch.get("episode_rewards", None),
                                uid=uid_arr,
                                v0_prior=self.opsd_v0_prior,
                                v0_per_traj=v0_per_traj,
                                seq_advantage_per_row=seq_advantage_per_row,
                                use_anchoring=self.opsd_use_anchoring,
                                use_group_norm=self.opsd_use_group_norm,
                                center_delta=self.opsd_center_delta,
                                belief_decay_gamma=self.opsd_belief_decay_gamma,
                                tether_seq=self.opsd_tether_seq,
                                tether_lambda=self.opsd_tether_lambda,
                                tether_band=self.opsd_tether_band,
                                belief_mult=self.opsd_belief_mult,
                                mult_lambda=current_mult_lambda,
                                mult_band=self.opsd_mult_band,
                                belief_gate=self.opsd_belief_gate,
                                mult_signed=self.opsd_mult_signed,
                                signed=self.opsd_signed,
                            )
                        else:
                            opsd_advantages = compute_opsd_token_advantage(
                                token_level_rewards=token_level_rewards,
                                student_log_probs=student_log_probs,
                                teacher_log_probs=teacher_log_probs,
                                response_mask=response_mask,
                                v0_prior=self.opsd_v0_prior,
                                v0_per_traj=v0_per_traj,
                                seq_advantage_per_row=seq_advantage_per_row,
                                uid=uid_arr,
                                use_anchoring=self.opsd_use_anchoring,
                                use_group_norm=self.opsd_use_group_norm,
                                center_delta=self.opsd_center_delta,
                                belief_decay_gamma=self.opsd_belief_decay_gamma,
                                tether_seq=self.opsd_tether_seq,
                                tether_lambda=self.opsd_tether_lambda,
                                tether_band=self.opsd_tether_band,
                                belief_mult=self.opsd_belief_mult,
                                mult_lambda=current_mult_lambda,
                                mult_band=self.opsd_mult_band,
                                belief_gate=self.opsd_belief_gate,
                                mult_signed=self.opsd_mult_signed,
                                signed=self.opsd_signed,
                            )

                        batch.batch["advantages"] = opsd_advantages

                        # Log OPSD-specific metrics
                        metrics["opsd/granularity"] = 1.0 if self.opsd_granularity == "turn" else 0.0
                        metrics["opsd/use_anchoring"] = 1.0 if self.opsd_use_anchoring else 0.0
                        metrics["opsd/use_group_norm"] = 1.0 if self.opsd_use_group_norm else 0.0
                        metrics["opsd/center_delta"] = 1.0 if self.opsd_center_delta else 0.0
                        metrics["opsd/belief_decay_gamma"] = float(self.opsd_belief_decay_gamma)
                        metrics["opsd/tether_seq"] = 1.0 if self.opsd_tether_seq else 0.0
                        metrics["opsd/belief_mult"] = 1.0 if self.opsd_belief_mult else 0.0
                        if self.opsd_belief_mult:
                            metrics["opsd/mult_lambda"] = float(current_mult_lambda)
                            metrics["opsd/mult_band"] = float(self.opsd_mult_band)
                            metrics["opsd/belief_gate"] = 1.0 if self.opsd_belief_gate else 0.0
                            metrics["opsd/mult_signed"] = 1.0 if self.opsd_mult_signed else 0.0
                            metrics["opsd/signed"] = 1.0 if self.opsd_signed else 0.0
                        if self.opsd_granularity == "turn" and "traj_uid" in batch.non_tensor_batch:
                            traj_uids_arr = batch.non_tensor_batch["traj_uid"]
                            n_turns = len(traj_uids_arr)
                            n_eps = len(set(traj_uids_arr.tolist())) if n_turns > 0 else 0
                            metrics["opsd/mean_turns_per_episode"] = (n_turns / n_eps) if n_eps > 0 else 0.0
                        delta_t = (teacher_log_probs - student_log_probs) * response_mask
                        metrics["opsd/teacher_student_gap_mean"] = masked_mean(delta_t, response_mask).item()
                        metrics["opsd/teacher_student_gap_std"] = masked_mean(delta_t ** 2, response_mask).sqrt().item()
                        if v0_per_traj is not None:
                            metrics["opsd/v0_mean"] = v0_per_traj.float().mean().item()
                        else:
                            metrics["opsd/v0_mean"] = float(self.opsd_v0_prior)

                        # Save per-token teacher/student log-prob gap if SAVE_CGTD_DEBUG=1, at test_freq interval
                        if os.environ.get("SAVE_CGTD_DEBUG", "0") == "1" and \
                                self.config.trainer.test_freq > 0 and \
                                self.global_steps % self.config.trainer.test_freq == 0:
                            save_dir = os.environ.get("SAVE_CGTD_DEBUG_DIR", "outputs/opsd_debug")
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"step_{self.global_steps}.jsonl")

                            bs = response_mask.shape[0]
                            turn_steps = batch.non_tensor_batch.get("turn_step", np.zeros(bs, dtype=object))
                            traj_uids = batch.non_tensor_batch.get("traj_uid", np.array([""] * bs, dtype=object))
                            episode_rewards_arr = batch.non_tensor_batch.get("episode_rewards", np.zeros(bs, dtype=object))
                            episode_lengths_arr = batch.non_tensor_batch.get("episode_lengths", np.zeros(bs, dtype=object))
                            response_ids = batch.batch["responses"]
                            token_advs = batch.batch["advantages"]

                            with open(save_path, "w") as f:
                                for i in range(bs):
                                    mask_i = response_mask[i].bool()
                                    valid_count = mask_i.sum().item()
                                    if valid_count == 0:
                                        continue
                                    token_ids_i = response_ids[i][mask_i].cpu().tolist()
                                    tokens_i = [self.tokenizer.decode([tid]) for tid in token_ids_i]
                                    gaps_i = delta_t[i][mask_i].cpu().tolist()
                                    teacher_lps_i = teacher_log_probs[i][mask_i].cpu().tolist()
                                    student_lps_i = student_log_probs[i][mask_i].cpu().tolist()
                                    token_advs_i = token_advs[i][mask_i].cpu().tolist()

                                    record = {
                                        "global_step": self.global_steps,
                                        "sample_idx": i,
                                        "turn_step": int(turn_steps[i]),
                                        "traj_uid": str(traj_uids[i]),
                                        "episode_reward": float(episode_rewards_arr[i]),
                                        "episode_length": float(episode_lengths_arr[i]),
                                        "seq_advantage": float(seq_advantages[i][mask_i][0].item()),
                                        "tokens": tokens_i,
                                        "token_ids": token_ids_i,
                                        "gaps": gaps_i,
                                        "teacher_log_probs": teacher_lps_i,
                                        "student_log_probs": student_lps_i,
                                        "token_advantages": token_advs_i,
                                    }
                                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                            print(f"[OPSD Debug] Saved per-token gap data to {save_path} ({bs} samples)")

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_outpu