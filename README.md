<h1 align="center">
AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning
</h1>
<div align='center' style="font-size:18px;">
<p>
    <a href="https://arxiv.org/abs/2608.05987">
      <img src="https://img.shields.io/badge/Paper-arXiv%3A2608.05987-blue" alt="Paper"/>
    </a>
    <a href="https://huggingface.co/papers/2608.05987">
      <img src="https://img.shields.io/badge/Daily%20Paper-huggingface-yellow" alt="HF Paper"/>
    </a>
    <a href="https://github.com/ZethWang/AgentOPSD">
      <img src="https://img.shields.io/badge/Code-GitHub-black" alt="Code"/>
    </a>
  </p>
</div>

## 🔥 Overview

**AgentOPSD** is a **critic-free, recursive turn-level credit assignment** method for agentic
reinforcement learning. In long-horizon multi-turn tasks, standard RL with verifiable rewards
only constructs a trajectory-level advantage and struggles to credit the few *pivotal* decisions
that actually drive the outcome.

AgentOPSD turns sparse outcome supervision into dense, turn-level credit:

1. It computes a **token-level teacher–student log-probability gap** using privileged
   self-distillation (the same self-oracle signal used by SDAR/RLSD).
2. It **aggregates these token-level gaps into a turn-level gap** for each interaction turn.
3. It **recursively updates a Bayesian belief state in log-odds space** across the episode's
   turns, and identifies pivotal turns by the marginal revision between consecutive belief states.

The resulting reweighting is fully compatible with standard policy optimization (e.g. GRPO) and
requires **no value network and no extra rollouts**.

<!-- TODO: add teaser + method figures at docs/agentopsd/teaser.png and docs/agentopsd/method.png -->

## 📖 Results

AgentOPSD improves over GRPO and strong self-distillation baselines on **ALFWorld**, **WebShop**,
and **Search-QA** with Qwen2.5 (3B / 7B).

<!-- TODO: numbers below are provisional — confirm against the final paper tables before release. -->

| Method | ALFWorld (7B) | WebShop (3B) | Search-QA (3B) |
|---|---|---|---|
| GRPO (baseline) | TODO | TODO | TODO |
| AgentOPSD (ours) | **89.1** | TODO | TODO |

<!-- TODO: add the summary metric figure at docs/agentopsd/metric.png -->

## 🛠️ Installation

### Python environment

```bash
conda create -n agentopsd python==3.12 -y
conda activate agentopsd

pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

Log in to Weights & Biases if you use WandB logging (scripts pass `trainer.logger=['console','wandb']`):

```bash
export WANDB_API_KEY=your_key_here
```

### Install Supported Environments

#### 1. ALFWorld
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip3 install alfworld
alfworld-download -f   # PDDL & game files + MaskRCNN detector, stored in ~/.cache/alfworld/
```

#### 2. WebShop
WebShop requires Python <=3.10, so create a separate environment:
```bash
conda create -n verl-webshop python==3.10 -y
conda activate verl-webshop

cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all

cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
```
The `typer` version warnings can be safely ignored.

#### 3. Search
```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2

cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py   # -> ~/data/searchR1_processed_direct
```

Set up a separate retriever environment (faiss-gpu is not available via pip). The retrieval server
uses ~6GB GPU memory per GPU:
```bash
conda create -n retriever python=3.10 -y
conda activate retriever
conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets pyserini huggingface_hub
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y
pip install uvicorn fastapi
```

Download the index and start the retrieval server:
```bash
conda activate retriever
local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz

bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log
```

## 🚀 Training

All scripts live under `examples/` and assume the repo root as the working directory.
AgentOPSD scripts are in `examples/agentopsd_trainer/`:

```bash
bash examples/agentopsd_trainer/run_alfworld_3b.sh
bash examples/agentopsd_trainer/run_alfworld_7b.sh
bash examples/agentopsd_trainer/run_search_3b.sh
bash examples/agentopsd_trainer/run_webshop_3b.sh
```

Each script runs the full method (`belief_mult` + `signed`). Remove the
`+algorithm.opsd.signed=true` line to run the unsigned variant. Key knobs are exposed at the top
of every script (`v0_prior`, `mult_lambda`, `granularity`, `skill_all`).

**Naming note.** The public/paper name is **AgentOPSD**; the implementation is internally named
`opsd` — the entry point is `verl.trainer.main_opsd`, the trainer is `OPSDRayTrainer`, and the
config namespace is `algorithm.opsd.*`. To re-brand run/experiment names in one place, set:

```bash
export AGENTOPSD_METHOD_NAME=AgentOPSD   # used for experiment_name prefixes and log banners
```

Baselines used in the paper (GRPO, Skill-GRPO, OPSD, GRPO+OPSD, Skill-SD, RLSD, and the SDAR
method) are also provided under `examples/` for reproduction.

### Merge checkpoints
See `scripts/model_merger.py` for FSDP/Megatron merge examples using paths under `./checkpoints/...`.

## ⭐️ Citation

If you find this project useful, please consider citing us:

```bibtex
@article{wang2026agentopsd,
  title={AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning},
  author={Wang, Zi-Han and Lu, Zhengxi and Yao, Zhiyuan and Wu, Jinyang and Wu, Jie and Cai, Zhengzhou and Sun, Yueqing and Ye, Ziang and Hao, Linji and Gu, Qi and others},
  journal={arXiv preprint arXiv:2608.05987},
  year={2026}
}
```

## 🤝 Acknowledgement

AgentOPSD is built on top of [SDAR](https://github.com/ZJU-REAL/SDAR),
[verl-agent](https://github.com/langfengQ/verl-agent), and [veRL](https://github.com/volcengine/verl),
and uses the [ALFWorld](https://github.com/alfworld/alfworld), WebShop, and
[Search-R1](https://github.com/PeterGriffinJin/Search-R1) environments. We thank the authors of
those projects.
