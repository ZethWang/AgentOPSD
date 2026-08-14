<h1 align="center">
AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning
</h1>

> 🚧 **Code coming soon.** The full training code and scripts will be released here shortly. Star / watch this repo to get notified.

## Overview

**AgentOPSD** is a **critic-free, recursive turn-level credit assignment** method for agentic
reinforcement learning. In long-horizon multi-turn tasks, standard RL with verifiable rewards only
constructs a trajectory-level advantage and struggles to credit the few *pivotal* decisions that
drive the outcome.

AgentOPSD aggregates token-level teacher–student log-probability **gaps** into a turn-level gap,
then recursively updates a Bayesian belief state in log-odds space across the episode's turns,
identifying pivotal turns by the marginal revision between consecutive belief states. It transforms
sparse outcome supervision into dense turn-level credit, is fully compatible with standard policy
optimization (e.g. GRPO), and requires **no value network and no extra rollouts**.

We evaluate AgentOPSD on **ALFWorld**, **WebShop**, and **Search-QA** with Qwen2.5 (3B/7B),
where it improves over GRPO and strong self-distillation baselines.

## Citation


```bibtex
@article{wang2026agentopsd,
  title={AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning},
  author={Wang, Zi-Han and Lu, Zhengxi and Yao, Zhiyuan and Wu, Jinyang and Wu, Jie and Cai, Zhengzhou and Sun, Yueqing and Ye, Ziang and Hao, Linji and Gu, Qi and others},
  journal={arXiv preprint arXiv:2608.05987},
  year={2026}
}
```
