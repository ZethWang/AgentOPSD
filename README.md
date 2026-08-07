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
@misc{wang2026agentopsdrecursiveselfdistillationagentic,
      title={AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning}, 
      author={Zi-Han Wang and Zhengxi Lu and Zhiyuan Yao and Jinyang Wu and Jie Wu and Zhengzhou Cai and Yueqing Sun and Ziang Ye and Linji Hao and Qi Gu and Xunliang Cai and Yongliang Shen and Yujiu Yang},
      year={2026},
      eprint={2608.05987},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2608.05987}, 
}
```
