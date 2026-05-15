# Orbit Wars PPO Self-Play (1v1)

This folder contains a minimal PPO training framework that uses the official
kaggle_environments Orbit Wars engine for rules and simulation.

Key points:
- 1v1 self-play with an opponent snapshot pool
- Legal action filtering before sending moves to the engine
- Multi-environment batched rollouts for higher throughput
- Deeper MLP policy/value network
- Discrete action space: per-source multi-launch sequences (stop or target+fraction)

Conda: 
```bash
conda create -n OrbitWars python=3.11
```

Install dependencies:

```bash
pip install "kaggle-environments>=1.28.0" torch numpy tensorboard
```

TensorBoard:

```bash
python -m rl.train --log-dir runs/orbit_wars
tensorboard --logdir runs/orbit_wars
```

Multi-env + multi-launch example:

```bash
python -m rl.train --num-envs 8 --max-launches-per-source 6
```
