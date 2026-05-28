# Orbit Wars PPO Self-Play (1v1)

This folder contains a minimal PPO training framework that uses the official
kaggle_environments Orbit Wars engine for rules and simulation.

Key points:
- 1v1 self-play with an opponent snapshot pool
- Legal action filtering before sending moves to the engine
- Multi-environment batched rollouts for higher throughput
- Deeper MLP policy/value network
- Discrete action space: per-source multi-launch sequences (stop or target+fraction)

Conda virtual environment: 
```bash
conda create -n OrbitWars python=3.11 -y
conda activate OrbitWars
```

Install dependencies:

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision --upgrade
# 将 .../whl/cu121换成对应的驱动版本，例如.../whl/cu130
pip install numpy tensorboard numba pyyaml "requests >= 2.25.1" "jsonschema>=3.0.1"
pip install -U setuptools
```

Train:
```bash
python -m rl.train --config configs/default.yaml --device cuda --resume checkpoints/ppo_orbit_wars_150.pt
```

TensorBoard:

```bash
tensorboard --logdir <logdir>
```

Multi-env + multi-launch example:

```bash
python -m rl.train --num-envs 8 --max-launches-per-source 6
```
