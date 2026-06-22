# Orbit Wars — PPO Agent Submission

## 文件结构

```
submission/
├── main.py          # entry point — 包含 agent(obs) 函数
├── model.pt         # ← 你的训练好的 checkpoint（需要你自己放入）
├── config.yaml      # 与 checkpoint 匹配的配置
├── rl/              # 推理所需的 RL 模块
│   ├── __init__.py
│   ├── config.py    # 配置 dataclass
│   ├── models.py    # ActorCritic / ActorCriticGNN
│   ├── obs.py       # encode_observation
│   └── action.py    # ActionBuilder, sample_action_discrete
└── README.md        # 本文件
```

## 提交前准备

### 1. 复制模型 checkpoint

```bash
# 从 checkpoints/ 目录复制你想要提交的模型
cp checkpoints/ppo_orbit_wars_1500.pt submission/model.pt
```

### 2. 确保 config.yaml 与 checkpoint 匹配

`config.yaml` 中的 `model.model_type`、`action.max_sources`、`action.max_targets`、`action.offset_bins` 等必须与**训练该 checkpoint 时的配置一致**。

如果 checkpoint 是用其他 config 训练的：

```bash
# 例如：用 default.yaml 训练的 MLP 模型
cp configs/default.yaml submission/config.yaml
```

### 3. 打包提交

```bash
cd submission
tar -czf ../submission.tar.gz .
cd ..
kaggle competitions submit orbit-wars -f submission.tar.gz -m "PPO agent v1"
```

## 容错设计

如果 `model.pt` 加载失败（文件缺失、shape mismatch 等），`agent()` 会自动降级为 **NearestPlanet** 策略，确保提交不会因环境差异而崩溃。

错误信息会打印到 stderr，可通过 `kaggle competitions logs <EPISODE_ID> 0` 查看。

## 单文件快速测试

```bash
# 本地测试（单文件 main.py）
cd submission
python -c "
from kaggle_environments import make
env = make('orbit_wars', configuration={'seed': 42}, debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"
```

## 注意事项

- 配置文件里 `model.model_type` 决定用 `ActorCritic`（mlp）还是 `ActorCriticGNN`（gnn）
- `action.max_sources` / `action.max_targets` / `action.offset_bins` 改变模型的输出维度，必须与 checkpoint 完全一致
- Kaggle 评估环境**只有 CPU**，模型会自动加载到 CPU
- 如果模型不加载，终端会输出 WARNING，agent 会自动使用 NearestPlanet 保底
