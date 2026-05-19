# Orbit-Wars 改进记录

本次对项目进行了 **1 个逻辑 Bug 修复** 和 **5 项 GPU 加速优化**，涉及 6 个文件的修改。

---

## 1. 🔴 严重 Bug 修复：目标优先级排序错误

**文件**: [rl/action.py](rl/action.py#L47-L50)

**问题**: `ActionBuilder.build()` 中目标星球按 `(is_own, dist)` 升序排列。`is_own=1`（己方）排在 `is_own=0`（敌方/中立）之后，而 `max_targets=8` 限制了可选目标数量，导致**己方星球永远无法被选为目标**——Agent 完全无法学会增援己方星球。

**修复**:

```python
# 修复前
return (is_own, dist)

# 修复后
return (-is_own, dist)  # 负值使己方星球排在最前
```

使得 `(-1, dist) < (0, dist)`，己方星球优先出现，可按需增援。

---

## 2. 🟢 混合精度训练 (AMP)

**文件**: [rl/ppo.py](rl/ppo.py#L84-L168), [rl/train.py](rl/train.py#L55)

**问题**: 全部使用 FP32 精度，未利用现代 GPU 的 Tensor Core 加速能力。

**修复**:

- `PPOTrainer.__init__` 新增 `use_amp` 参数和 `GradScaler`
- 模型 forward 使用 `torch.cuda.amp.autocast()` 包裹，自动混合 FP16/FP32
- 反向传播使用 `scaler.scale(loss).backward()` + `scaler.unscale_()` + `scaler.step()` + `scaler.update()`
- `train.py` 中自动检测 CUDA 设备并启用 AMP

**预期效果**: 训练速度提升 **1.5–2 倍**，显存占用降低约 30%。

---

## 3. 🟢 CuDNN Benchmark 自动调优

**文件**: [rl/train.py](rl/train.py#L57-L58)

**问题**: 观测维度固定（1110），网络结构固定，但未开启 cuDNN 自动算法搜索。

**修复**:

```python
if device == "cuda":
    torch.backends.cudnn.benchmark = True
```

让 cuDNN 在首次运行时自动寻找最优卷积/矩阵运算算法。

**预期效果**: 额外 **10–30%** 加速。

---

## 4. 🟡 对手批量 GPU 推理

**文件**: [rl/opponents.py](rl/opponents.py#L89-L147), [rl/envs/orbit_wars_env.py](rl/envs/orbit_wars_env.py#L61-L77), [rl/envs/orbit_wars_env.py](rl/envs/orbit_wars_env.py#L103-L127), [rl/train.py](rl/train.py#L140-L167)

**问题**: 原来每个环境的对手推理调用 `PolicyOpponent.act()` 是**串行单样本 GPU 推理**——每步每个环境 × 每个对手，各自独立跑一次 GPU forward，GPU 利用率极低。

**修复**:

| 改动 | 说明 |
|------|------|
| `PolicyOpponent.batch_act()` | 新增静态方法，按 `id(policy)` 分组，同一 checkpoint 的对手合并为一个 batch 前向推理 |
| `OrbitWarsSelfPlayEnv.get_opponents_data()` | 新增方法，暴露 `(opp_idx, opponent, obs)` 元组供批量推理收集 |
| `OrbitWarsSelfPlayEnv.step(opponent_actions=...)` | `step()` 新增可选参数，传入预计算的对手动作则跳过 `opponent.act()` 调用 |
| `train.py` 训练循环 | 每步先收集所有环境的对手观测，PolicyOpponent 批量 GPU 推理，非策略对手（NearestPlanet/Random）原地 CPU 执行 |

数据流变为：

```
收集所有对手观测 → 按 policy_id 分组 → 每组一次 GPU batch forward → 分发结果到各 env.step()
```

**预期效果**: 对手推理时间减少 **80%+**（多个环境共享同一个 GPU batch）。

---

## 5. 🟡 对手池策略网络复用

**文件**: [rl/opponents.py](rl/opponents.py#L150-L183)

**问题**: `OpponentPool.sample()` 每次调用都 `policy_factory().to(device)` 创建新网络 + 加载权重后移到 GPU，旧网络被 Python GC 回收。训练 2000 个 update × 每 10 步刷新 = **大量 GPU 显存分配/释放抖动和 CPU→GPU 拷贝开销**。

**修复**:

- `add()`: 在将 state_dict 存入快照列表时，同步创建并预加载一个策略实例到 `_policy_instances` 池
- `sample()`: 从 `_policy_instances` 池随机选取已就位的策略实例，无需创建新网络
- 池满时（capacity=5），淘汰最旧快照的同时释放对应 GPU 策略实例

**预期效果**: 消除训练过程中的 GPU 显存分配抖动，避免重复 `policy_factory().to(device)` 开销。

---

## 6. 🔵 Step 归一化硬编码修复

**文件**: [rl/obs.py](rl/obs.py#L44-L50), [rl/obs.py](rl/obs.py#L117)

**问题**: `encode_observation()` 中 `step / 500.0` 硬编码了 episode_steps 值，若修改配置则不准确。

**修复**:

- `encode_observation()` 新增 `episode_steps: int = 500` 参数
- 归一化改为 `step / max(episode_steps, 1)`
- `train.py` 调用处传入 `config.env.episode_steps`

---

## 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| [rl/action.py](rl/action.py#L50) | 目标排序 `return (-is_own, dist)` |
| [rl/obs.py](rl/obs.py#L44-L50) | `encode_observation` 新增 `episode_steps` 参数 |
| [rl/ppo.py](rl/ppo.py#L84-L168) | PPOTrainer 新增 `use_amp` + GradScaler + autocast |
| [rl/opponents.py](rl/opponents.py#L89-L183) | PolicyOpponent 新增 `batch_act()`；OpponentPool 预创建策略实例池 |
| [rl/envs/orbit_wars_env.py](rl/envs/orbit_wars_env.py#L61-L77) | 新增 `get_opponents_data()`；`step()` 新增 `opponent_actions` 参数 |
| [rl/train.py](rl/train.py#L55-L58) | cudnn benchmark + AMP 启用 + batch 对手推理 + episode_steps 传参 |

---

## 预期总加速效果

| 优化项 | 加速幅度 | 类型 |
|--------|---------|------|
| CuDNN benchmark | 10–30% | 训练 + 推理 |
| AMP 混合精度 | 1.5–2× | 训练 |
| 对手批量推理 | 80%+ 对手推理耗时降低 | 推理（rollout 阶段） |
| 对手池策略复用 | 消除显存抖动 | 训练全程 |

综合来看，在 GPU 环境下训练，整体 **每 step 耗时预计降低 30–50%**，单次 PPO update 的训练时间预计降低 **40–60%**。
