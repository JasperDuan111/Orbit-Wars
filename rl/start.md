# Orbit Wars PPO Self-Play — 代码框架与开发指南

## 1. 项目概览

这是一个基于 PPO (Proximal Policy Optimization) 的 1v1 自对弈训练框架，使用 Kaggle 官方 `kaggle_environments` 作为游戏引擎。核心思路：训练一个 Actor-Critic 网络，通过与历史版本的自己（Opponent Pool）对弈来持续变强。

**关键设计决策：**
- 离散动作空间：每个己方星球独立输出一串发射指令（停止 / 向某目标发送某比例的飞船）
- 固定维度观测编码：48 星球 × 10 特征 + 64 舰队 × 7 特征 + 6 全局特征 = 964 维向量
- GAE 优势估计 + PPO clip loss + 价值函数裁剪
- 飞船数量差作为稠密 reward（差分），终局附加终值 reward

---

## 2. 文件结构与职责

```
rl/
├── __init__.py              # 包声明
├── config.py                # 配置分组与默认配置（OrbitWarsConfig 等）
├── models.py                # Actor-Critic 网络定义
├── action.py                # 动作空间：构建合法动作模板、编解码、采样/对数概率
├── obs.py                   # 观测编码：原始 dict → 固定维度 numpy 向量
├── ppo.py                   # RolloutBuffer（经验存储）+ PPOTrainer（更新逻辑）
├── train.py                 # 主训练入口：多环境 rollout + 自对弈循环
├── opponents.py             # 对手策略：最近星球 / 随机 / 策略网络 + 对手池
├── envs/
│   ├── __init__.py
│   └── orbit_wars_env.py    # Gym 风格环境封装（self-play + 计分 + 动作校验）
└── README.md                # 简要使用说明
```

---

## 3. 各模块详细说明

### 3.1 `config.py` — 配置

配置被拆成多个 dataclass，并通过 `OrbitWarsConfig` 统一管理，便于扩展与复用。

**核心结构：**
- `GameConfig`：地图与中心点等几何常量
- `ObsConfig`：观测规模（max_planets / max_fleets / feature dims）
- `ActionSpaceConfig`：动作规模（max_sources / max_targets / ship_fractions / max_launches_per_source）
- `ModelConfig`：网络结构（hidden_sizes / dropout）
- `TrainConfig`：PPO 训练超参数
- `RewardConfig`：reward 各项缩放
- `EnvConfig`：环境运行参数（num_players / episode_steps / act_timeout / seed / debug）

`OrbitWarsConfig` 聚合上述配置，并提供：
- `obs_dim`：由 `ObsConfig` 自动计算
- `actions_per_source`：由 `ActionSpaceConfig` 自动计算

为兼容旧用法，仍保留 `MAX_PLANETS`、`OBS_DIM` 等常量，值来自 `DEFAULT_CONFIG`。

**TrainConfig 常用字段：**

| 参数 | 默认值 | 含义 |
|---|---|---|
| `seed` | 42 | 随机种子 |
| `num_envs` | 2 | 并行环境数（同时跑几局） |
| `total_updates` | 2000 | 总 PPO 更新轮数 |
| `rollout_steps` | 64 | 每轮收集多少步经验 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ 参数 |
| `clip_range` | 0.2 | PPO clip 范围 |
| `learning_rate` | 3e-4 | Adam 学习率 |
| `ent_coef` | 0.01 | 熵正则系数 |
| `vf_coef` | 0.5 | 价值损失权重 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `batch_size` | 64 | 每轮 PPO 更新的 mini-batch 大小 |
| `epochs` | 4 | 每轮数据重复训练几次 |
| `save_every` | 50 | 每 N 轮保存一次 checkpoint |
| `opponent_refresh` | 10 | 每 N 轮重新采样对手 |

**RewardConfig 常用字段：**

| 参数 | 默认值 | 含义 |
|---|---|---|
| `reward_scale` | 0.01 | 稠密 reward 缩放 |
| `invalid_action_penalty` | 0.05 | 非法动作惩罚 |
| `terminal_reward_scale` | 0.01 | 终局 reward 缩放 |
| `planet_control_scale` | 0.0 | 星球控制奖励 |
| `production_scale` | 0.0 | 产量奖励 |
| `survival_reward` | 0.0 | 生存奖励 |

---

### 3.2 `models.py` — 神经网络

**`_build_mlp(input_dim, hidden_sizes, output_dim, dropout)`**
- 构建一个 MLP 序列：Linear → LayerNorm → GELU → Dropout（循环每层），最后 Linear 输出。
- 每层 hidden_size 后用 LayerNorm + GELU + Dropout。

**`class ActorCritic(nn.Module)`**
- 输入：观测向量 `(batch, 964)`
- Body：3 层 MLP (964 → 512 → 512 → 256)，输出 256 维特征
- 策略头 `policy_head`：`Linear(256, 8×33)`，reshape 为 `(batch, 8, 33)` → 每个源星球的 33 维 logits
- 价值头 `value_head`：`Linear(256, 1)` → 状态价值
- `forward(obs)` 返回 `(logits, value)`

---

### 3.3 `action.py` — 动作空间

这是整个框架最核心的模块，负责将神经网络的离散索引映射为实际的游戏指令，以及反过来。

**辅助函数：**
- `_get_field(obs, name, default)` — 兼容 dict/object 两种观测格式
- `_parse_planets(obs)` — 将原始观测中的 planets 列表解析为 `Planet` namedtuple 列表

**`class ActionBuilder`** — 动作模板构建与解码器

- `build(obs)` → `(actions, source_ships)`
  - 输入原始观测，返回 **合法动作模板列表** 和 **各源星球飞船数**。
  - 流程：
    1. 取出己方星球，按飞船数降序排序
    2. 对前 `ActionSpaceConfig.max_sources`（默认 8）个己方星球，生成 `actions_per_source` 维动作槽位：
       - 槽位 0：停止发射（恒为合法）
       - 槽位 1..：对排序后的前 `ActionSpaceConfig.max_targets` 个目标星球 × `ship_fractions` 组合（目标排序：己方优先 → 距离近优先）
    3. 每个槽位存储一个 `ActionTemplate(source_id, angle, fraction)`
  - 返回 `actions` (`max_sources` 行 × `actions_per_source` 列的模板矩阵) 和 `source_ships` (每个源星球的飞船数)

- `decode(action_indices, actions, source_ships, max_launches)` → `moves`
  - 将模型输出的动作索引转换为 Kaggle 引擎需要的 `[[from_id, angle, ships], ...]` 格式。
  - 按序解码每个源星球的发射序列，从飞船余额中扣除，余额不足时停止。
  - 遇到 action_idx=0（停止）时中断当前源星球的后续发射。

**`class ActionTemplate`** (frozen dataclass)
- 不可变数据类，存储 `source_id`, `angle`, `fraction`

**`_ships_to_send(remaining_ships, fraction)`**
- 根据剩余飞船数和比例计算发送数量，最少 1 艘，最多全部剩余。

**`_mask_tensor(source_actions, remaining_ships, device)` → `Tensor`**
- 为给定源星球生成 `actions_per_source` 维合法动作 mask（0=非法, 1=合法）。
- 槽位 0（停止）始终合法；其余槽位根据余额是否够发 1 艘判断合法性。

**`sample_action_sequence(logits, actions, source_ships, max_launches, deterministic)`**
- 从 logits 中自回归采样动作序列。每个源星球逐次采样，每次用 mask 屏蔽非法动作，通过 `Categorical` 分布采样。
- `deterministic=True` 时取 argmax。
- 返回 `(action_indices, logprob_sum, entropy_sum)`

**`logprob_for_action_sequence(logits, actions, source_ships, action_indices, max_launches)`**
- 给定已采样的 action_indices，计算其对数概率和熵（用于 PPO 更新时的 importance sampling ratio）。
- 和 `sample_action_sequence` 的自回归逻辑完全一致，但不采样，只算概率。

---

### 3.4 `obs.py` — 观测编码

将 Kaggle 环境的原始观测 dict 编码为固定维度的 numpy float32 向量。

**辅助函数：**
- `_get_field(obs, name, default)` — 同上
- `_log_scale(value, max_value=1000.0)` — 对数缩放 `log(x+1) / log(max+1)`

**`parse_entities(obs)` → `(planets, fleets)`**
- 解析原始观测中的 planets 和 fleets 为 namedtuple 列表。

**`ship_totals(obs, player_id)` → `(my_total, enemy_total)`**
- 统计某玩家的总飞船数（星球 garrison + 飞行中舰队）。

**`encode_observation(obs, max_planets=None, max_fleets=None, obs_config=None, game_config=None)` → `np.ndarray` (964,)**
- 星球特征（48 × 10）：`[is_me, is_enemy, is_neutral, x_norm, y_norm, radius_norm, ships_log, production_norm, is_comet, is_inner]`
  - `is_inner`：星球是否在内圈（距中心 < 50）
  - 排序优先级：己方 → 敌方 → 距中心近
- 舰队特征（64 × 7）：`[is_me, is_enemy, x_norm, y_norm, cos_angle, sin_angle, ships_log]`
  - 排序按 (x, y)
- 全局特征（6）：`[step/500, my_planets_ratio, enemy_planets_ratio, neutral_ratio, my_ships_log, enemy_ships_log]`
- 不足 max 数量的用零填充。

---

### 3.5 `ppo.py` — PPO 实现

**`class RolloutBuffer`** — 经验存储

- 存储 `rollout_steps × num_envs` 步的观测、动作、logprob、reward、done、value。
- `add_batch(...)` — 存入一步经验（所有并行环境的一帧）
- `compute_returns_and_advantages(last_values, gamma, gae_lambda)` — 用 GAE 计算 advantages 和 returns
- `get(batch_size)` — 生成器，随机打乱后按 batch_size 产出 mini-batch
- `clear()` — 重置缓冲区位置指针

**`class PPOTrainer`** — 训练更新

- `update(buffer)` → `stats dict`
  1. 标准化 advantages（减均值除标准差）
  2. 对每个 epoch × mini-batch：
     - 前向传播得到新 logits 和 value
     - 对 batch 中每个样本，调用 `logprob_for_action_sequence` 计算新策略的 logprob 和熵
     - 计算 PPO clip 策略损失 `-min(ratio*adv, clip(ratio)*adv)`
     - 计算 clipped value loss
     - 总损失 = policy_loss + vf_coef × value_loss - ent_coef × entropy
     - 梯度裁剪 + 反向传播 + 优化器更新
  3. 返回平均 `policy_loss`, `value_loss`, `entropy`

---

### 3.6 `train.py` — 主训练循环

**`set_seed(seed)`** — 固定 random/numpy/torch 随机种子。

**`main()`** — 训练入口

流程：
1. 初始化 `OrbitWarsConfig`，并用命令行参数覆盖 `TrainConfig` / `ActionSpaceConfig`
2. 创建 `SummaryWriter`（TensorBoard 日志）
3. 创建 `num_envs` 个 `OrbitWarsSelfPlayEnv`，初始对手为 `NearestPlanetOpponent`
4. 初始化 `ActorCritic` 网络 + Adam 优化器 + `PPOTrainer` + `RolloutBuffer`
5. 初始化 `OpponentPool`（capacity=5），为每个环境采样对手
6. 主循环（`total_updates` 轮）：
   - 收集 `rollout_steps` 步经验（多环境并行）：
     - 将观测编码为向量 → 网络前向 → 采样动作 → 环境 step → 存储到 buffer
     - 若某环境 done，从 opponent pool 采样新对手并 reset
   - 计算 GAE returns & advantages
   - 执行 PPO 更新
   - 记录 TensorBoard 标量
   - 每 `save_every` 轮保存 checkpoint 并加入 opponent pool
   - 每 `opponent_refresh` 轮重新采样对手
   - 每 10 轮打印训练指标

**运行命令：**
```bash
python -m rl.train --num-envs 8 --max-launches-per-source 6 --log-dir runs/orbit_wars
tensorboard --logdir runs/orbit_wars
```

---

### 3.7 `opponents.py` — 对手系统

**`class NearestPlanetOpponent`**
- 规则型对手策略。
- `act(obs)` → `moves`
- 对每个己方星球，找最近的非己方星球，若飞船数大于目标 garrison，发送刚好足够的飞船去占领。

**`class RandomOpponent`**
- `act(obs)` → `[]`，永远不发射。作为最弱的 baseline。

**`class PolicyOpponent`**
- 用训练好的网络作为对手。
- `act(obs)` → `moves`
- 流程：build action templates → encode obs → 网络前向 → 确定性采样 → decode。
- 使用 `deterministic=True`（贪婪策略）。

**`class OpponentPool`**
- 对手快照池，存储最近 N 个 checkpoint 的 state_dict。
- `add(state_dict)` — 加入新快照，超过容量时移除最旧的（FIFO）。
- `sample()` → 对手实例
  - 若池为空：随机返回 `NearestPlanetOpponent` 或 `RandomOpponent`
  - 若池非空：随机选一个历史快照，加载到新网络，返回 `PolicyOpponent`

---

### 3.8 `envs/orbit_wars_env.py` — 环境封装

**`class OrbitWarsSelfPlayEnv`**

封装 Kaggle 的 `make("orbit_wars")` 环境，提供类 Gym 接口。

**`__init__` 参数：**
- `opponent`：对手策略实例
- `env_config`：环境配置（`EnvConfig`，默认使用 `DEFAULT_CONFIG.env`）
- `reward_config`：reward 配置（`RewardConfig`，默认使用 `DEFAULT_CONFIG.reward`）
- `action_config`：动作空间配置（`ActionSpaceConfig`，默认使用 `DEFAULT_CONFIG.action`）

如需局部覆盖，可使用 `dataclasses.replace`：

```python
from dataclasses import replace
from rl.config import DEFAULT_CONFIG

env_config = replace(DEFAULT_CONFIG.env, seed=123, episode_steps=300)
reward_config = replace(DEFAULT_CONFIG.reward, reward_scale=0.02)
```

**关键方法：**
- `reset()` → `obs`：重置环境，初始化飞船差基准线，构建动作模板。
- `step(action_indices)` → `(obs, reward, done, info)`
  1. 用 `ActionBuilder.decode` 将 action_indices 转为 moves
  2. 用 `_sanitize_action` 过滤非法 moves（统计 invalid_count）
  3. 为对手调用 `opponent.act()` 获取对手的 moves
  4. 执行 `env.step(actions)`
  5. 计算 reward = `(当前飞船差 - 上一步飞船差) × reward_scale`（差分稠密奖励）
  6. 若终局：额外加 `飞船差 × terminal_reward_scale`
  7. 若有非法动作：扣 `invalid_action_penalty × invalid_count`
  8. 构建下一步的动作模板

**`_sanitize_action(action, obs, player_id)` → `(clean_moves, invalid_count)`**
- 校验每条 move：格式正确、来源星球存在且有足够飞船。
- 防止重复使用同一星球的飞船资源（同一星球多条 move 共享余额检查）。
- 返回清洗后的 moves 和非法计数。
- **安全关键点**：此函数防止模型输出不合法指令导致环境崩溃或被对手利用。

---

## 4. 动作空间图解

模型输出 logits shape: `(batch, 8, 33)`

```
源星球 0: [停止 | 目标1×0.25 | 目标1×0.5 | 目标1×0.75 | 目标1×1.0 |
                  目标2×0.25 | 目标2×0.5 | 目标2×0.75 | 目标2×1.0 | ... | 目标8×1.0]
源星球 1: [停止 | ...]
...
源星球 7: [停止 | ...]
```

每次采样是自回归的：
1. 对当前源星球，用 mask 屏蔽非法动作（飞船不够发送的）
2. 采样一个 action_idx
3. 如果 action_idx=0（停止），跳到下一个源星球
4. 否则扣除对应比例的飞船，继续在当前源星球上循环采样
5. 直到发射次数达到 `max_launches_per_source` 或飞船耗尽

最后 decode 汇总所有源星球的所有发射序列为 `[[from_id, angle, ships], ...]`

---

## 5. 数据流全貌

```
Kaggle 环境
   ↓ obs (dict)
ObsWarsSelfPlayEnv.reset/step
   ↓ raw obs (dict)
encode_observation() [obs.py]
   ↓ obs_vector (np.float32, 964)
ActorCritic.forward()
   ↓ logits (8×33), value (scalar)
sample_action_sequence() [action.py]
   ↓ action_indices (8×max_launches) + logprob, entropy
ActionBuilder.decode()
   ↓ moves [[from_id, angle, ships], ...]
_sanitize_action()
   ↓ verified moves → Kaggle env
   ↓ reward, done
RolloutBuffer 收集 (obs, actions, logprobs, rewards, dones, values)
   ↓ rollout complete
compute GAE advantages & returns
   ↓
PPOTrainer.update() → policy improvement
   ↓
OpponentPool.add(state_dict)  ← 定期加入对手池
```

---

## 6. 后续开发方向

### 6.1 训练优化
- **调整超参数**：修改 `PPOConfig` 中的参数（如增大 `rollout_steps`、调整 `ent_coef`）
- **增大模型**：修改 `MODEL_HIDDEN_SIZES`，增加层数或宽度
- **学习率调度**：当前是固定 lr，可加入 cosine annealing 或 linear decay

### 6.2 观测与特征
- **增加历史帧**：将过去 N 步的观测作为额外输入（帧堆叠），捕捉时序动态
- **图神经网络**：用 GNN 替代 MLP 编码星球/舰队关系，天然处理变长输入（不再需要 zero-padding 到固定长度）
- **加入 comet 轨迹预测**：当前只标记 `is_comet`，可以编码彗星的椭圆轨道参数
- **加入太阳/碰撞区域编码**：帮助网络更好地理解内圈危险区域

### 6.3 动作空间改进
- **连续动作空间**：直接用角度 + 飞船比例作为连续输出（Beta 分布 / Gaussian），替代离散化
- **飞船数量直接预测**：当前只支持固定比例 (25%, 50%, 75%, 100%)，可让网络直接输出具体船只数量
- **全局动作协调**：当前每个源星球独立决策，可加入 attention 机制让源星球之间协调

### 6.4 多智能体与对手
- **多玩家训练**：扩展到 4 人局，训练时在不同阵营中轮换
- **对手多样化**：加入更多规则型对手（如 defensively turtling、aggressive zerg）丰富训练分布
- **Elo 评分匹配**：为 opponent pool 中的每个快照维护 Elo 评分，优先匹配实力相近的对手
- **League training**：参考 AlphaStar，维护多类策略池（主策略、反策略等）

### 6.5 工程改进
- **WandB 集成**：替换或补充 TensorBoard，方便云端实验追踪
- **配置文件**：将 `PPOConfig` 从代码中抽到 YAML/JSON 配置文件
- **恢复训练**：添加 `--resume` 参数，从 checkpoint 恢复模型+优化器+对手池状态
- **评估模式**：添加独立的评估脚本，在固定对手集合上评测 Elo
- **对战 replay 保存**：保存精彩对局用于可视化分析

### 6.6 奖励设计
- **领土奖励**：根据控制的星球数量/生产力给予奖励
- **击杀奖励**：消灭敌方舰队时给予正奖励
- **效率惩罚**：发射过多飞船（超出占领所需的）给予微小惩罚，鼓励精确计算
- **生存奖励**：每步给予小额正奖励，鼓励存活更久
- **内圈风险惩罚**：在内圈星球上滞留过多飞船时给予惩罚（太阳可能摧毁）

### 6.7 观测编码改进
- **星球所属标签**：当前是 one-hot（is_me, is_enemy, is_neutral），可加入按 owner 编号的 learnable embedding
- **相对位置编码**：舰队位置相对于各源星球的偏移，而非绝对坐标
- **全局视野**：加入更多全局统计量（如所有制中心、前线位置等）

---

## 7. 快速开始

```bash
# 安装依赖
pip install "kaggle-environments>=1.28.0" torch numpy tensorboard

# 启动训练（默认参数）
python -m rl.train

# 多环境并行训练
python -m rl.train --num-envs 8 --max-launches-per-source 6

# 使用 GPU（自动检测）
python -m rl.train --device cuda

# 查看训练曲线
tensorboard --logdir runs/orbit_wars
```

Checkpoint 会保存到 `checkpoints/ppo_orbit_wars_{update}.pt`。

---

## 8. 关键注意事项

1. **动作非法过滤是必须的**：网络输出的动作可能不合法（飞船不够、目标不存在），`_sanitize_action` 和 `_mask_tensor` 确保不发生崩溃。
2. **对手池是自对弈的关键**：opponent pool 存储训练历史中的旧策略，形成持续增强的对抗对手。容量设为 5 意味着对抗最近 5 个 checkpoint 时代的自己。
3. **Observation 兼容性**：框架同时支持 `obs` 为 dict 或对象，因为 Kaggle 环境在不同模式下返回不同类型。
4. **飞船差作为稠密 reward**：这种设计使得每一步都能收到反馈，而非稀疏的终局胜负。但需要配合 `reward_scale` 调参。
5. **自回归动作采样与 logprob 计算必须一致**：`sample_action_sequence` 和 `logprob_for_action_sequence` 的自回归循环逻辑完全一致，确保 PPO 的 ratio 计算正确。
