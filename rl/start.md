# Orbit Wars PPO Self-Play — 代码框架与开发指南

## 1. 项目概览

基于 PPO 的 1v1 自对弈训练框架，使用 Kaggle 官方 `kaggle_environments` 作为游戏引擎。

**关键设计决策：**

- **两层层次化离散动作空间**：第一层由 GNN 学习选择发射源星球（Gumbel top-k），第二层在每个槽位上输出自回归发射序列（停止 / 向某目标发送某比例的飞船）
- **GNN + Self-Attention + Cross-Attention 模型**：图卷积编码星球关系，自注意力编码舰队关系，交叉注意力融合两者。MLP 模型为回退选项
- **六维归一化奖励函数**：经济优势 / 领土事件 / 战斗效率 / 闲置惩罚 / 分阶段生存 / 温和终局（±5）
- **固定维度观测编码**：48 星球 × 11 特征 + 64 舰队 × 9 特征 + 6 全局特征 = 1110 维向量
- **GAE 优势估计 + PPO clip loss + 价值函数裁剪**

---

## 2. 文件结构与职责

```
rl/
├── __init__.py              # 包声明
├── config.py                # 配置分组与默认配置（OrbitWarsConfig 等）
├── models.py                # Actor-Critic 网络定义（MLP + GNN 双模型）
├── action.py                # 动作空间：源选择 + 槽位动作编解码 + 采样/对数概率
├── obs.py                   # 观测编码：原始 dict → 固定维度 numpy 向量 (1110,)
├── ppo.py                   # RolloutBuffer（经验存储）+ PPOTrainer（更新逻辑）
├── train.py                 # 主训练入口：批量 rollout + 自对弈循环
├── opponents.py             # 对手策略：最近星球 / 随机 / 策略网络 + 对手池
├── envs/
│   ├── __init__.py
│   └── orbit_wars_env.py    # Gym 风格环境封装（self-play + 六维奖励 + 动作校验）
└── README.md                # 简要使用说明
```

---

## 3. 各模块详细说明

### 3.1 `config.py` — 配置

配置拆分为多个 dataclass，通过 `OrbitWarsConfig` 统一管理，支持 YAML 加载。

**核心结构：**

- `GameConfig`：地图与几何常量（`board_size=100`, `center_x/center_y=50`）
- `ObsConfig`：观测规模（`max_planets=48`, `max_fleets=64`, `planet_features=11`, `fleet_features=9`, `global_features=6`）。`obs_dim` 属性自动计算 = 1110
- `ActionSpaceConfig`：动作规模（`max_sources=20`, `max_targets=20`, `ship_fractions`, `max_launches_per_source=3`）。`actions_per_source` = `1 + max_targets × len(ship_fractions)`
- `ModelConfig`：模型类型（`model_type` = "mlp" 或 "gnn"）及超参（`hidden_sizes`, `dropout`, `gnn` 子配置）
- `GNNConfig`：GNN 超参（`hg=128`, `hf=128`, `ha=64`, `num_gcn_layers=2`）
- `TrainConfig`：PPO 训练超参数
- `RewardConfig`：六维奖励缩放参数
- `EnvConfig`：环境运行参数（`num_players`, `episode_steps`, `act_timeout`, `seed`, `debug`）

**`OrbitWarsConfig.from_yaml(path)`** — 从 YAML 文件构建完整配置。

为兼容旧用法，仍保留 `MAX_PLANETS`, `OBS_DIM` 等模块级常量，值来自 `DEFAULT_CONFIG`。

**TrainConfig 常用字段：**

| 参数 | 默认值 | 含义 |
|---|---|---|
| `seed` | 42 | 随机种子 |
| `num_envs` | 10 | 并行环境数 |
| `total_updates` | 2000 | 总 PPO 更新轮数 |
| `rollout_steps` | 64 | 每轮收集多少步经验 |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE λ 参数 |
| `clip_range` | 0.2 | PPO clip 范围 |
| `learning_rate` | 3e-4 | Adam 学习率 |
| `ent_coef` | 0.01 | 熵正则系数 |
| `vf_coef` | 0.5 | 价值损失权重 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `batch_size` | 64 | mini-batch 大小 |
| `epochs` | 4 | 每轮数据重复训练次数 |
| `save_every` | 50 | 每 N 轮保存 checkpoint |
| `opponent_refresh` | 10 | 每 N 轮重新采样对手 |

**RewardConfig 字段（六维奖励）：**

| 参数 | 默认值 | 维度 |
|---|---|---|
| `economic_scale` | 1.0 | ① 经济优势 |
| `territory_scale` | 2.0 | ② 领土事件 |
| `combat_efficiency_scale` | 0.5 | ③ 战斗效率 |
| `idle_penalty_scale` | 0.1 | ④ 闲置惩罚 |
| `production_weight` | 0.5 | ② 星球质量倍率 |
| `idle_threshold` | 0.5 | ④ 闲置触发阈值 |
| `early_game_steps` / `mid_game_steps` | 50 / 200 | ⑤ 分阶段边界 |
| `survival_reward_early/mid/late` | 0.0 / 0.01 / 0.05 | ⑤ 分阶段生存奖励 |
| `terminal_win_scale` / `terminal_lose_scale` | ±5.0 | ⑥ 终局 |
| `invalid_action_penalty` | 0.1 | 非法动作惩罚 |

---

### 3.2 `models.py` — 神经网络

#### `_build_mlp(input_dim, hidden_sizes, output_dim, dropout)`

构建 MLP 序列：每层 `Linear → LayerNorm → GELU → Dropout`（循环），最后 `Linear` 输出。

#### `class ActorCritic(nn.Module)` — MLP 回退模型

- 输入：观测向量 `(batch, 1110)`
- Body：3 层 MLP (1110 → 512 → 512 → 256)，输出 256 维特征
- 槽位策略头 `slot_policy_head`：`Linear(256, max_sources × actions_per_source)` → reshape 为 `(batch, max_sources, actions_per_source)`
- 源选择 logits：`fallback_source_logits` 为非空间回退（MLP 无每行星嵌入结构），`ActionBuilder` 检测到后会回退到按舰船数排序
- 价值头 `value_head`：`Linear(256, 1)`
- 所有权掩码：dummy `ones(B, 1)`（MLP 不使用真实行星结构）
- `forward(obs)` 返回四元组 `(source_logits, slot_logits, value, ownership_mask)`

#### `class ActorCriticGNN(nn.Module)` — GNN + Attention 主模型

结构化处理：将扁平的观测向量拆分回 `Zp(B, 48, 11)` 行星张量和 `Zf(B, 64, 9)` 舰队张量。

**处理流程：**

1. **掩码构建**：
   - `planet_mask`：行星特征 L1 范数 > 1e-6（区分真实行星与零填充）
   - `fleet_mask`：同上
   - `ownership_mask`：取 `is_me` 特征 > 0.5

2. **行星 GNN**（图卷积）：
   - 可学习邻接矩阵：`Wa(Zp) @ Zp^T` → 掩码 → softmax 归一化
   - N 层 GCN：每层 `Linear(hg) → LayerNorm → GELU → Dropout`，消息传递 `Ag @ Zp`

3. **舰队自注意力**：QKV 全来自 `Zf`，掩码 softmax → 加权聚合

4. **交叉注意力**：行星 Q 查询舰队 KV → 残差连接 `Wp(cross_attn_output) + Zp` → 行星嵌入 `Z(B, 48, 128)`

5. **源选择**（第一层动作）：
   - 可学习槽位查询向量 `source_query(max_sources, hg)` × `source_key(Z)` → `source_logits(B, max_sources, 48)`
   - 训练时由 `select_sources()` 做 Gumbel top-k 选出 `max_sources` 个发射源
   - 推理时 argmax top-k

6. **槽位嵌入**：softmax(source_logits, mask=ownership) @ Z → `slot_embs(B, max_sources, hg)`

7. **槽位策略头** `slot_policy_head`：共享 MLP `Linear(hg→hg) → LayerNorm → GELU → Dropout → Linear(hg→201)` → `slot_logits(B, max_sources, 201)`

8. **价值头** `value_head`：mean-pool(Z) → MLP → `value(B, 1)`

**参数量**（默认 GNN 配置）：~0.9M trainable params。

---

### 3.3 `action.py` — 动作空间

两层层次化离散动作空间。整个框架最核心的模块。

#### 辅助函数

- `_get_field(obs, name, default)` — 兼容 dict/object 两种观测格式
- `_parse_planets(obs)` — 将原始 planets 列表解析为 `Planet` namedtuple

#### `class ActionBuilder` — 动作模板构建与解码

- **`build(obs, source_planet_ids=None)`** → `(actions, source_ships)`
  - 若 `source_planet_ids` 提供（GNN 模型输出），将其作为优先源，其余己方行星按舰船数降序填充
  - 若为 None（MLP 回退），全部按舰船数降序排序取前 `max_sources`
  - 对每个源，构建 `actions_per_source` 维动作槽位：
    - 槽位 0：停止发射
    - 槽位 1..：前 `max_targets` 个目标 × `ship_fractions`（目标排序：敌方优先 → 距离近优先）
  - 每个槽位存储 `ActionTemplate(source_id, angle, fraction)`

- **`decode(action_indices, actions, source_ships, max_launches)`** → `moves`
  - 将动作索引转换为 `[[from_id, angle, ships], ...]` 格式
  - 自回归解码：每步从剩余舰船中扣除，余额不足时停止

#### `@dataclass(frozen=True) class ActionTemplate`

不可变数据类：`source_id: int`, `angle: float`, `fraction: float`

#### `select_sources(source_logits, ownership_mask, k, deterministic)` → `Tensor` (k,) or None

第一层动作 — 源选择：
- GNN 模型：`source_logits` shape = `(max_sources, max_planets)`，取行均值得到每行星分数，Gumbel top-k（训练）或 argmax top-k（推理）选出 k 个己方行星
- MLP 模型：`source_logits` shape = `(max_sources, 1)` → 返回 None，`ActionBuilder` 回退到舰船排序

#### `source_selection_logprob(source_logits, ownership_mask, source_indices)` → scalar

Plackett-Luce 不放回次序 log-prob：对选出的每个行星，计算其在剩余候选池中的 softmax log-prob，累加。

#### `sample_action_sequence(logits, actions, source_ships, max_launches, deterministic)` → `(action_indices, logprob_sum, entropy_sum)`

第二层动作 — 槽位动作采样：
- 每个源独立自回归：从 `max_launches_per_source` 步循环
- 每步用 `_mask_tensor` 屏蔽非法动作（剩余舰船不够发送的），然后 `Categorical` 采样
- 选中 index=0（STOP）时终止当前源

#### `logprob_for_action_sequence(logits, actions, source_ships, action_indices, max_launches)` → `(logprob_sum, entropy_sum)`

给定已采样的 action_indices，计算对数概率和熵（用于 PPO 更新）。

#### `_ships_to_send(remaining_ships, fraction)` → int

`max(1, min(int(remaining × fraction), remaining))` — 最少发 1 艘，最多全部。

#### `_mask_tensor(source_actions, remaining_ships, device)` → `Tensor`

合法动作 mask：`remaining × fraction > 0` 的动作合法，槽位 0 始终合法。

---

### 3.4 `obs.py` — 观测编码

将 Kaggle 环境的原始 dict 编码为固定维度 `np.float32(1110,)` 向量。

**辅助函数：**
- `_log_scale(value, max_value=1000.0)` — `log(x+1) / log(max+1)`
- `parse_entities(obs)` — 解析 planets 和 fleets
- `ship_totals(obs, player_id)` — 统计某玩家总舰船数（驻守 + 飞行中）

**`encode_observation(obs, ...)` → `np.ndarray(1110,)`**

| 组成部分 | 维度 | 特征列表 |
|---------|------|---------|
| 行星特征 | 48 × 11 | `[planet_id_norm, is_me, is_enemy, is_neutral, x_norm, y_norm, radius_norm, ships_log, production_norm, is_comet, is_inner]` |
| 舰队特征 | 64 × 9 | `[fleet_id_norm, from_planet_id_norm, is_me, is_enemy, x_norm, y_norm, cos_a, sin_a, ships_log]` |
| 全局特征 | 6 | `[step/500, my_planets/48, enemy_planets/48, neutral/48, my_ships_log, enemy_ships_log]` |

- 行星排序优先级：己方 → 敌方 → 距中心近
- 舰队排序按 `(x, y)`
- 不足 max 数量的用零填充
- `is_comet` 标记彗星行星（沿轨道运行的临时行星），`is_inner` 标记内圈行星（距中心 < 50，可能被太阳摧毁）

---

### 3.5 `ppo.py` — PPO 实现

#### `class RolloutBuffer` — 经验存储

- 存储 `rollout_steps × num_envs` 步的完整经验
- 缓存 `action_templates`、`source_ships`、`source_indices`，避免训练时重复 `ActionBuilder.build()`
- **关键方法：**
  - `add_batch(obs, raw_obs, actions, logprobs, rewards, dones, values, action_templates_list, source_ships_list, source_indices_list)` — 存入一步经验
  - `compute_returns_and_advantages(last_values, gamma, gae_lambda)` — GAE 计算
  - `get(batch_size)` — 生成器，随机打乱产出 mini-batch（含 templates/ships/source_indices）
  - `clear()` — 重置位置指针

#### `class PPOTrainer` — 训练更新

- **`update(buffer)` → `stats dict`**
  1. 标准化 advantages
  2. 对每个 epoch × mini-batch：
     - 前向传播 → `(source_logits, slot_logits, values, ownership_mask)`
     - 对每个样本：`source_lp = source_selection_logprob(...)` + `slot_lp = logprob_for_action_sequence(...)` → `new_logprob = source_lp + slot_lp`
     - PPO clip loss：`-min(ratio×adv, clip(ratio)×adv)`
     - Clipped value loss
     - 总损失 = `policy_loss + vf_coef × value_loss - ent_coef × entropy`
     - 梯度裁剪 + 反向传播 + 优化器更新（支持 AMP）
  3. 返回平均 `policy_loss`, `value_loss`, `entropy`

---

### 3.6 `train.py` — 主训练循环

**`main()`** — 训练入口

**流程：**

1. 解析 YAML 配置 + 命令行覆盖 → `OrbitWarsConfig`
2. 创建 `SummaryWriter`（TensorBoard）
3. 创建 `num_envs` 个 `OrbitWarsSelfPlayEnv`，初始对手 `NearestPlanetOpponent`
4. 初始化模型（GNN 或 MLP）+ Adam 优化器 + `PPOTrainer` + `RolloutBuffer`
5. 初始化 `OpponentPool`（capacity=5），为每个环境采样对手
6. 主循环（`total_updates` 轮）：

**Rollout 阶段（`rollout_steps` 步）：**

- 将所有环境观测编码为向量 → 批量前向 → `(source_logits, slot_logits, values, ownership_masks)`
- 批量对手推断（`PolicyOpponent.batch_act`）
- 对每个环境：
  1. `select_sources()` — 从 GNN 输出中选出 `max_sources` 个发射源
  2. `ActionBuilder.build(obs, source_planet_ids)` — 用选定源构建动作模板
  3. `sample_action_sequence()` — 从槽位 logits 中采样动作
  4. `source_selection_logprob()` — 计算源选择 logprob
  5. `env.step()` — 执行，传入模板/舰船覆盖
  6. 若 done → 从对手池采样新对手 → `env.reset()`
- 将观测/动作/模板/舰船/源索引存入 `RolloutBuffer`

**训练阶段：**

- GAE 计算 returns & advantages
- `PPOTrainer.update(buffer)` — PPO 更新
- 记录 TensorBoard 标量

**定期操作：**

- 每 `save_every` 轮保存 checkpoint（含 model/optimizer/pool_snapshots）并加入对手池
- 每 `opponent_refresh` 轮重新采样对手

**运行命令：**

```bash
python -m rl.train --config configs/default.yaml --device cuda
python -m rl.train --config configs/default.yaml --device cuda --resume checkpoints/ppo_orbit_wars_150.pt
tensorboard --logdir runs/OrbitWars
```

---

### 3.7 `opponents.py` — 对手系统

#### `class NearestPlanetOpponent`

规则型对手：对每个己方星球找最近的非己方星球，若舰船数大于目标驻军就刚好发送足够占领的数量。

#### `class RandomOpponent`

永不发射（`act(obs) → []`），最弱 baseline。

#### `class PolicyOpponent`

训练好的网络作为对手。使用**确定性**策略（argmax）。

`act(obs)` 流程：
1. 编码观测 → 前向 → `(source_logits, slot_logits, _, ownership_mask)`
2. `select_sources(..., deterministic=True)` → 选出源行星
3. `ActionBuilder.build(obs, source_planet_ids)` → 构建模板
4. `sample_action_sequence(..., deterministic=True)` → 采样动作
5. `ActionBuilder.decode()` → 返回 moves

#### `class PolicyOpponent.batch_act()`

批量对手推断：按策略身份分组，同一模型 checkpoint 的对手合并为一个 batch 前向。

#### `class OpponentPool`

对手快照池（FIFO，capacity=5）：
- `add(state_dict)` — 加入新快照（保存到 CPU），加载为新策略实例
- `sample()` — 随机返回对手（空池时随机 `NearestPlanetOpponent` / `RandomOpponent`）
- `restore_snapshots(snapshots)` — 从 checkpoint 恢复池状态

---

### 3.8 `envs/orbit_wars_env.py` — 环境封装

封装 Kaggle 的 `make("orbit_wars")` 环境，提供类 Gym 接口。

**`__init__` 参数：**
- `opponent`：对手策略实例
- `env_config` / `reward_config` / `action_config`：各配置 dataclass

**关键方法：**

- **`reset()`** → `obs`
  - 初始化跟踪状态：`_last_planet_owners`（星球归属快照）、`_last_econ_ratio`（产量比率）、`_last_my_total` / `_last_enemy_total`（总舰船数）

- **`step(action_indices, opponent_actions, action_templates_override, source_ships_override)`** → `(obs, reward, done, info)`
  - 支持模板/舰船覆盖（配合训练循环中按选定源构建的模板）
  - `_sanitize_action` 过滤非法 moves
  - 调用 `_compute_reward` 计算六维奖励

- **`_compute_reward(obs, player_id)`** — **六维归一化奖励**：

| 维度 | 每步范围 | 机制 |
|------|---------|------|
| ① 经济优势 | ±0.02 | `Δ(我方产量 / (我方+敌方产量+1)) × economic_scale` |
| ② 领土事件 | ±1.5~12（稀疏） | 星球归属变更：`±(1 + production × weight) × territory_scale` |
| ③ 战斗效率 | ±0.25 | `(敌方损失 / 总损失 - 0.5) × combat_scale`，有战斗时触发 |
| ④ 闲置惩罚 | −0.05~0 | `max(0, 驻守比例 − threshold) × idle_scale` |
| ⑤ 生存奖励 | 0~+0.05 | 早期 0 / 中期 0.01 / 后期 0.05 每步 |
| ⑥ 终局 | ±5 | 胜利/失败（温和，不碾压过程信号） |

- **`_sanitize_action(action, obs, player_id)`** → `(clean_moves, invalid_count)`
  - 校验每条 move：格式正确、来源星球存在、舰船足够
  - 同星球多条 move 共享余额检查，防止重复使用资源

**环境辅助函数（模块级）**：
- `_planet_owner_map(obs)` — 返回 `{planet_id: owner}` 快照
- `_planet_production_totals(obs, player_id)` — 返回 `(my_prod, eny_prod)`
- `_planet_production(obs, planet_id)` — 返回指定星球的产量
- `_idle_ship_ratio(obs, player_id)` — 返回驻守舰船 / 总舰船比例

---

## 4. 动作空间图解

### 第一层：源选择（模型学习）

```
GNN 输出 per-planet embedding Z (B, 48, 128)
   ↓ source_query (20, 128) · source_key(Z)^T
source_logits (B, 20, 48)  ← 每个槽位对每个行星的注意力分数
   ↓ select_sources (Gumbel top-k / argmax)
选出 max_sources=20 个己方行星作为发射源（带顺序）
```

### 第二层：每槽位动作

```
softmax(source_logits, mask=ownership) @ Z
   → slot_embs (B, 20, 128)  ← 每个槽位独立嵌入
   ↓ slot_policy_head (共享)
slot_logits (B, 20, 201)  ← 201 路 categorical per slot

每个槽位：
┌─────┬──────────────────────────────────────────────────┐
│  0  │  STOP（停止发射）                                   │
├─────┼──────────────────────────────────────────────────┤
│  1  │  目标[0], 0.1                                    │
│  2  │  目标[0], 0.2                                    │
│ ... │  ...                                             │
│ 10  │  目标[0], 1.0                                    │
│ 11  │  目标[1], 0.1                                    │
│ ... │  ...                                             │
│200  │  目标[19], 1.0                                   │
└─────┴──────────────────────────────────────────────────┘
actions_per_source = 1 + 20 targets × 10 fractions = 201
```

### 自回归发射

```
每源最多 max_launches_per_source=3 步：
  step 0: Categorical(201-way, masked) → 选中 action_idx
           → 扣除 ships → 如果 action_idx=0 或剩余=0 → 停止
  step 1: 同上（mask 根据新剩余舰船更新）
  step 2: 同上
```

---

## 5. 数据流全貌

```
Kaggle 环境
   ↓ obs (dict)
OrbitWarsSelfPlayEnv
   ↓ raw obs (dict)
encode_observation() [obs.py]
   ↓ obs_vector (np.float32, 1110)
ActorCriticGNN.forward()
   ↓ source_logits (20×48), slot_logits (20×201), value, ownership_mask
select_sources() + ActionBuilder.build() + sample_action_sequence()
   ↓ action_indices (20×3), source_indices (20,), logprob
ActionBuilder.decode()
   ↓ moves [[from_id, angle, ships], ...]
_sanitize_action()
   ↓ verified moves → Kaggle env
   ↓ reward (六维), done
RolloutBuffer 收集 (obs, actions, logprobs, rewards, dones, values,
                     action_templates, source_ships, source_indices)
   ↓ rollout complete
compute GAE advantages & returns
   ↓
PPOTrainer.update()
   ↓ logprob = source_lp + slot_lp
   ↓ PPO clip loss → policy improvement
OpponentPool.add(state_dict)  ← 定期加入对手池
```

---

## 6. 奖励函数设计

六维归一化过程奖励，所有组件 ∈ [−1, +1] 量级，避免单一维度劫持梯度。

### ① 经济优势（每步）
```
cur_econ = 我方产量 / (我方产量 + 敌方产量 + 1)
reward   = Δ(econ_ratio) × economic_scale(1.0)
```
- 激励：打压敌方经济，建设己方经济
- 用比值而非绝对值：双方总量同时增长时无虚假奖励

### ② 领土事件（事件驱动）
```
quality = 1 + planet.production × production_weight(0.5)
捕获:    +quality × territory_scale(2.0)    (≈ +2~+12)
丢失:    −quality × territory_scale(2.0)    (≈ −2~−12)
```
- 只在归属变更的步触发，稀疏但高价值信号
- 高生产力星球价值是低生产力星球的 3-6 倍

### ③ 战斗效率（有战斗时触发）
```
效率 = 敌方损失 / (我方损失 + 敌方损失 + ε) − 0.5     [−0.5, +0.5]
reward = 效率 × combat_scale(0.5)                    [−0.25, +0.25]
```
- 10:1 碾压 ≈ +0.23；1:10 惨败 ≈ −0.23
- 只在双方总损失 > 1 时触发

### ④ 闲置惩罚（每步）
```
idle = 驻守舰船 / 总舰船
penalty = max(0, idle − 0.5) × idle_scale(0.1)   [−0.05, 0]
```
- 50% 以下不触发，避免不必要的防守惩罚

### ⑤ 分阶段生存（每步）
```
早期 (0-50)   : 0          ← 允许探索试错
中期 (50-200) : 0.01/步    ← 保存实力
后期 (200-500): 0.05/步    ← 活下来才有机会翻盘
```
- 后期累计可达 +15，比一次终局胜负 +5 更高 → 激励不放弃

### ⑥ 终局
```
胜利: +5.0    失败: −5.0
```
- 旧版 ±100 → 新版 ±5：不碾压过程信号，让中期高质量战斗的价值超过终局

---

## 7. 后续开发方向

### 7.1 训练优化
- **学习率调度**：cosine annealing 或 linear decay 替代固定 lr
- **熵退火**：`ent_coef` 从 0.05 线性退火到 0.001
- **PPO dual-clip**：负 advantage 时加 ratio 下界
- **Phasic Policy Gradient (PPG)**：增加 auxiliary phase 解耦 policy/value 训练

### 7.2 动作空间改进
- **目标选择学习化**：类似源选择的 Gumbel top-k 机制应用于目标选择
- **连续舰船分配**：用 softmax 分配替代离散 fraction，天然支持多目标
- **全局协调**：让槽位之间通过 attention 协调（当前独立决策）

### 7.3 观测与特征
- **彗星轨迹编码**：当前只标记 `is_comet`，可加入速度方向 / 剩余寿命 / 轨道参数
- **帧堆叠**：将过去 N 步观测作为额外输入，捕捉时序动态
- **彗星感知掩码**：GNN 中给彗星分配独立 attention mask
- **相对位置编码**：舰队位置相对于各源星球的偏移

### 7.4 奖励函数增强
- **彗星过滤**：领土奖励中排除彗星归属变更（误报"丢失星球"）
- **舰队碰撞惩罚**：发送中的舰队撞彗星被摧毁时给予信号

### 7.5 多智能体与对手
- **League Training**：参考 AlphaStar，维护主策略/反策略等多类池
- **Elo 加权采样**：优先匹配实力相近的对手
- **多玩家扩展**：4 人局训练

### 7.6 工程改进
- **WandB 集成**：替代或补充 TensorBoard
- **评估脚本**：在固定对手集合上评测 Elo
- **对战 replay 保存**：保存精彩对局用于分析

---

## 8. 快速开始

```bash
# 安装依赖
pip install "kaggle-environments>=1.28.0" torch numpy tensorboard pyyaml

# 启动训练（默认 GNN 模型）
python -m rl.train --config configs/default.yaml --device cuda

# 恢复训练
python -m rl.train --config configs/default.yaml --device cuda \
    --resume checkpoints/ppo_orbit_wars_150.pt

# 使用 MLP 回退模型
python -m rl.train --config configs/default.yaml --device cuda --model-type mlp

# 查看训练曲线
tensorboard --logdir runs/OrbitWars
```

Checkpoint 保存到 `checkpoints/ppo_orbit_wars_{update}.pt`。

---

## 9. 关键注意事项

1. **源选择是学习化的**：GNN 通过 attention 为每个槽位学习"哪个行星最好"，替代旧版的硬编码舰船排序。MLP 模型自动回退到舰船排序。

2. **奖励尺度一致性**：所有六维组件归一化到同一量级，终局信号 ±5 不碾压过程信号。避免旧版中"占一个星球 +10 但有 100 艘船的优势变更只有 +0.02"的问题。

3. **对手池是自对弈的关键**：存储历史 checkpoint 的策略快照，形成持续增强的对抗对手。capacity=5 对抗最近 5 个 version 的自己。

4. **模板缓存避免重复计算**：rollout 时将 `action_templates` / `source_ships` / `source_indices` 存入 buffer，PPO 更新时直接复用，消除训练内循环中数万次 `build()` 调用。

5. **自回归采样与 logprob 必须一致**：`sample_action_sequence` 和 `logprob_for_action_sequence` 的 mask 更新和 ship 扣除逻辑完全相同。

6. **动作非法过滤是必须的**：`_sanitize_action` 和 `_mask_tensor` 确保不发生崩溃。即使 mask 层出错，sanitize 也会兜底。

7. **非法动作惩罚**：自动从奖励中扣除 `invalid_action_penalty × invalid_count`。
