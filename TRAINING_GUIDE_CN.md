# ODP-JAX 训练流程（基于 FQL JAX）

整体顺序：

```text
FQL/OGBench transition dataset
  ├─ Planner：学习轨迹扩散分布
  ├─ Kernel：学习 p(s'|s,a)，判断计划是否符合数据分布
  ├─ Reward：学习 r(s,a)
  └─ Critic：学习 V(s)
          ↓
当前 Planner 采样 → Kernel 约束 → Reward/Critic 评价
          ↓
Adjoint Matching 更新 Planner → rollout → 更新 buffer 和辅助模型
```

## 1. Planner 预训练

入口是：

```bash
python odp_main.py --stage planner --env_name antmaze-large-navigate-singletask-task4-v0
```

`odp/data.py` 将 FQL 的 transition arrays 按 terminal 切成轨迹，再生成长度为
`horizon` 的滑动窗口。每个时间步拼成 `[state, action]`；第一个状态作为
条件，状态按 ODP 的均值/标准差标准化，动作保持原值。

`odp/models.py` 中的 Flax DiT 对应原 ODP 的 `DiT1d`：`emb_dim=128`、
`d_model=256`、`n_heads=4`、`depth=4`，使用 Fourier time embedding、
adaLN-Zero 和零初始化输出层。训练时随机采样 VP 时间 `t`，构造：

```text
x_t = alpha(t) * x_0 + sigma(t) * epsilon
score_target = -(x_t - alpha(t) * x_0) / sigma(t)^2
```

第一个状态坐标固定，其余坐标使用 `sigma²` 加权 MSE。优化器、cosine
learning-rate、AdamW weight decay、梯度累积和 EMA 与原 ODP 对齐。

## 2. Kernel 训练

入口：

```bash
python odp_main.py --stage kernel --env_name antmaze-large-navigate-singletask-task4-v0 \
  --restore_path exp/odp_jax/planner --restore_epoch 1000000
```

`KernelDataset` 将每条轨迹变成 `(s_t, a_t, s_{t+1})`，只标准化状态。
当前支持两种模型：

- `robust`：对角高斯转移 ensemble，输出 `mu` 和 `log_std`，优化高斯 NLL。
- `mog`：混合高斯转移 ensemble，输出多个均值、方差和混合权重，优化 mixture NLL。

默认的 `odp_antmaze_task4` profile 使用 ensemble10、10 modes、4 层宽度
514 的 kernel、`noise_floor=5e-4` 和 `lambda_reg=1e-3` 对应的分歧正则。

## 3. Reward model 训练

入口：

```bash
python odp_main.py --stage reward --env_name antmaze-large-navigate-singletask-task4-v0 \
  --restore_path exp/odp_jax/kernel --restore_epoch 20000
```

每个动作形成 `(s_t, a_t, r_t)` 样本。OGBench 单任务数据沿用原 ODP 的全局
奖励平移；状态标准化，奖励支持 Gaussian smoothing（`--reward_sigma`）、
EMA smoothing（`--reward_alpha`）和目标缩放（`--target_reward`）。

Flax `RewardModel` 对应原 `SimpleReward`：状态动作拼接后经过 SiLU/LayerNorm
MLP，输出标量奖励；使用 Smooth L1、AdamW、cosine learning-rate 和梯度更新。

## 4. Critic 训练

入口：

```bash
python odp_main.py --stage critic --env_name antmaze-large-navigate-singletask-task4-v0 \
  --restore_path exp/odp_jax/reward --restore_epoch 40000
```

`critic_transitions` 使用长度为 `horizon` 的折扣 n-step target：

```text
target = R[t:t+H] + gamma^H * V_target(s[t+H])
```

Critic 输入标准化状态，使用 Smooth L1、Adam、cosine learning-rate、梯度更新
和 `tau=0.005` 的 soft target update。finetune 阶段还可以直接用当前 Planner
生成的计划构造多长度 bootstrap target；profile 默认启用 symlog value target
和可配置的 Q scale，保持原 `train_critic_with_planner7` 的计划驱动更新逻辑。

## 5. Planner Finetune

入口：

```bash
python odp_main.py --stage finetune \
  --env_name antmaze-large-navigate-singletask-task4-v0 \
  --restore_path exp/odp_jax/critic --restore_epoch 20
```

每个 finetune round 执行：

1. 从离线窗口或 replay buffer 取初始状态，并按 `batch_per_sample` 重复。
2. 当前 Planner 用 Karras/cosine SDE 生成完整扩散路径，固定第一个状态，动作裁剪到 `[-1,1]`。
3. Reward 给出奖励梯度；Kernel 用 ensemble log-likelihood 形成 soft constraint，
   由 lambda 加权；Critic 模式加入末状态 bootstrap value。
4. 冻结上一轮 Planner，用 JAX VJP/Jacobian-transpose 反向传播离散 adjoint。
5. 训练新的 Planner，使 score vector field 匹配旧 field 加上 adjoint 修正；初始状态坐标不进入 loss。
6. 使用 AdamW、EMA、loss clipping 和梯度累积更新 Planner，并保存 FQL 风格 checkpoint。
7. 在线模式用新 Planner rollout，把轨迹加入 buffer；`--update_aux` 会按配置更新 Reward、Kernel 和 Critic。

默认 `odp_antmaze_task4` profile 对齐原 finetune 脚本：30 个 round、每轮 3 次
更新、扩散 10 步、Karras 1 步、batch 32、每个状态 8 个计划、`eta=0.8`、
reward scaling 50、`min_log_prob=-110`、lambda 更新率 0.5、前 70% loss clipping。

## 关键注意事项

- Planner、Kernel、Reward、Critic 必须使用相同的环境、`task_id`、维度和状态统计量；阶段 checkpoint 按 README 的顺序串接。
- `horizon` 同时影响 planner 窗口、critic target、kernel 约束和 adjoint matching。
- `ODP_JAX_PARITY.md` 保存了原 ODP 脚本与 JAX profile 的逐项参数对照。
- 已在 `/data/josephsuh/odp-jax-env` 的 CPU JAX 环境完成模块级检查；正式实验仍需运行 OGBench 短训练并确认 GPU 数值。
