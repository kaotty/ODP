# ODP 与 ODP-jax 参数校对

这个 profile 对应当前仓库里用于 antmaze large task4 的主训练脚本：
`Pretrain/pretrain_script2.py`、`Pretrain/train_kernel_script2.py`、
`Pretrain/train_reward_script2.py` 和 `Finetuning/finetune_script.py`。

| 阶段 | 原 ODP 配置 | ODP-jax `odp_antmaze_task4` profile |
| --- | --- | --- |
| Planner | `horizon=32`, DiT `emb=128`, `d_model=256`, `heads=4`, `depth=4`, `steps=1,000,000`, `batch=128`, `lr=2e-4`, AdamW weight decay `1e-5`, gradient accumulation `2`, EMA `0.9999`, EMA start `1000`, update every `2` | 相同；Flax DiT 使用相同的输入/输出维度、Fourier time embedding、adaLN-Zero、深度和 attention heads |
| Kernel | MoG ensemble，`ensemble=10`, `modes=10`, hidden `4×514`, `batch=1024`, `steps=20,000`, `lr=1e-4`, `noise_floor=5e-4`, disagreement regularization `1e-3` | 相同；通过 `--kernel_type mog --kernel_num_modes 10` 选择 |
| Reward | `SimpleReward`，hidden `4×512`, `batch=256`, `steps=40,000`, `lr=5e-5`, `min_lr=5e-8`, `sigma=3`, `target_reward=200`, AdamW weight decay `1e-4` | 相同；state normalization、Gaussian smoothing 和 target scaling 均保留 |
| Critic | hidden `4×512`, `batch=256`, `steps=20`, `lr=1e-4`, `min_lr=1e-6`, `gamma=0.99`, `tau=0.005`；Planner7 使用 symlog target 和 Q scale | 相同；使用 n-step target、plan bootstrap、symlog/Q-scale 选项和 soft target update |
| Finetune | `rounds=30`, `steps=90`, 每轮 `3` 次更新，`horizon=32`, diffusion steps `10`, Karras steps `1`, `eta=.8`, batch `32`, 每个初始状态重复 `8` 次，lr `2e-5`, reward scale `50`, `min_log_prob=-110`, loss clip `70%`, 初始 lambda `0`, lambda update rate `.5` | 相同；`odp_main.py --profile odp_antmaze_task4` 自动设置这些值 |

表中的 `hidden=4×512` 表示原 Torch 参数 `hidden_layers=4`；原网络还包含一个输入 hidden block，因此 JAX 配置使用五个 512 宽的 MLP hidden blocks 来保持层数一致。Kernel 的 `num_hidden_layers=4` 则对应四个 514 宽 backbone blocks。

逻辑对应关系也保持一致：Planner 用 VP score matching，Kernel 用 ensemble
转移 likelihood 形成软约束，Reward 对状态动作输出即时奖励，Critic 使用
n-step bootstrap；finetune 采样完整扩散路径，用旧 Planner 的 VJP/Jacobian
transpose 求 adjoint，再用 vector-field matching 更新新 Planner。在线模式
在每轮 rollout 后更新 replay trajectories，并按 `--update_aux` 重训辅助模型。

两处实现差异来自框架迁移：Flax attention 和参数初始化与 PyTorch 的底层
随机数实现不同，模型权重不能直接互相加载；训练目标、张量形状、默认超参、
归一化、schedule、EMA、约束和 finetune 计算顺序保持对应。已在
`/data/josephsuh/odp-jax-env` 的 CPU JAX 环境验证 Planner/Reward/Kernel/Critic
更新、robust/MoG Kernel、adjoint、finetune 和 FQL checkpoint restore；真实
OGBench/GPU 数值还需要短训练确认。
