# ODP on the FQL JAX base

`ODP-jax` keeps FQL's JAX/Flax layout and environment interface while adding
the ODP stages. The original `ODP` Torch tree is untouched.

The new code is organized as follows:

```text
odp/data.py      OGBench transition arrays -> trajectories and ODP datasets
odp/models.py    Flax planner score, reward, kernel ensemble, and critic nets
odp/agent.py     TrainState updates, SDE sampling, constraints, and adjoints
odp_main.py      planner/kernel/reward/critic pretraining and finetune driver
```

Install the FQL dependencies first:

```bash
cd ODP-jax
pip install -r requirements.txt
```

Run the four pretraining stages from the repository root. The same environment
name is used for every stage so that the state and action dimensions match:

```bash
python odp_main.py --stage planner --env_name antmaze-large-navigate-singletask-task4-v0
python odp_main.py --stage kernel --env_name antmaze-large-navigate-singletask-task4-v0 --restore_path exp/odp_jax/planner --restore_epoch 1000000
python odp_main.py --stage reward --env_name antmaze-large-navigate-singletask-task4-v0 --restore_path exp/odp_jax/kernel --restore_epoch 20000
python odp_main.py --stage critic --env_name antmaze-large-navigate-singletask-task4-v0 --restore_path exp/odp_jax/reward --restore_epoch 40000
```

The JAX implementation saves each stage under `exp/odp_jax/`. Continue from a
saved state with `--restore_path`; for example, use the planner checkpoint when
starting the finetune stage:

```bash
python odp_main.py --stage finetune \
  --env_name antmaze-large-navigate-singletask-task4-v0 \
  --restore_path exp/odp_jax/critic --restore_epoch 20
```

The planner loss follows ODP's VP score matching with a fixed initial state.
The kernel supports the robust diagonal Gaussian ensemble and the MoG ensemble
(`--kernel_type mog --kernel_num_modes 8`), the reward model is a normalized
state-action regressor, and the critic uses n-step targets with a soft target
update. Reward smoothing and scaling are available with `--reward_sigma` and
`--target_reward`. Finetuning samples plans with the ODP SDE,
adds kernel log-likelihood penalties to the learned reward, computes the
adjoint with JAX VJP products, and updates the planner by adjoint matching.
`--online_steps N --update_aux` additionally collects environment rollouts and
updates the reward, kernel, and critic buffers after each finetune round.

An isolated environment at `/data/josephsuh/odp-jax-env` now contains the FQL
JAX dependencies. It has passed synthetic CPU forward/update, robust and MoG
kernel, adjoint, finetune, and FQL-style checkpoint restore checks. A real
OGBench/GPU run still needs to be launched with the commands above.

The parameter-by-parameter comparison with the original ODP scripts is in
`ODP_JAX_PARITY.md`.

The Chinese training workflow is documented in `TRAINING_GUIDE_CN.md`.

## Pulling the `jax` branch on another service

```bash
git clone --branch jax --single-branch https://github.com/kaotty/ODP.git ODP-jax
cd ODP-jax
git branch --show-current
```

For an authenticated SSH checkout:

```bash
git clone --branch jax --single-branch git@github.com:kaotty/ODP.git ODP-jax
```

To update an existing checkout:

```bash
git fetch origin jax
git switch jax
git pull --ff-only origin jax
```

## Running on an H800 node

The current `odp_main.py` uses one JAX process and one device. Set one GPU
explicitly for a single pipeline:

```bash
python3 -m venv /data/$USER/odp-jax-env
source /data/$USER/odp-jax-env/bin/activate
pip install -r requirements.txt

# Use the CUDA wheel matching the node driver. For a CUDA 12 node:
pip install -U "jax[cuda12]"

export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0
python - <<'PY'
import jax
print(jax.devices())
PY
```

Run the stages sequentially in one output directory so each stage can restore
the previous FQL-style checkpoint:

```bash
export ENV_NAME=antmaze-large-navigate-singletask-task4-v0
export SAVE_DIR=exp/odp_jax/antmaze_task4

python odp_main.py --stage planner --env_name "$ENV_NAME" --profile odp_antmaze_task4 --save_dir "$SAVE_DIR"
python odp_main.py --stage kernel --env_name "$ENV_NAME" --profile odp_antmaze_task4 --save_dir "$SAVE_DIR" --restore_path "$SAVE_DIR/planner" --restore_epoch 1000000
python odp_main.py --stage reward --env_name "$ENV_NAME" --profile odp_antmaze_task4 --save_dir "$SAVE_DIR" --restore_path "$SAVE_DIR/kernel" --restore_epoch 20000
python odp_main.py --stage critic --env_name "$ENV_NAME" --profile odp_antmaze_task4 --save_dir "$SAVE_DIR" --restore_path "$SAVE_DIR/reward" --restore_epoch 40000
python odp_main.py --stage finetune --env_name "$ENV_NAME" --profile odp_antmaze_task4 --save_dir "$SAVE_DIR" --restore_path "$SAVE_DIR/critic" --restore_epoch 20
```

An eight H800 node can run eight independent task or hyperparameter jobs by
assigning a different `CUDA_VISIBLE_DEVICES` and `SAVE_DIR` to each job. The
current ODP-JAX agent does not yet implement synchronized `jax.pmap` updates;
launching eight copies of one pipeline would create eight unsynchronized
optimizers. Keep one GPU per pipeline until a data-parallel update layer is
added. The CUDA wheel choice should follow the installed driver and the
[official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
