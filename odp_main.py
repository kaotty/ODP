"""Train ODP's JAX implementation on top of FQL's environment interface.

Examples:
  python odp_main.py --stage planner --env_name antmaze-large-navigate-singletask-task4-v0
  python odp_main.py --stage kernel --env_name antmaze-large-navigate-singletask-task4-v0
  python odp_main.py --stage reward --env_name antmaze-large-navigate-singletask-task4-v0
  python odp_main.py --stage critic --env_name antmaze-large-navigate-singletask-task4-v0
  python odp_main.py --stage finetune --env_name antmaze-large-navigate-singletask-task4-v0
"""

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from envs.env_utils import make_env_and_datasets
from odp.agent import ODPAgent
from odp.data import (
    critic_transitions,
    full_state_normalization,
    kernel_transitions,
    make_trajectories,
    planner_windows,
    reward_transitions,
    state_normalization,
)
from utils.flax_utils import restore_agent, save_agent


def _batch(data, batch_size, rng):
    size = len(next(iter(data.values())))
    idx = rng.integers(size, size=batch_size)
    return {key: jnp.asarray(value[idx]) for key, value in data.items()}


def _save(agent, save_dir, name, step):
    stage_dir = save_dir / name
    stage_dir.mkdir(parents=True, exist_ok=True)
    save_agent(agent, str(stage_dir), step)


def _prepare(env_name, horizon, config):
    _, _, dataset, _ = make_env_and_datasets(env_name)
    reward_mode = 'ogbench' if 'singletask' in env_name else 'raw'
    trajectories = make_trajectories(dataset, reward_mode=reward_mode)
    planner_stats = state_normalization(trajectories)
    reward_stats = state_normalization(trajectories)
    kernel_stats = state_normalization(trajectories)
    critic_stats = full_state_normalization(trajectories)
    planner = planner_windows(trajectories, horizon, planner_stats)
    reward = reward_transitions(
        trajectories,
        reward_stats,
        config.get('reward_sigma'),
        config.get('target_reward'),
        config.get('reward_alpha'),
    )
    kernel = kernel_transitions(trajectories, kernel_stats)
    critic = critic_transitions(trajectories, critic_stats, horizon, config['gamma'])
    config.update({
        'critic_horizon': horizon,
        'planner_mean': jnp.asarray(planner_stats.mean),
        'planner_std': jnp.asarray(planner_stats.std),
        'reward_mean': jnp.asarray(reward_stats.mean),
        'reward_std': jnp.asarray(reward_stats.std),
        'kernel_mean': jnp.asarray(kernel_stats.mean),
        'kernel_std': jnp.asarray(kernel_stats.std),
        'critic_mean': jnp.asarray(critic_stats.mean),
        'critic_std': jnp.asarray(critic_stats.std),
    })
    return trajectories, planner, reward, kernel, critic, config


def _create_agent(planner, config, seed):
    windows = jnp.asarray(planner['windows'][: min(4, len(planner['windows']))])
    conditions = jnp.asarray(planner['conditions'][: len(windows)])
    return ODPAgent.create(seed, windows, conditions, config)


def _train_stage(
    agent,
    stage,
    data,
    steps,
    batch_size,
    seed,
    save_dir,
    gradient_accumulation=1,
):
    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        batches = [_batch(data, batch_size, rng) for _ in range(gradient_accumulation)]
        if stage == 'planner':
            if gradient_accumulation > 1:
                batch = {
                    key: jnp.stack([item[key] for item in batches])
                    for key in batches[0]
                }
                agent, info = agent.update_planner_accumulated(batch)
            else:
                agent, info = agent.update_planner(batches[0])
        elif stage == 'reward':
            agent, info = agent.update_reward(batches[0])
        elif stage == 'kernel':
            agent, info = agent.update_kernel(batches[0])
        elif stage == 'critic':
            agent, info = agent.update_critic(batches[0])
        else:
            raise ValueError(f'unknown stage: {stage}')
        if step == 1 or step % max(1, steps // 10) == 0:
            print(f'{stage} step {step}/{steps}: {info}')
        if step % 10000 == 0:
            _save(agent, save_dir, stage, step)
    _save(agent, save_dir, stage, steps)
    return agent


def _collect_rollouts(env, agent, steps, seed):
    """Collect environment trajectories using the current planner."""
    trajectories = []
    observation, _ = env.reset(seed=seed)
    observations = [np.asarray(observation, dtype=np.float32)]
    actions, rewards = [], []
    rng = jax.random.PRNGKey(seed)
    for _ in range(steps):
        rng, key = jax.random.split(rng)
        action = np.asarray(agent.sample_actions(observation, seed=key))
        next_observation, reward, terminated, truncated, _ = env.step(action.copy())
        actions.append(action)
        rewards.append(float(reward))
        observations.append(np.asarray(next_observation, dtype=np.float32))
        observation = next_observation
        if terminated or truncated:
            trajectories.append({
                'observations': np.asarray(observations),
                'actions': np.asarray(actions),
                'rewards': np.asarray(rewards, dtype=np.float32),
            })
            observation, _ = env.reset()
            observations, actions, rewards = [np.asarray(observation, dtype=np.float32)], [], []
    if actions:
        trajectories.append({
            'observations': np.asarray(observations),
            'actions': np.asarray(actions),
            'rewards': np.asarray(rewards, dtype=np.float32),
        })
    return trajectories


def _finetune(agent, env, trajectories, planner, config, args, save_dir):
    rng = np.random.default_rng(args.seed)
    jax_rng = jax.random.PRNGKey(args.seed)
    lam = float(args.initial_lambda)
    for round_id in range(1, args.rounds + 1):
        agent = agent.freeze_old_planner()
        last_plans = None
        for update in range(args.steps_per_round):
            indices = rng.integers(len(planner['conditions']), size=args.batch_size)
            conditions = jnp.asarray(planner['conditions'][indices])
            conditions = jnp.repeat(conditions, args.batch_per_sample, axis=0)
            paths, adjoints = [], []
            constraint_values = []
            for condition in conditions:
                jax_rng, sample_key = jax.random.split(jax_rng)
                path, _ = agent.sample_plan_path(condition, sample_key, use_ema=False)
                global_update = (round_id - 1) * args.steps_per_round + update + 1
                if args.alpha_decay:
                    decay_rate = max(args.alpha_total_steps / 10.0, 1.0)
                    alpha = args.alpha_end + (
                        args.alpha_start - args.alpha_end
                    ) * np.exp(-global_update / decay_rate)
                    alpha = max(alpha, args.alpha_end)
                else:
                    alpha = args.alpha_start
                adjoint = agent.compute_adjoint(path, condition, lam, alpha=alpha)
                paths.append(path)
                adjoints.append(adjoint)
                _, constraint = agent._reward_constraint(path[-1], lam)
                constraint_values.append(float(jnp.mean(constraint)))
            agent, info = agent.finetune_step(
                jnp.stack(paths), conditions, jnp.stack(adjoints)
            )
            last_plans = jnp.stack([path[-1] for path in paths])
            mean_constraint = float(np.mean(constraint_values))
            lam = max(
                args.initial_lambda,
                min(5.0, lam + args.lambda_lr * mean_constraint),
            )
            print(
                f'finetune round {round_id}/{args.rounds}, update {update + 1}/{args.steps_per_round}: '
                f'loss={info}, constraint={mean_constraint:.4f}, lambda={lam:.4f}'
            )

        if args.update_critic and last_plans is not None:
            agent, critic_info = agent.update_critic_from_plans(last_plans)
            print(f'critic update round {round_id}/{args.rounds}: {critic_info}')

        if args.online_steps > 0:
            new_trajectories = _collect_rollouts(env, agent, args.online_steps, args.seed + round_id)
            trajectories.extend(new_trajectories)
            planner = planner_windows(
                trajectories, args.horizon, state_normalization(trajectories)
            )
            if args.update_aux and new_trajectories:
                reward_stats = state_normalization(trajectories)
                kernel_stats = state_normalization(trajectories)
                critic_stats = full_state_normalization(trajectories)
                agent, _ = _train_stage(
                    agent, 'reward',
                    reward_transitions(
                        trajectories,
                        reward_stats,
                        config.get('reward_sigma'),
                        config.get('target_reward'),
                        config.get('reward_alpha'),
                    ),
                    args.aux_reward_steps, args.batch_size, args.seed + round_id, save_dir,
                )
                agent, _ = _train_stage(
                    agent, 'kernel', kernel_transitions(trajectories, kernel_stats),
                    args.aux_kernel_steps, args.batch_size, args.seed + round_id, save_dir,
                )
                if not args.update_critic:
                    agent, _ = _train_stage(
                        agent, 'critic',
                        critic_transitions(trajectories, critic_stats, args.horizon, config['gamma']),
                        args.aux_critic_steps, args.batch_size, args.seed + round_id, save_dir,
                    )
        _save(agent, save_dir, 'finetune', round_id)
    return agent


def main():
    parser = argparse.ArgumentParser(description='JAX ODP training pipeline')
    parser.add_argument('--stage', choices=['planner', 'reward', 'kernel', 'critic', 'finetune'], required=True)
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--profile', choices=['odp_antmaze_task4', 'custom'], default='odp_antmaze_task4')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--aux_steps', type=int, default=None)
    parser.add_argument('--steps_per_round', type=int, default=None)
    parser.add_argument('--batch_per_sample', type=int, default=None)
    parser.add_argument('--planner_lr', type=float, default=None)
    parser.add_argument('--reward_lr', type=float, default=None)
    parser.add_argument('--kernel_lr', type=float, default=None)
    parser.add_argument('--critic_lr', type=float, default=None)
    parser.add_argument('--reward_min_lr', type=float, default=None)
    parser.add_argument('--critic_min_lr', type=float, default=None)
    parser.add_argument('--critic_q_scale', type=float, default=1.0)
    parser.add_argument('--reward_scaling_factor', type=float, default=None)
    parser.add_argument('--min_log_prob', type=float, default=None)
    parser.add_argument('--finetune_diffusion_steps', type=int, default=None)
    parser.add_argument('--num_karras', type=int, default=None)
    parser.add_argument('--eta', type=float, default=None)
    parser.add_argument('--loss_clip_percent', type=float, default=None)
    parser.add_argument('--alpha_start', type=float, default=1.0)
    parser.add_argument('--alpha_end', type=float, default=0.1)
    parser.add_argument('--alpha_total_steps', type=int, default=300)
    parser.add_argument('--alpha_decay', action='store_true')
    parser.add_argument('--online_steps', type=int, default=0)
    parser.add_argument('--update_aux', action='store_true')
    parser.add_argument('--kernel_type', choices=['robust', 'mog'], default=None)
    parser.add_argument('--kernel_num_modes', type=int, default=None)
    parser.add_argument('--reward_sigma', type=float, default=None)
    parser.add_argument('--reward_alpha', type=float, default=None)
    parser.add_argument('--target_reward', type=float, default=None)
    parser.add_argument('--initial_lambda', type=float, default=None)
    parser.add_argument('--lambda_lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_dir', default='exp/odp_jax')
    parser.add_argument('--restore_path', default=None, help='FQL-style checkpoint directory.')
    parser.add_argument('--restore_epoch', type=int, default=None)
    args = parser.parse_args()

    if args.profile == 'odp_antmaze_task4':
        profile = {
            'planner_steps': 1_000_000,
            'planner_batch_size': 128,
            'planner_lr': 2e-4,
            'reward_steps': 40_000,
            'reward_batch_size': 256,
            'reward_lr': 5e-5,
            'reward_min_lr': 5e-8,
            'reward_sigma': 3.0,
            'target_reward': 200.0,
            'kernel_steps': 20_000,
            'kernel_batch_size': 1024,
            'kernel_lr': 1e-4,
            'kernel_type': 'mog',
            'kernel_ensemble_size': 10,
            'kernel_hidden_dims': (514, 514, 514, 514),
            'kernel_num_modes': 10,
            'kernel_noise_floor': 5e-4,
            'critic_steps': 20,
            'critic_batch_size': 256,
            'critic_lr': 1e-4,
            'critic_min_lr': 1e-6,
            'critic_hidden_dims': (512, 512, 512, 512, 512),
            'rounds': 30,
            'steps_per_round': 3,
            'batch_size': 32,
            'batch_per_sample': 8,
            'finetune_diffusion_steps': 10,
            'num_karras': 1,
            'eta': 0.8,
            'loss_clip_percent': 0.7,
            'reward_scaling_factor': 50.0,
            'min_log_prob': -110.0,
            'initial_lambda': 0.0,
            'lambda_lr': 0.5,
            'update_critic': True,
            'critic_use_symlog': True,
        }
    else:
        profile = {}

    def pick(name, default=None):
        value = getattr(args, name, None)
        return value if value is not None else profile.get(name, default)

    args.rounds = pick('rounds', args.rounds)
    args.steps_per_round = pick('steps_per_round', 1)
    args.batch_per_sample = pick('batch_per_sample', 1)
    requested_batch_size = args.batch_size
    args.batch_size = pick('batch_size', 256)
    args.stage_batch_size = requested_batch_size
    requested_aux_steps = args.aux_steps
    args.aux_steps = pick('aux_steps', 100)
    if requested_aux_steps is None:
        args.aux_reward_steps = profile.get('reward_steps', args.aux_steps)
        args.aux_kernel_steps = profile.get('kernel_steps', args.aux_steps)
        args.aux_critic_steps = profile.get('critic_steps', args.aux_steps)
    else:
        args.aux_reward_steps = args.aux_kernel_steps = args.aux_critic_steps = args.aux_steps
    args.steps = pick('steps', None)
    if args.steps is None:
        args.steps = profile.get(f'{args.stage}_steps', args.rounds * args.steps_per_round)
    args.planner_lr = pick('planner_lr', 2e-4)
    args.reward_lr = pick('reward_lr', 2e-4)
    args.kernel_lr = pick('kernel_lr', 1e-3)
    args.critic_lr = pick('critic_lr', 5e-5)
    args.reward_min_lr = pick('reward_min_lr', 1e-5)
    args.critic_min_lr = pick('critic_min_lr', 1e-5)
    args.reward_sigma = pick('reward_sigma', None)
    args.reward_alpha = pick('reward_alpha', None)
    args.target_reward = pick('target_reward', None)
    args.reward_scaling_factor = pick('reward_scaling_factor', 50.0)
    args.min_log_prob = pick('min_log_prob', -10.0)
    args.initial_lambda = pick('initial_lambda', 0.01)
    args.lambda_lr = pick('lambda_lr', 0.001)
    args.update_critic = profile.get('update_critic', False)
    args.kernel_type = pick('kernel_type', 'robust')
    args.kernel_num_modes = pick('kernel_num_modes', 8)
    if args.stage_batch_size is None:
        args.stage_batch_size = profile.get(f'{args.stage}_batch_size', args.batch_size)
    args.finetune_diffusion_steps = pick('finetune_diffusion_steps', 30)
    args.num_karras = pick('num_karras', 2)
    args.eta = pick('eta', 0.8)
    args.loss_clip_percent = pick('loss_clip_percent', 0.0)

    config = {
        'gamma': 0.99,
        'critic_gamma': 0.99,
        'schedule_offset': 0.008,
        'diffusion_steps': args.finetune_diffusion_steps,
        'num_karras': args.num_karras,
        'sigma_min': 0.01,
        'sigma_max': 30.0,
        'eta': args.eta,
        'min_log_prob': args.min_log_prob,
        'constraint_beta': 1.0,
        'kernel_noise_floor': 5e-4 if args.kernel_type == 'mog' else 1e-2,
        'kernel_disagreement_weight': 1e-3,
        'kernel_type': args.kernel_type,
        'kernel_num_modes': args.kernel_num_modes,
        'critic_tau': 0.005,
        'critic_use_symlog': profile.get('critic_use_symlog', False),
        'critic_q_scale': args.critic_q_scale,
        'use_critic': profile.get('update_critic', False),
        'reward_scaling_factor': args.reward_scaling_factor,
        'loss_clip_steps': math.ceil(args.finetune_diffusion_steps * args.loss_clip_percent) + 1,
        'planner_lr': args.planner_lr,
        'planner_gradient_accumulation': 2 if args.profile == 'odp_antmaze_task4' else 1,
        'reward_lr': args.reward_lr,
        'kernel_lr': args.kernel_lr,
        'critic_lr': args.critic_lr,
        'reward_min_lr': args.reward_min_lr,
        'critic_min_lr': args.critic_min_lr,
        'planner_steps': profile.get('planner_steps', 1_000_000),
        'reward_steps': profile.get('reward_steps', 40_000),
        'kernel_steps': profile.get('kernel_steps', 5_000),
        'critic_steps': profile.get('critic_steps', 3_000),
        'planner_hidden_dims': profile.get('planner_hidden_dims', (256, 256, 256, 256)),
        'reward_hidden_dims': profile.get('reward_hidden_dims', (512, 512, 512, 512, 512)),
        'kernel_hidden_dims': profile.get('kernel_hidden_dims', (256, 256)),
        'critic_hidden_dims': profile.get('critic_hidden_dims', (512, 512, 512, 512, 512)),
        'kernel_ensemble_size': profile.get('kernel_ensemble_size', 10),
        'planner_emb_dim': 128,
        'planner_d_model': 256,
        'planner_heads': 4,
        'planner_depth': 4,
        'reward_sigma': args.reward_sigma,
        'reward_alpha': args.reward_alpha,
        'target_reward': args.target_reward,
    }
    env, _, _dataset, _ = make_env_and_datasets(args.env_name)
    trajectories, planner, reward, kernel, critic, config = _prepare(
        args.env_name, args.horizon, config
    )
    agent = _create_agent(planner, config, args.seed)
    if args.restore_path is not None:
        if args.restore_epoch is None:
            raise ValueError('--restore_epoch is required with --restore_path')
        agent = restore_agent(agent, args.restore_path, args.restore_epoch)
    save_dir = Path(args.save_dir)

    if args.stage == 'planner':
        agent = _train_stage(
            agent,
            'planner',
            planner,
            args.steps,
            args.stage_batch_size,
            args.seed,
            save_dir,
            gradient_accumulation=config['planner_gradient_accumulation'],
        )
    elif args.stage == 'reward':
        agent = _train_stage(agent, 'reward', reward, args.steps, args.stage_batch_size, args.seed, save_dir)
    elif args.stage == 'kernel':
        agent = _train_stage(agent, 'kernel', kernel, args.steps, args.stage_batch_size, args.seed, save_dir)
    elif args.stage == 'critic':
        agent = _train_stage(agent, 'critic', critic, args.steps, args.stage_batch_size, args.seed, save_dir)
    else:
        agent = _finetune(agent, env, trajectories, planner, config, args, save_dir)
    _save(agent, save_dir, 'final', args.steps)


if __name__ == '__main__':
    main()
