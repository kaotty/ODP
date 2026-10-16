"""JAX ODP agent.

The implementation keeps the ODP stages separate while using FQL's Flax
``TrainState`` and network conventions.  Planner, reward, kernel, and critic
updates can therefore be scheduled independently by ``odp_main.py``.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from odp.models import CriticModel, KernelEnsemble, MoGKernelEnsemble, RewardModel, ScoreModel
from utils.flax_utils import TrainState, nonpytree_field


def cosine_alpha_sigma(t, offset=0.008):
    t = jnp.clip(t, 0.0, 1.0 - 1e-3)
    factor = (t + offset) / (1.0 + offset)
    f_t = jnp.cos(factor * jnp.pi / 2) ** 2
    f_0 = jnp.cos((offset / (1.0 + offset)) * jnp.pi / 2) ** 2
    alpha_bar = jnp.clip(f_t / f_0, 0.0, 1.0 - 1e-3)
    return jnp.sqrt(alpha_bar), jnp.sqrt(1.0 - alpha_bar)


def cosine_beta(t, offset=0.008):
    t = jnp.clip(t, 0.0, 1.0 - 1e-3)
    angle = jnp.pi / 2 * ((t + offset) / (1.0 + offset))
    return jnp.pi / (1.0 + offset) * jnp.tan(angle)


def karras_beta_schedule(num_steps, sigma_min=0.01, sigma_max=30.0):
    if num_steps < 1 or not (0 < sigma_min < sigma_max):
        raise ValueError('invalid Karras schedule arguments')
    times = jnp.linspace(1.0, 0.0, num_steps + 1)
    rho = jnp.log(sigma_max / sigma_min)
    sigma_k = sigma_min * jnp.exp(rho * times)
    sigma_sq = sigma_k ** 2 / (1.0 + sigma_k ** 2)
    return times, 2.0 * rho * sigma_sq, jnp.sqrt(sigma_sq)


def _smooth_l1(diff):
    abs_diff = jnp.abs(diff)
    return jnp.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)


def _symlog(value):
    return jnp.sign(value) * jnp.log1p(jnp.abs(value))


def _symexp(value):
    return jnp.sign(value) * jnp.expm1(jnp.abs(value))


def _optimizer(config, prefix):
    learning_rate = config[f'{prefix}_lr']
    steps = max(1, int(config[f'{prefix}_steps']))
    min_lr = config[f'{prefix}_min_lr']
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=steps,
        alpha=min_lr / max(learning_rate, 1e-12),
    )
    return optax.adamw(schedule, weight_decay=config[f'{prefix}_weight_decay'])


def _ema_update(current, previous, step, decay, start_step, update_every):
    def warm_start(_):
        return current

    def update(_):
        return jax.lax.cond(
            step % update_every == 0,
            lambda __: jax.tree_util.tree_map(
                lambda p, q: decay * q + (1.0 - decay) * p,
                current,
                previous,
            ),
            lambda __: previous,
            operand=None,
        )

    return jax.lax.cond(step < start_step, warm_start, update, operand=None)


class ODPAgent(flax.struct.PyTreeNode):
    """ODP's four trainable components and frozen old planner parameters."""

    rng: Any
    planner: Any
    reward: Any
    kernel: Any
    critic: Any
    target_critic: Any
    planner_ema_params: Any
    old_planner_params: Any
    planner_mean: Any
    planner_std: Any
    reward_mean: Any
    reward_std: Any
    kernel_mean: Any
    kernel_std: Any
    critic_mean: Any
    critic_std: Any
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, example_window, example_condition, config):
        config = {
            'planner_lr': 2e-4,
            'planner_hidden_dims': (256, 256, 256, 256),
            'planner_depth': 4,
            'planner_min_lr': 0.0,
            'planner_steps': 1_000_000,
            'planner_weight_decay': 1e-5,
            'reward_lr': 2e-4,
            'reward_hidden_dims': (512, 512, 512, 512, 512),
            'reward_min_lr': 1e-5,
            'reward_steps': 40_000,
            'reward_weight_decay': 1e-4,
            'kernel_lr': 1e-3,
            'kernel_hidden_dims': (256, 256),
            'kernel_ensemble_size': 10,
            'kernel_min_lr': 0.0,
            'kernel_steps': 5_000,
            'kernel_weight_decay': 1e-5,
            'critic_lr': 5e-5,
            'critic_hidden_dims': (512, 512, 512, 512, 512),
            'critic_min_lr': 1e-5,
            'critic_steps': 3_000,
            'critic_weight_decay': 0.0,
            'schedule_offset': 0.008,
            'diffusion_steps': 30,
            'num_karras': 2,
            'sigma_min': 0.01,
            'sigma_max': 30.0,
            'eta': 0.8,
            'kernel_noise_floor': 1e-2,
            'kernel_disagreement_weight': 1e-3,
            'kernel_type': 'robust',
            'kernel_num_modes': 8,
            'critic_tau': 0.005,
            'critic_horizon': 32,
            'critic_use_symlog': False,
            'critic_q_scale': 1.0,
            'use_critic': False,
            'ema_decay': 0.9999,
            'ema_start_step': 1000,
            'ema_update_every': 2,
            'gamma': 0.99,
            'critic_gamma': 0.99,
            'min_log_prob': -10.0,
            'constraint_beta': 1.0,
            'reward_scaling_factor': 50.0,
            'loss_clip_steps': 0,
            **dict(config),
        }
        stats = {
            key: config.pop(key, default)
            for key, default in {
                'planner_mean': jnp.zeros((example_condition.shape[-1],)),
                'planner_std': jnp.ones((example_condition.shape[-1],)),
                'reward_mean': jnp.zeros((example_condition.shape[-1],)),
                'reward_std': jnp.ones((example_condition.shape[-1],)),
                'kernel_mean': jnp.zeros((example_condition.shape[-1],)),
                'kernel_std': jnp.ones((example_condition.shape[-1],)),
                'critic_mean': jnp.zeros((example_condition.shape[-1],)),
                'critic_std': jnp.ones((example_condition.shape[-1],)),
            }.items()
        }
        rng = jax.random.PRNGKey(seed)
        rng, planner_rng, reward_rng, kernel_rng, critic_rng = jax.random.split(rng, 5)
        horizon, transition_dim = example_window.shape[1:]
        state_dim = example_condition.shape[-1]
        action_dim = transition_dim - state_dim

        planner_def = ScoreModel(
            horizon=horizon,
            transition_dim=transition_dim,
            state_dim=state_dim,
            emb_dim=int(config.get('planner_emb_dim', 128)),
            d_model=int(config.get('planner_d_model', 256)),
            n_heads=int(config.get('planner_heads', 4)),
            depth=int(config.get('planner_depth', 4)),
        )
        reward_def = RewardModel(tuple(config.get('reward_hidden_dims', (256, 256, 256))))
        kernel_class = MoGKernelEnsemble if config.get('kernel_type') == 'mog' else KernelEnsemble
        kernel_kwargs = {
            'state_dim': state_dim,
            'ensemble_size': int(config.get('kernel_ensemble_size', 5)),
            'hidden_dims': tuple(config.get('kernel_hidden_dims', (256, 256))),
        }
        if config.get('kernel_type') == 'mog':
            kernel_kwargs['num_modes'] = int(config.get('kernel_num_modes', 8))
        kernel_def = kernel_class(**kernel_kwargs)
        critic_def = CriticModel(tuple(config.get('critic_hidden_dims', (256, 256))))

        time_example = jnp.zeros((len(example_window), 1), dtype=jnp.float32)

        planner = TrainState.create(
            planner_def,
            planner_def.init(
                planner_rng,
                example_window,
                example_condition,
                time_example,
            )['params'],
            tx=_optimizer(config, 'planner'),
        )
        reward = TrainState.create(
            reward_def,
            reward_def.init(
                reward_rng,
                example_condition,
                jnp.zeros((len(example_window), action_dim)),
            )['params'],
            tx=_optimizer(config, 'reward'),
        )
        kernel = TrainState.create(
            kernel_def,
            kernel_def.init(
                kernel_rng,
                example_condition,
                jnp.zeros((len(example_window), action_dim)),
            )['params'],
            tx=_optimizer(config, 'kernel'),
        )
        critic = TrainState.create(
            critic_def,
            critic_def.init(critic_rng, example_condition)['params'],
            tx=_optimizer(config, 'critic'),
        )
        target_critic = critic.replace(tx=None, opt_state=None)
        config.update({
            'horizon': horizon,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'transition_dim': transition_dim,
        })
        return cls(
            rng=rng,
            planner=planner,
            reward=reward,
            kernel=kernel,
            critic=critic,
            target_critic=target_critic,
            planner_ema_params=planner.params,
            old_planner_params=planner.params,
            planner_mean=jnp.asarray(stats['planner_mean']),
            planner_std=jnp.asarray(stats['planner_std']),
            reward_mean=jnp.asarray(stats['reward_mean']),
            reward_std=jnp.asarray(stats['reward_std']),
            kernel_mean=jnp.asarray(stats['kernel_mean']),
            kernel_std=jnp.asarray(stats['kernel_std']),
            critic_mean=jnp.asarray(stats['critic_mean']),
            critic_std=jnp.asarray(stats['critic_std']),
            config=flax.core.FrozenDict(config),
        )

    def _planner_loss(self, params, batch, rng):
        windows = batch['windows']
        conditions = batch['conditions']
        batch_size = windows.shape[0]
        rng, noise_rng, time_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(noise_rng, windows.shape)
        times = jax.random.uniform(time_rng, (batch_size, 1), minval=1e-5, maxval=1.0 - 1e-5)
        alpha, sigma = cosine_alpha_sigma(times[:, 0], self.config['schedule_offset'])
        alpha = alpha[:, None, None]
        sigma = sigma[:, None, None]
        noisy = alpha * windows + sigma * noise
        mask = jnp.zeros_like(noisy).at[:, 0, :self.config['state_dim']].set(1.0)
        conditioned = jnp.zeros_like(noisy).at[:, 0, :self.config['state_dim']].set(conditions)
        noisy = mask * conditioned + (1.0 - mask) * noisy
        target = -(noisy - alpha * windows) / (sigma ** 2 + 1e-8)
        prediction = self.planner(noisy, conditions, times, params=params)
        error = (prediction - target) * (1.0 - mask)
        per_sample = jnp.sum(error ** 2, axis=(1, 2))
        weight = sigma[:, 0, 0] ** 2
        denominator = self.config['horizon'] * self.config['transition_dim'] - self.config['state_dim']
        return jnp.mean(weight * per_sample) / denominator

    @jax.jit
    def update_planner(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        state, info = self.planner.apply_loss_fn(
            loss_fn=lambda p: (self._planner_loss(p, batch, rng), {}),
        )
        ema_params = _ema_update(
            state.params,
            self.planner_ema_params,
            state.step,
            self.config['ema_decay'],
            self.config['ema_start_step'],
            self.config['ema_update_every'],
        )
        return self.replace(
            planner=state,
            planner_ema_params=ema_params,
            rng=new_rng,
        ), {'planner_loss': info}

    @jax.jit
    def update_planner_accumulated(self, batches):
        new_rng, rng = jax.random.split(self.rng)
        batch_count = batches['windows'].shape[0]
        keys = jax.random.split(rng, batch_count)

        def loss_fn(params):
            losses = jax.vmap(
                lambda batch, key: self._planner_loss(params, batch, key)
            )(batches, keys)
            return jnp.mean(losses), {'planner_loss': jnp.mean(losses)}

        state, info = self.planner.apply_loss_fn(loss_fn=loss_fn)
        ema_params = _ema_update(
            state.params,
            self.planner_ema_params,
            state.step,
            self.config['ema_decay'],
            self.config['ema_start_step'],
            self.config['ema_update_every'],
        )
        return self.replace(
            planner=state,
            planner_ema_params=ema_params,
            rng=new_rng,
        ), info

    @jax.jit
    def update_reward(self, batch):
        def loss_fn(params):
            prediction = self.reward(
                batch['observations'], batch['actions'], params=params
            )
            return jnp.mean(_smooth_l1(prediction - batch['rewards'])), {}
        state, info = self.reward.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(reward=state), {'reward_loss': info}

    @jax.jit
    def update_kernel(self, batch):
        def loss_fn(params):
            outputs = self.kernel(batch['observations'], batch['actions'], params=params)
            if self.config['kernel_type'] == 'mog':
                means, log_stds, weights = outputs
                variance = jnp.exp(2.0 * log_stds) + self.config['kernel_noise_floor']
                residual = jnp.clip(batch['next_observations'][None, :, None, None] - means, -10.0, 10.0)
                mode_log_prob = -0.5 * jnp.sum(
                    jnp.log(2.0 * jnp.pi * variance) + residual ** 2 / variance,
                    axis=-1,
                )
                member_log_prob = jax.nn.logsumexp(
                    mode_log_prob + jnp.log(weights + 1e-8), axis=-1
                )
                loss = -jnp.mean(member_log_prob)
                disagreement = jnp.mean((means - means.mean(axis=2, keepdims=True)) ** 2)
            else:
                means, log_stds = outputs
                variance = jnp.exp(2.0 * log_stds) + self.config['kernel_noise_floor']
                residual = jnp.clip(batch['next_observations'][None] - means, -10.0, 10.0)
                nll = 0.5 * (jnp.log(2.0 * jnp.pi * variance) + residual ** 2 / variance)
                loss = jnp.mean(jnp.sum(nll, axis=-1))
                disagreement = jnp.mean((means - means.mean(axis=0, keepdims=True)) ** 2)
            loss = loss + self.config['kernel_disagreement_weight'] * disagreement
            return loss, {}
        state, info = self.kernel.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(kernel=state), {'kernel_loss': info}

    @jax.jit
    def update_critic(self, batch):
        next_value = self.target_critic(batch['next_observations'])
        if self.config['critic_use_symlog']:
            next_value = _symexp(next_value)
        target = batch['returns'] + (
            self.config['gamma'] ** self.config['critic_horizon']
        ) * batch['masks'] * next_value
        if self.config['critic_use_symlog']:
            target = _symlog(target)
        target = jax.lax.stop_gradient(target)

        def loss_fn(params):
            prediction = self.critic(batch['observations'], params=params)
            return jnp.mean(_smooth_l1(prediction - target)), {}
        state, info = self.critic.apply_loss_fn(loss_fn=loss_fn)
        target_params = jax.tree_util.tree_map(
            lambda p, q: self.config['critic_tau'] * p + (1.0 - self.config['critic_tau']) * q,
            state.params,
            self.target_critic.params,
        )
        target_critic = self.target_critic.replace(params=target_params)
        return self.replace(critic=state, target_critic=target_critic), {'critic_loss': info}

    @jax.jit
    def update_critic_from_plans(self, plans):
        state_dim = self.config['state_dim']
        states = plans[:, :, :state_dim]
        actions = plans[:, :, state_dim:]
        reward_states = (states - self.reward_mean) / self.reward_std
        rewards = self.reward(reward_states, actions)
        targets = []
        for length in range(1, self.config['horizon']):
            discounts = self.config['critic_gamma'] ** jnp.arange(length)
            returns = jnp.sum(rewards[:, :length] * discounts[None], axis=1)
            if self.config['critic_use_symlog']:
                returns = returns / self.config['critic_q_scale']
            raw_state = states[:, length] * self.planner_std + self.planner_mean
            critic_state = (raw_state - self.critic_mean) / self.critic_std
            bootstrap = self.target_critic(critic_state)
            if self.config['critic_use_symlog']:
                bootstrap = _symexp(bootstrap)
            targets.append(returns + self.config['critic_gamma'] ** length * bootstrap)
        target = jax.lax.stop_gradient(jnp.mean(jnp.stack(targets, axis=1), axis=1))
        if self.config['critic_use_symlog']:
            target = _symlog(target)
        initial_raw = states[:, 0] * self.planner_std + self.planner_mean
        initial_state = (initial_raw - self.critic_mean) / self.critic_std

        def loss_fn(params):
            prediction = self.critic(initial_state, params=params)
            return jnp.mean(_smooth_l1(prediction - target)), {}

        state, info = self.critic.apply_loss_fn(loss_fn=loss_fn)
        target_params = jax.tree_util.tree_map(
            lambda p, q: self.config['critic_tau'] * p + (1.0 - self.config['critic_tau']) * q,
            state.params,
            self.target_critic.params,
        )
        return self.replace(
            critic=state,
            target_critic=self.target_critic.replace(params=target_params),
        ), {'critic_plan_loss': info}

    def freeze_old_planner(self):
        return self.replace(old_planner_params=self.planner_ema_params)

    def sample_plan_path(self, condition, seed, eta=None, use_ema=True):
        """Sample a planner trajectory and retain every diffusion state."""
        eta = self.config['eta'] if eta is None else eta
        times, beta_karras, sigma_grid = karras_beta_schedule(
            self.config['diffusion_steps'], self.config['sigma_min'], self.config['sigma_max']
        )
        beta_cosine = cosine_beta(times, self.config['schedule_offset'])
        planner_params = self.planner_ema_params if use_ema else self.planner.params
        key, noise_key = jax.random.split(seed)
        x = jax.random.normal(
            noise_key, (self.config['horizon'], self.config['transition_dim'])
        ) * sigma_grid[0]
        mask = jnp.zeros_like(x).at[0, :self.config['state_dim']].set(1.0)
        conditioned = jnp.zeros_like(x).at[0, :self.config['state_dim']].set(condition)
        x = mask * conditioned + (1.0 - mask) * x
        path = [x]
        for i in range(self.config['diffusion_steps']):
            t_now, t_next = times[i], times[i + 1] if i + 1 < len(times) else 0.0
            dt = t_next - t_now
            beta = beta_karras[i] if i < self.config['num_karras'] else beta_cosine[i]
            score = self.planner(
                x[None], condition[None], jnp.asarray([[t_now]]),
                params=planner_params,
            )[0]
            drift = -0.5 * beta * x
            step_noise = jax.random.normal(jax.random.fold_in(key, i), x.shape)
            x = x + (drift - beta * score) * dt
            if eta > 0:
                x = x + eta * jnp.sqrt(beta * (-dt)) * step_noise
            x = mask * conditioned + (1.0 - mask) * x
            x = x.at[:, self.config['state_dim']:].set(
                jnp.clip(x[:, self.config['state_dim']:], -1.0, 1.0)
            )
            path.append(x)
        return jnp.stack(path), times

    def _reward_constraint(self, plan, lam):
        state_dim = self.config['state_dim']
        states = plan[:, :state_dim]
        actions = plan[:, state_dim:]
        reward_states = (states - self.reward_mean) / self.reward_std
        kernel_states = (states - self.kernel_mean) / self.kernel_std
        rewards = self.reward(reward_states, actions)
        outputs = self.kernel(kernel_states[:-1], actions[:-1])
        if self.config['kernel_type'] == 'mog':
            means, log_stds, weights = outputs
            variance = jnp.exp(2.0 * log_stds) + self.config['kernel_noise_floor']
            residual = jnp.clip(kernel_states[1:][None, :, None, :] - means, -10.0, 10.0)
            mode_log_prob = -0.5 * jnp.sum(
                residual ** 2 / variance + jnp.log(2.0 * jnp.pi * variance), axis=-1
            )
            log_prob = jax.nn.logsumexp(
                mode_log_prob + jnp.log(weights + 1e-8), axis=-1
            ).mean(axis=0)
        else:
            means, log_stds = outputs
            variance = jnp.exp(2.0 * log_stds) + self.config['kernel_noise_floor']
            residual = jnp.clip(kernel_states[1:][None] - means, -10.0, 10.0)
            log_prob = -0.5 * jnp.sum(
                residual ** 2 / variance + jnp.log(2.0 * jnp.pi * variance), axis=-1
            ).mean(axis=0)
        beta = self.config['constraint_beta']
        constraint_input = self.config['min_log_prob'] - log_prob.mean(axis=0)
        constraint = jax.nn.softplus(beta * constraint_input) / beta
        discounts = self.config['critic_gamma'] ** jnp.arange(len(rewards))
        objective = self.config['reward_scaling_factor'] * (
            jnp.mean(discounts * rewards) - lam * jnp.mean(constraint)
        )
        if self.config['use_critic']:
            critic_state = (
                states[-1] * self.planner_std + self.planner_mean - self.critic_mean
            ) / self.critic_std
            critic_value = self.critic(critic_state[None])[0]
            if self.config['critic_use_symlog']:
                critic_value = _symexp(critic_value)
            objective = objective + (
                self.config['critic_gamma'] ** (len(states) - 1)
                * self.config['critic_q_scale'] * critic_value
            )
        return objective, constraint

    def compute_adjoint(self, path, condition, lam, alpha=1.0):
        """Solve ODP's discrete adjoint with JAX VJP products."""
        terminal = path[-1]
        _, terminal_grad = jax.value_and_grad(
            lambda plan: self._reward_constraint(plan, lam)[0]
        )(terminal)
        adjoints = [-terminal_grad / alpha]
        reversed_path = path[::-1]
        reversed_times = jnp.linspace(0.0, 1.0, path.shape[0])
        for i in range(path.shape[0] - 1):
            t_now = reversed_times[i]
            t_next = reversed_times[i + 1]
            dt = t_now - t_next
            k = -0.5 * cosine_beta(t_now, self.config['schedule_offset'])
            state = reversed_path[i]

            def score_fn(value, t_now=t_now):
                return self.planner(
                    value[None], condition[None], jnp.asarray([[t_now]]),
                    params=self.old_planner_params,
                )[0]

            _, pullback = jax.vjp(score_fn, state)
            jt_adjoint = pullback(adjoints[-1])[0]
            current = adjoints[-1] + dt * (k * adjoints[-1] + 2.0 * k * jt_adjoint)
            current = current.at[0, :self.config['state_dim']].set(0.0)
            adjoints.append(current)
        return jnp.stack(adjoints[::-1])

    @jax.jit
    def finetune_step(self, paths, conditions, adjoints):
        times = jnp.linspace(1.0, 0.0, paths.shape[1])
        k = -0.5 * cosine_beta(times, self.config['schedule_offset'])
        sigma = jnp.sqrt(jnp.maximum(-2.0 * k, 1e-8))
        batch, steps = paths.shape[:2]
        flat_paths = paths.reshape(batch * steps, self.config['horizon'], self.config['transition_dim'])
        flat_conditions = jnp.repeat(conditions, steps, axis=0)
        flat_times = jnp.repeat(times[None], batch, axis=0).reshape(-1, 1)

        def loss_fn(params):
            new_score = self.planner(flat_paths, flat_conditions, flat_times, params=params)
            old_score = self.planner(
                flat_paths, flat_conditions, flat_times, params=self.old_planner_params
            )
            kk = jnp.repeat(k[None], batch, axis=0).reshape(-1, 1, 1)
            ss = jnp.repeat(sigma[None], batch, axis=0).reshape(-1, 1, 1)
            target = (new_score - old_score) * kk * (2.0 / ss) + ss * adjoints.reshape(
                batch * steps, self.config['horizon'], self.config['transition_dim']
            )
            mask = jnp.ones_like(target).at[:, 0, :self.config['state_dim']].set(0.0)
            denom = self.config['horizon'] * self.config['transition_dim'] - self.config['state_dim']
            losses = jnp.sum((target * mask) ** 2, axis=(1, 2)) / denom
            losses = losses.reshape(batch, steps)
            clip_steps = min(self.config['loss_clip_steps'], steps)
            cap = self.config['reward_scaling_factor'] ** 2 * 1.6
            clipped = jnp.minimum(losses[:, :clip_steps], cap)
            remainder = losses[:, clip_steps:]
            return (jnp.sum(clipped) + jnp.sum(remainder)) / (batch * steps), {}

        state, info = self.planner.apply_loss_fn(loss_fn=loss_fn)
        ema_params = _ema_update(
            state.params,
            self.planner_ema_params,
            state.step,
            self.config['ema_decay'],
            self.config['ema_start_step'],
            self.config['ema_update_every'],
        )
        return self.replace(planner=state, planner_ema_params=ema_params), {'adjoint_loss': info}

    def sample_actions(self, observations, seed=None, temperature=1.0):
        condition = (jnp.asarray(observations) - self.planner_mean) / self.planner_std
        plan, _ = self.sample_plan_path(condition, seed, eta=0.0)
        return jnp.clip(plan[-1, 0, self.config['state_dim']:] * temperature, -1.0, 1.0)
