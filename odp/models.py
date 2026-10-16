"""Flax models used by the JAX ODP implementation."""

import flax.linen as nn
import jax
import jax.nn
import jax.numpy as jnp

from utils.networks import MLP, ensemblize


class SinusoidalEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, values):
        half_dim = self.dim // 2
        frequencies = jnp.exp(
            -jnp.log(10000.0) * jnp.arange(half_dim) / max(half_dim - 1, 1)
        )
        angles = values[..., None] * frequencies
        return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class FourierEmbedding(nn.Module):
    dim: int
    scale: float = 16.0

    @nn.compact
    def __call__(self, values):
        frequencies = self.param(
            'frequencies',
            lambda key, shape: self.scale * jax.random.normal(key, shape),
            (self.dim // 8,),
        )
        angles = values.reshape(-1, 1) * (2.0 * jnp.pi * frequencies)[None]
        embedding = jnp.concatenate([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        embedding = nn.Dense(
            self.dim,
            kernel_init=nn.initializers.normal(0.02),
        )(embedding)
        embedding = jax.nn.mish(embedding)
        return nn.Dense(
            self.dim,
            kernel_init=nn.initializers.normal(0.02),
        )(embedding)


class DiTBlock(nn.Module):
    d_model: int
    n_heads: int

    @nn.compact
    def __call__(self, x, embedding):
        modulation = nn.Dense(
            6 * self.d_model,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(jax.nn.silu(embedding))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation, 6, axis=-1
        )
        normalized = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=1e-6
        )(x)
        normalized = normalized * (1.0 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
        )(normalized, normalized, deterministic=True)
        x = x + gate_msa[:, None, :] * attention
        normalized = nn.LayerNorm(
            use_scale=False, use_bias=False, epsilon=1e-6
        )(x)
        normalized = normalized * (1.0 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        mlp = nn.Dense(self.d_model * 4)(normalized)
        mlp = nn.gelu(mlp, approximate=True)
        mlp = nn.Dense(self.d_model)(mlp)
        return x + gate_mlp[:, None, :] * mlp


class ScoreModel(nn.Module):
    horizon: int
    transition_dim: int
    state_dim: int
    emb_dim: int = 128
    d_model: int = 256
    n_heads: int = 4
    depth: int = 4

    @nn.compact
    def __call__(self, noisy_window, condition, times):
        del condition
        positions = SinusoidalEmbedding(self.d_model)(
            jnp.arange(noisy_window.shape[1], dtype=noisy_window.dtype)
        )
        x = nn.Dense(self.d_model)(noisy_window) + positions[None]
        time_embedding = FourierEmbedding(self.emb_dim)(times.reshape(-1))
        time_embedding = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.normal(0.02),
        )(time_embedding)
        time_embedding = jax.nn.mish(time_embedding)
        time_embedding = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.normal(0.02),
        )(time_embedding)
        time_embedding = jax.nn.mish(time_embedding)
        for _ in range(self.depth):
            x = DiTBlock(self.d_model, self.n_heads)(x, time_embedding)
        shift_scale = nn.Dense(
            2 * self.d_model,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(jax.nn.silu(time_embedding))
        shift, scale = jnp.split(shift_scale, 2, axis=-1)
        x = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x)
        x = x * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return nn.Dense(
            self.transition_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(x)


class RewardModel(nn.Module):
    hidden_dims: tuple = (512, 512, 512, 512, 512)

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        return MLP(
            (*self.hidden_dims, 1),
            activations=nn.silu,
            activate_final=False,
            layer_norm=True,
        )(inputs).squeeze(-1)


class KernelMember(nn.Module):
    state_dim: int
    hidden_dims: tuple = (256, 256)
    min_log_std: float = -6.0
    max_log_std: float = 4.0

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        features = nn.Dense(self.hidden_dims[0])(inputs)
        features = nn.LayerNorm()(features)
        features = nn.relu(features)
        for hidden_dim in self.hidden_dims[1:]:
            features = nn.relu(nn.Dense(hidden_dim)(features))
        mean = nn.Dense(self.state_dim)(features)
        raw_log_std = nn.Dense(self.state_dim)(features)
        log_std = self.min_log_std + jax.nn.softplus(raw_log_std - self.min_log_std)
        return mean, jnp.clip(log_std, a_max=self.max_log_std)


class KernelEnsemble(nn.Module):
    state_dim: int
    ensemble_size: int = 5
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, observations, actions):
        ensemble = ensemblize(KernelMember, self.ensemble_size)
        return ensemble(state_dim=self.state_dim, hidden_dims=self.hidden_dims)(observations, actions)


class MoGKernelMember(nn.Module):
    state_dim: int
    num_modes: int = 8
    hidden_dims: tuple = (256, 256)
    min_log_std: float = -6.0
    max_log_std: float = 4.0

    @nn.compact
    def __call__(self, observations, actions):
        inputs = jnp.concatenate([observations, actions], axis=-1)
        features = MLP(
            self.hidden_dims,
            activations=nn.relu,
            activate_final=True,
            layer_norm=True,
        )(inputs)
        output = nn.Dense(self.num_modes * (2 * self.state_dim + 1))(features)
        output = output.reshape(
            output.shape[:-1] + (self.num_modes, 2 * self.state_dim + 1)
        )
        means = output[..., :self.state_dim]
        raw_log_stds = output[..., self.state_dim:2 * self.state_dim]
        log_stds = self.min_log_std + jax.nn.softplus(raw_log_stds - self.min_log_std)
        log_stds = jnp.clip(log_stds, a_max=self.max_log_std)
        weights = jax.nn.softmax(output[..., -1], axis=-1)
        return means, log_stds, weights


class MoGKernelEnsemble(nn.Module):
    state_dim: int
    ensemble_size: int = 5
    num_modes: int = 8
    hidden_dims: tuple = (256, 256)

    @nn.compact
    def __call__(self, observations, actions):
        ensemble = ensemblize(MoGKernelMember, self.ensemble_size)
        return ensemble(
            state_dim=self.state_dim,
            num_modes=self.num_modes,
            hidden_dims=self.hidden_dims,
        )(observations, actions)


class CriticModel(nn.Module):
    hidden_dims: tuple = (512, 512, 512, 512, 512)

    @nn.compact
    def __call__(self, observations):
        return MLP(
            (*self.hidden_dims, 1),
            activations=nn.silu,
            activate_final=False,
            layer_norm=True,
        )(observations).squeeze(-1)
