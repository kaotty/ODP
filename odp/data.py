"""Dataset preparation for ODP's planner, reward, kernel, and critic stages."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Normalization:
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, value):
        return (np.asarray(value) - self.mean) / self.std


def _array(dataset: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in dataset:
        raise KeyError(f'dataset is missing {key!r}')
    return np.asarray(dataset[key])


def make_trajectories(
    dataset: Mapping[str, Any],
    reward_mode: str = 'raw',
) -> list[dict[str, np.ndarray]]:
    """Convert FQL transition arrays into state/action trajectories.

    FQL stores one ``observations`` and one ``next_observations`` row per
    transition. ODP needs the terminal state as well, so every returned
    trajectory contains ``len(actions) + 1`` states.
    """
    observations = _array(dataset, 'observations')
    next_observations = _array(dataset, 'next_observations')
    actions = _array(dataset, 'actions')
    rewards = _array(dataset, 'rewards')
    terminals = _array(dataset, 'terminals').astype(bool)
    if not (len(observations) == len(next_observations) == len(actions) == len(rewards) == len(terminals)):
        raise ValueError('transition fields must have the same length')
    if reward_mode == 'ogbench' and len(rewards):
        rewards = rewards - np.min(rewards)
    elif reward_mode != 'raw':
        raise ValueError(f'unsupported reward_mode: {reward_mode}')

    trajectories = []
    start = 0
    boundaries = np.flatnonzero(terminals)
    for boundary in np.concatenate([boundaries, np.array([len(actions) - 1])]):
        end = int(boundary) + 1
        if end <= start:
            continue
        states = np.concatenate(
            [observations[start:end], next_observations[end - 1:end]], axis=0
        )
        trajectories.append({
            'observations': states,
            'actions': actions[start:end],
            'rewards': rewards[start:end],
            'terminals': terminals[start:end],
        })
        start = end
        if start >= len(actions):
            break
    return trajectories


def state_normalization(trajectories: Iterable[Mapping[str, np.ndarray]]) -> Normalization:
    states = [np.asarray(t['observations'][:-1]) for t in trajectories]
    if not states:
        raise ValueError('cannot compute normalization from an empty trajectory set')
    values = np.concatenate(states, axis=0).astype(np.float32)
    return Normalization(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-3))


def full_state_normalization(trajectories: Iterable[Mapping[str, np.ndarray]]) -> Normalization:
    """Normalize all stored states, including terminal states for the critic."""
    states = [np.asarray(t['observations']) for t in trajectories]
    if not states:
        raise ValueError('cannot compute normalization from an empty trajectory set')
    values = np.concatenate(states, axis=0).astype(np.float32)
    return Normalization(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-3))


def planner_windows(
    trajectories: Iterable[Mapping[str, np.ndarray]],
    horizon: int,
    normalization: Normalization | None = None,
):
    """Build ODP planner windows and their initial-state conditions."""
    if horizon < 1:
        raise ValueError('horizon must be positive')
    trajectories = list(trajectories)
    normalization = normalization or state_normalization(trajectories)
    windows, conditions = [], []
    for trajectory in trajectories:
        states = np.asarray(trajectory['observations'])
        actions = np.asarray(trajectory['actions'])
        length = min(len(actions), len(states) - 1)
        states = normalization.normalize(states[:length])
        actions = actions[:length]
        if length < horizon:
            continue
        transitions = np.concatenate([states, actions], axis=-1)
        for start in range(length - horizon + 1):
            windows.append(transitions[start:start + horizon])
            conditions.append(states[start])
    if not windows:
        raise ValueError('no planner windows were produced; check horizon and data')
    return {
        'windows': np.asarray(windows, dtype=np.float32),
        'conditions': np.asarray(conditions, dtype=np.float32),
        'normalization': normalization,
    }


def reward_transitions(
    trajectories,
    normalization: Normalization,
    sigma: float | None = None,
    target_reward: float | None = None,
    alpha: float | None = None,
):
    """Return normalized ``(state, action, reward)`` samples."""
    states, actions, rewards = [], [], []
    for trajectory in trajectories:
        obs = np.asarray(trajectory['observations'])
        act = np.asarray(trajectory['actions'])
        rew = np.asarray(trajectory['rewards'], dtype=np.float32)
        length = min(len(act), len(obs) - 1, len(rew))
        raw = rew[:length]
        if target_reward is not None:
            raw = raw * target_reward
        if sigma is not None and sigma > 0:
            from scipy.ndimage import gaussian_filter1d
            rew = gaussian_filter1d(
                raw, sigma, mode='nearest', truncate=200.0 / sigma
            )
        elif alpha is not None:
            if not 0.0 < alpha < 1.0:
                raise ValueError('reward EMA alpha must be in (0, 1)')
            smoothed = np.empty_like(raw)
            smoothed[0] = raw[0]
            for index in range(1, len(raw)):
                smoothed[index] = alpha * smoothed[index - 1] + (1.0 - alpha) * raw[index]
            rew = smoothed
        else:
            rew = raw
        states.append(normalization.normalize(obs[:length]))
        actions.append(act[:length])
        rewards.append(rew)
    if not states:
        raise ValueError('no reward transitions were produced')
    return {
        'observations': np.concatenate(states).astype(np.float32),
        'actions': np.concatenate(actions).astype(np.float32),
        'rewards': np.concatenate(rewards).astype(np.float32),
    }


def kernel_transitions(trajectories, normalization: Normalization):
    """Return normalized ``(s_t, a_t, s_{t+1})`` samples."""
    states, actions, next_states = [], [], []
    for trajectory in trajectories:
        obs = np.asarray(trajectory['observations'])
        act = np.asarray(trajectory['actions'])
        length = min(len(act), len(obs) - 1)
        states.append(normalization.normalize(obs[:length]))
        next_states.append(normalization.normalize(obs[1:length + 1]))
        actions.append(act[:length])
    if not states:
        raise ValueError('no kernel transitions were produced')
    return {
        'observations': np.concatenate(states).astype(np.float32),
        'actions': np.concatenate(actions).astype(np.float32),
        'next_observations': np.concatenate(next_states).astype(np.float32),
    }


def critic_transitions(trajectories, normalization: Normalization, horizon: int, gamma: float):
    """Build n-step value targets used by the base critic trainer."""
    observations, returns, next_observations, masks = [], [], [], []
    for trajectory in trajectories:
        obs = np.asarray(trajectory['observations'])
        rew = np.asarray(trajectory['rewards'], dtype=np.float32)
        length = min(len(rew), len(obs) - 1)
        for t in range(length):
            end = min(t + horizon, length)
            discounts = gamma ** np.arange(end - t, dtype=np.float32)
            observations.append(normalization.normalize(obs[t]))
            returns.append(np.sum(discounts * rew[t:end]))
            next_observations.append(normalization.normalize(obs[end]))
            masks.append(float(end < length))
    if not observations:
        raise ValueError('no critic transitions were produced')
    return {
        'observations': np.asarray(observations, dtype=np.float32),
        'returns': np.asarray(returns, dtype=np.float32),
        'next_observations': np.asarray(next_observations, dtype=np.float32),
        'masks': np.asarray(masks, dtype=np.float32),
    }
