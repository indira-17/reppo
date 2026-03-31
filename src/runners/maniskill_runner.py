import time
import torch
import gymnasium
import jax
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
import sys
import os
import imageio
import h5py

from src.common import (
    EvalFn,
    Key,
    Policy,
    RolloutFn,
    TrainState,
    Transition,
)
from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax

def _compute_action_bounds(demo_path, env_id="PushCube-v1", filter_success=True):
    """Compute action bounds from demo file, matching BC pretraining exactly.
    
    Uses ManiSkillDemoLoader with filter_success_only (which also cuts
    trajectories at first success), then computes bounds from raw trajectories
    with 10% safety margin — identical to pretrain-jax.py.
    """
    from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
    
    config = DemoConfig(device=torch.device("cpu"), filter_success_only=filter_success)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories, _ = loader.load_demo_dataset(demo_path)
    
    all_actions = np.concatenate(
        [traj['actions'].numpy() for traj in trajectories], axis=0
    )
    
    data_low = all_actions.min(axis=0)
    data_high = all_actions.max(axis=0)
    
    # Add 10% safety margin to bounds (matching pretrain-jax.py)
    margin = 0.1 * (data_high - data_low)
    low_with_margin = data_low - margin
    high_with_margin = data_high + margin

    # Ensure bounds are at least [-1, 1] in each dimension
    dataset_low = torch.from_numpy(np.minimum(low_with_margin, -1.0).astype(np.float32))
    dataset_high = torch.from_numpy(np.maximum(high_with_margin, 1.0).astype(np.float32))
    
    return dataset_low, dataset_high

def denormalize_action(action, dataset_low, dataset_high):
    # Denormalize action from [-1, 1] to dataset bounds.
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(np.ascontiguousarray(action.copy()))
    denormalized = (action + 1.0) * (dataset_high - dataset_low) / 2.0 + dataset_low
    return torch_to_numpy(denormalized)

def normalize_action(action, dataset_low, dataset_high):
    # Normalize action from dataset bounds to [-1, 1].
    if isinstance(action, np.ndarray):
        action = torch.from_numpy(np.ascontiguousarray(action.copy()))
    normalized = 2.0 * (action - dataset_low) / (dataset_high - dataset_low) - 1.0
    return torch_to_numpy(normalized) 

def torch_to_numpy(tensor):
    # Convert a torch tensor to numpy, handling CUDA tensors
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    return np.array(tensor)

def get_demo_obs_keys(demo_path):
    """Extract which observation keys are actually in the demo file."""
    import h5py
    
    with h5py.File(demo_path, 'r') as f:
        traj_group = f['traj_0']
        obs_group = traj_group['obs']
        
        demo_keys = {'agent': [], 'extra': []}
        for key in sorted(obs_group.keys()):
            if key in ('agent', 'extra'):
                sub_group = obs_group[key]
                demo_keys[key] = sorted(sub_group.keys())
        
        return demo_keys

def flatten_obs(obs_dict, env, demo_obs_keys):
    # Flatten the observation dictionary to match BC pretraining's _load_observations().
    # Only includes: agent/qpos, agent/qvel, and ALL extra fields.
    # Sorted alphabetically at each level (matching h5py iteration order in BC loader).
    if not hasattr(obs_dict, "keys"):
        raise TypeError(
            f"flatten_obs expects a dict observation, received {type(obs_dict)}"
        )

    obs_list = []
    for key in sorted(obs_dict.keys()):
        if key in ('agent', 'extra'):
            sub_group = obs_dict[key]
            for sub_key in sorted(sub_group.keys()):
                if key == 'agent' and sub_key in ['qpos', 'qvel']:
                    data = torch_to_numpy(sub_group[sub_key])
                    # reshape(N, -1) matches BC loader: (batch, features)
                    data_flat = data.reshape(data.shape[0], -1)
                    obs_list.append(data_flat)
                elif key == 'extra':
                    data = torch_to_numpy(sub_group[sub_key])
                    # reshape(N, -1) handles both 2D (N, D) -> (N, D)
                    # and 1D scalars (N,) -> (N, 1), matching BC loader
                    data_flat = data.reshape(data.shape[0], -1)
                    obs_list.append(data_flat)
    
    if not obs_list:
        raise ValueError(f"No observation data found in obs_dict with keys: {obs_dict.keys()}")
    
    result = np.concatenate(obs_list, axis=1)
    return result

def make_rollout_fn(env: gymnasium.Env, num_steps: int, num_envs: int, demo_path: str = None, bc_indicator: bool = False, env_id: str = None, filter_success: bool = True) -> RolloutFn:
    # BC-specific rollout function with demo observation flattening
    if bc_indicator:
        demo_obs_keys = get_demo_obs_keys(demo_path) if demo_path else None
        # Compute action bounds once at function creation time
        _env_id = env_id or (env.spec.id if hasattr(env, 'spec') and env.spec else "PushCube-v1")
        dataset_low, dataset_high = _compute_action_bounds(demo_path, _env_id, filter_success)
        
        def collect_rollout(
            key: Key, train_state: TrainState, policy: Policy
        ) -> tuple[Transition, TrainState]:
            transitions = []
            obs_dict = train_state.last_obs
            obs = flatten_obs(obs_dict, env=env, demo_obs_keys=demo_obs_keys)
            
            prev_step = train_state.time_steps
            prev_time = time.perf_counter()
            for i in range(num_steps):
                key, act_key = jax.random.split(key)
                action, _ = policy(act_key, obs)

                # Keep normalized policy action for storage; denormalize only for env step.
                action_norm = np.asarray(action)
                action_env = denormalize_action(action_norm, dataset_low, dataset_high)
                # Get raw dict from base env
                next_obs_dict, reward, done, truncated, info = env.step(action_env)
                if "final_observation" in info:
                    _next_obs = to_jax(flatten_obs(info["final_observation"], env=env, demo_obs_keys=demo_obs_keys))
                else:
                    _next_obs = flatten_obs(next_obs_dict, env=env, demo_obs_keys=demo_obs_keys)
            
                # Convert torch tensors to numpy, handling CUDA
                reward = torch_to_numpy(reward)
                done = torch_to_numpy(done)
                truncated = torch_to_numpy(truncated)
                
                # Record the transition
                transition = Transition(
                    obs=obs,
                    next_obs=_next_obs,
                    action=action_norm,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    extras={},
                )
                transitions.append(transition)
                obs = flatten_obs(next_obs_dict, env=env, demo_obs_keys=demo_obs_keys)

            transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
            train_state = train_state.replace(
                last_obs=next_obs_dict,  # Store dict observation, not flattened
                time_steps=train_state.time_steps + num_steps * num_envs,
            )

            return transitions, train_state
    else:
        # Non-BC rollout function - matches upstream behavior
        def collect_rollout(
            key: Key, train_state: TrainState, policy: Policy
        ) -> tuple[Transition, TrainState]:
            transitions = []
            obs = train_state.last_obs
            prev_step = train_state.time_steps
            prev_time = time.perf_counter()
            for i in range(num_steps):
                key, act_key = jax.random.split(key)
                action, _ = policy(act_key, obs)
                # Take a step in the environment
                next_obs, reward, done, truncated, info = env.step(action)
                if "final_observation" in info:
                    _next_obs = to_jax(info["final_observation"])
                else:
                    _next_obs = next_obs
                # Record the transition
                transition = Transition(
                    obs=obs,
                    next_obs=_next_obs,
                    action=action,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    extras={},
                )
                transitions.append(transition)
                obs = next_obs
            transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
            train_state = train_state.replace(
                last_obs=obs,
                time_steps=train_state.time_steps + num_steps * num_envs,
            )

            return transitions, train_state

    return collect_rollout


def make_eval_fn(env: gymnasium.Env, max_episode_steps: int, demo_path: str = None, bc_indicator: bool = False, env_id: str = None, filter_success: bool = True) -> EvalFn:
    # BC-specific evaluation function with demo observation flattening
    if bc_indicator:
        demo_obs_keys = get_demo_obs_keys(demo_path) if demo_path else None
        # Compute action bounds once at function creation time
        _env_id = env_id or (env.spec.id if hasattr(env, 'spec') and env.spec else "PushCube-v1")
        dataset_low, dataset_high = _compute_action_bounds(demo_path, _env_id, filter_success)

        def evaluate(key: Key, policy: Policy) -> dict:
            obs_dict, _ = env.reset()
            obs = flatten_obs(obs_dict, env=env, demo_obs_keys=demo_obs_keys)
            online_trajectories = []
            
            metrics = defaultdict(list)
            num_episodes = 0
            for i in range(max_episode_steps):
                key, act_key = jax.random.split(key)
                action, log_prob = policy(act_key, obs)
                # Keep normalized policy action for storage; denormalize only for env step.
                action_norm = np.asarray(action)
                action_env = denormalize_action(action_norm, dataset_low, dataset_high)
                # Get raw dict from base env
                next_obs_dict, reward, done, truncated, info = env.step(action_env)
                reward = torch_to_numpy(reward)
                done = torch_to_numpy(done)
                truncated = torch_to_numpy(truncated)
                if "final_observation" in info:
                    _next_obs = to_jax(flatten_obs(info["final_observation"], env=env, demo_obs_keys=demo_obs_keys))
                else:
                    _next_obs = flatten_obs(next_obs_dict, env=env, demo_obs_keys=demo_obs_keys)
                if "final_info" in info:
                    mask = info["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in info["final_info"]["episode"].items():
                        metrics[k].append(v)

                transition = Transition(
                    obs=obs,
                    next_obs=_next_obs,
                    action=action_norm,
                    reward=reward,
                    done=done,
                    truncated=truncated,
                    extras={
                        "log_prob": log_prob['log_prob'],
                        "behavior_log_prob": log_prob['log_prob'],
                    },
                )
                online_trajectories.append(transition)
                obs = flatten_obs(next_obs_dict, env=env, demo_obs_keys=demo_obs_keys)

            eval_metrics = {}
            for k, v in metrics.items():
                v_array = np.array([torch_to_numpy(item) for item in v])
                eval_metrics[f"{k}_std"] = v_array.std()
                eval_metrics[k] = v_array.mean()
            eval_metrics["episode_return"] = eval_metrics.pop("return", 0.0)
            eval_metrics["episode_return_std"] = eval_metrics.pop("return_std", 0.0)
            eval_metrics["episode_length"] = eval_metrics.pop("episode_len", 0.0)
            eval_metrics["episode_length_std"] = eval_metrics.pop("episode_len_std", 0.0)
            return eval_metrics, online_trajectories
    else:
        # Non-BC evaluation function - matches upstream behavior
        def evaluate(key: Key, policy: Policy) -> dict:
            obs, _ = env.reset()
            metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(max_episode_steps):
                key, act_key = jax.random.split(key)
                action, _ = policy(act_key, obs)
                next_obs, reward, terminated, truncated, infos = env.step(action)
                if "final_info" in infos:
                    mask = infos["_final_info"]
                    num_episodes += mask.sum()
                    for k, v in infos["final_info"]["episode"].items():
                        metrics[k].append(v)
                obs = next_obs

            eval_metrics = {}
            for k, v in metrics.items():
                v_array = np.array([torch_to_numpy(item) for item in v])
                eval_metrics[f"{k}_std"] = v_array.std()
                eval_metrics[k] = v_array.mean()
            eval_metrics["episode_return"] = eval_metrics.pop("return", 0.0)
            eval_metrics["episode_return_std"] = eval_metrics.pop("return_std", 0.0)
            eval_metrics["episode_length"] = eval_metrics.pop("episode_len", 0.0)
            eval_metrics["episode_length_std"] = eval_metrics.pop("episode_len_std", 0.0)
            return eval_metrics

    return evaluate