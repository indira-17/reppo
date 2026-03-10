import os
import pickle
import numpy as np
import hydra
import jax
import jax.numpy as jnp
import gymnasium as gym
import imageio
import wandb
import torch
from pathlib import Path
from collections import defaultdict
from flax import nnx
from omegaconf import OmegaConf

from src.networks.common import MLP
from src.networks.policy_heads import TanhGaussianPolicyHead
from src.algorithms.reppo.networks import Actor
from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
import h5py


def create_actor(n_obs, n_act, cfg, rngs):
    """Create a JAX Actor matching the REPPO architecture."""
    hparams = cfg.algorithm
    feature_encoder = MLP(
        in_features=n_obs,
        out_features=n_act * 2,
        hidden_dim=hparams.actor_hidden_dim,
        hidden_activation=nnx.swish,
        output_activation=None,
        use_norm=hparams.use_actor_norm,
        use_output_norm=False,
        layers=hparams.num_actor_layers,
        hidden_skip=hparams.use_actor_skip,
        output_skip=hparams.use_actor_skip,
        rngs=rngs,
    )
    actor = Actor(
        feature_encoder=feature_encoder,
        policy_head=TanhGaussianPolicyHead(
            min_std=hparams.actor_min_std,
            fixed_std=hparams.fixed_actor_std,
        ),
        kl_start=hparams.kl_start,
        ent_start=hparams.ent_start,
    )
    return actor


def load_actor(actor, path):
    """Load JAX actor state from pickle file."""
    with open(path, 'rb') as f:
        state = pickle.load(f)
    nnx.update(actor, state)
    return actor


def make_eval_env(env_id, control_mode="pd_joint_pos", seed=0, render=True):
    env = gym.make(
        env_id,
        obs_mode="state_dict",
        control_mode=control_mode,
        render_mode="rgb_array" if render else None,
        reward_mode="normalized_dense",
        max_episode_steps=100,
    )
    env.reset(seed=seed)
    return env


def get_demo_obs_keys(demo_path):
    """Extract which observation keys are actually in the demo file."""
    with h5py.File(demo_path, 'r') as f:
        traj_group = f['traj_0']
        obs_group = traj_group['obs']

        demo_keys = {'agent': [], 'extra': []}
        for key in sorted(obs_group.keys()):
            if key in ('agent', 'extra'):
                sub_group = obs_group[key]
                if hasattr(sub_group, 'keys'):
                    demo_keys[key] = sorted(sub_group.keys())

        return demo_keys


def flatten_obs(obs_dict, env, demo_obs_keys):
    """Flatten the observation dictionary to match the demo data format.
    
    Must match the observation loading in ManiSkillDemoLoader._load_observations().
    Only includes: agent.qpos, agent.qvel, and ALL extra fields.
    """
    obs_list = []
    
    for key in sorted(obs_dict.keys()):
        if key in ('agent', 'extra'):
            sub_group = obs_dict[key]
            for sub_key in sorted(sub_group.keys()):
                if key == 'agent' and sub_key in ['qpos', 'qvel']:
                    data = sub_group[sub_key]
                    if not isinstance(data, np.ndarray):
                        data = np.asarray(data)
                    data_flat = data.reshape(data.shape[0], -1)
                    obs_list.append(data_flat)
                elif key == 'extra':
                    data = sub_group[sub_key]
                    if not isinstance(data, np.ndarray):
                        data = np.asarray(data)
                    data_flat = data.reshape(data.shape[0], -1)
                    obs_list.append(data_flat)

    if not obs_list:
        raise ValueError(f"No observation data found in obs_dict with keys: {obs_dict.keys()}")

    return np.concatenate(obs_list, axis=1)


def test(cfg, env_id=None, model_path=None, demo_path=None):
    # Initialize wandb
    wandb.init(
        config=dict(cfg),
        entity=cfg.logging.entity,
        project="reppo_v2",
        name=f"bc_eval_{cfg.env.name}",
        mode=cfg.logging.mode,
    )

    if cfg is None:
        config_path = Path(__file__).parent.parent / "config" / "default" / "reppo_maniskill.yaml"
        cfg = OmegaConf.load(str(config_path))

    if env_id:
        cfg.env.name = env_id
    if demo_path:
        cfg.env.demo.demo_path = demo_path

    env_id = cfg.env.name
    demo_path = cfg.env.demo.demo_path

    # Get demo observation keys
    demo_obs_keys = get_demo_obs_keys(demo_path)
    print(f"Demo observation keys: {demo_obs_keys}")

    # Determine obs/act dims from demo data
    filter_success = cfg.env.demo.get('filter_success', True)
    config = DemoConfig(device=torch.device("cpu"), filter_success_only=filter_success)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories_for_dims, _ = loader.load_demo_dataset(demo_path)
    n_obs = trajectories_for_dims[0]["observations"].shape[1]
    n_act = trajectories_for_dims[0]["actions"].shape[1]

    # Create JAX actor and load weights
    rngs = nnx.Rngs(0)
    actor = create_actor(n_obs, n_act, cfg, rngs)
    print(f"Actor created with n_obs={n_obs}, n_act={n_act}")

    # Load trained weights
    if model_path is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
        if outputs_dir.exists():
            best_candidates = list(
                outputs_dir.glob(f"*/*/{env_id}/saved_models/*/bc_model_actor_best.pkl")
            )
            if best_candidates:
                model_path = str(max(best_candidates, key=lambda p: p.stat().st_mtime))
        else:
            raise ValueError(f"Outputs directory not found: {outputs_dir}")

    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    actor = load_actor(actor, str(model_path))
    print(f"Loaded JAX actor from {model_path}")

    # Get action bounds from environment
    temp_env = gym.make(env_id, control_mode=cfg.env.get('control_mode'), obs_mode="state_dict")
    env_low = temp_env.action_space.low
    env_high = temp_env.action_space.high
    temp_env.close()

    print(f"\n=== ENV ACTION SPACE INFO ===")
    print(f"Action bounds - Low: {env_low}")
    print(f"Action bounds - High: {env_high}")
    print(f"Action space range: {env_high - env_low}")
    print(f"Action space center: {(env_high + env_low) / 2}")
    print(f"=" * 50)

    # Compute dataset action bounds with safety margin (same as pretrain-jax.py)
    all_dataset_actions = np.concatenate(
        [traj['actions'].numpy() for traj in trajectories_for_dims], axis=0
    )
    data_low = all_dataset_actions.min(axis=0)
    data_high = all_dataset_actions.max(axis=0)

    margin = 0.1 * (data_high - data_low)
    low_with_margin = data_low - margin
    high_with_margin = data_high + margin

    dataset_low = np.minimum(low_with_margin, -1.0)
    dataset_high = np.maximum(high_with_margin, 1.0)

    print(f"\n=== OFFLINE DATASET ACTION BOUNDS (with 10% safety margin) ===")
    print(f"Dataset Action bounds - Low: {dataset_low}")
    print(f"Dataset Action bounds - High: {dataset_high}")
    print(f"Dataset space range: {dataset_high - dataset_low}")
    print(f"Dataset space center: {(dataset_high + dataset_low) / 2}")
    print(f"=" * 50)

    num_episodes = 500
    max_steps = 100

    # Save videos in the same directory as the model
    model_parent_dir = Path(model_path).parent
    save_dir = model_parent_dir / "eval_videos"
    save_dir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(list)
    rng = jax.random.PRNGKey(0)

    for ep in range(num_episodes):
        env = make_eval_env(env_id, control_mode=cfg.env.get('control_mode'), seed=ep, render=True)
        obs, _ = env.reset()
        obs_flat = flatten_obs(obs, env, demo_obs_keys)
        obs_jax = jnp.array(obs_flat, dtype=jnp.float32)

        frames = []
        ep_reward = 0.0
        success = False

        for t in range(max_steps):
            # Get distribution from actor
            dist = actor(obs_jax)

            # Sample action from distribution (in [-1, 1] normalized space)
            rng, sample_key = jax.random.split(rng)
            normalized_action = dist.sample(seed=sample_key)  # [1, action_dim]
            normalized_action = np.asarray(normalized_action).squeeze(0)

            # Rescale from [-1, 1] to original per-dimension dataset bounds
            action = (normalized_action + 1.0) * (dataset_high - dataset_low) / 2.0 + dataset_low

            obs, reward, terminated, truncated, info = env.step(action)

            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            ep_reward += reward
            success |= info.get("success", False)

            frame = env.render()
            if frame is not None:
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                frames.append(frame)

            if terminated or truncated:
                break

            obs_flat = flatten_obs(obs, env, demo_obs_keys)
            obs_jax = jnp.array(obs_flat, dtype=jnp.float32)

        # Save video
        if frames:
            video_path = str(save_dir / f"ep_{ep:03d}.mp4")
            frames_stacked = np.stack(frames, axis=0)
            imageio.mimsave(video_path, frames_stacked, fps=30)

        # Log stats
        reward_val = ep_reward.item() if hasattr(ep_reward, 'item') else float(ep_reward)
        stats["reward"].append(reward_val)
        stats["success"].append(float(success))
        stats["length"].append(t + 1)

        wandb.log({
            "eval/reward": reward_val,
            "eval/success": float(success),
            "eval/episode_length": t + 1,
        }, step=ep)

        print(
            f"EP {ep:02d} | reward {reward_val:.2f} | "
            f"success {success} | steps {t+1}"
        )

        env.close()

    print(f"\n=== EVALUATION SUMMARY ===")
    print(f"Average reward: {np.mean(stats['reward']):.3f}")
    print(f"Success rate: {np.mean(stats['success']):.1%}")
    print(f"Average episode length: {np.mean(stats['length']):.1f}")
    print(f"=" * 50)

    wandb.log({
        "eval/avg_reward": np.mean(stats['reward']),
        "eval/success_rate": np.mean(stats['success']),
        "eval/avg_episode_length": np.mean(stats['length']),
    })

    wandb.finish()


@hydra.main(version_base=None, config_path="../config/default", config_name="reppo_maniskill")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    model_path = OmegaConf.select(cfg, "model_path")
    test(cfg=cfg, env_id=None, model_path=model_path, demo_path=cfg.env.demo.demo_path)


if __name__ == "__main__":
    main()
