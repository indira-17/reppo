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


def make_eval_env(env_id, control_mode="pd_joint_pos", seed=0, render=True, sim_backend=None, max_steps=50):
    env_kwargs = {}
    if sim_backend is not None:
        env_kwargs["sim_backend"] = sim_backend
    env = gym.make(
        env_id,
        obs_mode="state_dict",
        control_mode=control_mode,
        render_mode="rgb_array" if render else None,
        reward_mode="normalized_dense",
        max_episode_steps=max_steps,
        **env_kwargs,
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


def flatten_obs(obs_dict, demo_obs_keys):
    """Flatten the observation dictionary to match the demo data format.
    
    Must match the observation loading in ManiSkillDemoLoader._load_observations().
    Only includes: agent.qpos, agent.qvel, and ALL extra fields.
    """
    obs_list = []

    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    
    for key in sorted(obs_dict.keys()):
        if key in ('agent', 'extra'):
            sub_group = obs_dict[key]
            for sub_key in sorted(sub_group.keys()):
                if key == 'agent' and sub_key in ['qpos', 'qvel']:
                    data = sub_group[sub_key]
                    if not isinstance(data, np.ndarray):
                        data = _to_numpy(data)
                    data_flat = data.reshape(data.shape[0], -1)
                    obs_list.append(data_flat)
                elif key == 'extra':
                    data = sub_group[sub_key]
                    if not isinstance(data, np.ndarray):
                        data = _to_numpy(data)
                    data_flat = data.reshape(data.shape[0], -1)
                    obs_list.append(data_flat)

    if not obs_list:
        raise ValueError(f"No observation data found in obs_dict with keys: {obs_dict.keys()}")

    return np.concatenate(obs_list, axis=1)


def get_eval_seeds(cfg):
    """Resolve evaluation seed list from config, defaulting to 5 seeds."""
    if "pretrain_seeds" in cfg.algorithm and cfg.algorithm.pretrain_seeds is not None:
        return [int(s) for s in cfg.algorithm.pretrain_seeds]
    if "pretrain_seeds" in cfg and cfg.pretrain_seeds is not None:
        return [int(s) for s in cfg.pretrain_seeds]
    return [0]


def find_seed_model_paths(env_id, seeds, model_path=None):
    """Find one best model per seed, preferring the most recent checkpoint."""
    seed_to_model = {}

    # If explicit file path is provided, use it for all seeds only when needed.
    if model_path is not None:
        model_path = Path(model_path).resolve()
        if model_path.is_file():
            for seed in seeds:
                seed_to_model[seed] = str(model_path)
            return seed_to_model

    outputs_dir = Path(__file__).parent.parent / "outputs"
    if not outputs_dir.exists():
        raise ValueError(f"Outputs directory not found: {outputs_dir}")

    for seed in seeds:
        candidates = list(
            outputs_dir.glob(f"*/*/{env_id}/saved_models/*_seed{seed}/bc_model_actor_best.pkl")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No seed model found for seed={seed}. "
                f"Expected pattern: {outputs_dir}/*/*/{env_id}/saved_models/*_seed{seed}/bc_model_actor_best.pkl"
            )
        seed_to_model[seed] = str(max(candidates, key=lambda p: p.stat().st_mtime).resolve())

    return seed_to_model


def evaluate_single_model(
    cfg,
    actor,
    env_id,
    sim_backend,
    dataset_low,
    dataset_high,
    demo_obs_keys,
    num_episodes,
    max_steps,
    save_dir,
    model_seed,
    step_offset=0,
):
    """Run rollout evaluation for one model and return episode stats."""
    stats = defaultdict(list)
    rng = jax.random.PRNGKey(model_seed)

    def _to_bool(x):
        if isinstance(x, torch.Tensor):
            return bool(x.detach().cpu().any().item())
        if isinstance(x, np.ndarray):
            return bool(np.any(x))
        return bool(x)

    for ep in range(num_episodes):
        env = make_eval_env(
            env_id,
            control_mode=cfg.env.get('control_mode'),
            seed=ep,
            render=True,
            sim_backend=sim_backend,
            max_steps=max_steps,
        )
        obs, _ = env.reset()
        obs_flat = flatten_obs(obs, demo_obs_keys)
        obs_jax = jnp.array(obs_flat, dtype=jnp.float32)

        frames = []
        ep_reward = 0.0
        success = False

        for t in range(max_steps):
            dist = actor(obs_jax)
            rng, sample_key = jax.random.split(rng)
            normalized_action = dist.sample(seed=sample_key)
            normalized_action = np.asarray(normalized_action).squeeze(0)

            action = (normalized_action + 1.0) * (dataset_high - dataset_low) / 2.0 + dataset_low

            obs, reward, terminated, truncated, info = env.step(action)

            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            ep_reward += reward
            step_success = _to_bool(info.get("success", False))
            success = bool(success or step_success)

            frame = env.render()
            if frame is not None:
                if isinstance(frame, torch.Tensor):
                    frame = frame.cpu().numpy()
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                frames.append(frame)

            if terminated or truncated:
                break

            obs_flat = flatten_obs(obs, demo_obs_keys)
            obs_jax = jnp.array(obs_flat, dtype=jnp.float32)

        if frames:
            video_path = str(save_dir / f"seed_{model_seed}_ep_{ep:03d}.mp4")
            frames_stacked = np.stack(frames, axis=0)
            imageio.mimsave(video_path, frames_stacked, fps=30)

        reward_val = ep_reward.item() if hasattr(ep_reward, 'item') else float(ep_reward)
        stats["reward"].append(reward_val)
        stats["success"].append(float(success))
        stats["length"].append(t + 1)

        wandb.log({
            f"seed_{model_seed}/eval/reward": reward_val,
            f"seed_{model_seed}/eval/success": float(success),
            f"seed_{model_seed}/eval/episode_length": t + 1,
            "model_seed": model_seed,
        }, step=step_offset + ep)

        print(
            f"SEED {model_seed} | EP {ep:02d} | reward {reward_val:.2f} | "
            f"success {success} | steps {t+1}"
        )

        env.close()

    return stats


def test(cfg, env_id=None, model_path=None, demo_path=None):
    # Initialize one wandb run for all seeds so all metrics share dashboards.
    wandb.init(
        config=dict(cfg),
        entity=cfg.logging.entity,
        project="reppo-baseline",
        name=f"bc_eval_{cfg.env.name}_multiseed",
        mode=cfg.logging.mode,
    )

    if env_id:
        cfg.env.name = env_id
    if demo_path:
        cfg.env.demo.demo_path = demo_path

    env_id = cfg.env.name
    demo_path = cfg.env.demo.demo_path
    sim_backend = cfg.env.get("env_kwargs", {}).get("sim_backend", None)
    eval_seeds = get_eval_seeds(cfg)

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

    # Resolve one trained model path per seed
    seed_model_paths = find_seed_model_paths(env_id, eval_seeds, model_path=model_path)
    print("\nUsing seed checkpoints:")
    for seed in eval_seeds:
        print(f"  seed={seed}: {seed_model_paths[seed]}")

    # Get action bounds from environment
    temp_env_kwargs = {}
    if sim_backend is not None:
        temp_env_kwargs["sim_backend"] = sim_backend
    temp_env = gym.make(
        env_id,
        control_mode=cfg.env.get('control_mode'),
        obs_mode="state_dict",
        **temp_env_kwargs,
    )
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

    seed_success_rates = []
    seed_episode_rewards = {}
    seed_episode_success = {}
    seed_episode_lengths = {}

    for seed_idx, model_seed in enumerate(eval_seeds):

        model_path_for_seed = Path(seed_model_paths[model_seed]).resolve()
        if not model_path_for_seed.exists():
            raise FileNotFoundError(f"Model file not found: {model_path_for_seed}")

        rngs = nnx.Rngs(model_seed)
        actor = create_actor(n_obs, n_act, cfg, rngs)
        actor = load_actor(actor, str(model_path_for_seed))
        print(f"\nLoaded JAX actor for seed={model_seed} from {model_path_for_seed}")

        model_parent_dir = model_path_for_seed.parent
        save_dir = model_parent_dir / "eval_videos"
        save_dir.mkdir(parents=True, exist_ok=True)

        stats = evaluate_single_model(
            cfg=cfg,
            actor=actor,
            env_id=env_id,
            sim_backend=sim_backend,
            dataset_low=dataset_low,
            dataset_high=dataset_high,
            demo_obs_keys=demo_obs_keys,
            num_episodes=num_episodes,
            max_steps=max_steps,
            save_dir=save_dir,
            model_seed=model_seed,
            step_offset=seed_idx * num_episodes,
        )

        seed_episode_rewards[model_seed] = list(stats['reward'])
        seed_episode_success[model_seed] = list(stats['success'])
        seed_episode_lengths[model_seed] = list(stats['length'])

        seed_avg_reward = float(np.mean(stats['reward']))
        seed_success_rate = float(np.mean(stats['success']))
        seed_avg_len = float(np.mean(stats['length']))
        seed_success_rates.append(seed_success_rate)

        print(f"\n=== SEED {model_seed} SUMMARY ===")
        print(f"Average reward: {seed_avg_reward:.3f}")
        print(f"Success rate: {seed_success_rate:.1%}")
        print(f"Average episode length: {seed_avg_len:.1f}")
        print(f"=" * 50)

        wandb.log({
            f"seed_{model_seed}/eval/avg_reward": seed_avg_reward,
            f"seed_{model_seed}/eval/success_rate": seed_success_rate,
            f"seed_{model_seed}/eval/avg_episode_length": seed_avg_len,
            "model_seed": model_seed,
        })

    reward_series = [seed_episode_rewards[s] for s in eval_seeds]
    success_series = [seed_episode_success[s] for s in eval_seeds]

    success_matrix = np.array(success_series, dtype=np.float32)  # [num_seeds, num_episodes]
    # Final scalar mean/variance across BOTH seeds and episodes.
    final_success_mean_all = float(success_matrix.mean())
    final_success_var_all = float(success_matrix.var())

    print("\n=== FINAL SUCCESS SUMMARY ===")
    for s, sr in zip(eval_seeds, seed_success_rates):
        print(f"seed={s} success_rate={sr:.1%}")
    print(f"Final mean success (across seeds + episodes): {final_success_mean_all:.1%}")
    print(f"Final variance success (across seeds + episodes): {final_success_var_all:.6f}")
    print("=" * 50)

    wandb.log({
        "eval/final_success_mean_across_seeds_and_episodes": final_success_mean_all,
        "eval/final_success_variance_across_seeds_and_episodes": final_success_var_all,
    })
    
    wandb.finish()


@hydra.main(version_base=None, config_path="../config/default", config_name="reppo_maniskill")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    model_path = OmegaConf.select(cfg, "model_path")
    test(cfg=cfg, env_id=None, model_path=model_path, demo_path=cfg.env.demo.demo_path)


if __name__ == "__main__":
    main()
