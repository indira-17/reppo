import hydra
import jax
import logging
import time
import wandb
import numpy as np
import h5py
import torch
from omegaconf import DictConfig, OmegaConf
from src.algorithms import envs, utils
from src.common import InitFn, LearnerFn, PolicyFn
from src.cfg_utils import fix_cfg
import jax.numpy as jnp
from flax import nnx
from src.algorithms.reppo.common import ReplayBuffer
from src.runners.maniskill_runner import (
    flatten_obs,
    get_demo_obs_keys,
    _compute_action_bounds,
)

logging.basicConfig(level=logging.INFO)


# Build initial replay buffer from expert demo transitions.
def _build_replay_buffer_from_trajectories(
    demo_path: str,
    filter_success: bool,
    replay_buffer_size: int | None = None,
    env_id: str | None = None,
) -> ReplayBuffer:
    # Read demo observation keys used by the runner flattening logic.
    demo_obs_keys = get_demo_obs_keys(demo_path)

    # Compute action bounds (matching runner and pretraining) so we can
    # normalize raw demo actions to [-1, 1].
    _env_id = env_id or "PushCube-v1"
    dataset_low, dataset_high = _compute_action_bounds(
        demo_path, _env_id, filter_success
    )
    # Convert to numpy for vectorised normalisation.
    dataset_low_np = dataset_low.numpy() if isinstance(dataset_low, torch.Tensor) else np.asarray(dataset_low)
    dataset_high_np = dataset_high.numpy() if isinstance(dataset_high, torch.Tensor) else np.asarray(dataset_high)

    obs_chunks = []
    next_obs_chunks = []
    action_chunks = []
    reward_chunks = []
    done_chunks = []
    trunc_chunks = []
    episode_start_chunks = []

    # Read raw trajectories from demo file.
    with h5py.File(demo_path, "r") as f:
        traj_keys = sorted(
            [k for k in f.keys() if k.startswith("traj_")],
            key=lambda x: int(x.split("_")[1]),
        )

        for traj_key in traj_keys:
            traj_group = f[traj_key]
            actions = np.array(traj_group["actions"])
            terminated = np.array(traj_group["terminated"])
            truncated = np.array(traj_group["truncated"])

            if len(actions) == 0:
                continue

            if filter_success and "success" in traj_group:
                success = np.array(traj_group["success"])
                if not np.any(success):
                    continue

            if "obs" not in traj_group:
                continue

            obs_group = traj_group["obs"]
            obs_dict = {"agent": {}, "extra": {}}
            for key in ("agent", "extra"):
                if key in obs_group:
                    sub_group = obs_group[key]
                    for sub_key in sub_group.keys():
                        obs_dict[key][sub_key] = np.array(sub_group[sub_key])

            if "rewards" in traj_group and traj_group["rewards"] is not None:
                rewards = np.array(traj_group["rewards"])
            else:
                rewards = np.full(len(actions), -0.01, dtype=np.float32)
                if "success" in traj_group:
                    success = np.array(traj_group["success"])
                    if np.any(success):
                        rewards[np.argmax(success)] = 1.0

            if "success" in traj_group:
                success = np.array(traj_group["success"])
                cut_idx = np.argmax(success) + 1 if np.any(success) else len(actions)
            else:
                cut_idx = len(actions)

            # Flatten observations exactly like ManiSkill runner.
            flat_obs = flatten_obs(obs_dict, env=None, demo_obs_keys=demo_obs_keys)

            obs = flat_obs[:cut_idx]
            act = actions[:cut_idx]
            rew = rewards[:cut_idx]
            done = terminated[:cut_idx]
            trunc = truncated[:cut_idx]

            if len(obs) < 2:
                continue

            obs_chunks.append(obs[:-1])
            next_obs_chunks.append(obs[1:])
            # Normalize raw demo actions to [-1, 1] using the same bounds as
            # the runner's denormalize/normalize pipeline and BC pretraining.
            act_normalized = 2.0 * (act[:-1] - dataset_low_np) / (dataset_high_np - dataset_low_np) - 1.0
            act_normalized = np.clip(act_normalized, -0.999, 0.999)
            traj_done = done[:-1].copy()
            traj_trunc = trunc[:-1].copy()
            traj_trunc[-1] = True
            traj_episode_start = np.zeros(len(act_normalized), dtype=np.float32)
            traj_episode_start[0] = 1.0
            action_chunks.append(act_normalized)
            reward_chunks.append(rew[:-1])
            done_chunks.append(traj_done)
            trunc_chunks.append(traj_trunc)
            episode_start_chunks.append(traj_episode_start)

    obs = jnp.asarray(np.concatenate(obs_chunks, axis=0))
    next_obs = jnp.asarray(np.concatenate(next_obs_chunks, axis=0))
    actions_normalized = jnp.asarray(np.concatenate(action_chunks, axis=0))
    rewards = jnp.asarray(np.concatenate(reward_chunks, axis=0))
    dones = jnp.asarray(np.concatenate(done_chunks, axis=0))
    truncations = jnp.asarray(np.concatenate(trunc_chunks, axis=0))
    episode_starts = jnp.asarray(np.concatenate(episode_start_chunks, axis=0))

    # Pick replay capacity from config or default rule.
    offline_count = obs.shape[0]
    capacity = replay_buffer_size if replay_buffer_size is not None else int(offline_count * 2)
    capacity = max(capacity, int(offline_count))

    # Create empty replay arrays, then fill with offline data.
    replay_buffer = ReplayBuffer(
        observations=jnp.zeros((capacity, *obs.shape[1:]), dtype=obs.dtype),
        next_observations=jnp.zeros((capacity, *next_obs.shape[1:]), dtype=next_obs.dtype),
        actions=jnp.zeros((capacity, *actions_normalized.shape[1:]), dtype=actions_normalized.dtype),
        rewards=jnp.zeros((capacity, *rewards.shape[1:]), dtype=rewards.dtype),
        dones=jnp.zeros((capacity, *dones.shape[1:]), dtype=dones.dtype),
        truncations=jnp.zeros((capacity, *truncations.shape[1:]), dtype=truncations.dtype),
        behavior_log_probs=jnp.zeros((capacity,), dtype=jnp.float32),
        episode_starts=jnp.zeros((capacity,), dtype=jnp.float32),
        ptr=jnp.array(offline_count % capacity),
        size=jnp.array(offline_count),
    )

    # Insert all offline transitions at the start of the buffer.
    replay_buffer = replay_buffer.replace(
        observations=replay_buffer.observations.at[:offline_count].set(obs),
        next_observations=replay_buffer.next_observations.at[:offline_count].set(next_obs),
        actions=replay_buffer.actions.at[:offline_count].set(actions_normalized),
        rewards=replay_buffer.rewards.at[:offline_count].set(rewards),
        dones=replay_buffer.dones.at[:offline_count].set(dones),
        truncations=replay_buffer.truncations.at[:offline_count].set(truncations),
        episode_starts=replay_buffer.episode_starts.at[:offline_count].set(episode_starts),
    )
    return replay_buffer


def _fill_offline_behavior_log_probs(
    replay_buffer: ReplayBuffer,
    init_fn: InitFn,
    key: jax.Array,
    batch_size: int = 4096,
) -> ReplayBuffer:
    init_state = init_fn(key)
    actor_model = nnx.merge(init_state.actor.graphdef, init_state.actor.params)
    actor_model.eval()

    size = int(replay_buffer.size)
    obs = replay_buffer.observations[:size]
    actions = jnp.clip(replay_buffer.actions[:size], -0.999, 0.999)
    log_probs = []
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        log_probs.append(actor_model(obs[start:end]).log_prob(actions[start:end]))
    behavior_log_probs = jnp.concatenate(log_probs, axis=0)
    return replay_buffer.replace(
        behavior_log_probs=replay_buffer.behavior_log_probs.at[:size].set(behavior_log_probs)
    )

@hydra.main(
    version_base=None,
    config_path="../config/default",
    config_name="reppo_continuous.yaml",
)
def main(cfg: DictConfig):
    cfg = fix_cfg(cfg)
    OmegaConf.resolve(cfg)
    logging.info("\n" + OmegaConf.to_yaml(cfg))
    
    # Modify run name based on bc_indicator
    run_name = f"bc-reppo-{cfg.env.name}-retrace" if cfg.algorithm.bc_indicator else f"reppo-{cfg.env.name}-retrace"
    
    wandb.init(
        mode=cfg.logging.mode,
        project="bc-reppo-v2",
        entity=cfg.logging.entity,
        tags=cfg.tags,
        config=OmegaConf.to_container(cfg),
        name=run_name,
        save_code=True,
    )

    key = jax.random.PRNGKey(cfg.seed)

    # Load demo settings and initialize replay buffer before training.
    logging.info(f"Loading dataset from {cfg.env.demo.demo_path}")
    filter_success = cfg.env.demo.get("filter_success", True)

    replay_size_cfg = OmegaConf.select(cfg, "algorithm.replay_buffer_size")
    replay_offline_fraction_cfg = OmegaConf.select(cfg, "algorithm.replay_offline_fraction")
    # Build the replay buffer from flattened expert trajectories.
    replay_buffer = _build_replay_buffer_from_trajectories(
        cfg.env.demo.demo_path,
        filter_success=filter_success,
        replay_buffer_size=int(replay_size_cfg) if replay_size_cfg is not None else None,
        env_id=cfg.env.name,
    )
    replay_offline_fraction = (
        float(replay_offline_fraction_cfg)
        if replay_offline_fraction_cfg is not None
        else 0.5
    )
    logging.info(
        f"Initialized replay buffer: size={int(replay_buffer.size)}, capacity={replay_buffer.observations.shape[0]}, offline_fraction={replay_offline_fraction}"
    )
    
    if cfg.algorithm.bc_indicator:
        n_obs_dataset = replay_buffer.observations.shape[1]
        logging.info(f"Dataset observation dimension: {n_obs_dataset}")
        # Create environment with the correct observation dimension
        env_setup = envs.make_env(cfg, n_obs_dataset=n_obs_dataset)
        obs_space = env_setup.observation_space
    else:
        env_setup = envs.make_env(cfg)
        obs_space = env_setup.observation_space
    
    init_fn: InitFn = hydra.utils.call(cfg.algorithm.init)(
        cfg=cfg,
        observation_space=obs_space,
        action_space=env_setup.action_space,
    )
    replay_buffer = _fill_offline_behavior_log_probs(replay_buffer, init_fn, jax.random.PRNGKey(cfg.seed))

    learner_fn: LearnerFn = hydra.utils.call(cfg.algorithm.learner)(
        cfg=cfg,
        observation_space=obs_space,
        action_space=env_setup.action_space,
    )
    policy_fn: PolicyFn = hydra.utils.call(cfg.algorithm.policy)(
        cfg=cfg,
        action_space=env_setup.action_space,
        observation_space=obs_space,
    )
    rollout_fn = hydra.utils.call(cfg.runner.rollout_fn)(env_setup.env, demo_path=cfg.env.demo.demo_path, bc_indicator=cfg.algorithm.bc_indicator)
    eval_fn = hydra.utils.call(cfg.runner.eval_fn)(env_setup.eval_env, demo_path=cfg.env.demo.demo_path, bc_indicator=cfg.algorithm.bc_indicator)
    make_train_fn = hydra.utils.call(cfg.runner.train_fn)
    # Pass replay buffer settings into the training loop.
    train_fn = make_train_fn(
        env=(env_setup.env, env_setup.eval_env),
        init_fn=init_fn,
        learner_fn=learner_fn,
        policy_fn=policy_fn,
        rollout_fn=rollout_fn,
        eval_fn=eval_fn,
        log_callback=utils.make_log_callback(),
        replay_buffer=replay_buffer,
        replay_offline_fraction=replay_offline_fraction,
        cfg=cfg,
    )
    start = time.perf_counter()
    _, metrics = train_fn(key)
    jax.block_until_ready(metrics)
    duration = time.perf_counter() - start
    logging.info(f"Training took {duration:.2f} seconds.")
    wandb.finish()


if __name__ == "__main__":
    main()