import logging
import torch
import gymnasium
import numpy as np
from gymnax.environments.environment import Environment
import jax
from src.common import (
    EvalFn,
    InitFn,
    Key,
    LearnerFn,
    LogCallback,
    PolicyFn,
    RolloutFn,
    TrainFn,
    TrainState,
)
from src.algorithms import utils
import jax.numpy as jnp

from src.algorithms.reppo.common import OfflineReplayBuffer
from src.common import Transition
from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax
from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
from src.runners.maniskill_runner import _compute_action_bounds, normalize_action

# stack each list has num_envs arrays of shape [num_steps, ...]
def stack_and_transpose(arrays):
    stacked = jnp.stack(arrays, axis=0)  # [num_envs, num_steps, ...]
    return jnp.swapaxes(stacked, 0, 1)   # [num_steps, num_envs, ...]

def align_partial_reset_flags(terminated, truncated):
    """Replicate online ManiSkill partial-reset semantics for offline data.

    When partial-reset is used online:
        - done is always 0
        - boundaries are carried by truncated := terminated OR truncated
    We mirror that exactly here and force a terminal boundary at cut end.
    """
    boundary = jnp.logical_or(terminated.astype(bool), truncated.astype(bool))
    boundary = boundary.at[-1].set(True)
    done = jnp.zeros_like(terminated)
    trunc = boundary.astype(truncated.dtype)
    return done, trunc

def create_replay_buffer_from_demos(demo_path, env_id, num_steps, policy_fn, state, key, filter_success, cut_at_first_success):
    config = DemoConfig(device=torch.device("cpu"), filter_success_only=filter_success, cut_at_first_success=cut_at_first_success)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories, _ = loader.load_demo_dataset(demo_path)
    obs_list, next_obs_list, act_list, rew_list, done_list, trunc_list = [], [], [], [], [], []

    curr_obs, curr_next_obs, curr_act = [], [], []
    curr_rew, curr_done, curr_trunc = [], [], []
    curr_len = 0

    for traj in trajectories:
        obs = to_jax(traj['observations'])
        next_obs = to_jax(traj['next_observations'])
        act = to_jax(traj['actions'])
        rew = to_jax(traj['rewards']).squeeze(-1)
        terminated = to_jax(traj['dones']).squeeze(-1)
        trunc = to_jax(traj['truncations']).squeeze(-1)

        traj_len = int(obs.shape[0])
        if traj_len == 0:
            continue

        done, trunc = align_partial_reset_flags(terminated, trunc)

        j = 0
        while j < traj_len:
            take = min(num_steps - curr_len, traj_len - j)
            end = j + take

            curr_obs.append(obs[j:end])
            curr_next_obs.append(next_obs[j:end])
            curr_act.append(act[j:end])
            curr_rew.append(rew[j:end])
            curr_done.append(done[j:end])
            curr_trunc.append(trunc[j:end])

            curr_len += take
            j = end

            if curr_len == num_steps:
                seg_trunc = jnp.concatenate(curr_trunc, axis=0)
                seg_trunc = seg_trunc.at[-1].set(jnp.array(1, dtype=seg_trunc.dtype))

                obs_list.append(jnp.concatenate(curr_obs, axis=0))
                next_obs_list.append(jnp.concatenate(curr_next_obs, axis=0))
                act_list.append(jnp.concatenate(curr_act, axis=0))
                rew_list.append(jnp.concatenate(curr_rew, axis=0))
                done_list.append(jnp.concatenate(curr_done, axis=0))
                trunc_list.append(seg_trunc)

                curr_obs, curr_next_obs, curr_act = [], [], []
                curr_rew, curr_done, curr_trunc = [], [], []
                curr_len = 0

    if len(obs_list) == 0:
        raise ValueError(
            f"Not enough offline transitions to build one segment of length {num_steps}."
        )

    replay_buffer = OfflineReplayBuffer(
        obs=obs_list,
        next_obs=next_obs_list,
        action=act_list,
        reward=rew_list,
        done=done_list,
        truncated=trunc_list,
        behavior_log_prob=[jnp.zeros(num_steps) for _ in obs_list],
        size=len(obs_list),
    )

    dataset_low, dataset_high = _compute_action_bounds(
        demo_path,
        env_id,
        filter_success=filter_success,
        cut_at_first_success=cut_at_first_success,
    )
    offline_normalized_actions = [
        to_jax(normalize_action(np.asarray(a), dataset_low, dataset_high))
        for a in replay_buffer.action
    ]

    policy_for_logprobs = policy_fn(state, True)
    for traj_idx in range(replay_buffer.size):
        traj_obs = replay_buffer.obs[traj_idx]
        traj_action = offline_normalized_actions[traj_idx]
        key, lp_key = jax.random.split(key)
        _, extras = policy_for_logprobs(lp_key, traj_obs, action_input=traj_action)
        if "behavior_log_prob" not in extras:
            raise KeyError("Policy must return 'behavior_log_prob' when action_input is provided.")
        replay_buffer.behavior_log_prob[traj_idx] = extras["behavior_log_prob"]
    print(f"[REPLAY] Offline behavior_log_probs computed for {replay_buffer.size} trajectories")

    print(f"[REPLAY] Offline buffer created: {replay_buffer.size} segments")
    print(f"  traj[0] obs={replay_buffer.obs[0].shape} action={replay_buffer.action[0].shape} "
              f"reward={replay_buffer.reward[0].shape} done={replay_buffer.done[0].shape} "
              f"behavior_log_prob={replay_buffer.behavior_log_prob[0].shape}")
    
    return replay_buffer, offline_normalized_actions

def make_scan_train_fn(
    env: gymnasium.Env | tuple[gymnasium.Env, gymnasium.Env],
    total_time_steps: int,
    num_seeds: int,
    num_steps: int,
    num_envs: int,
    num_eval: int,
    max_episode_steps: int,
    stochastic_eval: bool,
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    rollout_fn: RolloutFn | None = None,
    eval_fn: EvalFn | None = None,
    log_callback: LogCallback | None = None,
    demo_path: str | None = None,
    filter_success: bool = True,
    wandb_run = None,
    cut_at_first_success: bool = True,
    critic_offline_warmup_iters: int = 0,
    data_type: str = "expert",
    max_buffer_size: int = 1_000_000,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
) -> TrainFn:
    from src.runners.gymnax_runner import (
        make_eval_fn as make_gymnax_eval_fn,
        make_rollout_fn as make_gymnax_rollout_fn,
    )

    # Initialize the environment and wrap it to admit vectorized behavior.
    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    eval_interval = int((total_time_steps / (num_steps * num_envs)) // num_eval)

    if eval_fn is None:
        eval_fn = make_gymnax_eval_fn(eval_env, max_episode_steps)

    if rollout_fn is None:
        rollout_fn = make_gymnax_rollout_fn(env, num_steps=num_steps, num_envs=num_envs)

    if log_callback is None:
        log_callback = lambda state, metrics: None

    num_segments_cap = (max_buffer_size + num_steps - 1) // num_steps
    
    # JAX-native ring buffer helpers
    def init_scan_buf(rt_template):
        """Zero-initialise a fixed-capacity JAX replay buffer with shapes inferred from a rollout."""
        blp = rt_template.extras.get("behavior_log_prob", None)
        seg_obs = rt_template.obs[:, 0]
        seg_nobs = rt_template.next_obs[:, 0]
        seg_act = rt_template.action[:, 0]
        seg_rew = rt_template.reward[:, 0]
        seg_done = rt_template.done[:, 0]
        seg_trunc = rt_template.truncated[:, 0]
        seg_blp = blp[:, 0] if blp is not None else jnp.zeros((num_steps,), dtype=jnp.float32)
        return {
            'obs': jnp.zeros((num_segments_cap,) + seg_obs.shape, seg_obs.dtype),
            'next_obs': jnp.zeros((num_segments_cap,) + seg_nobs.shape,  seg_nobs.dtype),
            'action': jnp.zeros((num_segments_cap,) + seg_act.shape,   seg_act.dtype),
            'reward': jnp.zeros((num_segments_cap,) + seg_rew.shape,   seg_rew.dtype),
            'done': jnp.zeros((num_segments_cap,) + seg_done.shape,  seg_done.dtype),
            'truncated': jnp.zeros((num_segments_cap,) + seg_trunc.shape, seg_trunc.dtype),
            'behavior_log_prob': jnp.zeros((num_segments_cap,) + seg_blp.shape,   seg_blp.dtype),
            # priority == 0 for empty slots so they are never sampled
            'priority': jnp.zeros((num_segments_cap,), dtype=jnp.float32),
            'write_idx': jnp.zeros((), dtype=jnp.int32),
            'buf_size': jnp.zeros((), dtype=jnp.int32),
        }

    def insert_scan_buf(buf, rollout_transitions):
        # FIFO-insert num_envs segments from one rollout into the JAX replay buffer.
        blp_all = rollout_transitions.extras.get("behavior_log_prob", None)
        # New segments get max-priority so they are sampled at least once
        max_p = jnp.maximum(buf['priority'].max(), jnp.ones((), dtype=jnp.float32))

        def insert_one(b, ei):
            idx = b['write_idx']  # current ring slot
            seg_blp = blp_all[:, ei] if blp_all is not None else jnp.zeros((num_steps,), dtype=jnp.float32)
            b_new = {
                'obs': b['obs'].at[idx].set(rollout_transitions.obs[:, ei]),
                'next_obs': b['next_obs'].at[idx].set(rollout_transitions.next_obs[:, ei]),
                'action': b['action'].at[idx].set(rollout_transitions.action[:, ei]),
                'reward': b['reward'].at[idx].set(rollout_transitions.reward[:, ei]),
                'done': b['done'].at[idx].set(rollout_transitions.done[:, ei]),
                'truncated': b['truncated'].at[idx].set(rollout_transitions.truncated[:, ei]),
                'behavior_log_prob': b['behavior_log_prob'].at[idx].set(seg_blp),
                'priority': b['priority'].at[idx].set(max_p),
                'write_idx': (b['write_idx'] + 1) % num_segments_cap,
                'buf_size': jnp.minimum(b['buf_size'] + 1, num_segments_cap),
            }
            return b_new, None

        buf, _ = jax.lax.scan(insert_one, buf, jnp.arange(num_envs))
        return buf

    def sample_scan_buf(buf, key):
        # Sample num_envs segment indices: uniform ('random') or priority-weighted ('PER').
        valid_mask = jnp.arange(num_segments_cap) < jnp.maximum(buf['buf_size'], 1)
        if data_type == 'PER':
            raw_p = (buf['priority'] ** per_alpha) * valid_mask
        else:
            raw_p = valid_mask.astype(jnp.float32)
        p = raw_p / jnp.maximum(raw_p.sum(), 1e-8)
        return jax.random.choice(key, num_segments_cap, shape=(num_envs,), replace=True, p=p)

    # Inner scan body: rollout → insert → sample → learn, carry = (TrainState, buf_dict)
    def train_step_replay(
        carry: tuple, key: Key
    ) -> tuple:
        state, buf = carry
        key, rollout_key, learn_key, sample_key = jax.random.split(key, 4)

        # Collect fresh rollout
        policy = policy_fn(state, False)
        rollout_transitions, state = rollout_fn(
            key=rollout_key, train_state=state, policy=policy
        )

        # FIFO-insert rollout segments into the JAX ring buffer
        buf = insert_scan_buf(buf, rollout_transitions)

        # Sample a batch of num_envs segments from the buffer
        sampled_indices = sample_scan_buf(buf, sample_key)
        batch_obs = jnp.swapaxes(buf['obs'][sampled_indices], 0, 1)
        batch_nobs = jnp.swapaxes(buf['next_obs'][sampled_indices], 0, 1)
        batch_act = jnp.swapaxes(buf['action'][sampled_indices], 0, 1)
        batch_rew = jnp.swapaxes(buf['reward'][sampled_indices], 0, 1)
        batch_done = jnp.swapaxes(buf['done'][sampled_indices], 0, 1)
        batch_trunc = jnp.swapaxes(buf['truncated'][sampled_indices], 0, 1)
        batch_blp = jnp.swapaxes(buf['behavior_log_prob'][sampled_indices], 0, 1)

        if data_type == 'PER':
            safe_size = jnp.maximum(buf['buf_size'], 1)
            valid_mask = jnp.arange(num_segments_cap) < safe_size
            raw_p = (buf['priority'] ** per_alpha) * valid_mask
            p = raw_p / jnp.maximum(raw_p.sum(), 1e-8)
            is_weight = (safe_size * p[sampled_indices]) ** (-per_beta)
            is_weight = is_weight / jnp.maximum(is_weight.max(), 1e-8)
            is_weight_batch = jnp.broadcast_to(is_weight[None], (num_steps, num_envs))
        else:
            is_weight_batch = jnp.ones((num_steps, num_envs), dtype=jnp.float32)

        transitions = Transition(
            obs=batch_obs, next_obs=batch_nobs, action=batch_act,
            reward=batch_rew, done=batch_done, truncated=batch_trunc,
            extras={
                "is_weight":         is_weight_batch,
                "behavior_log_prob": batch_blp,
            },
        )

        state, update_metrics, per_env_td_error = learner_fn(
            key=learn_key, train_state=state, batch=transitions
        )
        state = state.replace(iteration=state.iteration + 1)

        # Update PER priorities with TD errors returned by the learner
        if data_type == 'PER':
            new_p = jnp.abs(per_env_td_error) + 1e-6
            buf = {**buf, 'priority': buf['priority'].at[sampled_indices].set(new_p)}

        namespaced = {k: v for k, v in update_metrics.items() if "/" in k}
        plain      = {k: v for k, v in update_metrics.items() if "/" not in k}
        jax.debug.callback(log_callback, state, {**utils.prefix_dict("train", plain), **namespaced})

        return (state, buf), update_metrics

    # Eval step: runs eval_interval train steps then evaluates
    def train_eval_step_replay(key, train_state, scan_buf):
        train_key, eval_key = jax.random.split(key)
        (train_state, scan_buf), stacked_metrics = jax.lax.scan(
            f=train_step_replay,
            init=(train_state, scan_buf),
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], stacked_metrics)
        policy = policy_fn(train_state, not stochastic_eval)
        eval_metrics = eval_fn(eval_key, policy)
        eval_return = eval_metrics["episode_return"]
        grad_updates = train_metrics.get("sys/grad_updates", jnp.array(1.0))
        transitions_used = train_state.time_steps
        log_metrics_eval = utils.prefix_dict("eval", eval_metrics)
        log_metrics_eval.update({
            "sys/perf_per_grad_update": eval_return / jnp.maximum(grad_updates, 1.0),
            "sys/perf_per_transaction": eval_return / jnp.maximum(transitions_used, 1),
        })
        jax.debug.callback(log_callback, train_state, log_metrics_eval)
        return train_state, scan_buf, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    # Outer scan body: vmaps eval step over seeds, threads (state, buf) carry
    def train_eval_loop_body(carry, key):
        train_state, scan_buf = carry
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_seeds)
        train_state, scan_buf, metrics = jax.vmap(train_eval_step_replay)(
            keys, train_state, scan_buf
        )
        jax.debug.callback(log_callback, train_state, metrics)
        return (train_state, scan_buf), metrics

    def init_train_state(key: Key) -> TrainState:
        key, env_key = jax.random.split(key)
        train_state = init_fn(key)
        # obs, env_state = utils.init_env_state(key=env_key, env=env, num_envs=num_envs)
        env_key = jax.random.split(env_key, num_envs)
        obs, env_state = env.reset(env_key)
        train_state = train_state.replace(last_obs=obs, last_env_state=env_state)
        return train_state

    # Define the training loop
    def scan_train_fn(key: Key) -> tuple[TrainState, dict]:
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key, warmup_key = jax.random.split(key, 3)
        init_keys = jax.random.split(init_key, num_seeds)
        train_state = jax.vmap(init_train_state)(init_keys)

        # One warm-up rollout to infer transition shapes for buffer initialisation.
        state0 = jax.tree.map(lambda x: x[0], train_state)
        warmup_policy = policy_fn(state0, False)
        rt_template, _ = rollout_fn(key=warmup_key, train_state=state0, policy=warmup_policy)
        # Zero-filled per-seed buffers with correct dtypes/shapes; each seed gets an independent buffer.
        seed_buf = init_scan_buf(rt_template)
        scan_buf_init = jax.tree.map(
            lambda x: jnp.zeros((num_seeds,) + x.shape, x.dtype), seed_buf
        )

        keys = jax.random.split(key, num_iterations)
        (state, _), metrics = jax.lax.scan(
            f=train_eval_loop_body, init=(train_state, scan_buf_init), xs=keys
        )
        return state, metrics

    return jax.jit(scan_train_fn)


def make_loop_train_fn(
    env: gymnasium.Env | tuple[gymnasium.Env, gymnasium.Env],
    total_time_steps: int,
    num_steps: int,
    num_envs: int,
    num_eval: int,
    max_episode_steps: int,
    stochastic_eval: bool,
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    rollout_fn: RolloutFn | None = None,
    eval_fn: EvalFn | None = None,
    log_callback: LogCallback | None = None,
    demo_path: str | None = None,
    filter_success: bool = True,
    wandb_run = None,
    cut_at_first_success: bool = True,
    critic_offline_warmup_iters: int = 0,
    data_type: str = "expert",
    max_buffer_size: int = 1_000_000,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
):
    from src.runners.gymnasium_runner import (
        make_eval_fn as make_gymnasium_eval_fn,
        make_rollout_fn as make_gymnasium_rollout_fn,
    )

    train_log_interval = (
        int((total_time_steps / (num_steps * num_envs)) // num_eval) // 4
    )

    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    if rollout_fn is None:
        rollout_fn = make_gymnasium_rollout_fn(env, num_steps, num_envs)

    if eval_fn is None:
        eval_fn = make_gymnasium_eval_fn(eval_env, max_episode_steps)

    def _offline_ratio_for_iteration(iter_idx: int, num_iterations: int) -> float:
        """
        Schedule:
            1) pure offline critic warmup for `critic_offline_warmup_iters`
            2) then linearly decay offline ratio from 1.0 → 0.0 by the final iteration

        Example (num_eval=20, critic_offline_warmup_iters=3):
            - iterations 0..2: offline ratio = 1.0
            - iteration 3:     offline ratio ≈ 0.94
            - iteration 19:    offline ratio = 0.0

        With critic_offline_warmup_iters=0, still linearly decays:
            - iteration 0:  ratio ≈ 0.95
            - iteration 19: ratio = 0.0
        """
        if data_type in ("PER", "random"):
            # SAC-style replay modes: learner batch comes entirely from replay.
            return 1.0

        if data_type == 'expert':
            off_end = max(0, int(critic_offline_warmup_iters))
            # Warmup: fully offline for the first `off_end` iterations.
            if iter_idx < off_end:
                return 1.0
            # Then linearly decay the offline ratio to 0.0 by the final iteration
            # (with off_end == 0 this simply decays from the start).
            remaining = max(1, num_iterations - off_end)
            progress = (iter_idx - off_end + 1) / remaining
            return float(max(0.0, 1.0 - progress))

        return 0.0

    def loop_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_eval
        train_steps_per_iteration = num_train_steps // num_iterations
        key, init_key = jax.random.split(key)
        state = init_fn(init_key)
        obs, _ = env.reset()
        state = state.replace(last_obs=to_jax(obs), last_env_state=None)
        logging.info(f"Starting training for {num_iterations} iterations.")
        logging.info(f"Train steps per iteration: {train_steps_per_iteration}.")
        logging.info(f"Total time steps: {total_time_steps}.")

        buffer_memory_gb = 0.0

        if data_type in ('expert', 'PER', 'random'):
            num_segments_cap = (max_buffer_size + num_steps - 1) // num_steps
            ring_write_idx = 0  # FIFO pointer for PER / random

            if data_type == 'expert':
                offline_replay_buffer, offline_normalized_actions = create_replay_buffer_from_demos(
                    demo_path, env.spec.id, num_steps, policy_fn, state, key,
                    filter_success=filter_success, cut_at_first_success=cut_at_first_success,
                )
                
            else:  # PER / random: FastTD3-style growing FIFO replay, no full prefill.
                logging.info(
                    f"[REPLAY] Initializing empty {data_type} buffer "
                    f"with capacity {max_buffer_size} transitions."
                )
                offline_normalized_actions = None
                offline_replay_buffer = OfflineReplayBuffer(
                    obs=[], next_obs=[], action=[], reward=[], done=[], truncated=[],
                    behavior_log_prob=[], priority=[], size=0,
                )

            buffer_memory_gb = sum(
                a.nbytes
                for field in ('obs', 'next_obs', 'action', 'reward', 'done', 'truncated', 'behavior_log_prob')
                for a in getattr(offline_replay_buffer, field, [])
            ) / 1e9
            logging.info(f"[REPLAY] {data_type} buffer: {offline_replay_buffer.size} segments, {buffer_memory_gb:.3f} GB")

        step = 0
        online_transitions_used = 0
        offline_transitions_used = 0

        for iter_idx in range(num_iterations):
            ratio = _offline_ratio_for_iteration(iter_idx, num_iterations) if data_type in ('expert', 'PER', 'random') else 0.0
            for _ in range(train_steps_per_iteration):
                key, rollout_key, learn_key, off_subkey = jax.random.split(key, 4)
                # Collect trajectories from 'state'
                policy = policy_fn(state, False)

                # Also collect fresh rollout transitions (for env stepping / state update)
                rollout_transitions, state = rollout_fn(
                    key=rollout_key, train_state=state, policy=policy
                )

                # For PER/random replay: insert the current rollout before sampling.
                if data_type in ('PER', 'random'):
                    max_p = max(offline_replay_buffer.priority) if (data_type == 'PER' and offline_replay_buffer.priority) else 1.0
                    blp = rollout_transitions.extras.get("behavior_log_prob", None)
                    for ei in range(num_envs):
                        w_idx = ring_write_idx % num_segments_cap
                        seg_blp = blp[:, ei] if blp is not None else jnp.zeros(num_steps)
                        if offline_replay_buffer.size < num_segments_cap:
                            offline_replay_buffer.obs.append(rollout_transitions.obs[:, ei])
                            offline_replay_buffer.next_obs.append(rollout_transitions.next_obs[:, ei])
                            offline_replay_buffer.action.append(rollout_transitions.action[:, ei])
                            offline_replay_buffer.reward.append(rollout_transitions.reward[:, ei])
                            offline_replay_buffer.done.append(rollout_transitions.done[:, ei])
                            offline_replay_buffer.truncated.append(rollout_transitions.truncated[:, ei])
                            offline_replay_buffer.behavior_log_prob.append(seg_blp)
                            offline_replay_buffer.priority.append(max_p if data_type == 'PER' else 1.0)
                            offline_replay_buffer.size += 1
                        else:
                            offline_replay_buffer.obs[w_idx] = rollout_transitions.obs[:, ei]
                            offline_replay_buffer.next_obs[w_idx] = rollout_transitions.next_obs[:, ei]
                            offline_replay_buffer.action[w_idx] = rollout_transitions.action[:, ei]
                            offline_replay_buffer.reward[w_idx] = rollout_transitions.reward[:, ei]
                            offline_replay_buffer.done[w_idx] = rollout_transitions.done[:, ei]
                            offline_replay_buffer.truncated[w_idx] = rollout_transitions.truncated[:, ei]
                            offline_replay_buffer.behavior_log_prob[w_idx] = seg_blp
                            offline_replay_buffer.priority[w_idx] = max_p if data_type == 'PER' else 1.0
                        ring_write_idx += 1

                if data_type in ('expert', 'PER', 'random'):
                    # Sample num_envs trajectories of fixed length num_steps.
                    if data_type in ('PER', 'random'):
                        num_offline = num_envs
                        num_online = 0
                    else:
                        num_offline = num_envs if ratio >= 1.0 else int(ratio * num_envs)
                        num_online = num_envs - num_offline
                    online_transitions_used += num_online * num_steps
                    offline_transitions_used += num_offline * num_steps

                    traj_obs, traj_next_obs, traj_action = [], [], []
                    traj_reward, traj_done, traj_truncated, traj_behavior_log_prob = [], [], [], []
                    traj_source_is_offline = []

                    # Sample num_offline trajectories from offline buffer (already fixed to num_steps)
                    sampled_offline_indices = []
                    traj_is_weight = []
                    if data_type == 'PER':
                        # proportional priority sampling
                        all_p = np.array([p ** per_alpha for p in offline_replay_buffer.priority])
                        all_p /= all_p.sum()
                        N = offline_replay_buffer.size
                        max_w = float((N * all_p.min()) ** (-per_beta))
                    for _ in range(num_offline):
                        off_subkey, pick_key = jax.random.split(off_subkey)
                        if data_type == 'PER':
                            idx = int(jax.random.choice(pick_key, N, p=jnp.asarray(all_p)))
                            w = float((N * all_p[idx]) ** (-per_beta)) / max_w
                            traj_is_weight.append(jnp.full((num_steps,), w, dtype=jnp.float32))
                        else:
                            idx = int(jax.random.choice(pick_key, offline_replay_buffer.size))
                        sampled_offline_indices.append(idx)
                        traj_obs.append(offline_replay_buffer.obs[idx])
                        traj_next_obs.append(offline_replay_buffer.next_obs[idx])
                        traj_action.append(offline_normalized_actions[idx]) if data_type == 'expert' else traj_action.append(offline_replay_buffer.action[idx])
                        traj_reward.append(offline_replay_buffer.reward[idx])
                        traj_done.append(offline_replay_buffer.done[idx])
                        traj_truncated.append(offline_replay_buffer.truncated[idx])
                        traj_behavior_log_prob.append(offline_replay_buffer.behavior_log_prob[idx])
                        traj_source_is_offline.append(jnp.ones((num_steps,), dtype=jnp.float32))

                    # Use current rollout transitions as on-policy online data
                    # rollout_transitions shape: [num_steps, num_envs, ...]
                    if num_online > 0:
                        off_subkey, env_pick_key = jax.random.split(off_subkey)
                        online_env_idxs = jax.random.permutation(env_pick_key, num_envs)[:num_online]
                        for env_idx in np.asarray(online_env_idxs):
                            env_idx = int(env_idx)
                            traj_obs.append(rollout_transitions.obs[:, env_idx])
                            traj_next_obs.append(rollout_transitions.next_obs[:, env_idx])
                            traj_action.append(rollout_transitions.action[:, env_idx])
                            traj_reward.append(rollout_transitions.reward[:, env_idx])
                            traj_done.append(rollout_transitions.done[:, env_idx])
                            traj_truncated.append(rollout_transitions.truncated[:, env_idx])
                            traj_behavior_log_prob.append(rollout_transitions.extras["behavior_log_prob"][:, env_idx])
                            traj_source_is_offline.append(jnp.zeros((num_steps,), dtype=jnp.float32))

                    # for online slots IS weight = 1.0
                    if data_type == 'PER':
                        for _ in range(num_online):
                            traj_is_weight.append(jnp.ones(num_steps, dtype=jnp.float32))
                        is_weight_batch = stack_and_transpose(traj_is_weight)
                    else:
                        # uniform: IS weight = 1 everywhere (no bias correction needed)
                        is_weight_batch = jnp.ones((num_steps, num_envs), dtype=jnp.float32)

                    source_is_offline_batch = stack_and_transpose(traj_source_is_offline)
                    behavior_lp_batch = stack_and_transpose(traj_behavior_log_prob)
                    # Only the expert path mixes offline+online, so only it carries
                    # `source_is_offline` (which drives the per-source metric split downstream).
                    # PER/random/online are single-source -> logged under one unified name.
                    extras = {
                        "is_weight": is_weight_batch,
                        "behavior_log_prob": behavior_lp_batch,
                    }
                    if data_type == 'expert':
                        extras["source_is_offline"] = source_is_offline_batch
                    transitions = Transition(
                        obs=stack_and_transpose(traj_obs),
                        next_obs=stack_and_transpose(traj_next_obs),
                        action=stack_and_transpose(traj_action),
                        reward=stack_and_transpose(traj_reward),
                        done=stack_and_transpose(traj_done),
                        truncated=stack_and_transpose(traj_truncated),
                        extras=extras,
                    )
                    if step == 0:
                        print(f"[REPLAY] Sampled batch (step={step}, ratio={ratio:.3f}): "
                              f"num_offline={num_offline}, num_online={num_online}")
                        print(f"  transitions.obs={transitions.obs.shape} "
                              f"action={transitions.action.shape} "
                              f"reward={transitions.reward.shape} "
                              f"done={transitions.done.shape}")
                        print(f"  extras behavior_log_prob={transitions.extras['behavior_log_prob'].shape}")
                        print(f"  Expected: [{num_steps}, {num_envs}, ...]")
                else:
                    transitions = rollout_transitions
                    online_transitions_used += num_envs * num_steps

                # total transitions consumed by the learner (online + offline replayed)
                num_samples = online_transitions_used + offline_transitions_used

                # Execute an update to the policy with `transitions`
                state, train_metrics, per_env_td_error = learner_fn(
                    key=learn_key, train_state=state, batch=transitions
                )

                if step % train_log_interval == 0:
                    log_metrics = train_metrics
                    # critic_diag/, actor_diag/, sys/ stay at top level and only plain scalar metrics get the "train/" prefix.
                    namespaced = {k: v for k, v in log_metrics.items() if "/" in k}
                    plain = {k: v for k, v in log_metrics.items() if "/" not in k}
                    log_callback(state, {**utils.prefix_dict("train", plain), **namespaced})

                # Update priorities only for sampled PER segments. New rollout segments were
                # already inserted before sampling, matching SAC/TD3-style online replay growth.
                if data_type == 'PER':
                    per_env_td = np.asarray(per_env_td_error)
                    for i, sidx in enumerate(sampled_offline_indices):
                        offline_replay_buffer.priority[sidx] = float(abs(per_env_td[i])) + 1e-6

                step += 1

            policy = policy_fn(state, not stochastic_eval)
            key, eval_key = jax.random.split(key)

            eval_metrics = eval_fn(eval_key, policy)

            state = state.replace(iteration=state.iteration + 1)

            eval_return = float(eval_metrics["episode_return"])
            grad_updates = float(train_metrics.get("sys/grad_updates", 1.0))

            eval_metrics["num_samples"] = num_samples

            if data_type in ('expert', 'PER', 'random'):
                buffer_memory_gb = sum(
                    a.nbytes
                    for field in ('obs', 'next_obs', 'action', 'reward', 'done', 'truncated', 'behavior_log_prob')
                    for a in getattr(offline_replay_buffer, field, [])
                ) / 1e9

            # Everything goes through log_callback so all logging shares the same wandb step (step=state.time_steps)
            log_metrics_eval = utils.prefix_dict("eval", eval_metrics)
            log_metrics_eval.update({
                "sys/online_transitions":     online_transitions_used,
                "sys/offline_transitions":    offline_transitions_used,
                "sys/replay_buffer_memory_gb": buffer_memory_gb,
                "sys/perf_per_gb":            eval_return / buffer_memory_gb if buffer_memory_gb > 0 else 0.0,
                "sys/perf_per_grad_update":   eval_return / max(grad_updates, 1.0),
                "sys/perf_per_offline_transaction": eval_return / max(offline_transitions_used, 1),
                "sys/perf_per_online_transaction": eval_return / max(online_transitions_used, 1),
            })
            log_callback(state, log_metrics_eval)

        return state, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    return loop_train_fn