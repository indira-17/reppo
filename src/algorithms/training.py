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
def _stack_and_transpose(arrays):
    stacked = jnp.stack(arrays, axis=0)  # [num_envs, num_steps, ...]
    return jnp.swapaxes(stacked, 0, 1)   # [num_steps, num_envs, ...]

def _align_partial_reset_flags(terminated, truncated):
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

def _create_replay_buffer_from_demos(demo_path, env_id, num_steps, filter_success, cut_at_first_success):
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

        done, trunc = _align_partial_reset_flags(terminated, trunc)

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
    print(f"[REPLAY] Offline buffer created: {replay_buffer.size} segments")
    print(f"  traj[0] obs={replay_buffer.obs[0].shape} action={replay_buffer.action[0].shape} "
              f"reward={replay_buffer.reward[0].shape} done={replay_buffer.done[0].shape} "
              f"behavior_log_prob={replay_buffer.behavior_log_prob[0].shape}")
    return replay_buffer

def make_scan_train_fn(
    env: Environment | tuple[Environment, Environment],
    total_time_steps: int,
    num_steps: int,
    num_envs: int,
    num_eval: int,
    num_seeds: int,
    max_episode_steps: int,
    stochastic_eval: bool,
    init_fn: InitFn,
    policy_fn: PolicyFn,
    learner_fn: LearnerFn,
    eval_fn: EvalFn | None = None,
    rollout_fn: RolloutFn | None = None,
    log_callback: LogCallback | None = None,
    demo_path: str | None = None,
    filter_success: bool = True,
    cut_at_first_success: bool = True,
    critic_offline_warmup_iters: int = 0,
    data_type: str = "online",
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

    offline_obs = offline_next_obs = offline_action = offline_reward = None
    offline_done = offline_truncated = offline_behavior_log_prob = None
    offline_size = 0
    buffer_memory_gb = 0.0

    if data_type == "expert":
        offline_replay_buffer = _create_replay_buffer_from_demos(
            demo_path,
            env.spec.id,
            num_steps,
            filter_success=filter_success,
            cut_at_first_success=cut_at_first_success,
        )
        dataset_low, dataset_high = _compute_action_bounds(
            demo_path,
            env.spec.id,
            filter_success=filter_success,
            cut_at_first_success=cut_at_first_success,
        )
        offline_normalized_actions = [
            to_jax(normalize_action(np.asarray(a), dataset_low, dataset_high))
            for a in offline_replay_buffer.action
        ]

        offline_obs = jnp.stack(offline_replay_buffer.obs, axis=0)
        offline_next_obs = jnp.stack(offline_replay_buffer.next_obs, axis=0)
        offline_action = jnp.stack(offline_normalized_actions, axis=0)
        offline_reward = jnp.stack(offline_replay_buffer.reward, axis=0)
        offline_done = jnp.stack(offline_replay_buffer.done, axis=0)
        offline_truncated = jnp.stack(offline_replay_buffer.truncated, axis=0)
        offline_behavior_log_prob = jnp.stack(offline_replay_buffer.behavior_log_prob, axis=0)
        offline_size = int(offline_obs.shape[0])

        buffer_memory_gb = (
            offline_obs.nbytes
            + offline_next_obs.nbytes
            + offline_action.nbytes
            + offline_reward.nbytes
            + offline_done.nbytes
            + offline_truncated.nbytes
            + offline_behavior_log_prob.nbytes
        ) / 1e9
        logging.info(
            f"[REPLAY] expert buffer for scan mode: {offline_size} segments, {buffer_memory_gb:.3f} GB"
        )

    def train_step_replay(
        state: TrainState, key: Key
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        key, rollout_key, learn_key, off_subkey = jax.random.split(key, 4)
        # Collect trajectories from 'state'
        policy = policy_fn(state, False)

        # Also collect fresh rollout transitions (for env stepping / state update)
        rollout_transitions, state = rollout_fn(
            key=rollout_key, train_state=state, policy=policy
        )

        if data_type == 'expert':
            # JAX scan mode: use fully-offline replay batches (fixed-size, jittable).
            num_offline = num_envs
            num_online = 0

            traj_obs, traj_next_obs, traj_action = [], [], []
            traj_reward, traj_done, traj_truncated, traj_behavior_log_prob = [], [], [], []
            traj_source_is_offline = [jnp.ones((num_steps,), dtype=jnp.float32) for _ in range(num_envs)]

            # Sample num_offline trajectories from offline buffer (already fixed to num_steps)
            off_subkey, pick_key = jax.random.split(off_subkey)
            sampled_offline_indices = jax.random.choice(
                pick_key, offline_size, shape=(num_offline,), replace=True
            )

            for i in range(num_offline):
                idx = sampled_offline_indices[i]
                traj_obs.append(offline_obs[idx])
                traj_next_obs.append(offline_next_obs[idx])
                traj_action.append(offline_action[idx])
                traj_reward.append(offline_reward[idx])
                traj_done.append(offline_done[idx])
                traj_truncated.append(offline_truncated[idx])
                traj_behavior_log_prob.append(offline_behavior_log_prob[idx])

            # for online slots IS weight = 1.0
            # uniform: IS weight = 1 everywhere (no bias correction needed)
            is_weight_batch = jnp.ones((num_steps, num_envs), dtype=jnp.float32)

            source_is_offline_batch = _stack_and_transpose(traj_source_is_offline)
            transitions = Transition(
                obs=_stack_and_transpose(traj_obs),
                next_obs=_stack_and_transpose(traj_next_obs),
                action=_stack_and_transpose(traj_action),
                reward=_stack_and_transpose(traj_reward),
                done=_stack_and_transpose(traj_done),
                truncated=_stack_and_transpose(traj_truncated),
                extras={
                    "is_weight": is_weight_batch,
                    "behavior_log_prob": _stack_and_transpose(traj_behavior_log_prob),
                    "source_is_offline": source_is_offline_batch,
                },
            )
            behavior_lp_batch = transitions.extras["behavior_log_prob"]
            offline_count = source_is_offline_batch.sum()
            online_count = (1.0 - source_is_offline_batch).sum()
            offline_mean = jnp.where(
                offline_count > 0,
                (behavior_lp_batch * source_is_offline_batch).sum() / offline_count,
                0.0,
            )
            online_mean = jnp.where(
                online_count > 0,
                (behavior_lp_batch * (1.0 - source_is_offline_batch)).sum() / online_count,
                0.0,
            )

            replay_logprob_stats = {
                "behavior_log_prob_offline_replay": offline_mean,
                "behavior_log_prob_online_replay": online_mean,
            }
        else:
            transitions = rollout_transitions
            replay_logprob_stats = {}

        # Execute an update to the policy with `transitions`
        state, update_metrics, _ = learner_fn(
            key=learn_key, train_state=state, batch=transitions
        )
        metrics = update_metrics
        state = state.replace(iteration=state.iteration + 1)
        log_metrics = {**metrics, **replay_logprob_stats}
        # critic_diag/, actor_diag/, sys/ stay at top level and only plain scalar metrics get the "train/" prefix.
        namespaced = {k: v for k, v in log_metrics.items() if "/" in k}
        plain = {k: v for k, v in log_metrics.items() if "/" not in k}
        log_callback(state, {**utils.prefix_dict("train", plain), **namespaced})

        return state, metrics

    def train_eval_step_replay(key, train_state):
        train_key, eval_key = jax.random.split(key)
        train_state, train_metrics = jax.lax.scan(
            f=train_step_replay,
            init=train_state,
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
        policy = policy_fn(train_state, not stochastic_eval)
        eval_metrics = eval_fn(eval_key, policy)
        eval_return = eval_metrics["episode_return"]
        grad_updates = train_metrics.get("sys/grad_updates", jnp.array(1.0))
        if data_type == 'expert':
            offline_transitions_used = train_state.time_steps
            offline_transitions_used = jnp.array(0)
        else:
            offline_transitions_used = jnp.array(0)
            offline_transitions_used = jnp.array(0)
            transitions_used = train_state.time_steps
        
        # Everything goes through log_callback so all logging shares the same wandb step (step=state.time_steps)
        log_metrics_eval = utils.prefix_dict("eval", eval_metrics)
        log_metrics_eval.update({
            "sys/perf_per_gb":            eval_return / buffer_memory_gb if buffer_memory_gb > 0 else 0.0,
            "sys/perf_per_grad_update":   eval_return / jnp.maximum(grad_updates, 1.0),
            "sys/perf_per_offline_transaction": eval_return / jnp.maximum(offline_transitions_used, 1) if data_type == 'expert' else 0.0,
            "sys/perf_per_online_transaction": eval_return / jnp.maximum(online_transitions_used, 1) if data_type == 'expert' else 0.0,
            "sys/perf_per_transaction": eval_return / jnp.maximum(transitions_used, 1),
        })
        log_callback(train_state, log_metrics_eval)
        return train_state, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    def train_eval_loop_body(
        train_state: TrainState, key: Key
    ) -> tuple[TrainState, dict]:
        # Map execution of the train+eval step across num_seeds (will be looped using jax.lax.scan)
        key, subkey = jax.random.split(key)
        train_state, metrics = jax.vmap(train_eval_step_replay)(
            jax.random.split(subkey, num_seeds), train_state
        )
        jax.debug.callback(log_callback, train_state, metrics)
        return train_state, metrics

    def init_train_state(key: Key) -> TrainState:
        key, env_key = jax.random.split(key)
        train_state = init_fn(key)
        obs, env_state = utils.init_env_state(key=env_key, env=env, num_envs=num_envs)
        train_state = train_state.replace(last_obs=obs, last_env_state=env_state)
        return train_state

    # Define the training loop
    def scan_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )
        key, init_key = jax.random.split(key)
        init_keys = jax.random.split(init_key, num_seeds)

        if data_type in ("PER", "random"):
            train_state = jax.vmap(init_train_state)(init_keys)

            def _seed_replay(ts, k):
                k, rk = jax.random.split(k)
                policy = policy_fn(ts, False)
                rt, _ = rollout_fn(key=rk, train_state=ts, policy=policy)
                return _init_replay_from_rollout(rt)

            replay_state = jax.vmap(_seed_replay)(train_state, init_keys)
            keys = jax.random.split(key, num_iterations)
            (state, _), metrics = jax.lax.scan(
                f=train_eval_loop_body_replay,
                init=(train_state, replay_state),
                xs=keys,
            )
            return state, metrics

        train_state = jax.vmap(init_train_state)(init_keys)
        keys = jax.random.split(key, num_iterations)
        # Run the training and evaluation loop from the initialized training state
        state, metrics = jax.lax.scan(f=train_eval_loop_body, init=train_state, xs=keys)
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
            if iter_idx < off_end:
                return 1.0

            if off_end == 0:
                return 0.5

            # Linear decay after warmup, reaching 0.0 at the final iteration.
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
                offline_replay_buffer = _create_replay_buffer_from_demos(demo_path, env.spec.id, num_steps, filter_success=filter_success, cut_at_first_success=cut_at_first_success)
                dataset_low, dataset_high = _compute_action_bounds(demo_path, env.spec.id, filter_success=filter_success, cut_at_first_success=cut_at_first_success)

                offline_normalized_actions = [
                    to_jax(normalize_action(np.asarray(a), dataset_low, dataset_high))
                    for a in offline_replay_buffer.action
                ]
                # compute behavior log-probs under the current actor
                policy_for_logprobs = policy_fn(state, True)
                for traj_idx in range(offline_replay_buffer.size):
                    traj_obs = offline_replay_buffer.obs[traj_idx]
                    traj_action = offline_normalized_actions[traj_idx]
                    key, lp_key = jax.random.split(key)
                    _, extras = policy_for_logprobs(lp_key, traj_obs, action_input=traj_action)
                    if "behavior_log_prob" not in extras:
                        raise KeyError("Policy must return 'behavior_log_prob' when action_input is provided.")
                    offline_replay_buffer.behavior_log_prob[traj_idx] = extras["behavior_log_prob"]
                print(f"[REPLAY] Offline behavior_log_probs computed for {offline_replay_buffer.size} trajectories")
                
            else:  # PER / random: seed buffer with initial rollouts, then grow via FIFO ring
                logging.info(f"[REPLAY] Collecting initial {data_type} buffer ({max_buffer_size} transitions)...")
                buf_obs, buf_next_obs, buf_action = [], [], []
                buf_reward, buf_done, buf_trunc, buf_blp = [], [], [], []
                buf_priority = [] if data_type == 'PER' else None
                while len(buf_obs) < num_segments_cap:
                    key, rk = jax.random.split(key)
                    policy = policy_fn(state, False)
                    rt, state = rollout_fn(key=rk, train_state=state, policy=policy)
                    for ei in range(num_envs):
                        if len(buf_obs) >= num_segments_cap:
                            break
                        buf_obs.append(rt.obs[:, ei])
                        buf_next_obs.append(rt.next_obs[:, ei])
                        buf_action.append(rt.action[:, ei])
                        buf_reward.append(rt.reward[:, ei])
                        buf_done.append(rt.done[:, ei])
                        buf_trunc.append(rt.truncated[:, ei])
                        blp = rt.extras.get("behavior_log_prob", None)
                        buf_blp.append(blp[:, ei] if blp is not None else jnp.zeros(num_steps))
                        if data_type == 'PER':
                            buf_priority.append(1.0)
                offline_replay_buffer = OfflineReplayBuffer(
                    obs=buf_obs, next_obs=buf_next_obs, action=buf_action,
                    reward=buf_reward, done=buf_done, truncated=buf_trunc,
                    behavior_log_prob=buf_blp, priority=buf_priority, size=len(buf_obs),
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
                        is_weight_batch = _stack_and_transpose(traj_is_weight)
                    else:
                        # uniform: IS weight = 1 everywhere (no bias correction needed)
                        is_weight_batch = jnp.ones((num_steps, num_envs), dtype=jnp.float32)

                    source_is_offline_batch = _stack_and_transpose(traj_source_is_offline)
                    transitions = Transition(
                        obs=_stack_and_transpose(traj_obs),
                        next_obs=_stack_and_transpose(traj_next_obs),
                        action=_stack_and_transpose(traj_action),
                        reward=_stack_and_transpose(traj_reward),
                        done=_stack_and_transpose(traj_done),
                        truncated=_stack_and_transpose(traj_truncated),
                        extras={
                            "is_weight": is_weight_batch,
                            "behavior_log_prob": _stack_and_transpose(traj_behavior_log_prob),
                            "source_is_offline": source_is_offline_batch,
                        },
                    )
                    behavior_lp_batch = transitions.extras["behavior_log_prob"]
                    offline_count = source_is_offline_batch.sum()
                    online_count = (1.0 - source_is_offline_batch).sum()
                    offline_mean = jnp.where(
                        offline_count > 0,
                        (behavior_lp_batch * source_is_offline_batch).sum() / offline_count,
                        0.0,
                    )
                    online_mean = jnp.where(
                        online_count > 0,
                        (behavior_lp_batch * (1.0 - source_is_offline_batch)).sum() / online_count,
                        0.0,
                    )

                    replay_logprob_stats = {
                        "behavior_log_prob_offline_replay": offline_mean,
                        "behavior_log_prob_online_replay": online_mean,
                    }
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
                    replay_logprob_stats = {}
                    online_transitions_used += num_envs * num_steps

                # total transitions consumed by the learner (online + offline replayed)
                num_samples = online_transitions_used + offline_transitions_used

                # Execute an update to the policy with `transitions`
                state, train_metrics, per_env_td_error = learner_fn(
                    key=learn_key, train_state=state, batch=transitions
                )

                if step % train_log_interval == 0:
                    log_metrics = {**train_metrics, **replay_logprob_stats}
                    # critic_diag/, actor_diag/, sys/ stay at top level and only plain scalar metrics get the "train/" prefix.
                    namespaced = {k: v for k, v in log_metrics.items() if "/" in k}
                    plain = {k: v for k, v in log_metrics.items() if "/" not in k}
                    log_callback(state, {**utils.prefix_dict("train", plain), **namespaced})

                # ring buffer FIFO-insert for replay buffer
                if data_type in ('PER', 'random'):
                    max_p = max(offline_replay_buffer.priority) if (data_type == 'PER' and offline_replay_buffer.priority) else 1.0
                    for ei in range(num_envs):
                        w_idx = ring_write_idx % num_segments_cap
                        if offline_replay_buffer.size < num_segments_cap:
                            offline_replay_buffer.obs.append(rollout_transitions.obs[:, ei])
                            offline_replay_buffer.next_obs.append(rollout_transitions.next_obs[:, ei])
                            offline_replay_buffer.action.append(rollout_transitions.action[:, ei])
                            offline_replay_buffer.reward.append(rollout_transitions.reward[:, ei])
                            offline_replay_buffer.done.append(rollout_transitions.done[:, ei])
                            offline_replay_buffer.truncated.append(rollout_transitions.truncated[:, ei])
                            blp = rollout_transitions.extras.get("behavior_log_prob", None)
                            offline_replay_buffer.behavior_log_prob.append(blp[:, ei] if blp is not None else jnp.zeros(num_steps))
                            if data_type == 'PER':
                                offline_replay_buffer.priority.append(max_p)
                            offline_replay_buffer.size += 1
                        else:
                            offline_replay_buffer.obs[w_idx] = rollout_transitions.obs[:, ei]
                            offline_replay_buffer.next_obs[w_idx] = rollout_transitions.next_obs[:, ei]
                            offline_replay_buffer.action[w_idx] = rollout_transitions.action[:, ei]
                            offline_replay_buffer.reward[w_idx] = rollout_transitions.reward[:, ei]
                            offline_replay_buffer.done[w_idx] = rollout_transitions.done[:, ei]
                            offline_replay_buffer.truncated[w_idx] = rollout_transitions.truncated[:, ei]
                            blp = rollout_transitions.extras.get("behavior_log_prob", None)
                            offline_replay_buffer.behavior_log_prob[w_idx] = blp[:, ei] if blp is not None else jnp.zeros(num_steps)
                            if data_type == 'PER':
                                offline_replay_buffer.priority[w_idx] = max_p
                        ring_write_idx += 1
                    
                    # update priorities for sampled segments using per-segment TD errors.
                    # segments occupy slots 0..num_offline-1 (matching batch assembly order).
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