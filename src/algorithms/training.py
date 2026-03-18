import logging
import jax
import jax.numpy as jnp
import gymnasium
from gymnax.environments.environment import Environment
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
    Transition
)
from src.algorithms import utils
from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax
from src.algorithms.reppo.common import ReplayBuffer
import numpy as np

def _concat_sequence_batches(batches: list[Transition]) -> Transition:
    extras = {}
    if batches and batches[0].extras:
        for key in batches[0].extras:
            extras[key] = jnp.concatenate([batch.extras[key] for batch in batches], axis=1)
    return Transition(
        obs=jnp.concatenate([batch.obs for batch in batches], axis=1),
        next_obs=jnp.concatenate([batch.next_obs for batch in batches], axis=1),
        action=jnp.concatenate([batch.action for batch in batches], axis=1),
        reward=jnp.concatenate([batch.reward for batch in batches], axis=1),
        done=jnp.concatenate([batch.done for batch in batches], axis=1),
        truncated=jnp.concatenate([batch.truncated for batch in batches], axis=1),
        extras=extras,
    )

def _sample_from_offline_batches(
    replay_buffer: ReplayBuffer,
    num_steps: int,
    num_sequences: int,
    cursor: int,
) -> tuple[Transition, int]:
    """
    Sample sequential fixed-length trajectories from the offline replay buffer.
    Each sampled sequence is contiguous in time and never crosses an episode boundary;
    short tails are zero-padded to length `num_steps`.
    Returns a Transition of shape (num_steps, num_sequences, ...).
    """
    size = int(replay_buffer.size)
    if size <= 0:
        raise ValueError("Replay buffer is empty.")

    starts = np.where(np.asarray(replay_buffer.episode_starts)[:size] == 1)[0]
    starts = starts.tolist()
    if len(starts) == 0:
        raise ValueError("No episodes found in replay buffer.")
    ends = starts[1:] + [size]

    # Keep cursor in range.
    cursor = int(cursor) % size

    def _episode_index_for(pos: int) -> int:
        # Rightmost start <= pos
        return int(np.searchsorted(starts, pos, side="right") - 1)

    def _advance_to_valid(pos: int) -> int:
        # Ensure cursor lands inside a known episode segment.
        if pos < starts[0]:
            return starts[0]
        if pos >= size:
            return starts[0]
        ep_idx = _episode_index_for(pos)
        if ep_idx < 0:
            return starts[0]
        ep_end = ends[ep_idx]
        if pos >= ep_end:
            next_idx = (ep_idx + 1) % len(starts)
            return starts[next_idx]
        return pos

    cursor = _advance_to_valid(cursor)

    def _alloc_like(field: np.ndarray, fill_value=0):
        shape = (num_steps, num_sequences) + tuple(field.shape[1:])
        return np.full(shape, fill_value, dtype=np.asarray(field).dtype)

    obs = _alloc_like(replay_buffer.observations, fill_value=0)
    next_obs = _alloc_like(replay_buffer.next_observations, fill_value=0)
    action = _alloc_like(replay_buffer.actions, fill_value=0)
    reward = _alloc_like(replay_buffer.rewards, fill_value=0)
    done = _alloc_like(replay_buffer.dones, fill_value=1)
    truncated = _alloc_like(replay_buffer.truncations, fill_value=1)
    behavior_log_prob = _alloc_like(replay_buffer.behavior_log_probs, fill_value=0)
    episode_start = _alloc_like(replay_buffer.episode_starts, fill_value=0)
    valid_mask = np.zeros((num_steps, num_sequences), dtype=np.float32)

    for b in range(num_sequences):
        cursor = _advance_to_valid(cursor)
        ep_idx = _episode_index_for(cursor)
        ep_start, ep_end = starts[ep_idx], ends[ep_idx]

        take_end = min(cursor + num_steps, ep_end)
        length = max(0, take_end - cursor)

        if length > 0:
            s = slice(cursor, take_end)
            obs[:length, b] = np.asarray(replay_buffer.observations[s])
            next_obs[:length, b] = np.asarray(replay_buffer.next_observations[s])
            action[:length, b] = np.asarray(replay_buffer.actions[s])
            reward[:length, b] = np.asarray(replay_buffer.rewards[s])
            done[:length, b] = np.asarray(replay_buffer.dones[s])
            truncated[:length, b] = np.asarray(replay_buffer.truncations[s])
            behavior_log_prob[:length, b] = np.asarray(replay_buffer.behavior_log_probs[s])
            episode_start[:length, b] = np.asarray(replay_buffer.episode_starts[s])
            valid_mask[:length, b] = 1.0

        # Step forward by one full sequence chunk without crossing boundary.
        if take_end >= ep_end:
            next_ep_idx = (ep_idx + 1) % len(starts)
            cursor = starts[next_ep_idx]
        else:
            cursor = take_end

    is_offline = jnp.ones((num_steps, num_sequences), dtype=jnp.float32)

    return Transition(
        obs=jnp.asarray(obs),
        next_obs=jnp.asarray(next_obs),
        action=jnp.asarray(action),
        reward=jnp.asarray(reward),
        done=jnp.asarray(done),
        truncated=jnp.asarray(truncated),
        extras={
            "behavior_log_prob": jnp.asarray(behavior_log_prob),
            "episode_start": jnp.asarray(episode_start),
            "is_offline": is_offline,
            "valid_mask": jnp.asarray(valid_mask),
        },
    ), int(cursor)


def _sample_from_online_batches(
    batches: list[Transition],
    num_steps: int,
    num_sequences: int,
    batch_cursor: int,
    env_cursor: int,
) -> tuple[Transition, int, int]:
    pieces = []
    remaining = num_sequences

    while remaining > 0:
        batch = batches[batch_cursor % len(batches)]
        available = batch.obs.shape[1] - env_cursor
        take = min(remaining, available)
        pieces.append(
            Transition(
                obs=batch.obs[:num_steps, env_cursor : env_cursor + take],
                next_obs=batch.next_obs[:num_steps, env_cursor : env_cursor + take],
                action=batch.action[:num_steps, env_cursor : env_cursor + take],
                reward=batch.reward[:num_steps, env_cursor : env_cursor + take],
                done=batch.done[:num_steps, env_cursor : env_cursor + take],
                truncated=batch.truncated[:num_steps, env_cursor : env_cursor + take],
                extras={
                    "behavior_log_prob": batch.extras["behavior_log_prob"][:num_steps, env_cursor : env_cursor + take],
                    "is_offline": batch.extras["is_offline"][:num_steps, env_cursor : env_cursor + take],
                    "valid_mask": jnp.ones_like(batch.reward[:num_steps, env_cursor : env_cursor + take]),
                },
            )
        )
        remaining -= take
        env_cursor += take
        if env_cursor >= batch.obs.shape[1]:
            batch_cursor += 1
            env_cursor = 0

    return _concat_sequence_batches(pieces), batch_cursor, env_cursor

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

    def train_step(
        state: TrainState, key: Key
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        key, rollout_key, learn_key = jax.random.split(key, 3)
        # Collect trajectories from `state`
        policy = policy_fn(state, False)
        transitions, state = rollout_fn(
            key=rollout_key, train_state=state, policy=policy
        )
        # Execute an update to the policy with `transitions`
        state, update_metrics = learner_fn(
            key=learn_key, train_state=state, batch=transitions
        )
        metrics = update_metrics
        state = state.replace(iteration=state.iteration + 1)
        return state, metrics

    def train_eval_step(key, train_state):
        train_key, eval_key = jax.random.split(key)
        train_state, train_metrics = jax.lax.scan(
            f=train_step,
            init=train_state,
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)
        policy = policy_fn(train_state, not stochastic_eval)
        eval_metrics = eval_fn(eval_key, policy)
        metrics = {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

        return train_state, metrics

    def train_eval_loop_body(
        train_state: TrainState, key: Key
    ) -> tuple[TrainState, dict]:
        # Map execution of the train+eval step across num_seeds (will be looped using jax.lax.scan)
        key, subkey = jax.random.split(key)
        train_state, metrics = jax.vmap(train_eval_step)(
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
        train_state = jax.vmap(init_train_state)(jax.random.split(init_key, num_seeds))
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
    replay_buffer: ReplayBuffer | None = None,
    replay_offline_fraction: float = 0.5,
    cfg = None,
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

    def loop_train_fn(key: Key) -> tuple[TrainState, dict]:
        offline_buffer = replay_buffer
        prev_iteration_online_batches = None
        offline_cursor = 0
        online_batch_cursor = 0
        online_env_cursor = 0
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

        step = 0
        for outer_iter in range(num_iterations):
            state = state.replace(
                rollout_actor={
                    'graphdef': state.actor.graphdef,
                    'params': state.actor.params,
                }
            )
            current_iteration_online_batches = []
            for _ in range(train_steps_per_iteration):
                key, rollout_key, learn_key = jax.random.split(key, 3)
                # Collect trajectories from `state`
                policy = policy_fn(state, False)
                transitions, state = rollout_fn(
                    key=rollout_key, train_state=state, policy=policy
                )

                current_iteration_online_batches.append(transitions)

                # Co-training with a fixed offline buffer and previous-iteration online rollouts.
                if offline_buffer is not None:
                    online_batches = prev_iteration_online_batches

                    # alpha_k = offline fraction. Iteration 0 is all offline; then decay
                    # linearly toward 0 so that bc_transition_iterations is fully online.
                    if cfg is not None and hasattr(cfg, 'algorithm'):
                        bc_ti = max(int(cfg.algorithm.bc_transition_iterations) - 1, 1)
                        progress = min(1.0, float(outer_iter) / float(bc_ti))
                        alpha_k = 1.0 - progress
                    else:
                        alpha_k = replay_offline_fraction

                    n_offline = int(num_envs * alpha_k)
                    n_online = num_envs - n_offline

                    if online_batches is None:
                        n_online = 0
                        n_offline = num_envs

                    online_batch = None
                    offline_batch = None

                    if n_online > 0:
                        online_batch, online_batch_cursor, online_env_cursor = _sample_from_online_batches(
                            online_batches,
                            num_steps,
                            n_online,
                            online_batch_cursor,
                            online_env_cursor,
                        )

                    if n_offline > 0 and int(offline_buffer.size) > 0:
                        offline_batch, offline_cursor = _sample_from_offline_batches(
                            offline_buffer, num_steps, n_offline, offline_cursor
                        )

                    if online_batch is None:
                        transitions = offline_batch
                    elif offline_batch is None:
                        transitions = online_batch
                    else:
                        transitions = _concat_sequence_batches([online_batch, offline_batch])

                    transitions.extras["is_offline"] = jnp.concatenate(
                        [
                            jnp.zeros((num_steps, n_online), dtype=jnp.float32),
                            jnp.ones((num_steps, n_offline), dtype=jnp.float32),
                        ],
                        axis=1,
                    )
                else:
                    transitions.extras["is_offline"] = jnp.zeros((num_steps, num_envs), dtype=jnp.float32)
                    transitions.extras["valid_mask"] = jnp.ones((num_steps, num_envs), dtype=jnp.float32)
                # Execute an update to the policy with `transitions`
                state, train_metrics = learner_fn(
                    key=learn_key, train_state=state, batch=transitions
                )

                if step % train_log_interval == 0:
                    log_callback(state, utils.prefix_dict("train", train_metrics))
                step += 1

            if current_iteration_online_batches:
                prev_iteration_online_batches = current_iteration_online_batches
                online_batch_cursor = 0
                online_env_cursor = 0
            policy = policy_fn(state, not stochastic_eval)
            key, eval_key = jax.random.split(key)
            eval_metrics = eval_fn(eval_key, policy)
            state = state.replace(iteration=state.iteration + 1)
            log_callback(state, utils.prefix_dict("eval", eval_metrics))
        return state, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    return loop_train_fn