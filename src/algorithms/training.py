import logging
import gymnasium
import numpy as np
from gymnax.environments.environment import Environment
import jax
import torch
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

from src.algorithms.reppo.common import OfflineReplayBuffer, OnlineReplayBuffer
from src.common import Transition
from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax
from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
from src.runners.maniskill_runner import _compute_action_bounds, normalize_action

def _pad_or_truncate(arr, length, fill=0.0):
    """Pad or truncate arr along axis 0 to `length`."""
    T = arr.shape[0]
    if T >= length:
        return arr[:length]
    pad_width = [(0, length - T)] + [(0, 0)] * (arr.ndim - 1)
    return jnp.pad(arr, pad_width, constant_values=fill)


def _prepare_padded_trajectory(obs, next_obs, action, reward, done, truncated, behavior_log_prob, num_steps):
    """Pad/truncate one trajectory to `num_steps` and mark synthetic padding via done/truncated."""
    traj_len = int(obs.shape[0])
    pad_steps = max(0, num_steps - traj_len)

    sampled_obs = _pad_or_truncate(obs, num_steps)
    sampled_next_obs = _pad_or_truncate(next_obs, num_steps)
    sampled_action = _pad_or_truncate(action, num_steps)
    sampled_reward = _pad_or_truncate(reward, num_steps)
    sampled_done = _pad_or_truncate(done, num_steps, 1.0)
    sampled_truncated = _pad_or_truncate(truncated, num_steps, 1.0)
    sampled_behavior_log_prob = _pad_or_truncate(behavior_log_prob, num_steps)

    if pad_steps > 0 and traj_len > 0:
        sampled_done = sampled_done.at[traj_len - 1].set(1.0)
        sampled_truncated = sampled_truncated.at[traj_len - 1].set(1.0)

    return (sampled_obs, sampled_next_obs, sampled_action, sampled_reward, sampled_done, sampled_truncated, sampled_behavior_log_prob)

# stack each list has num_envs arrays of shape [num_steps, ...]
def _stack_and_transpose(arrays):
    stacked = jnp.stack(arrays, axis=0)  # [num_envs, num_steps, ...]
    return jnp.swapaxes(stacked, 0, 1)   # [num_steps, num_envs, ...]

def _create_replay_buffer_from_demos(demo_path, env_id):
    config = DemoConfig(device=torch.device("cpu"), filter_success_only=True)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories, _ = loader.load_demo_dataset(demo_path)
    obs_list, next_obs_list, act_list, rew_list, done_list, trunc_list = [], [], [], [], [], []
    for traj in trajectories:
        obs_list.append(to_jax(traj['observations']))
        next_obs_list.append(to_jax(traj['next_observations']))
        # Keep raw (unnormalized) demo actions in replay buffer.
        act_list.append(to_jax(traj['actions']))
        rew_list.append(to_jax(traj['rewards']).squeeze(-1))
        done_list.append(to_jax(traj['dones']).squeeze(-1))
        trunc_list.append(to_jax(traj['truncations']).squeeze(-1))
    
    replay_buffer = OfflineReplayBuffer(
        obs=obs_list,
        next_obs=next_obs_list,
        action=act_list,
        reward=rew_list,
        done=done_list,
        truncated=trunc_list,
        behavior_log_prob=[jnp.zeros(t['observations'].shape[0]) for t in trajectories],
        size=len(trajectories),
    )
    print(f"[REPLAY] Offline buffer created: {replay_buffer.size} trajectories")
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
        metrics = {**update_metrics, **update_metrics}
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
    demo_path: str | None = None,
    bc_indicator: bool = False,
    decay_rate: float = 1.0,
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
        ratio = 1.0
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

        if bc_indicator:
            # initialise the offline replay buffer before the training loop starts
            offline_replay_buffer = _create_replay_buffer_from_demos(demo_path, env.spec.id)
            dataset_low, dataset_high = _compute_action_bounds(
                demo_path, env.spec.id, filter_success=True
            )
            offline_normalized_actions = [
                to_jax(normalize_action(np.asarray(a), dataset_low, dataset_high))
                for a in offline_replay_buffer.action
            ]

            # Compute initial behavior_log_probs for offline data using current policy
            policy_for_logprobs = policy_fn(state, True)
            for traj_idx in range(offline_replay_buffer.size):
                traj_obs = offline_replay_buffer.obs[traj_idx]  # [T, obs_dim]
                traj_action = offline_normalized_actions[traj_idx]  # normalized to [-1, 1]
                key, lp_key = jax.random.split(key)
                # Compute behavior log-prob on dataset action under BC-initialized actor.
                _, extras = policy_for_logprobs(lp_key, traj_obs, action_input=traj_action)
                if "behavior_log_prob" not in extras:
                    raise KeyError(
                        "Policy must return 'behavior_log_prob' when action_input is provided."
                    )
                offline_replay_buffer.behavior_log_prob[traj_idx] = extras["behavior_log_prob"]

            print(f"[REPLAY] Offline behavior_log_probs computed for {offline_replay_buffer.size} trajectories")
            print(f"  traj[0] behavior_log_prob shape={offline_replay_buffer.behavior_log_prob[0].shape} "
                f"min={float(offline_replay_buffer.behavior_log_prob[0].min()):.4f} "
                f"max={float(offline_replay_buffer.behavior_log_prob[0].max()):.4f}")

            online_replay_buffer = None  # will be created after first eval

        step = 0
        for i in range(num_iterations):
            for _ in range(train_steps_per_iteration):
                key, rollout_key, learn_key, sample_key = jax.random.split(key, 4)
                # Collect trajectories from 'state'
                policy = policy_fn(state, False)

                # Also collect fresh rollout transitions (for env stepping / state update)
                rollout_transitions, state = rollout_fn(
                    key=rollout_key, train_state=state, policy=policy
                )

                if bc_indicator:
                    # Sample num_envs trajectories, pad/truncate each to num_steps so we get [num_steps, num_envs, ...] with correct temporal structure.
                    num_offline = num_envs if ratio >= 1.0 else int(ratio * num_envs)
                    num_online = num_envs - num_offline

                    sample_key, off_subkey, on_subkey = jax.random.split(sample_key, 3)

                    traj_obs, traj_next_obs, traj_action = [], [], []
                    traj_reward, traj_done, traj_truncated, traj_behavior_log_prob = [], [], [], []
                    traj_source_is_offline = []

                    # Sample num_offline trajectories from offline buffer
                    for _ in range(num_offline):
                        off_subkey, pick_key = jax.random.split(off_subkey)
                        idx = int(jax.random.choice(pick_key, offline_replay_buffer.size))
                        sampled_obs, sampled_next_obs, sampled_action, sampled_reward, sampled_done, sampled_truncated, sampled_behavior_log_prob = _prepare_padded_trajectory(obs=offline_replay_buffer.obs[idx], next_obs=offline_replay_buffer.next_obs[idx], action=offline_normalized_actions[idx], reward=offline_replay_buffer.reward[idx], done=offline_replay_buffer.done[idx], truncated=offline_replay_buffer.truncated[idx], behavior_log_prob=offline_replay_buffer.behavior_log_prob[idx], num_steps=num_steps)
                        traj_obs.append(sampled_obs)
                        traj_next_obs.append(sampled_next_obs)
                        traj_action.append(sampled_action)
                        traj_reward.append(sampled_reward)
                        traj_done.append(sampled_done)
                        traj_truncated.append(sampled_truncated)
                        traj_behavior_log_prob.append(sampled_behavior_log_prob)
                        traj_source_is_offline.append(jnp.ones((num_steps,), dtype=jnp.float32))

                    # Sample num_online trajectories from online buffer
                    if num_online > 0 and online_replay_buffer is not None and online_replay_buffer.size > 0:
                        for _ in range(num_online):
                            on_subkey, pick_key = jax.random.split(on_subkey)
                            idx = int(jax.random.choice(pick_key, online_replay_buffer.size))
                            sampled_obs, sampled_next_obs, sampled_action, sampled_reward, sampled_done, sampled_truncated, sampled_behavior_log_prob = _prepare_padded_trajectory(obs=online_replay_buffer.obs[idx], next_obs=online_replay_buffer.next_obs[idx], action=online_replay_buffer.action[idx], reward=online_replay_buffer.reward[idx], done=online_replay_buffer.done[idx], truncated=online_replay_buffer.truncated[idx], behavior_log_prob=online_replay_buffer.behavior_log_prob[idx], num_steps=num_steps)
                            traj_obs.append(sampled_obs)
                            traj_next_obs.append(sampled_next_obs)
                            traj_action.append(sampled_action)
                            traj_reward.append(sampled_reward)
                            traj_done.append(sampled_done)
                            traj_truncated.append(sampled_truncated)
                            traj_behavior_log_prob.append(sampled_behavior_log_prob)
                            traj_source_is_offline.append(jnp.zeros((num_steps,), dtype=jnp.float32))
                    else:
                        # Fill remaining slots with more offline trajectories
                        for _ in range(num_online):
                            off_subkey, pick_key = jax.random.split(off_subkey)
                            idx = int(jax.random.choice(pick_key, offline_replay_buffer.size))
                            sampled_obs, sampled_next_obs, sampled_action, sampled_reward, sampled_done, sampled_truncated, sampled_behavior_log_prob = _prepare_padded_trajectory(obs=offline_replay_buffer.obs[idx], next_obs=offline_replay_buffer.next_obs[idx], action=offline_normalized_actions[idx], reward=offline_replay_buffer.reward[idx], done=offline_replay_buffer.done[idx], truncated=offline_replay_buffer.truncated[idx], behavior_log_prob=offline_replay_buffer.behavior_log_prob[idx], num_steps=num_steps)
                            traj_obs.append(sampled_obs)
                            traj_next_obs.append(sampled_next_obs)
                            traj_action.append(sampled_action)
                            traj_reward.append(sampled_reward)
                            traj_done.append(sampled_done)
                            traj_truncated.append(sampled_truncated)
                            traj_behavior_log_prob.append(sampled_behavior_log_prob)
                            traj_source_is_offline.append(jnp.ones((num_steps,), dtype=jnp.float32))

                    source_is_offline_batch = _stack_and_transpose(traj_source_is_offline)
                    transitions = Transition(
                        obs=_stack_and_transpose(traj_obs),
                        next_obs=_stack_and_transpose(traj_next_obs),
                        action=_stack_and_transpose(traj_action),
                        reward=_stack_and_transpose(traj_reward),
                        done=_stack_and_transpose(traj_done),
                        truncated=_stack_and_transpose(traj_truncated),
                        extras={
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

                # Execute an update to the policy with `transitions`
                state, train_metrics = learner_fn(
                    key=learn_key, train_state=state, batch=transitions
                )

                if step % train_log_interval == 0:
                    log_metrics = {**train_metrics, **replay_logprob_stats}
                    log_callback(state, utils.prefix_dict("train", log_metrics))
                step += 1
            policy = policy_fn(state, not stochastic_eval)
            key, eval_key = jax.random.split(key)

            if bc_indicator:
                eval_metrics, online_trajectories = eval_fn(eval_key, policy)
            else:
                eval_metrics = eval_fn(eval_key, policy)

            state = state.replace(iteration=state.iteration + 1)
            log_callback(state, utils.prefix_dict("eval", eval_metrics))

            if bc_indicator:
                # online_trajectories is a flat list of per-timestep Transitions, each with shape [num_eval_envs, ...].
                # stack into [max_episode_steps, num_eval_envs, ...] then split by env to get num_eval_envs trajectories of [max_episode_steps, ...].
                stacked_online = jax.tree.map(
                    lambda *xs: jnp.stack(xs), *online_trajectories
                )
                n_eval_envs = stacked_online.obs.shape[1]
                obs_list, next_obs_list, act_list = [], [], []
                rew_list, done_list, trunc_list, lp_list = [], [], [], []
                for env_idx in range(n_eval_envs):
                    obs_list.append(stacked_online.obs[:, env_idx])
                    next_obs_list.append(stacked_online.next_obs[:, env_idx])
                    act_list.append(stacked_online.action[:, env_idx])
                    rew_list.append(stacked_online.reward[:, env_idx])
                    done_list.append(stacked_online.done[:, env_idx])
                    trunc_list.append(stacked_online.truncated[:, env_idx])
                    lp_list.append(stacked_online.extras['behavior_log_prob'][:, env_idx])

                online_replay_buffer = OnlineReplayBuffer(
                    obs=obs_list,
                    next_obs=next_obs_list,
                    action=act_list,
                    reward=rew_list,
                    done=done_list,
                    truncated=trunc_list,
                    behavior_log_prob=lp_list,
                    size=n_eval_envs,
                )
                print(f"[REPLAY] Online buffer created: {online_replay_buffer.size} trajectories "
                      f"(from {n_eval_envs} eval envs)")
                print(f"  traj[0] obs={online_replay_buffer.obs[0].shape} "
                        f"action={online_replay_buffer.action[0].shape} "
                        f"reward={online_replay_buffer.reward[0].shape} "
                        f"done={online_replay_buffer.done[0].shape} "
                        f"behavior_log_prob={online_replay_buffer.behavior_log_prob[0].shape}")
                # Decay offline sampling ratio faster (reach 0% at ~60% of training).
                ratio = max(0.0, ratio - (float(decay_rate) / num_iterations))
                print(f"[REPLAY] ratio updated to {ratio:.4f}")

        return state, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    return loop_train_fn