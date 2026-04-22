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

from src.algorithms.reppo.common import OfflineReplayBuffer
from src.common import Transition
from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax
from src.maniskill_utils.maniskill_dataloader_shabnam import DemoConfig, ManiSkillDemoLoader
from src.runners.maniskill_runner import _compute_action_bounds, normalize_action

def _take_fixed_horizon(arr, length):
    """Take first `length` steps. Caller guarantees arr.shape[0] >= length."""
    return arr[:length]

# stack each list has num_envs arrays of shape [num_steps, ...]
def _stack_and_transpose(arrays):
    stacked = jnp.stack(arrays, axis=0)  # [num_envs, num_steps, ...]
    return jnp.swapaxes(stacked, 0, 1)   # [num_steps, num_envs, ...]

def _create_replay_buffer_from_demos(demo_path, env_id, num_steps, filter_success, cut_at_first_success):
    config = DemoConfig(device=torch.device("cpu"), filter_success_only=filter_success, cut_at_first_success=cut_at_first_success)
    loader = ManiSkillDemoLoader(config, env_id)
    trajectories, _ = loader.load_demo_dataset(demo_path)
    obs_list, next_obs_list, act_list, rew_list, done_list, trunc_list = [], [], [], [], [], []
    dropped_short = 0
    curr_obs, curr_next_obs, curr_act = [], [], []
    curr_rew, curr_done, curr_trunc = [], [], []
    curr_len = 0
    dropped_tail = 0
    for traj in trajectories:
        obs = to_jax(traj['observations'])
        next_obs = to_jax(traj['next_observations'])
        act = to_jax(traj['actions'])
        rew = to_jax(traj['rewards']).squeeze(-1)
        done = to_jax(traj['dones']).squeeze(-1)
        trunc = to_jax(traj['truncations']).squeeze(-1)

        traj_len = int(obs.shape[0])
        if traj_len == 0:
            dropped_short += 1
            continue

        # Match online ManiSkill partial-reset semantics:
        # done is always 0; episode boundaries are encoded via truncated=1.
        boundary = jnp.logical_or(done.astype(bool), trunc.astype(bool))
        boundary = boundary.at[-1].set(True)
        done = jnp.zeros_like(done)
        trunc = boundary.astype(trunc.dtype)

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
                obs_list.append(jnp.concatenate(curr_obs, axis=0))
                next_obs_list.append(jnp.concatenate(curr_next_obs, axis=0))
                # Keep raw (unnormalized) demo actions in replay buffer.
                act_list.append(jnp.concatenate(curr_act, axis=0))
                rew_list.append(jnp.concatenate(curr_rew, axis=0))
                done_list.append(jnp.concatenate(curr_done, axis=0))
                trunc_list.append(jnp.concatenate(curr_trunc, axis=0))

                curr_obs, curr_next_obs, curr_act = [], [], []
                curr_rew, curr_done, curr_trunc = [], [], []
                curr_len = 0

    if curr_len > 0:
        dropped_tail = curr_len

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
    print(f"[REPLAY] Offline buffer created: {replay_buffer.size} trajectories")
    if dropped_short > 0:
        print(f"[REPLAY] Dropped {dropped_short} short trajectories (< {num_steps} steps)")
    if dropped_tail > 0:
        print(f"[REPLAY] Dropped {dropped_tail} leftover transitions (< {num_steps}) after packing")
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
    filter_success: bool = True,
    cut_at_first_success: bool = False,
    critic_offline_warmup_iters: int = 2,
    critic_mixed_warmup_iters: int = 2,
    mixed_offline_start_ratio: float = 0.75,
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
        Phase schedule:
          1) warmup the critic on offline data only
          2) warmup the critic on mixed offline+online data, decaying offline ratio
          3) continue decaying via `decay_rate`
        Actor freezing/unfreezing is handled in learner via `bc_actor_update_delay`.
        """
        off_end = max(0, int(critic_offline_warmup_iters))
        mix_len = max(0, int(critic_mixed_warmup_iters))
        mix_end = off_end + mix_len

        if iter_idx < off_end:
            return 1.0

        if iter_idx < mix_end and mix_len > 0:
            mix_progress = (iter_idx - off_end) / max(1, mix_len - 1)
            return float(max(0.0, mixed_offline_start_ratio * (1.0 - mix_progress)))

        # After explicit warmup phases, keep old global decay behaviour.
        post_start_ratio = 0.0 if mix_len > 0 else float(mixed_offline_start_ratio)
        post_steps = iter_idx - mix_end
        return float(max(0.0, post_start_ratio - (float(decay_rate) / max(1, num_iterations)) * post_steps))

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

        if bc_indicator:
            # initialise the offline replay buffer before the training loop starts
            offline_replay_buffer = _create_replay_buffer_from_demos(demo_path, env.spec.id, num_steps, filter_success=filter_success, cut_at_first_success=cut_at_first_success)
            dataset_low, dataset_high = _compute_action_bounds(
                demo_path, env.spec.id, filter_success=filter_success
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

        step = 0
        for iter_idx in range(num_iterations):
            ratio = _offline_ratio_for_iteration(iter_idx, num_iterations) if bc_indicator else 0.0
            for _ in range(train_steps_per_iteration):
                key, rollout_key, learn_key, off_subkey = jax.random.split(key, 4)
                # Collect trajectories from 'state'
                policy = policy_fn(state, False)

                # Also collect fresh rollout transitions (for env stepping / state update)
                rollout_transitions, state = rollout_fn(
                    key=rollout_key, train_state=state, policy=policy
                )

                if bc_indicator:
                    # Sample num_envs trajectories of fixed length num_steps.
                    num_offline = num_envs if ratio >= 1.0 else int(ratio * num_envs)
                    num_online = num_envs - num_offline

                    traj_obs, traj_next_obs, traj_action = [], [], []
                    traj_reward, traj_done, traj_truncated, traj_behavior_log_prob = [], [], [], []
                    traj_source_is_offline = []

                    # Sample num_offline trajectories from offline buffer (already fixed to num_steps)
                    for _ in range(num_offline):
                        off_subkey, pick_key = jax.random.split(off_subkey)
                        idx = int(jax.random.choice(pick_key, offline_replay_buffer.size))
                        traj_obs.append(offline_replay_buffer.obs[idx])
                        traj_next_obs.append(offline_replay_buffer.next_obs[idx])
                        traj_action.append(offline_normalized_actions[idx])
                        traj_reward.append(offline_replay_buffer.reward[idx])
                        traj_done.append(offline_replay_buffer.done[idx])
                        traj_truncated.append(offline_replay_buffer.truncated[idx])
                        traj_behavior_log_prob.append(offline_replay_buffer.behavior_log_prob[idx])
                        traj_source_is_offline.append(jnp.ones((num_steps,), dtype=jnp.float32))

                    # Use current rollout transitions as truly on-policy online data
                    # rollout_transitions shape: [num_steps, num_envs, ...]
                    if num_online > 0:
                        for env_idx in range(num_online):
                            traj_obs.append(rollout_transitions.obs[:, env_idx])
                            traj_next_obs.append(rollout_transitions.next_obs[:, env_idx])
                            traj_action.append(rollout_transitions.action[:, env_idx])
                            traj_reward.append(rollout_transitions.reward[:, env_idx])
                            traj_done.append(rollout_transitions.done[:, env_idx])
                            traj_truncated.append(rollout_transitions.truncated[:, env_idx])
                            traj_behavior_log_prob.append(rollout_transitions.extras["behavior_log_prob"][:, env_idx])
                            traj_source_is_offline.append(jnp.zeros((num_steps,), dtype=jnp.float32))

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

            eval_metrics = eval_fn(eval_key, policy)

            state = state.replace(iteration=state.iteration + 1)
            log_callback(state, utils.prefix_dict("eval", eval_metrics))

            if bc_indicator:
                print(f"[REPLAY] ratio for iteration {iter_idx}: {ratio:.4f}")

        return state, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    return loop_train_fn