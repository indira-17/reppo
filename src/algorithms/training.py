import logging
import torch
import gymnasium
import numpy as np
from gymnax.environments.environment import Environment
import jax
import flashbax as fbx
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

def to_jax(x):
    if isinstance(x, np.ndarray):
        return jnp.array(x)
    elif isinstance(x, jax.Array):
        return x
    elif isinstance(x, torch.Tensor):
        return jnp.asarray(x.detach().cpu().numpy()) # jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))
    elif isinstance(x, dict) or isinstance(x, list):
        return jax.tree.map(to_jax, x)
    else:
        return jnp.array(x)

def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, jax.Array):
        return torch.from_numpy(np.array(x))
    else:
        raise ValueError(f"Cannot convert type {type(x)} to torch.Tensor")

def _update_initial_obs_pool(initial_obs_pool: jax.Array, reset_obs: jax.Array, episode_ended: jax.Array) -> jax.Array:
    """Replace pool entries only with post-auto-reset observations."""
    mask = episode_ended.astype(bool).reshape(
        (episode_ended.shape[0],) + (1,) * (reset_obs.ndim - 1)
    )
    return jnp.where(mask, reset_obs, initial_obs_pool)

def _sample_initial_obs(key: Key, initial_obs_pool: jax.Array) -> jax.Array:
    return jnp.take(initial_obs_pool, jax.random.permutation(key, initial_obs_pool.shape[0]), axis=0)


def _contains_humanoid_bench(*objects) -> bool:
    """Best-effort check for HumanoidBench runner/config context.

    Hydra does not pass the config name into this training helper directly, so
    check the runner callables/env objects and, when available, Hydra runtime
    metadata. This keeps ManiSkill unchanged while automatically grouping CPU
    HumanoidBench collection blocks.
    """
    token = "humanoid_bench"

    def object_strings(obj):
        if obj is None:
            return
        if isinstance(obj, (tuple, list)):
            for item in obj:
                yield from object_strings(item)
            return
        for attr in ("__module__", "__qualname__", "__name__"):
            value = getattr(obj, attr, None)
            if value is not None:
                yield str(value)
        func = getattr(obj, "func", None)
        if func is not None:
            yield from object_strings(func)
        yield type(obj).__module__
        yield type(obj).__name__
        yield repr(obj)

    for obj in objects:
        for value in object_strings(obj):
            if token in value.lower():
                return True

    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            if token in str(HydraConfig.get()).lower():
                return True
    except Exception:
        pass

    return False

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
    bc_indicator: bool = False,
    decay_rate: float = 1.0,
    filter_success: bool = True,
    wandb_run = None,
    cut_at_first_success: bool = True,
    critic_offline_warmup_iters: int = 0,
    data_type: str = "expert",
    max_buffer_size: int = 1_000_000,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    num_epochs: int = 4,
    learning_starts: int = 1,
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

    if data_type not in ("random", "PER"):
        raise ValueError(
            "Flashbax replay supports data_type='random' or data_type='PER'. "
            "The expert replay path is separate from this SAC-style buffer path."
        )
    if num_steps != 1:
        raise ValueError(
            "Flashbax item replay here is one-step only; set algorithm.num_steps=1."
        )

    if data_type == "random":
        buffer_fn = fbx.make_item_buffer(
            max_length=max_buffer_size,
            min_length=num_envs,
            sample_batch_size=num_envs,
            add_sequences=True,
            add_batches=True,
        )
    else:
        buffer_fn = fbx.make_prioritised_item_buffer(
            max_length=max_buffer_size,
            min_length=num_envs,
            sample_batch_size=num_envs,
            add_sequences=True,
            add_batches=True,
            priority_exponent=per_alpha,
            device="gpu",
        )

    # One collection step, then num_epochs replay updates against one fixed actor_target.
    def train_step_replay(carry: tuple, key: Key) -> tuple:
        state, buffer_state, initial_obs_pool = carry
        key, rollout_key, update_key = jax.random.split(key, 3)

        policy = policy_fn(state, False)
        rollout_transitions, state = rollout_fn(
            key=rollout_key, train_state=state, policy=policy
        )

        # In the auto-reset Gymnasium/HumanoidBench runners, `transition.next_obs`
        # is the terminal observation, whereas `state.last_obs` is the post-reset
        # observation carried into the next action selection. Hence, for every
        # completed episode, `state.last_obs[j]` is a genuine s_0 ~ d_0 sample.
        episode_ended = jnp.logical_or(rollout_transitions.done[-1].astype(bool), rollout_transitions.truncated[-1].astype(bool))
        initial_obs_pool = _update_initial_obs_pool(initial_obs_pool, state.last_obs, episode_ended)

        replay_transitions = Transition(
            obs=rollout_transitions.obs,
            next_obs=rollout_transitions.next_obs,
            action=rollout_transitions.action,
            reward=rollout_transitions.reward,
            done=rollout_transitions.done,
            truncated=rollout_transitions.truncated,
            extras={
                "behavior_log_prob": rollout_transitions.extras["behavior_log_prob"],
            },
        )
        buffer_state = buffer_fn.add(
            buffer_state,
            jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), replay_transitions),
        )

        def replay_epoch(carry, epoch_key):
            state, buffer_state = carry
            learn_key, sample_key, initial_key = jax.random.split(epoch_key, 3)
            # initial_obs is sampled only from the post-auto-reset state pool, so s₀ ∼ d₀.
            initial_obs = _sample_initial_obs(initial_key, initial_obs_pool)

            if data_type == "PER":
                def _sample_from_buffer(_):
                    sampled = buffer_fn.sample(buffer_state, sample_key)
                    num_valid = jnp.where(
                        buffer_state.is_full,
                        max_buffer_size,
                        buffer_state.current_index,
                    )
                    is_weight = (jnp.maximum(num_valid, 1) * jnp.maximum(sampled.probabilities, 1e-8)) ** (-per_beta)
                    is_weight = is_weight / jnp.maximum(is_weight.max(), 1e-8)
                    transitions = Transition(
                        obs=sampled.experience.obs[None],
                        next_obs=sampled.experience.next_obs[None],
                        action=sampled.experience.action[None],
                        reward=sampled.experience.reward[None],
                        done=sampled.experience.done[None],
                        truncated=sampled.experience.truncated[None],
                        extras={
                            "behavior_log_prob": sampled.experience.extras["behavior_log_prob"][None],
                            "initial_obs": initial_obs,
                            "is_weight": is_weight[None],
                        },
                    )
                    return transitions, sampled.indices, jnp.array(True)

                def _use_fresh_rollout(_):
                    transitions = replay_transitions.replace(
                        extras={
                            **replay_transitions.extras,
                            "initial_obs": initial_obs,
                            "is_weight": jnp.ones((1, num_envs), dtype=jnp.float32),
                        }
                    )
                    return (
                        transitions,
                        jnp.zeros((num_envs,), dtype=jnp.int32),
                        jnp.array(False),
                    )

                transitions, sampled_indices, used_replay = jax.lax.cond(
                    buffer_fn.can_sample(buffer_state),
                    _sample_from_buffer,
                    _use_fresh_rollout,
                    operand=None,
                )
            else:
                def _sample_from_buffer(_):
                    sampled = buffer_fn.sample(buffer_state, sample_key)
                    return Transition(
                        obs=sampled.experience.obs[None],
                        next_obs=sampled.experience.next_obs[None],
                        action=sampled.experience.action[None],
                        reward=sampled.experience.reward[None],
                        done=sampled.experience.done[None],
                        truncated=sampled.experience.truncated[None],
                        extras={
                            "behavior_log_prob": sampled.experience.extras["behavior_log_prob"][None],
                            "initial_obs": initial_obs,
                        },
                    )

                transitions = jax.lax.cond(
                    buffer_fn.can_sample(buffer_state),
                    _sample_from_buffer,
                    lambda _: replay_transitions.replace(extras={**replay_transitions.extras, "initial_obs": initial_obs}),
                    operand=None,
                )

            state, update_metrics, per_env_td_error = learner_fn(
                key=learn_key, train_state=state, batch=transitions
            )

            if data_type == "PER":
                buffer_state = jax.lax.cond(
                    used_replay,
                    lambda b: buffer_fn.set_priorities(
                        b,
                        sampled_indices,
                        jnp.abs(per_env_td_error).reshape(-1) + 1e-6,
                    ),
                    lambda b: b,
                    buffer_state,
                )

            return (state, buffer_state), update_metrics

        (state, buffer_state), epoch_metrics = jax.lax.scan(
            replay_epoch,
            (state, buffer_state),
            jax.random.split(update_key, num_epochs),
        )
        update_metrics = jax.tree.map(lambda x: x[-1], epoch_metrics)
        state = state.replace(iteration=state.iteration + 1)

        namespaced = {k: v for k, v in update_metrics.items() if "/" in k}
        plain = {k: v for k, v in update_metrics.items() if "/" not in k}
        jax.debug.callback(
            log_callback, state, {**utils.prefix_dict("train", plain), **namespaced}
        )
        return (state, buffer_state, initial_obs_pool), update_metrics

    # Eval step: runs eval_interval train steps then evaluates
    def train_eval_step_replay(key, train_state, scan_buf):
        buffer_state, initial_obs_pool = scan_buf
        train_key, eval_key = jax.random.split(key)
        (train_state, buffer_state, initial_obs_pool), stacked_metrics = jax.lax.scan(
            f=train_step_replay,
            init=(train_state, buffer_state, initial_obs_pool),
            xs=jax.random.split(train_key, eval_interval),
        )
        scan_buf = (buffer_state, initial_obs_pool)
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

        # One rollout supplies the Flashbax template only. The first actual
        # scan update falls back to fresh data until the buffer can sample.
        state0 = jax.tree.map(lambda x: x[0], train_state)
        warmup_policy = policy_fn(state0, False)
        rt_template, _ = rollout_fn(
            key=warmup_key, train_state=state0, policy=warmup_policy
        )
        rt_template = Transition(
            obs=rt_template.obs,
            next_obs=rt_template.next_obs,
            action=rt_template.action,
            reward=rt_template.reward,
            done=rt_template.done,
            truncated=rt_template.truncated,
            extras={
                "behavior_log_prob": rt_template.extras["behavior_log_prob"],
            },
        )
        seed_buf = buffer_fn.init(jax.tree.map(lambda x: x[0, 0], rt_template))
        scan_buf_init = (
            jax.tree.map(lambda x: jnp.broadcast_to(x, (num_seeds,) + x.shape), seed_buf),
            train_state.last_obs,
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
    bc_indicator: bool = False,
    decay_rate: float = 1.0,
    filter_success: bool = True,
    wandb_run = None,
    cut_at_first_success: bool = True,
    critic_offline_warmup_iters: int = 0,
    data_type: str = "expert",
    max_buffer_size: int = 1_000_000,
    per_alpha: float = 0.6,
    per_beta: float = 0.4,
    num_epochs: int = 4,
    learning_starts: int = 1,
    num_collection_blocks: int = 1,
):
    from src.runners.gymnasium_runner import (
        make_eval_fn as make_gymnasium_eval_fn,
        make_rollout_fn as make_gymnasium_rollout_fn,
    )

    train_log_interval = 1000 # 125 for maniskill # max(
    #     1, int((total_time_steps / (num_steps * num_envs)) // num_eval) // 4
    # )

    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    if rollout_fn is None:
        rollout_fn = make_gymnasium_rollout_fn(env, num_steps, num_envs)

    if eval_fn is None:
        eval_fn = make_gymnasium_eval_fn(eval_env, max_episode_steps)

    if _contains_humanoid_bench(env, eval_env, rollout_fn, eval_fn):
        if num_collection_blocks != 8:
            logging.info(
                "HumanoidBench runner/config detected; overriding num_collection_blocks from %d to 8.",
                num_collection_blocks,
            )
        num_collection_blocks = 8

    if data_type not in ("random", "PER"):
        raise ValueError(
            "Flashbax replay supports data_type='random' or data_type='PER'. "
            "The expert replay path is separate from this SAC-style buffer path."
        )
    if num_steps != 1:
        raise ValueError(
            "Flashbax item replay here is one-step only; set algorithm.num_steps=1."
        )

    if data_type == "random":
        buffer_fn = fbx.make_item_buffer(
            max_length=max_buffer_size,
            min_length=num_envs*num_collection_blocks,
            sample_batch_size=num_envs*num_collection_blocks,
            add_sequences=True,
            add_batches=True,
        )
    else:
        buffer_fn = fbx.make_prioritised_item_buffer(
            max_length=max_buffer_size,
            min_length=num_envs*num_collection_blocks,
            sample_batch_size=num_envs*num_collection_blocks,
            add_sequences=True,
            add_batches=True,
            priority_exponent=per_alpha,
            device="gpu",
        )

    # The Python ManiSkill loop calls these repeatedly. Compile them once and donate the old state so Flashbax can reuse its device allocation.
    buffer_add = jax.jit(buffer_fn.add, donate_argnums=(0,))
    buffer_sample = jax.jit(buffer_fn.sample)
    if data_type == "PER":
        buffer_set_priorities = jax.jit(buffer_fn.set_priorities, donate_argnums=(0,))

    def loop_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Initialize the policy, environment and map that across the number of random seeds
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_eval
        train_steps_per_iteration = max(
            1, num_train_steps // (num_iterations * num_collection_blocks)
        )
        key, init_key = jax.random.split(key)
        state = init_fn(init_key)
        obs, _ = env.reset()
        state = state.replace(last_obs=to_jax(obs), last_env_state=None)
        initial_obs_pool = to_jax(obs)
        logging.info(f"Starting training for {num_iterations} iterations.")
        logging.info(f"Collection blocks per update: {num_collection_blocks}.")
        logging.info(f"Train update blocks per iteration: {train_steps_per_iteration}.")
        logging.info(f"Env transitions per update block: {num_collection_blocks * num_steps * num_envs}.")
        logging.info(f"Total time steps: {total_time_steps}.")

        buffer_memory_gb = 0.0
        buffer_state = None

        step = 0
        online_transitions_used = 0
        offline_transitions_used = 0
        prefill_stds = [] # [0.6, 0.8, 1.0, 1.2]

        for _ in range(learning_starts):
            key, rollout_key = jax.random.split(key)
            policy = policy_fn(state, False)
            rollout_transitions, state = rollout_fn(
                key=rollout_key, train_state=state, policy=policy
            )
            episode_ended = jnp.logical_or(rollout_transitions.done[-1].astype(bool), rollout_transitions.truncated[-1].astype(bool))
            initial_obs_pool = _update_initial_obs_pool(initial_obs_pool, state.last_obs, episode_ended)
            replay_transitions = Transition(
                obs=rollout_transitions.obs,
                next_obs=rollout_transitions.next_obs,
                action=rollout_transitions.action,
                reward=rollout_transitions.reward,
                done=rollout_transitions.done,
                truncated=rollout_transitions.truncated,
                extras={
                    "behavior_log_prob": rollout_transitions.extras["behavior_log_prob"],
                },
            )
            if buffer_state is None:
                buffer_state = buffer_fn.init(jax.tree.map(lambda x: x[0, 0], replay_transitions))
            
            buffer_state = buffer_add(buffer_state, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), replay_transitions))

        for iter_idx in range(num_iterations):
            for _ in range(train_steps_per_iteration):
                for _ in range(num_collection_blocks):
                    key, rollout_key = jax.random.split(key)
                    policy = policy_fn(state, False)
                    rollout_transitions, state = rollout_fn(
                        key=rollout_key, train_state=state, policy=policy
                    )
                    episode_ended = jnp.logical_or(rollout_transitions.done[-1].astype(bool), rollout_transitions.truncated[-1].astype(bool))
                    initial_obs_pool = _update_initial_obs_pool(initial_obs_pool, state.last_obs, episode_ended)

                    replay_transitions = Transition(
                        obs=rollout_transitions.obs,
                        next_obs=rollout_transitions.next_obs,
                        action=rollout_transitions.action,
                        reward=rollout_transitions.reward,
                        done=rollout_transitions.done,
                        truncated=rollout_transitions.truncated,
                        extras={
                            "behavior_log_prob": rollout_transitions.extras[
                                "behavior_log_prob"
                            ],
                        },
                    )
                    if buffer_state is None:
                        buffer_state = buffer_fn.init(
                            jax.tree.map(lambda x: x[0, 0], replay_transitions)
                        )

                    buffer_state = buffer_add(buffer_state, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), replay_transitions))

                for _ in range(num_epochs):
                    key, learn_key, sample_key, initial_key = jax.random.split(key, 4)
                    # initial_obs is sampled only from the post-auto-reset state pool, so s₀ ∼ d₀.
                    initial_obs = _sample_initial_obs(initial_key, initial_obs_pool)

                    if bool(buffer_fn.can_sample(buffer_state)):
                        sampled = buffer_sample(buffer_state, sample_key)
                        transitions = Transition(
                            obs=sampled.experience.obs[None],
                            next_obs=sampled.experience.next_obs[None],
                            action=sampled.experience.action[None],
                            reward=sampled.experience.reward[None],
                            done=sampled.experience.done[None],
                            truncated=sampled.experience.truncated[None],
                            extras={
                                "behavior_log_prob": sampled.experience.extras["behavior_log_prob"][None],
                                "initial_obs": initial_obs,
                            },
                        )
                        used_replay = True
                        if data_type == "PER":
                            num_valid = (
                                max_buffer_size
                                if bool(buffer_state.is_full)
                                else int(buffer_state.current_index)
                            )
                            is_weight = ( max(num_valid, 1) * jnp.maximum(sampled.probabilities, 1e-8)) ** (-per_beta)
                            is_weight = is_weight / jnp.maximum(is_weight.max(), 1e-8)
                            transitions = transitions.replace(
                                extras={
                                    **transitions.extras,
                                    "is_weight": is_weight[None],
                                }
                            )
                            sampled_indices = sampled.indices
                    else:
                        transitions = replay_transitions.replace(extras={**replay_transitions.extras, "initial_obs": initial_obs})
                        used_replay = False

                    # One sampled N-transition batch is handled by learner_fn: target construction, shuffle, critic/actor minibatches and target-critic Polyak updates.
                    state, train_metrics, per_env_td_error = learner_fn(
                        key=learn_key, train_state=state, batch=transitions
                    )

                    if data_type == "PER" and used_replay:
                        buffer_state = buffer_set_priorities(
                            buffer_state,
                            sampled_indices,
                            jnp.abs(per_env_td_error).reshape(-1) + 1e-6,
                        )

                # `iteration` counts collection/update blocks, matching the scan path.
                state = state.replace(iteration=state.iteration + 1)

                # This Flashbax path trains on online environment data stored
                # in replay. Count newly collected environment transitions, not
                # replay samples reused by the learner.
                online_transitions_used += num_collection_blocks * num_steps * num_envs
                offline_transitions_used += 0
                num_samples = online_transitions_used + offline_transitions_used

                if step % train_log_interval == 0:
                    log_metrics = train_metrics
                    namespaced = {k: v for k, v in log_metrics.items() if "/" in k}
                    plain = {k: v for k, v in log_metrics.items() if "/" not in k}
                    log_callback(state, {**utils.prefix_dict("train", plain), **namespaced})

                step += 1

            policy = policy_fn(state, not stochastic_eval)
            key, eval_key = jax.random.split(key)

            eval_metrics = eval_fn(eval_key, policy)

            eval_return = float(eval_metrics["episode_return"])
            grad_updates = float(train_metrics.get("sys/grad_updates", 1.0))

            eval_metrics["num_samples"] = num_samples

            buffer_memory_gb = sum(
                leaf.nbytes for leaf in jax.tree.leaves(buffer_state) if hasattr(leaf, "nbytes")
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