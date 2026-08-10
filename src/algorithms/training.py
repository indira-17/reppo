import logging
import torch
import gymnasium
import numpy as np
import wandb
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

def _pearson_correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end

    return ranks

def _spearman_correlation(x, y):
    if len(x) < 2:
        return None
    return _pearson_correlation(_rankdata(x), _rankdata(y))

def _update_geometry_history_and_log(geometry_history, wandb_run, state, train_metrics, eval_return):
    feature_correlation = float(
        np.asarray(
            jax.device_get(train_metrics["feature_correlation"]),
            dtype=np.float64,
        ).mean()
    )
    Aphi_min_real_eigenval = float(
        np.asarray(
            jax.device_get(train_metrics["Aphi_min_real_eigenval"]),
            dtype=np.float64,
        ).mean()
    )
    gram_min_eigenval = float(
        np.asarray(
            jax.device_get(train_metrics["gram_min_eigenval"]),
            dtype=np.float64,
        ).mean()
    )
    gram_max_eigenval = float(
        np.asarray(
            jax.device_get(train_metrics["gram_max_eigenval"]),
            dtype=np.float64,
        ).mean()
    )
    invariance_loss = float(
        np.asarray(
            jax.device_get(train_metrics["invariance_loss"]),
            dtype=np.float64,
        ).mean()
    )

    geometry_history["feature_correlation"].append(feature_correlation)
    geometry_history["Aphi_min_real_eigenval"].append(Aphi_min_real_eigenval)
    geometry_history["gram_min_eigenval"].append(gram_min_eigenval)
    geometry_history["gram_max_eigenval"].append(gram_max_eigenval)
    geometry_history["invariance_loss"].append(invariance_loss)
    geometry_history["eval_return"].append(eval_return)

    if wandb_run is None:
        return

    feature_table = wandb.Table(
        data=list(zip(geometry_history["feature_correlation"], geometry_history["eval_return"])),
        columns=["Feature correlation", "Evaluation return"],
    )
    eigenvalue_table = wandb.Table(
        data=list(zip(geometry_history["Aphi_min_real_eigenval"], geometry_history["eval_return"])),
        columns=["Minimum real eigenvalue", "Evaluation return"],
    )
    gram_min_table = wandb.Table(
        data=list(zip(geometry_history["gram_min_eigenval"], geometry_history["eval_return"])),
        columns=["Minimum Gram eigenvalue", "Evaluation return"],
    )
    gram_max_table = wandb.Table(
        data=list(zip(geometry_history["gram_max_eigenval"], geometry_history["eval_return"])),
        columns=["Maximum Gram eigenvalue", "Evaluation return"],
    )
    invariance_table = wandb.Table(
        data=list(zip(geometry_history["invariance_loss"], geometry_history["eval_return"])),
        columns=["Invariance loss", "Evaluation return"],
    )

    geometry_logs = {
        "geometry/feature_correlation_vs_eval_return": wandb.plot.scatter(
            feature_table,
            "Feature correlation",
            "Evaluation return",
            title="Feature correlation vs evaluation return",
        ),
        "geometry/min_real_eigenvalue_vs_eval_return": wandb.plot.scatter(
            eigenvalue_table,
            "Minimum real eigenvalue",
            "Evaluation return",
            title="Minimum real eigenvalue vs evaluation return",
        ),
        "geometry/gram_min_eigenvalue_vs_eval_return": wandb.plot.scatter(
            gram_min_table,
            "Minimum Gram eigenvalue",
            "Evaluation return",
            title="Minimum Gram eigenvalue vs evaluation return",
        ),
        "geometry/gram_max_eigenvalue_vs_eval_return": wandb.plot.scatter(
            gram_max_table,
            "Maximum Gram eigenvalue",
            "Evaluation return",
            title="Maximum Gram eigenvalue vs evaluation return",
        ),
        "geometry/invariance_loss_vs_eval_return": wandb.plot.scatter(
            invariance_table,
            "Invariance loss",
            "Evaluation return",
            title="Invariance loss vs evaluation return",
        ),
    }

    for metric_name in (
        "feature_correlation",
        "Aphi_min_real_eigenval",
        "gram_min_eigenval",
        "gram_max_eigenval",
        "invariance_loss",
    ):
        pearson = _pearson_correlation(geometry_history[metric_name], geometry_history["eval_return"])
        spearman = _spearman_correlation(geometry_history[metric_name], geometry_history["eval_return"])
        if pearson is not None:
            geometry_logs[f"geometry_correlation/{metric_name}_pearson"] = pearson
        if spearman is not None:
            geometry_logs[f"geometry_correlation/{metric_name}_spearman"] = spearman

    # Leave the step open so the existing log_callback commits the
    # evaluation metrics at the same state.time_steps value.
    wandb_run.log(geometry_logs, step=int(np.asarray(jax.device_get(state.time_steps)).reshape(-1)[0]), commit=False)

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
    mask = episode_ended.astype(bool).reshape((episode_ended.shape[0],) + (1,) * (reset_obs.ndim - 1))
    return jnp.where(mask, reset_obs, initial_obs_pool)

def _sample_initial_obs(key: Key, initial_obs_pool: jax.Array, sample_size: int) -> jax.Array:
    indices = jax.random.randint(key, shape=(sample_size,), minval=0, maxval=initial_obs_pool.shape[0])
    return jnp.take(initial_obs_pool, indices, axis=0)

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
    env: Environment | tuple[Environment, Environment],
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
    data_type: str,
    max_buffer_size: int,
    replay_batch_size: int,
    per_alpha: float,
    per_beta: float,
    num_epochs: int,
    prefill_buffer: int,
    num_collection_blocks: int,
    num_replay_updates: int,
    rollout_fn: RolloutFn | None = None,
    eval_fn: EvalFn | None = None,
    log_callback: LogCallback | None = None,
) -> TrainFn:
    from src.runners.gymnax_runner import (
        make_eval_fn as make_gymnax_eval_fn,
        make_rollout_fn as make_gymnax_rollout_fn,
    )

    # Keep the online runner schedule and logging unchanged.
    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    eval_interval = int((total_time_steps / (num_steps * num_envs)) // num_eval)

    if eval_fn is None:
        eval_fn = make_gymnax_eval_fn(eval_env, max_episode_steps)

    if rollout_fn is None:
        rollout_fn = make_gymnax_rollout_fn(
            env, num_steps=num_steps, num_envs=num_envs
        )

    if log_callback is None:
        log_callback = lambda state, metrics: None

    if num_epochs < 1:
        raise ValueError("num_epochs must be >= 1.")
    if prefill_buffer < 0:
        raise ValueError("prefill_buffer must be >= 0.")
    if num_collection_blocks < 1:
        raise ValueError("num_collection_blocks must be >= 1.")
    if num_replay_updates < 1:
        raise ValueError("num_replay_updates must be >= 1.")
    if data_type not in ("random", "PER"):
        raise ValueError("data_type must be 'random' or 'PER'.")
    if replay_batch_size > max_buffer_size:
        raise ValueError("replay_batch_size must be <= max_buffer_size.")

    # One Flashbax item is one transition. Sampling starts after one configured
    # rollout has been inserted. Item-buffer sampling is with replacement.
    min_buffer_length = num_steps * num_envs
    if max_buffer_size < min_buffer_length:
        raise ValueError(
            "max_buffer_size must hold at least one collected rollout."
        )

    if data_type == "random":
        buffer_fn = fbx.make_item_buffer(
            max_length=max_buffer_size,
            min_length=min_buffer_length,
            sample_batch_size=replay_batch_size,
            add_sequences=False,
            add_batches=True,
        )
    else:
        buffer_fn = fbx.make_prioritised_item_buffer(
            max_length=max_buffer_size,
            min_length=min_buffer_length,
            sample_batch_size=replay_batch_size,
            add_sequences=False,
            add_batches=True,
            priority_exponent=per_alpha,
        )

    def _buffer_transition(transitions: Transition) -> Transition:
        return Transition(
            obs=transitions.obs,
            next_obs=transitions.next_obs,
            action=transitions.action,
            reward=transitions.reward,
            done=transitions.done,
            truncated=transitions.truncated,
            extras={
                "behavior_log_prob": transitions.extras["behavior_log_prob"],
            },
        )

    def _add_to_buffer(buffer_state, transitions: Transition):
        transitions = _buffer_transition(transitions)
        flat_transitions = jax.tree.map(
            lambda x: x.reshape((num_steps * num_envs, *x.shape[2:])),
            transitions,
        )
        return buffer_fn.add(buffer_state, flat_transitions)

    def _sample_from_buffer(buffer_state, sample_key: Key) -> tuple[Transition, object]:
        sampled = buffer_fn.sample(buffer_state, sample_key)
        transitions = jax.tree.map(
            lambda x: x.reshape((1, replay_batch_size, *x.shape[1:])),
            sampled.experience,
        )

        if data_type == "PER":
            num_valid = jnp.where(
                buffer_state.is_full,
                max_buffer_size,
                buffer_state.current_index,
            )
            is_weight = (
                jnp.maximum(num_valid, 1)
                * jnp.maximum(sampled.probabilities, 1e-8)
            ) ** (-per_beta)
            is_weight = is_weight / jnp.maximum(is_weight.max(), 1e-8)
            transitions = transitions.replace(
                extras={
                    **transitions.extras,
                    "is_weight": is_weight.reshape((1, replay_batch_size)),
                }
            )

        return transitions, sampled

    def _collect_and_add(state, buffer_state, rollout_key):
        policy = policy_fn(state, False)
        transitions, state = rollout_fn(
            key=rollout_key, train_state=state, policy=policy
        )

        # Exactly the online SR-DICE d0-pool update.
        episode_ended = jnp.logical_or(
            transitions.done[-1].astype(bool),
            transitions.truncated[-1].astype(bool),
        )
        initial_obs_pool = state.params["sr_dice_initial_obs"]
        initial_obs_mask = episode_ended.reshape(
            (episode_ended.shape[0],)
            + (1,) * (state.last_obs.ndim - 1)
        )
        state = state.replace(
            params={
                **state.params,
                "sr_dice_initial_obs": jnp.where(
                    initial_obs_mask,
                    state.last_obs,
                    initial_obs_pool,
                ),
            }
        )

        buffer_state = _add_to_buffer(buffer_state, transitions)
        return state, buffer_state

    def train_step(
        carry: tuple[TrainState, object], key: Key
    ) -> tuple[tuple[TrainState, object], dict[str, jax.Array]]:
        state, buffer_state = carry
        key, collection_key, replay_key = jax.random.split(key, 3)

        # The YAML controls how many fresh rollout blocks precede replay.
        def collection_step(carry, rollout_key):
            state, buffer_state = carry
            state, buffer_state = _collect_and_add(
                state, buffer_state, rollout_key
            )
            return (state, buffer_state), None

        (state, buffer_state), _ = jax.lax.scan(
            collection_step,
            (state, buffer_state),
            jax.random.split(collection_key, num_collection_blocks),
        )

        # Replace online reuse by fresh replay draws. actor_target refresh stays
        # exactly where it was in the off-policy pipeline: before each learner call.
        def replay_update(carry, update_key):
            state, buffer_state = carry
            sample_key, learn_key = jax.random.split(update_key)

            replay_batch, sampled = _sample_from_buffer(
                buffer_state, sample_key
            )

            state = state.replace(
                actor_target=state.actor_target.replace(
                    params=state.actor.params
                )
            )

            state, update_metrics, per_env_td_error = learner_fn(
                key=learn_key,
                train_state=state,
                batch=replay_batch,
            )

            if data_type == "PER":
                buffer_state = buffer_fn.set_priorities(
                    buffer_state,
                    sampled.indices,
                    jnp.abs(per_env_td_error).reshape(-1) + 1e-6,
                )

            return (state, buffer_state), update_metrics

        (state, buffer_state), update_metrics = jax.lax.scan(
            replay_update,
            (state, buffer_state),
            jax.random.split(replay_key, num_replay_updates),
        )

        # Match online train_step: expose the last learner-call metrics and
        # increment iteration once per environment collection/update step.
        update_metrics = jax.tree.map(lambda x: x[-1], update_metrics)
        state = state.replace(iteration=state.iteration + 1)
        return (state, buffer_state), update_metrics

    def train_eval_step(key, train_state, buffer_state):
        train_key, eval_key = jax.random.split(key)
        (train_state, buffer_state), train_metrics = jax.lax.scan(
            f=train_step,
            init=(train_state, buffer_state),
            xs=jax.random.split(train_key, eval_interval),
        )
        train_metrics = jax.tree.map(lambda x: x[-1], train_metrics)

        policy = policy_fn(train_state, not stochastic_eval)
        eval_metrics = eval_fn(eval_key, policy)
        metrics = {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }
        return train_state, buffer_state, metrics

    def train_eval_loop_body(carry, key):
        train_state, buffer_state = carry
        key, subkey = jax.random.split(key)
        train_state, buffer_state, metrics = jax.vmap(train_eval_step)(
            jax.random.split(subkey, num_seeds),
            train_state,
            buffer_state,
        )
        jax.debug.callback(log_callback, train_state, metrics)
        return (train_state, buffer_state), metrics

    def init_train_state(key: Key) -> TrainState:
        # Keep online initialization exactly, including SR-DICE initial states.
        key, env_key = jax.random.split(key)
        train_state = init_fn(key)
        env_key = jax.random.split(env_key, num_envs)
        obs, env_state = env.reset(env_key)
        train_state = train_state.replace(
            last_obs=obs,
            last_env_state=env_state,
            params={
                **train_state.params,
                "sr_dice_initial_obs": obs,
            },
        )
        return train_state

    def scan_train_fn(key: Key) -> tuple[TrainState, dict]:
        # Keep the online evaluation/collection scheduling exactly.
        num_train_steps = total_time_steps // (num_steps * num_envs)
        num_iterations = num_train_steps // eval_interval + int(
            num_train_steps % eval_interval != 0
        )

        key, init_key = jax.random.split(key)
        train_state = jax.vmap(init_train_state)(
            jax.random.split(init_key, num_seeds)
        )

        # Infer replay item shapes without executing an uncounted rollout.
        state0 = jax.tree.map(lambda x: x[0], train_state)
        template_key = jax.random.fold_in(init_key, 1)

        def rollout_for_shape(shape_key, shape_state):
            return rollout_fn(
                key=shape_key,
                train_state=shape_state,
                policy=policy_fn(shape_state, False),
            )

        rollout_shape, _ = jax.eval_shape(
            rollout_for_shape,
            template_key,
            state0,
        )
        rollout_template = jax.tree.map(
            lambda x: jnp.zeros(x.shape, x.dtype),
            rollout_shape,
        )
        replay_template = _buffer_transition(rollout_template)
        item_template = jax.tree.map(
            lambda x: x[0, 0],
            replay_template,
        )
        seed_buffer_state = buffer_fn.init(item_template)
        buffer_state = jax.tree.map(
            lambda x: jnp.broadcast_to(x, (num_seeds,) + x.shape),
            seed_buffer_state,
        )

        # Optional prefill is controlled only by YAML. For the paper baseline
        # prefill_buffer=0, so this executes zero environment interactions.
        if prefill_buffer > 0:
            prefill_seed_keys = jax.random.split(jax.random.fold_in(init_key, 2), num_seeds)

            def prefill_seed(state, seed_buffer, seed_key):
                def prefill_step(carry, _):
                    state, seed_buffer, seed_key = carry
                    seed_key, rollout_key = jax.random.split(seed_key)
                    state, seed_buffer = _collect_and_add(
                        state, seed_buffer, rollout_key
                    )
                    return (state, seed_buffer, seed_key), None

                (state, seed_buffer, _), _ = jax.lax.scan(
                    prefill_step,
                    (state, seed_buffer, seed_key),
                    xs=None,
                    length=prefill_buffer,
                )
                return state, seed_buffer

            train_state, buffer_state = jax.vmap(prefill_seed)(
                train_state,
                buffer_state,
                prefill_seed_keys,
            )

        keys = jax.random.split(key, num_iterations)
        (state, _), metrics = jax.lax.scan(
            f=train_eval_loop_body,
            init=(train_state, buffer_state),
            xs=keys,
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
    data_type: str,
    max_buffer_size: int,
    replay_batch_size: int,
    per_alpha: float,
    per_beta: float,
    num_epochs: int,
    prefill_buffer: int,
    num_collection_blocks: int,
    num_replay_updates: int,
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
):
    from src.runners.gymnasium_runner import (
        make_eval_fn as make_gymnasium_eval_fn,
        make_rollout_fn as make_gymnasium_rollout_fn,
    )

    train_log_interval = max(
        1, int((total_time_steps / (num_steps * num_envs)) // num_eval) // 4
    )

    if isinstance(env, tuple):
        env, eval_env = env
    else:
        eval_env = env

    if rollout_fn is None:
        rollout_fn = make_gymnasium_rollout_fn(env, num_steps, num_envs)

    if eval_fn is None:
        eval_fn = make_gymnasium_eval_fn(eval_env, max_episode_steps)

    if data_type not in ("random", "PER"):
        raise ValueError(
            "Flashbax replay supports data_type='random' or data_type='PER'. "
            "The expert replay path is separate from this SAC-style buffer path."
        )

    # Each buffer item is one transition; sample a full IID TD(0) training batch.
    if data_type == "random":
        buffer_fn = fbx.make_item_buffer(
            max_length=max_buffer_size,
            min_length=replay_batch_size,
            sample_batch_size=replay_batch_size,
            add_sequences=False,
            add_batches=True,
        )
    else:
        buffer_fn = fbx.make_prioritised_item_buffer(
            max_length=max_buffer_size,
            min_length=replay_batch_size,
            sample_batch_size=replay_batch_size,
            add_sequences=False,
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
        if data_type == "random":
            logging.info("Replay sampling: uniform IID from Flashbax replay buffer.")

        buffer_memory_gb = 0.0
        buffer_state = None

        step = 0
        online_transitions_used = 0
        offline_transitions_used = 0
        prefill_stds = [] # [0.6, 0.8, 1.0, 1.2]
        geometry_history = {
            "feature_correlation": [],
            "Aphi_min_real_eigenval": [],
            "gram_min_eigenval": [],
            "gram_max_eigenval": [],
            "invariance_loss": [],
            "eval_return": [],
        }

        for _ in range(prefill_buffer):
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
            
            flat_replay_transitions = jax.tree.map(
                lambda x: x.reshape((num_steps * num_envs, *x.shape[2:])),
                replay_transitions,
            )
            buffer_state = buffer_add(buffer_state, flat_replay_transitions)

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

                    flat_replay_transitions = jax.tree.map(
                        lambda x: x.reshape((num_steps * num_envs, *x.shape[2:])),
                        replay_transitions,
                    )
                    buffer_state = buffer_add(buffer_state, flat_replay_transitions)

                # Resample a fresh replay batch for every learner update.
                for _ in range(num_replay_updates):
                    key, learn_key, sample_key = jax.random.split(key, 3)

                    if bool(buffer_fn.can_sample(buffer_state)):
                        if data_type == "PER":
                            sampled = buffer_sample(buffer_state, sample_key)
                            sampled_experience = sampled.experience
                            sampled_indices = sampled.indices
                        else:
                            sampled = buffer_sample(buffer_state, sample_key)
                            sampled_experience = sampled.experience

                        transitions = jax.tree.map(
                            lambda x: x.reshape((1, replay_batch_size, *x.shape[1:])),
                            sampled_experience,
                        )
                        used_replay = True
                        if data_type == "PER":
                            num_valid = max_buffer_size if bool(buffer_state.is_full) else int(buffer_state.current_index)
                            is_weight = (max(num_valid, 1) * jnp.maximum(sampled.probabilities, 1e-8)) ** (-per_beta)
                            is_weight = is_weight / jnp.maximum(is_weight.max(), 1e-8)
                            transitions = transitions.replace(
                                extras={
                                    **transitions.extras,
                                    "is_weight": is_weight.reshape((1, replay_batch_size)),
                                }
                            )
                    else:
                        transitions = replay_transitions.replace(extras={**replay_transitions.extras})
                        used_replay = False

                    # The actor_target is updated to the current actor params before each learner_fn call.
                    state = state.replace(
                        actor_target=state.actor_target.replace(params=state.actor.params)
                    )

                    state, train_metrics, per_env_td_error = learner_fn(
                        key=learn_key, train_state=state, batch=transitions
                    )

                    if data_type == "PER" and used_replay:
                        buffer_state = buffer_set_priorities(
                            buffer_state,
                            sampled_indices,
                            jnp.abs(per_env_td_error).reshape(-1) + 1e-6,
                        )

                    state = state.replace(iteration=state.iteration + 1)

                    if step % train_log_interval == 0:
                        log_metrics = train_metrics
                        namespaced = {k: v for k, v in log_metrics.items() if "/" in k}
                        plain = {k: v for k, v in log_metrics.items() if "/" not in k}
                        log_callback(state, {**utils.prefix_dict("train", plain), **namespaced})

                    step += 1

                online_transitions_used += num_collection_blocks * num_steps * num_envs
                offline_transitions_used += 0
                num_samples = online_transitions_used + offline_transitions_used

            policy = policy_fn(state, not stochastic_eval)
            key, eval_key = jax.random.split(key)

            eval_metrics = eval_fn(eval_key, policy)

            eval_return = float(eval_metrics["episode_return"])
            grad_updates = float(train_metrics.get("sys/grad_updates", 1.0))

            _update_geometry_history_and_log(geometry_history, wandb_run, state, train_metrics, eval_return)

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
                "sys/perf_per_offline_transition": eval_return / max(offline_transitions_used, 1),
                "sys/perf_per_online_transition": eval_return / max(online_transitions_used, 1),
            })
            log_callback(state, log_metrics_eval)

        return state, {
            **utils.prefix_dict("train", train_metrics),
            **utils.prefix_dict("eval", eval_metrics),
        }

    return loop_train_fn