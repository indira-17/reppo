from gymnax.environments.environment import Environment
import jax
import jax.numpy as jnp

from src.common import EvalFn, Key, Policy, RolloutFn, TrainState, Transition


def _merge_extras(info: dict, policy_extras: dict | None) -> dict:
    extras = dict(info) if info is not None else {}
    extras.update(policy_extras or {})
    return extras


def make_eval_fn(
    env: Environment,
    max_episode_steps: int,
    demo_path: str | None = None,
    data_type: str = "online",
    bc_indicator: bool = False,
    filter_success: bool = True,
    cut_at_first_success: bool = True,
) -> EvalFn:
    def evaluation_fn(key: Key, policy: Policy):
        def step_env(carry, _):
            key, env_state, obs = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, _ = policy(act_key, obs)
            env_key = jax.random.split(env_key, env.num_envs)
            obs, env_state, reward, done, info = env.step(env_key, env_state, action)
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        init_key = jax.random.split(init_key, env.num_envs)
        obs, env_state = env.reset(init_key)
        _, infos = jax.lax.scan(
            f=step_env,
            init=(key, env_state, obs),
            xs=None,
            length=max_episode_steps,
        )

        returned = infos["returned_episode"]
        returns = infos["returned_episode_returns"]
        lengths = infos["returned_episode_lengths"]
        lin_vels = infos["returned_episode_tracking_lin_vel"]
        ang_vels = infos["returned_episode_tracking_ang_vel"]
        num_episodes = returned.sum()
        safe_den = jnp.maximum(num_episodes, 1)

        return {
            "episode_return": jnp.where(
                num_episodes > 0, (returns * returned).sum() / safe_den, 0.0
            ),
            "episode_return_std": returns.std(where=returned),
            "episode_length": jnp.where(
                num_episodes > 0, (lengths * returned).sum() / safe_den, 0.0
            ),
            "episode_length_std": lengths.std(where=returned),
            "num_episodes": num_episodes,
            "episode_tracking_lin_vel": jnp.where(
                num_episodes > 0, (lin_vels * returned).sum() / safe_den, 0.0
            ),
            "episode_tracking_ang_vel": jnp.where(
                num_episodes > 0, (ang_vels * returned).sum() / safe_den, 0.0
            ),
        }

    return evaluation_fn


def make_rollout_fn(
    env: Environment,
    num_steps: int,
    num_envs: int,
    demo_path: str | None = None,
    data_type: str = "online",
    bc_indicator: bool = False,
    filter_success: bool = True,
    cut_at_first_success: bool = True,
    decay_rate: float = 1.0,
) -> RolloutFn:
    def collect_rollout(
        key: Key, train_state: TrainState, policy: Policy
    ) -> tuple[Transition, TrainState]:
        def step_env(carry, _) -> tuple[tuple, Transition]:
            key, env_state, train_state, obs = carry
            key, act_key, step_key = jax.random.split(key, 3)
            action, policy_extras = policy(act_key, obs)
            step_key = jax.random.split(step_key, num_envs)
            next_obs, next_env_state, reward, done, info = env.step(
                step_key, env_state, action
            )

            transition = Transition(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=reward,
                done=done,
                truncated=next_env_state.truncated,
                extras=_merge_extras(info, policy_extras),
            )
            return (key, next_env_state, train_state, next_obs), transition

        rollout_state, transitions = jax.lax.scan(
            f=step_env,
            init=(
                key,
                train_state.last_env_state,
                train_state,
                train_state.last_obs,
            ),
            length=num_steps,
        )
        _, last_env_state, train_state, last_obs = rollout_state
        train_state = train_state.replace(
            last_env_state=last_env_state,
            last_obs=last_obs,
            time_steps=train_state.time_steps + num_steps * num_envs,
        )
        return transitions, train_state

    return collect_rollout