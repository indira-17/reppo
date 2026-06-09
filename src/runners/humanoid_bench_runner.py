import numpy as np
import jax
import jax.numpy as jnp
import torch
import gymnasium

from src.common import EvalFn, Key, Policy, RolloutFn, TrainState, Transition
from src.env_utils.torch_wrappers.maniskill_wrapper import to_jax

def make_rollout_fn(
    env: gymnasium.Env,
    num_steps: int,
    num_envs: int,
    demo_path: str = None,
    data_type: str = None,
    env_id: str = None,
    filter_success: bool = True,
    cut_at_first_success: bool = True,
) -> RolloutFn:
    def collect_rollout(key, train_state, policy):
        transitions = []
        obs = train_state.last_obs  # jax array (reset stored via to_jax)
        for _ in range(num_steps):
            key, act_key = jax.random.split(key)
            action, policy_extras = policy(act_key, obs)
            action_torch = torch.as_tensor(np.asarray(action), device=env.sim_device)

            next_obs, reward, done, info = env.step(action_torch)   # 4-tuple
            truncated = info["time_outs"]
            # wrapper already put terminal obs into this for truncated envs:
            bootstrap_next_obs = to_jax(info["observations"]["raw"]["obs"])

            reward_j = to_jax(reward)
            transitions.append(Transition(
                obs=obs,
                next_obs=bootstrap_next_obs,
                action=action,
                reward=reward_j,
                done=to_jax(done),
                truncated=to_jax(truncated),
                extras={"behavior_log_prob":
                        policy_extras.get("behavior_log_prob", jnp.zeros_like(reward_j))},
            ))
            obs = to_jax(next_obs)   # carry the auto-reset obs

        transitions = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
        train_state = train_state.replace(
            last_obs=obs,
            time_steps=train_state.time_steps + num_steps * num_envs,
        )
        return transitions, train_state

    return collect_rollout


def make_eval_fn(
    env: gymnasium.Env,
    max_episode_steps: int,
    demo_path: str = None,
    data_type: str = None,
    env_id: str = None,
    filter_success: bool = True,
    cut_at_first_success: bool = True,
) -> EvalFn:
    def evaluate(key, policy):
        obs, _ = to_jax(env.reset())                 # no args; single return
        n = env.num_envs
        ep_return = np.zeros(n, dtype=np.float64)
        done_mask = np.zeros(n, dtype=bool)
        for _ in range(max_episode_steps):
            key, act_key = jax.random.split(key)
            action, _ = policy(act_key, obs)
            action_torch = torch.as_tensor(np.asarray(action), device=env.sim_device)
            next_obs, reward, done, info = env.step(action_torch)
            r = np.asarray(to_jax(reward)); d = np.asarray(to_jax(done))
            ep_return += r * (~done_mask)          # count only until first done
            done_mask |= d
            obs = to_jax(next_obs)
            if done_mask.all():
                break

        return {
            "episode_return": float(ep_return.mean()),
            "episode_return_std": float(ep_return.std()),
        }

    return evaluate