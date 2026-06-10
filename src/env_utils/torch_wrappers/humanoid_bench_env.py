from __future__ import annotations
import humanoid_bench
import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit
from loguru import logger as log
from stable_baselines3.common.vec_env import SubprocVecEnv

# Disable all logging below CRITICAL level
log.remove()
log.add(lambda msg: False, level="CRITICAL")


def make_env(env_name, rank, render_mode=None, seed=0):
    """
    Utility function for multiprocessed env.

    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    """

    if env_name in [
        "h1hand-push-v0",
        "h1-push-v0",
        "h1hand-cube-v0",
        "h1cube-v0",
        "h1hand-basketball-v0",
        "h1-basketball-v0",
        "h1hand-kitchen-v0",
        "h1-kitchen-v0",
    ]:
        max_episode_steps = 500
    else:
        max_episode_steps = 1000

    def _init():
        env = gym.make(env_name, render_mode=render_mode)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env.unwrapped.seed(seed + rank)

        return env

    return _init


class HumanoidBenchEnv:
    """Wraps HumanoidBench environment to support parallel environments."""

    def __init__(self, env_name, num_envs=1, render_mode=None, device=None):
        # NOTE: HumanoidBench action space is already normalized to [-1, 1]
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sim_device = device
        self.num_envs = num_envs

        # Create the base environment
        self.envs = SubprocVecEnv(
            [make_env(env_name, i, render_mode=render_mode) for i in range(num_envs)]
        )

        if env_name in [
            "h1hand-push-v0",
            "h1-push-v0",
            "h1hand-cube-v0",
            "h1cube-v0",
            "h1hand-basketball-v0",
            "h1-basketball-v0",
            "h1hand-kitchen-v0",
            "h1-kitchen-v0",
        ]:
            self.max_episode_steps = 500
        else:
            self.max_episode_steps = 1000

        # For compatibility with MuJoCo Playground
        self.asymmetric_obs = False  # For comptatibility with MuJoCo Playground
        self.num_obs = self.envs.observation_space.shape[-1]
        self.num_actions = self.envs.action_space.shape[-1]

        # Per-env elapsed step counter (used to split time-limit vs true termination).
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self):
        """Reset the environment."""
        observations = self.envs.reset()
        self.episode_steps[:] = 0
        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        return observations, {} # to be compatible with the training files

    def render(self):
        assert self.num_envs == 1, (
            "Currently only supports single environment rendering"
        )
        return self.envs.render()

    def step(self, actions):
        assert isinstance(actions, torch.Tensor)
        actions = actions.cpu().numpy()

        observations, rewards, dones, raw_infos = self.envs.step(actions)
        dones = np.asarray(dones, dtype=bool)

        # Split time-limit (truncation) from true termination using a per-env step counter, so it does not depend on the (often missing) "TimeLimit.truncated".
        self.episode_steps += 1
        truncateds = np.logical_and(dones, self.episode_steps >= self.max_episode_steps)
        terminateds = np.logical_and(dones, np.logical_not(truncateds))

        # substitute the terminal observation on every boundary.
        infos = dict()
        infos["observations"] = {"raw": {"obs": observations.copy()}}
        for i in range(self.num_envs):
            if dones[i] and "terminal_observation" in raw_infos[i]:
                infos["observations"]["raw"]["obs"][i] = raw_infos[i][
                    "terminal_observation"
                ]

        # reset the step counter for envs that just finished an episode
        self.episode_steps[dones] = 0

        observations = torch.from_numpy(observations).to(
            device=self.sim_device, dtype=torch.float
        )
        rewards = torch.from_numpy(rewards).to(
            device=self.sim_device, dtype=torch.float
        )
        dones = torch.from_numpy(dones).to(device=self.sim_device)
        truncateds = torch.from_numpy(truncateds).to(device=self.sim_device)
        terminateds = torch.from_numpy(terminateds).to(device=self.sim_device)
        infos["observations"]["raw"]["obs"] = torch.from_numpy(
            infos["observations"]["raw"]["obs"]
        ).to(device=self.sim_device, dtype=torch.float)
        infos["time_outs"] = truncateds      # time-limit only  -> Transition.truncated
        infos["terminated"] = terminateds    # true termination -> Transition.done

        return observations, rewards, dones, infos