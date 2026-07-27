import mujoco
from src.env_utils.torch_wrappers.humanoid_bench_env import HumanoidBenchEnv

def main():
    env = HumanoidBenchEnv("h1hand-hurdle-v0", num_envs=2)
    # base_env = env.unwrapped[0]
    obs, info = env.reset()

if __name__ == "__main__":
    main()