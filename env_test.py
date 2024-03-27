from agtlib.utils.stable_baselines.vec_env import SubprocVecEnv
import multigrid.multigrid.envs
import gymnasium as gym
from agtlib.utils.env import DecentralizedMGWrapper

def test():
    env = SubprocVecEnv([lambda: DecentralizedMGWrapper(gym.make("MultiGrid-Empty-3x3-TeamWins", disable_env_checker=True)) for _ in range(5)])
    # env = DecentralizedMGWrapper(gym.make("MultiGrid-Empty-3x3-TeamWins"))
    env.reset()
    print(env.step([{0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 2}, {0: 1, 1: 1, 2: 2}])[0])
if __name__ == "__main__":
    test()