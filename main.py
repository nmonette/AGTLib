import multiprocessing as mp

import numpy as np
# from agtlib.competitive.mwu import MultiplicativeWeights 
import gymnasium as gym

import torch

from agtlib.cooperative.ppo import PPO
from agtlib.utils.rollout import RolloutManager
from agtlib.utils.env import SingleAgentEnvWrapper

from agtlib.utils.stable_baselines.vec_env.subproc_vec_env import SubprocVecEnv
from agtlib.utils.stable_baselines.monitor import Monitor

def foo():
    lock = mp.Lock()
    lock.acquire(block=True)
    for i in range(1000):
        print(i)
    lock.release()


if __name__ == "__main__":
    # game = np.stack([
    #     np.array([0, 1, -1]), 
    #     np.array([1, 0, -1]), 
    #     np.array([-1, 1, 0]), 
    # ])
    # mw1 = MultiplicativeWeights(game, 0.5)
    # mw2 = MultiplicativeWeights(-game, 0.5)
    
    # converged1 = False
    # converged2 = False
    # for i in range(100000):
    #     move1 = mw1.get_action().item()
    #     move2 = mw2.get_action().item()

    #     converged1 = mw1.step(move2)
    #     converged2 = mw2.step(move1)

    #     # if converged1 and converged2:
    #     #     break

    #     print("1: ", mw1.get_strategy())
    #     print("2: ", mw2.get_strategy())

    # env = gym.make("CartPole-v1", render_mode="human")
    # env = SingleAgentEnvWrapper(env)
    ppo = PPO(2, 4, policy_hl_dims=[16], value_hl_dims=[16])

    def create_env():
        env = Monitor(gym.make("CartPole-v1"))
        # env = SingleAgentEnvWrapper(env)
        return env

    multi_env = SubprocVecEnv([create_env for _ in range(10)])
    next_obs = multi_env.reset()
    for epoch in range(100):
        rollout = RolloutManager(5, multi_env, [ppo.policy], [ppo.value])
        buffer, next_obs = rollout.rollout(next_obs)
        buffer = buffer[0]

        ppo.train(buffer, 5, 64)

    x = [torch.tensor(y) for y in multi_env.env_method("get_episode_rewards")]
    print(torch.mean(torch.cat(x)))

    multi_env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    for demo in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            obs, reward, done, trunc, _ = env.step(ppo.policy.get_action(torch.from_numpy(obs))[0].item())
    
            if done or trunc:
                break
            
    print("end")
    # pdoc --docformat numpy agtlib