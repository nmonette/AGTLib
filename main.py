import multiprocessing as mp

import numpy as np
# from agtlib.competitive.mwu import MultiplicativeWeights 
import gymnasium as gym

import torch

from agtlib.cooperative.ppo import PPO
from agtlib.utils.rollout import RolloutManager
from agtlib.utils.env import SingleAgentEnvWrapper

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

    env = gym.make("CartPole-v1", render_mode="human")
    env = SingleAgentEnvWrapper(env)
    ppo = PPO(2, 4)

    for epoch in range(100):
        rollout = RolloutManager(30, env, [ppo.policy], [ppo.value])
        # buffer = rollout.rollout()[0]
        p =  mp.Pool(5)
        buffers = p.map(rollout.rollout) # Figure out how to use map
        p.close()
        p.join()
        ppo.train(buffers[0])
        '''
        mp.set_start_method('spawn')
        p = [mp.Process(target=foo) for i in range(5)]
        for i in p:
            i.start()
            i.join()
        # for i in p:
        '''
# pdoc --docformat numpy agtlib