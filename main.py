import numpy as np
# from agtlib.competitive.mwu import MultiplicativeWeights 
import gymnasium as gym

import torch

from agtlib.cooperative.ppo import PPO
from agtlib.utils.rollout import RolloutManager

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

    env = gym.make("CartPole-v1")
    ppo = PPO(2, 4)

    for episode in range(100):
        while True:

            rollout = RolloutManager(10, env, [ppo.policy], [ppo.value])
            buffer = rollout.rollout()[0]

            ppo.train(buffer)

        






    
        
# pdoc --docformat numpy agtlib
