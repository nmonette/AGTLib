import multiprocessing as mp
from random import randint
from time import sleep

import numpy as np
# from agtlib.competitive.mwu import MultiplicativeWeights 
import gymnasium as gym
from gymnasium import register

import torch

from tests.gd_test import test_gd
from agtlib.cooperative.base import PolicyNetwork
from agtlib.cooperative.ppo import PPO, IPPO
from agtlib.utils.rollout import RolloutManager
from agtlib.utils.env import SingleAgentEnvWrapper, MultiGridWrapper

from agtlib.utils.stable_baselines.vec_env.subproc_vec_env import SubprocVecEnv
from agtlib.utils.stable_baselines.monitor import Monitor


from multigrid.multigrid.envs.team_empty import TeamEmptyEnv
import multigrid

if __name__ == "__main__":
    
    # CONFIGURATIONS = {
    #         'MultiGrid-Empty-8x8-Team': (TeamEmptyEnv, {'size': 8, "agents": 4, "allow_agent_overlap":True, "max_steps":3000})
    #     }
        
    # for name, (env_cls, config) in CONFIGURATIONS.items():
    #     register(id=name, entry_point=env_cls, kwargs=config)
    
    ippo = IPPO(4, 148, 4)
    ippo.train(lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-8x8-Team")), n_envs = 32, n_updates=1000, rollout_length=30)
   
    # policies = []
    # for i in range(len(ippo.ppo)):
    #     policies.append(ippo.ppo[i].policy)
    #     torch.save(policies[-1].state_dict(), f"policy_{i}.pt")
    
    
    # policies = [PolicyNetwork(148, 4) for i in range(4)] # []
    # for i in range(len(policies)):
    #     policies[i].load_state_dict(torch.load(f"policy_{i}.pt"))
        
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-8x8-Team", render_mode="human"))
    for i in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            action = {}
            for j in range(4):
                action[j] = randint(0,3)# policies[j].get_action(torch.from_numpy(obs[j]).float())[0].int().item()

            obs, reward, trunc, done, _ = env.step(action)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break


    # test_gd()
    print("end")
    # pdoc --docformat numpy agtlib