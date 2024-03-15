import multiprocessing as mp
import warnings
from random import randint
from time import sleep

# from agtlib.competitive.mwu import MultiplicativeWeights 
import gymnasium as gym
import numpy as np
import torch
from gymnasium import register
# from pettingzoo.test import render_test
# from pettingzoo.mpe import simple_adversary_v3


import multigrid
from agtlib.cooperative.base import PolicyNetwork
from agtlib.cooperative.ppo import IPPO, PPO
from agtlib.runners.gdmax_experiments import test_n_reinforce, test_reinforce, ngdmax_experiment, lgdmax_grid_experiment, nlgdmax_grid_experiment, test_lgdmax_weights, gdmax_experiment, reinforce_experiment, n_reinforce_experiment# ,mpe_experiment
from agtlib.utils.env import (MultiGridWrapper, SingleAgentEnvWrapper,
                              generate_reward)
from agtlib.utils.rollout import RolloutManager
from agtlib.utils.reward_table import generate_reward_3x3
from agtlib.utils.stable_baselines.monitor import Monitor
from agtlib.utils.stable_baselines.vec_env.subproc_vec_env import SubprocVecEnv
from multigrid.multigrid.envs.team_empty import TeamEmptyEnv
from tests.gd_test import test_gd
from tests.gdmax_test import test_gdmax
from treasure_hunt import TeamEmptyEnv

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    # test_lgdmax_weights()
    # test_reinforce()
    # test_n_reinforce()
    # test_n_reinforce()
    # torch.autograd.set_detect_anomaly(True)
    n_reinforce_experiment()
    # reinforce_experiment()
    # nlgdmax_grid_experiment()
    # 5 actions, 5 states
    # generate_reward_3x3()
    # print(torch.einsum("ij,jk->jik ", lambda_, x).reshape((S,A**3)))

    # print(torch.ger(lambda_, (x @ reward)).shape)

    # mpe_experiment()
    # generate_reward_3x3()
    # print("done")
    # env_dict = gym.envs.registration.registry.copy()

    # # for env in env_dict:
    # #     if 'MultiGrid-Empty-8x8-Team' in env:
    # #         print("Remove {} from registry".format(env))
    # #         del gym.envs.registration.registry[env]
    
    # CONFIGURATIONS = {
    #         'MultiGrid-Empty-6x6-Team': (TeamEmptyEnv, {'size': 8, "agents": 3, "allow_agent_overlap":True, "max_steps":60}),
    #         'MultiGrid-Empty-4x4-Team': (TeamEmptyEnv, {'size': 6, "agents": 3, "allow_agent_overlap":True, "max_steps":40}),
    #         'MultiGrid-Empty-3x3-Team': (TeamEmptyEnv, {'size': 5, "agents": 3, "allow_agent_overlap":True, "max_steps":12})
    #     }
    
    # for name, (env_cls, config) in CONFIGURATIONS.items():
    #     register(id=name, entry_point=env_cls, kwargs=config)

    # gym.make("MultiGrid-Empty-8x8-Team", render_mode="human")

    # MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3))
    
    # gdmax_experiment()
    # SubprocVecEnv([lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3))])
    # ippo = IPPO(4, 15, 3)
    # ippo.train(lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-8x8-Team", agents=3)), n_envs = 32, n_updates=1000, rollout_length=100)
   
    # policies = []
    # for i in range(len(ippo.ppo)):
    #     policies.append(ippo.ppo[i].policy)
    #     torch.save(policies[-1].state_dict(), f"policy_{i}.pt")
    
    
    # policies = [PolicyNetwork(148, 4) for i in range(4)] # []
    # for i in range(len(policies)):
    #     policies[i].load_state_dict(torch.load(f"policy_{i}.pt"))
    
    # policies = [0] * 3

    # env = MultiGridWrapper(gym.make("MultiGrid-Empty-4x4-Team", render_mode="human"))
    # for i in range(100):
    #     obs, _ = env.reset()
    #     env.render()
    #     while True:
    #         action = {}
    #         for j in range(len(policies)):
    #             action[j] = randint(0,3)# policies[j].get_action(torch.from_numpy(obs[j]).float())[0].int().item()

    #         obs, reward, trunc, done, _ = env.step(action)
    #         if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
    #             break

    # test_gd()
    print("end")
    
 