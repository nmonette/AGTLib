from time import sleep
import gym
import torch
from agtlib.cooperative.lambda_pg import TwoHeadPolicy
from agtlib.cooperative.pg import MAPolicyNetwork
from agtlib.utils.env import MultiGridWrapper
import multiprocessing as mp
import warnings
from random import randint
from time import sleep

# from agtlib.competitive.mwu import MultiplicativeWeights 
import gymnasium as gym
import numpy as np
import torch
from gymnasium import register
import multigrid
from agtlib.cooperative.base import PolicyNetwork
from agtlib.cooperative.ppo import IPPO, PPO
from agtlib.runners.gdmax_experiments import grid_experiment_3x3, lgdmax_grid_experiment, nlgdmax_grid_experiment# ,mpe_experiment
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



def test_lgdmax_weights():
    team = MAPolicyNetwork(15, 4*4, [(i,j) for i in range(4) for j in range(4)])
    adv = TwoHeadPolicy(15, 4, fm_dim1=64, fm_dim2=128)

    team.load_state_dict(torch.load("./output/4500-3x3-team-policy-final-nlambda.pt"))
    adv.load_state_dict(torch.load("./output/4500-3x3-adv-policy-final-nlambda.pt"))

    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
            adv_action = adv_action.item()
            action = {i: team_action[i] for i in range(len(team_action))}
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

test_lgdmax_weights()