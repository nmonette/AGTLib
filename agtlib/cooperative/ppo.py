from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from .base import RLBase, PolicyNetwork
from ..utils.rollout import RolloutBuffer, RolloutManager

class PPO(RLBase):
    """
    Base Implementation of Proximal Policy Optimization. Inspired by the OpenAI stable baselines 
    implementation: https://github.com/DLR-RM/stable-baselines3, as well as the seminal paper:
    https://arxiv.org/abs/1707.06347.Implementation. Also
    used the https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ blog post as a reference.
    """
    def __init__(self, action_size: int, obs_size: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    lambda_: float = 0.95, clip_range: float = 0.2):
        super().__init__(self, action_size, obs_size, v_obs_size=v_obs_size, policy_hl_dims=policy_hl_dims, value_hl_dims=value_hl_dims, 
                         linear_value=linear_value, gamma=gamma)
        if not (0 <= lambda_ <= 1):
            raise ValueError("Parameter 'lambda_' is not in the range `[0,1]`")
        else:
            self.lambda_ = lambda_
        
        if not (0 <= clip_range < 1):
            raise ValueError("Parameter 'clip_range' is not in the range `[0,1)lkjnlkjnsk`")
        else:
            self.clip_range = clip_range
    
    def step(self, buffer: RolloutBuffer) -> None:
        pass
    # note that the "old" policy is the policy from the last 
    # minibatch, NOT the last rolloutå



class MAPPO(PPO):
    ## TO DO: fix this, code was written for sake of explanation
    def __init__(self, teams: [int, ], env: gym.Env, use_centralized_v: bool = False):
        self.policy_groups = teams

        self.policies = [PolicyNetwork(...) for i in range(len(set(teams)))]

        self.rollout = RolloutManager(self.policies...)

    def train(self):
        for i in range(epochs):
            data = self.rollout.rollout()
            for j in range(len()):
                self.policies[j].train(data[j])

class IPPO:
    pass