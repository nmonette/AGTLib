from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .base import RLBase
from ..utils.rollout import RolloutBuffer

class PPO(RLBase):
    """
    Base Implementation of Proximal Policy Optimization. Inspired by the OpenAI stable baselines 
    implementation: https://github.com/DLR-RM/stable-baselines3, as well as the seminal paper:
    https://arxiv.org/abs/1707.06347.

    """
    def __init__(self, action_size: int, obs_size: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    lambda_: float = 0.95):
        super().__init__(self, action_size, obs_size, v_obs_size=v_obs_size, policy_hl_dims=policy_hl_dims, value_hl_dims=value_hl_dims, 
                         linear_value=linear_value, gamma=gamma)
        if not (0 <= lambda_ <= 1):
            raise ValueError("Parameter 'lambda_' is not in the range `[0,1]`")
        else:
            self.lambda_ = lambda_
    
    def step(self, buffer: RolloutBuffer) -> None:
        pass

