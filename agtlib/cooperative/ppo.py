from typing import Iterable

import torch
import torch.nn as nn
import numpy as np

from .base import RLBase

class PPO(RLBase):
    def __init__(self, action_size: int, obs_size: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    lambda_: float = 0.95):
        super().__init__(self, action_size, obs_size, v_obs_size=v_obs_size, policy_hl_dims=policy_hl_dims, value_hl_dims=value_hl_dims, 
                         linear_value=linear_value, gamma=gamma)

        if not isinstance(lambda_, float):
            raise TypeError("Parameter 'lambda_' is not type float")
        elif not (0 <= lambda_ <= 1):
            raise ValueError("Parameter 'lambda_' is not in the range [0,1]")
        else:
            self.lambda_ = lambda_
    
    def step(self, utility):
        pass

