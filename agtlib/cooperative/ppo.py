from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from .base import RLBase, PolicyNetwork, ValueNetwork
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
                    lambda_: float = 0.95, clip_range: float = 0.2, action_latent_dim: int = 100): # write docstrings for parameters
        super().__init__(self, action_size, obs_size, v_obs_size=v_obs_size, policy_hl_dims=policy_hl_dims, value_hl_dims=value_hl_dims, 
                         linear_value=linear_value, gamma=gamma)
        if not (0 <= lambda_ <= 1):
            raise ValueError("Parameter 'lambda_' is not in the range `[0,1]`")
        else:
            self.lambda_ = lambda_
        
        if not (0 <= clip_range < 1):
            raise ValueError("Parameter 'clip_range' is not in the range `[0,1)`")
        else:
            self.clip_range = clip_range

        # self.actor_extractor = PolicyNetwork(obs_size, action_latent_dim)
        # self.critic_extractor = ValueNetwork(obs_size)

        # self.latent_net = nn.Linear(action_size)

    def preprocess(self, obs: np.ndarray):
        '''
        Preprocesses observations.
        Parameters
        -------
        obs: np.ndarray
            Array containing the flattened observation at each time step.

        Returns
        -------
        torch.Tensor
            Tensor that contains the output of actor_extractor
        torch.Tensor
            Tensor that contains the output of critic_extractor
        '''
        obs = torch.Flatten()(obs).float()

    def _evaluate_actions(self, actions: np.ndarray, obs: np.ndarray):
        """
        Evaluates analysis of the rollout data that is necessary for 
        calculation of the PPO loss. 
        Parameters
        ----------
        actions: np.ndarray
            Array containing the integer index of each action at each time step.
        obs: np.ndarray
            Array containing the flattened observation at each time step.

        Returns
        -------
        torch.Tensor
            Tensor containing the values for each state at each time step.
        torch.Tensor
            Tensor containing the log probabilities for each action at each time step.
        torch.Tensor
            Tensor containing the entropy of the action distribution at each time step.
        """
        obs = self.preprocess(obs)
        values = self.value(obs)
        action_logits = self.policy(obs)

        dist = torch.distributions.Categorical(logits=action_logits)
        
        log_prob = dist.log_prob(actions) #
        entropy = dist.entropy()

        return values, log_prob, entropy 

    def train(self, buffer: RolloutBuffer, num_epochs: int = 1, batch_size: int = 32) -> None:
        """
        Performs a training update with rollout data.
        Intended to be performed in parallel with 
        other agent. 
        Parameter
        ---------
        buffer: RolloutBuffer
            Data collected from the rollout.
        num_epochs: int, optional
            Number of passes for the model to go through 
            the rollout data. Defaults to `1`.
        batch_size: int, optional
            Size of each minibatch for the model to train
            on. Defaults to `32`.
        """

        for epoch in range(num_epochs):
            for data in buffer.get_data(batch_size):
                actions = data.action_buffer
                
                self.
    






class MAPPO(PPO):
    ## TO DO: fix this, code was written for sake of explanation
    def __init__(self, teams: [int, ], env: gym.Env, ctde: bool = False):
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

