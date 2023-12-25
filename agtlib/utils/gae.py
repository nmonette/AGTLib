from typing import Iterable, List
from collections import defaultdict

import torch
import gymnasium as gym

from ..cooperative.base import PolicyNetwork, ValueNetwork, LinearValue

class GAE:
    """
    Implementation of monte carlo rollouts for the Generalized Advantage Estimation method,
    where the rewards of a several episodes are collected, discounted, and averaged. For 
    more details, refer to the paper https://arxiv.org/abs/1506.02438. 
    
    """
    def __init__(self, rollout_length: int, env: gym.Env, policies: Iterable[PolicyNetwork, ], values: Iterable[ValueNetwork | LinearValue, ], policy_groups: Iterable[PolicyNetwork, ] = None, value_groups: Iterable[PolicyNetwork, ] = None,lambda_: float = 0.95) -> None:
        """
        rollout_length: int
            Amount of episodes to simulate per rollout. 
        env: gym.Env
            The environment to be used in simulation.
            Supports multi-agent environments.
        policies: Iterable(PolicyNetwork)
            The list of policies to be used in simulation.
        values: Iterable(ValueNetwork or LinearValue)   
            The list of value estimators to be used in advantage calculation. 
        policy_groups: Iterable(int), optional
            An iterable such that the ith element represents the policy
            of the ith agent. Used to specify which agents share which
            policy. Defaults to `None`.
        value_groups: Iterable(int), optional
            An iterable such that the ith element represents the value
            estimator of the ith agent. Used to specify which agents share which
            value estimator. Defaults to `None`.
        lambda_: float, optional, keyword only
            The hyperparameter for Generalized Advantage Estimation. Must be in the range `[0,1]`.
            Defaults to `0.95`.
        """
        self.rollout_length = rollout_length
        self.env = env

        self.policies = policies
        if policy_groups is None:
            self.policy_groups = list(range(len(self.policy_groups)))
        else:
            self.policy_groups = policy_groups

        self.values = values
        if value_groups is None:
            self.value_groups = list(range(len(self.value_groups)))
            self.value_groups_u = set(policy_groups)
        else:
            self.value_groups = value_groups
            self.value_groups_u = set(value_groups)
        

    def calculate_adv(self, states: [dict, ], actions: [dict, ], rewards: [dict, ]) -> dict[int, List[int, ]]:
        """
        Calculates the advantages given the states, actions, and rewards of a given episode.
        """
        adv = defaultdict(list)
        if len(self.values) == len(self.value_groups):
            for agent in range(len(self.values)):
                for t in range(len(states) - 1, -1, 0):
                    obs = torch.flatten(torch.tensor([states[t][i] for i in range(len(self.value_groups)) if self.value_groups[i] == agent]))
                    adv[agent].append(rewards[t] + self.gamma * self.values[agent].forward(obs))
        else:
            for agent in range(len(self.values)):
                for t in range(len(states) - 1, -1, 0):
                    adv[agent].append(rewards[t] + self.gamma * self.values[agent].forward(states[t][agent]))

    def rollout(self):
        """
        Performs a monte carlo rollout, and then solves for the advantage estimator
        at each time step of the episode.
        Returns
        -------
        list(list(dict))
            list of episodes, where each element of the episode-list is the 
            corresponding observation(s) of each agent at each time step. 
        list(list(dict))
            list of episodes, where each element of the episode-list is the 
            corresponding observation(s) of each agent. 
        """

        for i in range(self.rollout_length):
            obs, _ = self.env.reset()
            states = []
            actions = []
            rewards = []
            while True:
                action = {
                    i : self.policies[self.policy_groups[i]].get_action() for i in range(len(self.policy_groups))
                }

                obs, reward, done, trunc, _ = self.env.step(action)

                states.append(obs)
                actions.append(action)
                rewards.append(reward)

                if done or trunc:
                    break
            
            advs = self.calculate_adv(states, actions, rewards)
        
        return state_buffer, action_buffer, reward_buffer
        
        

                
                

            


