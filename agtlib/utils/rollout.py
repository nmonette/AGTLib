from typing import Iterable, List, Dict, DefaultDict
from collections import defaultdict

import torch
import gymnasium as gym

from ..cooperative.base import PolicyNetwork, ValueNetwork, LinearValue

class RolloutBuffer:
    """
    Class to hold all of the data from a rollout.  The adv_buffer holds the form 
    {value_fn_index: [[value, ], ]}
    """
    def __init__(self, obs_buffer: DefaultDict[int, List[int, ]], log_prob_buffer: DefaultDict[int, List[int, ]], action_buffer: DefaultDict[int, List[int, ]], reward_buffer: DefaultDict[int, List[int, ]], adv_buffer: DefaultDict[int, List[int, ]]) -> None:
        """
        Parameters
        ----------
        obs_buffer: DefaultDict(int: [int, ])
            Holds the form: `{policy_index: [[value, ],  ]}`, 
            where the keys are the agents and the innermost list 
            represents the observations of a single episode. 
        log_prob_buffer: DefaultDict(int: [int, ])
            Holds the form: `{policy_index: [[value, ],  ]}`, 
            where the keys are the agents and the innermost list 
            represents the log probabilties of a single episode. 
        action_buffer: DefaultDict(int: [int, ])
            Holds the form: `{policy_index: [[value, ],  ]}`, 
            where the keys are the agents and the innermost list 
            represents the actions of a single episode. 
        reward_buffer: DefaultDict(int: [int, ])
            Holds the form: `{policy_index: [[value, ],  ]}`, 
            where the keys are the agents and the innermost list 
            represents the reward of a single episode. 
        adv_buffer: DefaultDict(int: [int, ])
            Holds the form `{value_fn_index: [[value, ], ]}` where 
            they keys are the value estimators and the innermost list
            represents the advantage estimates of a single episode.
        """
        self.obs_buffer = log_prob_buffer
        self.log_prob_buffer = log_prob_buffer
        self.action_buffer = action_buffer
        self.reward_buffer = reward_buffer
        self.adv_buffer = adv_buffer
    
## TO DO: make it so that we don't have to do multiple dictionaries and then translate, 
## just make a RolloutBuffer for each agent or vf so that we can pass it into the update function
class RolloutManager:
    """
    Implementation of monte carlo rollouts for the Generalized Advantage Estimation method. For 
    more details, refer to the paper https://arxiv.org/abs/1506.02438. Implementation heavily 
    inspired by the OpenAI stable baselines: https://github.com/DLR-RM/stable-baselines3. Also
    used the "37 implementation details for PPO" blog post as a reference.
    """
    def __init__(self, rollout_length: int, env: gym.Env, policies: Iterable[PolicyNetwork, ], values: Iterable[ValueNetwork | LinearValue, ], policy_groups: Iterable[PolicyNetwork, ] = None, value_groups: Iterable[PolicyNetwork, ] = None, gamma: float = 0.99, lambda_: float = 0.95) -> None:
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
        gamma: float, optional, keyword only
            The hyperparameter for discounting return. Must be in the range `[0,1]`. Defaults to
            `0.99`.
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
        else:
            self.value_groups = value_groups
        

    def calculate_adv(self, states: [dict, ], rewards: [dict, ]) -> Dict[int, List[int, ]]:
        """
        Calculates the advantages given the states, actions, and rewards of a given episode.
        """
        adv = defaultdict(lambda: [0])
        if len(self.values) == len(self.value_groups):
            for agent in range(len(self.values)):
                coef = (self.gamma * self.lambda_)**(len(states) - 1)
                for t in range(len(states) - 1, -1):
                    obs = torch.flatten(torch.tensor([states[t][i] for i in range(len(self.value_groups)) if self.value_groups[i] == agent]))
                    adv[agent].append(adv[agent][len(states) - t - 1] + coef * (rewards[t] + self.gamma * self.values[agent].forward(obs)))
                    coef /= (self.gamma * self.lambda_)
        else:
            for agent in range(len(self.values)):
                coef = (self.gamma * self.lambda_)**(len(states) - 1)
                for t in range(len(states) - 1, -1):
                    adv[agent].append(adv[agent][len(states) - t - 1] + coef * (rewards[t] + self.gamma * self.values[agent].forward(states[t][agent])))
                    coef /= (self.gamma * self.lambda_)
        return adv

    def rollout(self) -> Dict[int, List[int, ]]:
        """
        Performs a monte carlo rollout, and then solves for the advantage estimator
        at each time step of the episode.
        Returns
        -------
        Dict(int: [int, ])
            Dictionary such that each key corresponds to a value function and 
            the ith element of the corresponding list is a list of the advantage
            estimates of the ith episode. 
        """
        buffer = RolloutBuffer(defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list))
        for i in range(self.rollout_length):
            obs, _ = self.env.reset()

            states = defaultdict(list)
            log_probs = defaultdict(list)
            actions = defaultdict(list)
            rewards = defaultdict(list)

            while True:

                action = {}
                for i in range(len(self.policy_groups)): # sampling actions and log probs
                    action, log_prob = self.policies[self.policy_groups[i]].get_action()
                    action[i] = action
                    log_probs[i].append(log_prob)

                obs, reward, done, trunc, _ = self.env.step(action)

                for i in range(len(self.policy_groups)):
                    states[i].append(obs[i])
                    rewards[i].append(reward[i])

                if done or trunc:
                    break

            episode_advs = self.calculate_adv(states, rewards)
            for i in range(len(episode_advs)):
                buffer.adv_buffer[i].append(episode_advs[i])
            for i in range(len(states)):
                buffer.obs_buffer[i].append(states[i])
                buffer.log_prob_buffer[i].append(log_probs[i])
                buffer.action_buffer[i].append(actions[i])
                buffer.reward_buffer[i].append(rewards[i])

        return buffer

    def get_data(self, batch_size: int):
        pass
        
        

                
                

            


