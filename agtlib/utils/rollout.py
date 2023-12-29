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
    def __init__(self) -> None:
        self.obs_buffer = defaultdict(list)
        self.log_prob_buffer = defaultdict(list)
        self.action_buffer = defaultdict(list)
        self.reward_buffer = defaultdict(list)
        self.adv_buffer = defaultdict(list)
        self.value_buffer = defaultdict(list)
    
## TO DO: make it so that we don't have to do multiple dictionaries and then translate, 
## just make a RolloutBuffer for each agent or vf so that we can pass it into the update function
class RolloutManager:
    """
    Implementation of monte carlo rollouts for the Generalized Advantage Estimation method. For 
    more details, refer to the paper https://arxiv.org/abs/1506.02438. Implementation heavily 
    inspired by the OpenAI stable baselines: https://github.com/DLR-RM/stable-baselines3. Also
    used the https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ blog post as a reference.
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
        

    def calculate_adv(self, states: DefaultDict[int, List[int, ]], rewards: DefaultDict[int, List[int, ]]) -> (Dict[int, List[int, ]], Dict[int, List[int, ]]):
        """
        Calculates the advantages given the states, actions, and rewards of a given episode.
        Parameters
        ----------
        states: DefaultDict(int: [int, ])
            Dictionary of lists such that each key refers to each agent and the correpsponding 
            lists hold the agents' respective observations at each time step. 
        rewards: DefaultDict(int: [int, ])
            Dictionary of lists such that each key refers to each agent and the correpsponding 
            lists hold the agents' respective rewards at each time step. 
        
        Returns
        -------
        DefaultDict(int: [int, ])
            Dictionary of lists such that each key refers to each agent and the corresponding 
            lists hold the agents' respective advantage values at each time step.
        DefaultDict(int: [int, ])
            Dictionary of lists such that each key refers to each agent and the corresponding 
            lists hold the agents' respective state-values at each time step.
        """
        adv = defaultdict(list)
        values = defaultdict(list)

        if len(self.values) == len(self.value_groups):
            for agent in range(len(self.values)):
                coef = (self.gamma * self.lambda_)**(len(states) - 1)
                for t in range(1, len(states)):
                    obs = torch.flatten(torch.tensor([states[i][-t] for i in range(len(self.value_groups)) if self.value_groups[i] == agent]))
                    value1 = values[agent][-t+1] if t != 1 else 0 # V(s_{t+1}) 
                    value2 = self.values[agent].forward(obs) # V(s_t)
                    adv[agent].append(adv[agent][-t] + coef * (rewards[-t] + self.gamma * value1 - value2))
                    values[agent].append(value1)
                    coef /= (self.gamma * self.lambda_)
        else:
            for agent in range(len(self.values)): 
                coef = (self.gamma * self.lambda_)**(len(states) - 1)
                for t in range(1, len(states)): # goes in reverse by utilizing negative indexing
                    value1 = values[agent][-t+1] if t != 1 else 0 # V(s_{t+1}) 
                    value2 = self.values[agent].forward(states[agent][-t]) # V(s_t)
                    adv[agent].append(adv[agent][-t] + coef * (rewards[-t] + self.gamma * value1 - value2))
                    values[agent].append(value1)
                    coef /= (self.gamma * self.lambda_)

        return {i: adv[i][::-1] for i in range(len(adv))}, {i: adv[i][::-1] for i in range(len(values))}

    def rollout(self) -> Dict[int, List[int, ]]:
        """
        Performs a monte carlo rollout, and then solves for the advantage estimator
        at each time step of the episode.
        Returns
        -------

        Dict(int: [int, ])
            Dictionary such that each key corresponds to an agent, and the key refers to a rollout buffer.
        """
        buffers = [RolloutBuffer(defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)) \
                   for _ in range(len(self.policy_groups))]
        
        for i in range(self.rollout_length):
            obs, _ = self.env.reset()

            states = defaultdict(list)
            log_probs = defaultdict(list)
            actions = defaultdict(list)
            rewards = defaultdict(list)

            while True: # completes one episode in the environment

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

            episode_advs, episodes_values = self.calculate_adv(states, rewards)
            for i in range(len(self.policy_groups)):
                buffers[i].adv_buffer.append(episode_advs[self.value_groups[i]])
                buffers[i].value_buffer.append(episodes_values[self.value_groups[i]])
                buffers[i].obs_buffer.append(states[i])
                buffers[i].log_prob_buffer.append(log_probs[i])
                buffers[i].action_buffer.append(actions[i])
                buffers[i].reward_buffer.append(rewards[i])

        return buffers

    def get_data(self, batch_size: int):
        pass
        
        

                
                

            


