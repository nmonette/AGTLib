from typing import Iterable, List, Dict, DefaultDict, Union
from collections import defaultdict

import torch
import gymnasium as gym
import numpy as np

from ..cooperative.base import PolicyNetwork, ValueNetwork, LinearValue

class RolloutBuffer:
    """
    Class to hold all of the data from a rollout.  The adv_buffer holds the form 
    {value_fn_index: [[value, ], ]}
    """
    def __init__(self, rollout_length: int) -> None: # will add another param for the number of parallel rollouts
        """
        Parameters
        ----------
        rollout_length: int
            Integer indicating the length of the rollouts, 
            in order to set the size of the buffer arrays.
        """
        self.rollout_length = rollout_length
        self.obs_buffer = [] 
        self.log_prob_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.adv_buffer = []
        self.value_buffer = []
        self.return_buffer = []

    def __getitem__(self, key: Union[List[int, ], int]) -> "RolloutBuffer":
        """
        Retrieves a portion of the rollout buffer based on an index or
        list of indexes.
        Parameters
        ---------
        key: [int, ] or int
            The indexes of the buffer to retrieve.
        Returns
        -------
        RolloutBuffer
            The shortened rollout buffer.
        """
        if isinstance(key, int):
            rb = RolloutBuffer(1)
            rb.obs_buffer[0] = self.obs_buffer[key]
            rb.log_prob_buffer[0] = self.log_prob_buffer[key]
            rb.action_buffer[0] = self.action_buffer[key]
            rb.reward_buffer[0] = self.reward_buffer[key]
            rb.adv_buffer[0] = self.adv_buffer[key]
            rb.value_buffer[0] = self.value_buffer[key]
            rb.return_buffer[0] = self.return_buffer[key]

        else:
            rb = RolloutBuffer(len(key))
            rb.obs_buffer = self.obs_buffer[key] 
            rb.log_prob_buffer = self.log_prob_buffer[key]
            rb.action_buffer = self.action_buffer[key]
            rb.reward_buffer = self.reward_buffer[key]
            rb.adv_buffer = self.adv_buffer[key]
            rb.value_buffer = self.value_buffer[key]
            rb.return_buffer = self.return_buffer[key]

        return rb
    
    def get_data(self, batch_size: int):
        """
        Randomly splits up data from a rollout.
        
        Yields
        ------
        RolloutBuffer
            Abbreviated RolloutBuffer object for each minibatch.
        """
        
        self.obs_buffer = torch.stack(self.obs_buffer) 
        self.log_prob_buffer = torch.tensor(self.log_prob_buffer) 
        self.action_buffer = torch.tensor(self.action_buffer) 
        self.reward_buffer = torch.tensor(self.reward_buffer) 
        self.adv_buffer = torch.tensor(self.adv_buffer[::-1]) 
        self.value_buffer = torch.tensor(self.value_buffer[::-1]) 
        self.return_buffer = torch.tensor(self.return_buffer[::-1]) 

        perm = np.random.permutation(self.rollout_length) 

        idx = 0
        while idx < self.rollout_length: # will multiply by n_envs once multithreading is added
            yield self[perm[idx:idx + batch_size]]
            idx += batch_size


class RolloutManager:
    """
    Implementation of monte carlo rollouts for the Generalized Advantage Estimation method. For 
    more details, refer to the paper https://arxiv.org/abs/1506.02438. Implementation heavily 
    inspired by the OpenAI stable baselines: https://github.com/DLR-RM/stable-baselines3. Also
    used the https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ blog post as a reference.
    """
    def __init__(self, rollout_length: int, env: gym.Env, policies: Iterable[PolicyNetwork, ], values: Iterable[ Union[ValueNetwork, LinearValue], ], policy_groups: Iterable[PolicyNetwork, ] = None, value_groups: Iterable[PolicyNetwork, ] = None, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
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
            policy. Defaults to `None`. Assumes that if none is specified,
            there is no shared policy.
        value_groups: Iterable(int), optional
            An iterable such that the ith element represents the value
            estimator of the ith agent. Used to specify which agents share which
            value estimator. Defaults to `None`. Assumes that if none is specified,
            there is no shared value function.
        gamma: float, optional, keyword only
            The hyperparameter for discounting return. Must be in the range `[0,1]`. Defaults to
            `0.99`.
        gae_lambda: float, optional, keyword only
            The hyperparameter for Generalized Advantage Estimation. Must be in the range `[0,1]`.
            Defaults to `0.95`.
        """
        self.rollout_length = rollout_length
        self.env = env

        self.policies = policies
        if policy_groups is None:
            self.policy_groups = list(range(len(policies)))
        else:
            self.policy_groups = policy_groups

        self.values = values
        if value_groups is None:
            self.value_groups = list(range(len(values)))
        else:
            self.value_groups = value_groups

        self.gamma = gamma
        self.gae_lambda = gae_lambda        

    def calculate_adv(self, buffers: [RolloutBuffer, ], timesteps: int):
        """
        Calculates the advantages given the states, actions, and rewards of a given episode, 
        and populates the give RolloutBuffer objects with them.
        Parameters
        ----------
        buffers: List(RolloutBuffer, )
            List of RolloutBuffer objects holding data collected from the episode.
        timesteps: int
            Integer representing the number of timesteps that data was collected in the last 
            rollout iteration.
        """
        coef = self.gamma * self.gae_lambda
        if len(self.values) != len(self.value_groups):
            for agent in range(len(self.values)):
                for t in range(1, timesteps):
                    obs = torch.flatten(torch.tensor([buffers[i].obs_buffer[-t] for i in range(len(self.value_groups)) if self.value_groups[i] == agent]))
                    value1 = buffers[agent].value_buffer[t-2] if t != 1 else 0 # V(s_{t+1}) 
                    value2 = self.values[agent].forward(obs).item() # V(s_t)
                    prev_adv = buffers[agent].adv_buffer[t-2] if t != 1 else 0 
                    prev_return = buffers[agent].return_buffer[t-2] if t != 1 else 0

                    buffers[agent].adv_buffer.append(coef * prev_adv + (buffers[agent].reward_buffer[-t] + self.gamma * value1 - value2))
                    buffers[agent].value_buffer.append(value2)
                    buffers[agent].return_buffer.append(self.gamma * prev_return + buffers[agent].reward_buffer[-t])
        else:
            for agent in range(len(self.values)): 
                for t in range(1, timesteps): # goes in reverse by utilizing negative indexing
                    value1 = buffers[agent].value_buffer[t-2] if t != 1 else 0 # V(s_{t+1}) 
                    value2 = self.values[agent].forward(buffers[agent].obs_buffer[-t]).item() # V(s_t)
                    prev_adv = buffers[agent].adv_buffer[t-2] if t != 1 else 0 
                    prev_return = buffers[agent].return_buffer[t-2] if t != 1 else 0
                    
                    buffers[agent].adv_buffer.append(coef * prev_adv + (buffers[agent].reward_buffer[-t] + self.gamma * value1 - value2))
                    buffers[agent].value_buffer.append(value2)
                    buffers[agent].return_buffer.append(self.gamma * prev_return + buffers[agent].reward_buffer[-t])

    def rollout(self) -> Dict[int, List[int, ]]:
        """
        Performs a monte carlo rollout, and then solves for the advantage estimator
        at each time step of the episode.
        Returns
        -------

        Dict(int: RolloutBuffer)
            Dictionary such that each key corresponds to an agent, and the key refers to a rollout buffer.
        """
        buffers = [RolloutBuffer(self.rollout_length) for _ in range(len(self.policy_groups))]
        
        for i in range(self.rollout_length): # we should turn the block of this code into a function so that we can parallelize it
            obs, _ = self.env.reset()
            
            for i in range(len(self.policy_groups)):
                obs[i] = torch.from_numpy(obs[i])
                buffers[i].obs_buffer.append(obs[i])

            timesteps = 0 
            while True: # completes one episode in the environment

                currrent_action = {}
                for j in range(len(self.policy_groups)): # sampling actions and log probs
                    action, log_prob = self.policies[self.policy_groups[j]].get_action(obs[j])
                    currrent_action[j] = action.item()
                    buffers[i].action_buffer.append(currrent_action[j])
                    buffers[i].log_prob_buffer.append(log_prob)

                obs, reward, done, trunc, _ = self.env.step(currrent_action)
                obs = {i: torch.from_numpy(obs[i]) for i in range(len(obs))}

                for j in range(len(buffers)):
                    buffers[j].obs_buffer.append(obs[j])
                    buffers[j].reward_buffer.append(reward[j])

                if done or trunc: # we may need to put in some kind of modification here for if the episode is truncated
                    break
                
                timesteps += 1

            self.calculate_adv(buffers, timesteps)

        return buffers

        
        

                
                

            


