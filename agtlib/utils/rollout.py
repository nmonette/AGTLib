from typing import Iterable, List, Dict, DefaultDict, Union

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from collections import defaultdict

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

        self.grabbed = False

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
        if not self.grabbed:
            self.obs_buffer = torch.stack(self.obs_buffer) 
            self.log_prob_buffer = torch.tensor(self.log_prob_buffer, dtype=torch.float32) 
            self.action_buffer = torch.tensor(self.action_buffer, dtype=torch.float32) 
            self.reward_buffer = torch.tensor(self.reward_buffer, dtype=torch.float32) 
            self.adv_buffer = torch.tensor(self.adv_buffer[::-1], dtype=torch.float32) 
            self.value_buffer = torch.tensor(self.value_buffer[::-1], dtype=torch.float32) 
            self.return_buffer = torch.tensor(self.return_buffer[::-1], dtype=torch.float32) 
            self.grabbed = True   

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
    def __init__(self, rollout_length: int, env: gym.Env, policies: Iterable[PolicyNetwork, ], values: Iterable[ Union[ValueNetwork, LinearValue], ], model_name: str, policy_groups: Iterable[PolicyNetwork, ] = None, value_groups: Iterable[PolicyNetwork, ] = None, gamma: float = 0.99, gae_lambda: float = 0.95, n_envs: int = 1) -> None:
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
        model_name: str
            The name of the model being used. Helps to check
            for different variations of parameter sharing. 
        policy_groups: Iterable(int), optional
            An iterable such that the ith element represents the integer index
            policy of the ith agent. Used to specify which agents share which
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
        n_envs: int, optional
            Number of environments to run in parallel during rollouts. 
            Defaults to `1`. 
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
        self.n_envs = n_envs    

        self.model_name = model_name

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
            for value in range(len(self.values)):
                for t in range(1, timesteps + 1):
                    obs = torch.flatten(torch.stack([buffers[i].obs_buffer[-t] for i in range(len(self.value_groups)) if self.value_groups[i] == value])) # need to make sure this works lmao
                    value1 = buffers[value].value_buffer[t-2] if t != 1 else 0 # V(s_{t+1}) 
                    value2 = self.values[value].forward(obs).item() # V(s_t)
                    prev_adv = buffers[value].adv_buffer[t-2] if t != 1 else 0 
                    prev_return = buffers[value].return_buffer[t-2] if t != 1 else 0

                    adv = coef * prev_adv + (buffers[value].reward_buffer[-t] + self.gamma * value1 - value2)
                    val = value2
                    ret = adv + val # self.gamma * prev_return + buffers[value].reward_buffer[-t]

                    for agent in self.value_groups:
                        if agent == value:
                            buffers[agent].adv_buffer.append(adv)
                            buffers[agent].value_buffer.append(val)
                            buffers[agent].return_buffer.append(ret)
        else:
            for agent in range(len(self.values)): 
                for t in range(1, timesteps + 1): # goes in reverse by utilizing negative indexing
                    value1 = buffers[agent].value_buffer[t-2] if t != 1 else 0 # V(s_{t+1}) 
                    value2 = self.values[agent].forward(buffers[agent].obs_buffer[-t]).item() # V(s_t)
                    prev_adv = buffers[agent].adv_buffer[t-2] if t != 1 else 0 
                    prev_return = buffers[agent].return_buffer[t-2] if t != 1 else 0
                    
                    buffers[agent].adv_buffer.append(coef * prev_adv + (buffers[agent].reward_buffer[-t] + self.gamma * value1 - value2))
                    buffers[agent].value_buffer.append(value2)
                    # buffers[agent].return_buffer.append(self.gamma * prev_return + buffers[agent].reward_buffer[-t])
                    buffers[agent].return_buffer.append(buffers[agent].value_buffer[-1] + buffers[agent].adv_buffer[-1])


    def _process(self, passed_obs):
        new_obs = defaultdict(list)
        for i in passed_obs:
            for j in passed_obs[i]:
                new_obs[i].append(torch.from_numpy(j).float())
                
        if self.model_name == "MAPPO":           
            group_obs = {}
            for group in range(len(self.values)):
                group_obs[group] = torch.cat([new_obs[i][0] for i in range(len(new_obs)) if self.policy_groups[i] == group])
            return group_obs
        
        return new_obs
    
    def rollout(self, init_obs: (Dict[int, np.ndarray]), action_map = None) -> Dict[int, List[int, ]]: # may want to add a "calculate_advs" parameter
        """
        Performs a monte carlo rollout, and then solves for the advantage estimator
        at each time step of the episode.
        Parameters
        ----------
        init_obs: Dict(int: (np.ndarray, ))
            Essentially the set of observations to calculate the 
            initial action of the agents with.
        
        Returns
        -------
        Dict(int: RolloutBuffer)
            Dictionary such that each key corresponds to an agent, and the key refers to a rollout buffer.
        """
        buffers = [RolloutBuffer(self.rollout_length) for _ in range(len(self.policy_groups))]
        """
        to obtain group observations:

        group_obs = {}
        for group in range(len(self.values)):
            group_obs[group] = torch.cat([init_obs[i][0] for i in range(len(init_obs)) if self.policy_groups[i] == group])
        """
        
        new_obs = self._process(init_obs)

        for t in range(self.rollout_length): # "fixed length trajectory segments (PPO)"
            current_action = []
            for j in range(self.n_envs):
                current_action.append({})
                if self.model_name == "advPPO":
                    # Team actions
                    action, log_prob = self.policies[0].get_action(new_obs[0][j])
                    team_actions = action_map[action.to(torch.int32).item()]
                    buffers[0].action_buffer.append(action.to(torch.int32).item())
                    buffers[0].log_prob_buffer.append(log_prob)
                    for k in range(len(team_actions)):
                        current_action[j][k] = team_actions[k]

                    # Adversary Action
                    action, log_prob = self.policies[1].get_action(new_obs[1][j])
                    buffers[1].action_buffer.append(action.to(torch.int32).item())
                    buffers[1].log_prob_buffer.append(log_prob)
                    current_action[j][len(team_actions)] = action.to(torch.int32).item()

                else:
                    for k in range(len(self.policy_groups)): # sampling actions and log probs
                        action, log_prob = self.policies[self.policy_groups[k]].get_action(new_obs[k][j])
                        current_action[j][k] = action.to(torch.int32).item()
                        buffers[k].action_buffer.append(current_action[j][k])
                        buffers[k].log_prob_buffer.append(log_prob)

            self.env.step_async(current_action)
            obs, reward, done, _ = self.env.step_wait()
            next_obs = obs

            new_obs = self._process(obs)

            for j in range(len(buffers)):
                for k in range(self.n_envs):
                    buffers[j].obs_buffer.append(new_obs[j][k])
                    buffers[j].reward_buffer.append(reward[k][j])

        self.calculate_adv(buffers, self.rollout_length)

        return buffers, next_obs


