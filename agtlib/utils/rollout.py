from typing import Iterable, List, Dict, DefaultDict, Union, Tuple

import torch
import gymnasium as gym
import numpy as np
from collections import defaultdict

from ..common.base import PolicyNetwork, ValueNetwork, LinearValue


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
            self.obs_buffer = torch.stack(self.obs_buffer[0:len(self.log_prob_buffer)]) 
            self.log_prob_buffer = torch.tensor(self.log_prob_buffer, dtype=torch.float32) 
            self.action_buffer = torch.tensor(self.action_buffer, dtype=torch.float32) 
            self.reward_buffer = torch.tensor(self.reward_buffer, dtype=torch.float32) 
            self.adv_buffer = torch.tensor(self.adv_buffer[::-1], dtype=torch.float32) 
            self.value_buffer = torch.tensor(self.value_buffer, dtype=torch.float32) 
            self.return_buffer = torch.tensor(self.return_buffer[::-1], dtype=torch.float32) 
            self.grabbed = True   

        perm = np.random.permutation(len(self.reward_buffer)) 

        idx = 0
        while idx < len(self.reward_buffer): # will multiply by n_envs once multithreading is added
            yield self[perm[idx:idx + batch_size]]
            idx += batch_size


class RolloutManager:
    """
    Implementation of monte carlo rollouts for the Generalized Advantage Estimation method. For 
    more details, refer to the paper https://arxiv.org/abs/1506.02438. Implementation heavily 
    inspired by the OpenAI stable baselines: https://github.com/DLR-RM/stable-baselines3. Also
    used the https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ blog post as a reference.
    """
    def __init__(self, rollout_length: int, env: gym.Env, policies: Iterable[PolicyNetwork, ], values: Iterable[ Union[ValueNetwork, LinearValue], ], policy_groups: Iterable[PolicyNetwork, ] = None, value_groups: Iterable[PolicyNetwork, ] = None, gamma: float = 0.99, gae_lambda: float = 0.95, n_envs: int = 1) -> None:
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

    def calculate_adv(self, buffers: List[RolloutBuffer, ], timesteps: int):
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
                for t in range(1, timesteps + 1):
                    obs = torch.flatten([buffers[i].obs_buffer[-t] for i in range(len(self.value_groups)) if self.value_groups[i] == agent])
                    value1 = buffers[agent].value_buffer[t-2] if t != 1 else 0 # V(s_{t+1}) 
                    value2 = self.values[agent].__call__(obs).item() # V(s_t)
                    prev_adv = buffers[agent].adv_buffer[t-2] if t != 1 else 0 
                    prev_return = buffers[agent].return_buffer[t-2] if t != 1 else 0

                    buffers[agent].adv_buffer.append(prev_adv + coef * (buffers[agent].reward_buffer[-t] + self.gamma * value1 - value2))
                    buffers[agent].value_buffer.append(value2)
                    buffers[agent].return_buffer.append(self.gamma * prev_return + buffers[agent].reward_buffer[-t])
        else:
            for agent in range(len(self.values)): 
                for t in range(1, timesteps + 1): # goes in reverse by utilizing negative indexing
                    value1 = buffers[agent].value_buffer[t-2] if t != 1 else 0 # V(s_{t+1}) 
                    value2 = self.values[agent].__call__(buffers[agent].obs_buffer[-t]).item() # V(s_t)
                    prev_adv = buffers[agent].adv_buffer[t-2] if t != 1 else 0 
                    prev_return = buffers[agent].return_buffer[t-2] if t != 1 else 0
                    
                    buffers[agent].adv_buffer.append(coef * prev_adv + (buffers[agent].reward_buffer[-t] + self.gamma * value1 - value2))
                    buffers[agent].value_buffer.append(value2)
                    # buffers[agent].return_buffer.append(self.gamma * prev_return + buffers[agent].reward_buffer[-t])
                    buffers[agent].return_buffer.append(buffers[agent].value_buffer[-1] + buffers[agent].adv_buffer[-1])

    def rollout(self, init_obs: Tuple[np.ndarray, ]) -> Dict[int, List[int, ]]: # may want to add a "calculate_advs" parameter
        """
        Performs a monte carlo rollout, and then solves for the advantage estimator
        at each time step of the episode.
        Returns
        -------

        Dict(int: RolloutBuffer)
            Dictionary such that each key corresponds to an agent, and the key refers to a rollout buffer.
        """
        buffers = [RolloutBuffer(self.rollout_length) for _ in range(len(self.policy_groups))]

        new_obs = []
        for k in range(self.n_envs): # will need to account for multi-agent later
            new_obs.append(torch.from_numpy(init_obs[k]).float())
            buffers[0].obs_buffer.append(new_obs[k])
        
        for t in range(self.rollout_length): # "fixed length trajectory segments (PPO)"
            
            currrent_action = defaultdict(lambda: np.ndarray((self.n_envs, ), dtype=np.int32))
            for j in range(self.n_envs):
                for k in range(len(self.policy_groups)): # sampling actions and log probs
                    action, log_prob = self.policies[self.policy_groups[k]].get_action(new_obs[j]) #[j][k]
                    currrent_action[k][j] = action.to(torch.int32).item()
                    buffers[k].action_buffer.append(currrent_action[k][j])
                    buffers[k].log_prob_buffer.append(log_prob)

            self.env.step_async(currrent_action[0])
            obs, reward, done, _ = self.env.step_wait() # removed trunc
            next_obs = obs
            obs = {j: torch.from_numpy(obs[j]).float() for j in range(self.n_envs)}

            for j in range(len(buffers)):
                for k in range(len(obs)):
                    buffers[j].obs_buffer.append(obs[k]) # need to modify once it is a dict of observations
                    buffers[j].reward_buffer.append(reward[k]) # same with this

        self.calculate_adv(buffers, len(buffers[0].reward_buffer))

        return buffers, next_obs

class GDmaxRollout(RolloutManager):
    def rollout(self, init_obs: Tuple[np.ndarray, ], opponent_policy: torch.nn.Module) -> Dict[int, List[int, ]]:
        buffer = RolloutBuffer(self.rollout_length)

        adv_obs = adv_obs = torch.tensor(np.concatenate([init_obs[k][len(init_obs[0]) - 1] for k in range(len(init_obs))]), dtype=torch.float32)
        buffer.obs_buffer.extend(adv_obs)
        
        team_obs = torch.tensor(torch.concatenate([init_obs[k][0] for k in range(len(init_obs))]), dtype=torch.float32)

        for t in range(self.rollout_length): # "fixed length trajectory segments (PPO)"
            
            actions = [defaultdict(lambda: np.ndarray((self.n_envs, ), dtype=np.int32)) for _ in range(self.n_envs)]

            adv_action, log_prob = self.policies[0].get_action(adv_obs)
            team_action, _ = opponent_policy.get_actions(team_obs)
            team_translated = torch.tensor(opponent_policy.action_map)[team_action.to(torch.int32)]

            actions = []
            for k in range(self.n_envs):
                action = {}
                for i in range(len(team_translated[k])):
                    action[i] = team_translated[k][i].item()
                action[i+1] = adv_action[k]
                actions.append(action)
            # print(actions)
            # while True:
            #     exec(input())
            buffer.action_buffer.extend(adv_action.tolist())
            buffer.log_prob_buffer.extend(log_prob)

            self.env.step_async(actions)
            obs, reward, done, _ = self.env.step_wait() # removed trunc

            adv_obs = torch.tensor(np.concatenate([obs[k][len(obs[0]) - 1] for k in range(len(obs))]), dtype=torch.float32)
            buffer.obs_buffer.extend(adv_obs)
            buffer.reward_buffer.extend([reward[k][len(reward[0]) - 1] for k in range(len(reward))])

            team_obs = torch.tensor(np.concatenate([obs[k][0] for k in range(len(obs))]), dtype=torch.float32)

        self.calculate_adv([buffer], self.rollout_length)

        return buffer, obs