"""
We need to figure out which feature mapping
gets optimized with which network... oops
"""
from typing import Iterable

import torch
import torch.nn as nn
import numpy as np

from .base import RLBase, PolicyNetwork
from .pg import MAPolicyNetwork

class TwoHeadPolicy(nn.Module):
    """
    Policy Network that can output either a set of features or a policy
    given a state. 

    fm stands for feature mapping
    """
    def __init__(self, obs_size:int, action_size:int, hl_dims:Iterable[int, ]=[64,128], fm_dim1:int =64, fm_dim2:int=128):
        super(TwoHeadPolicy, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size

        # self.state_fm = nn.Linear(obs_size, fm_dim1)
        self.reward_fm = nn.Linear(obs_size + 2, fm_dim2)
        self.state_action_fm = nn.Linear(obs_size + 1, fm_dim2)

        self.init_head = nn.Linear(hl_dims[-1], fm_dim1)
        self.t_head1 = nn.Linear(hl_dims[-1], action_size)
        self.t_head2 = nn.Softmax(-1)

        prev_dim = obs_size
        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]

    # def state_mapping(self, x):
    #     """
    #     Returns the feature mapping of the given observation `x`.
    #     """
    #     return self.state_fm(x)
    
    def state_action_mapping(self, obs, action):
        """
        Returns the feature mapping of the given observation
        and action $\phi_{s,a}: S,A \to \mathbb{R}^{d'} 
        """
        x = torch.cat((obs, action.reshape(1)))
        return self.state_action_fm(x)
    
    def reward_mapping(self, obs, action, reward):
        """
        Returns the feature mapping of the given observation, 
        action, and reward $\phi_{r}: S,A,r \to \mathbb{R}^{d'}$
        """
        x = torch.cat((obs, action.reshape(1), reward))
        return self.reward_fm(x)
    
    def forward_t(self, x):
        # x = self.state_mapping(x)
        for i in range(len(self.layers)):
            x = torch.nn.ReLU()(self.layers[i](x))
        return self.t_head2(self.t_head1(x))

    def forward_init(self, x):
        # x = self.state_mappping(x)
        for i in range(len(self.layers)):
            x = torch.nn.ReLU()(self.layers[i](x))
        return self.init_head(x)
    
    def get_action(self, x):
        dist = torch.distributions.Categorical(self.forward_t(x))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
class LambdaNetwork(nn.Module):
    """
    Network to approximate lambda given team policy x
    and initial state s_0.
    """
    def __init__(self, input_size, ouptut_dim:int=64, hl_dims:Iterable[int, ]=[64,128]):
        super(LambdaNetwork, self).__init__()
        self.input_size = input_size
        
        prev_dim = input_size
        hl_dims.append(ouptut_dim)
        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the neural network.
        Parameters
        ----------
        x: torch.Tensor
            The flattened observation of the agent(s).

        Returns
        -------
        torch.Tensor
            Probability vector containing the action-probabilities.
        """
        for i in range(len(self.layers) - 1):
            x = torch.nn.ReLU()(self.layers[i](x))
        return self.layers[-1](x)
    
class NLGDmax:
    """
    Lambda GDmax with Neural Networks... lots of them.
    """
    def __init__(self, obs_size, action_size, action_map, env, gamma=0.9, lr=0.1, rollout_length=50, fm_dim1=64, fm_dim2=128):
        self.obs_size = obs_size
        self.action_size = action_size
        self.action_map = action_map
        
        self.env = env()

        self.gamma = gamma
        self.lr = lr
        self.rollout_length = rollout_length
        
        self.fm_dim1 = fm_dim1
        self.fm_dim2 = fm_dim2

        self.lambda_network = LambdaNetwork(fm_dim1, fm_dim2)
        self.team_policy = MAPolicyNetwork(obs_size, action_size*action_size, action_map=self.action_map)
        self.adv_policy = TwoHeadPolicy(obs_size, action_size, fm_dim1=fm_dim1, fm_dim2=fm_dim2)

        self.lambda_optimizer = torch.optim.Adam(self.lambda_network.parameters(), lr=lr)
        self.team_optimizer = torch.optim.Adam(self.team_policy.parameters(), lr=lr)
        self.adv_optimizer = torch.optim.Adam(self.adv_policy.parameters(), lr=lr)

        # Metric for tracking progress
        self.reward = []
    
    def find_lambda(self):
        obs_data = []
        init_obs = []
        reward_data = []
        for _ in range(self.rollout_length):
            gamma = 1
            lambda_ = torch.zeros((self.fm_dim2, ))
            reward_vec = torch.zeros((self.fm_dim2,))
            obs, _ = self.env.reset()
            init_obs.append(torch.tensor(obs[len(obs)-1]).float())

            rewards = []
            while True:
                team_action, team_log_prob = self.team_policy.get_actions(torch.tensor(obs[0]).float())
                action = {}
                for i in range(len(team_action)):
                    action[i] = team_action[i]
                action[i+1], adv_log_prob = self.adv_policy.get_action(torch.tensor(obs[len(obs)-1]).float())
                obs, reward, done, trunc, _ = self.env.step(action) 

                rewards.append(reward[0])

                obs_features = self.adv_policy.state_action_mapping(torch.tensor(obs[len(obs)-1]).float(), action[len(action)-1])
                reward_features = self.adv_policy.reward_mapping(torch.tensor(obs[len(obs)-1]).float(), action[len(action)-1], torch.tensor(reward[len(reward)-1]).reshape(1))

                lambda_ += gamma * obs_features
                reward_vec += gamma * reward_features

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break

                gamma *= self.gamma

            obs_data.append(lambda_)
            reward_data.append(reward_vec)

            self.reward.append(sum(rewards) / len(rewards))

        lambda_data = torch.stack(obs_data)
        init_data = torch.stack(init_obs)

        policy_features = self.adv_policy.forward_init(init_data)
        predictions = self.lambda_network.forward(policy_features)

        
        self.lambda_optimizer.zero_grad()
        loss = nn.MSELoss()(predictions, lambda_data)
        loss.backward()
        self.lambda_optimizer.step()

        return reward_data, init_data

    def calculate_loss(self, log_probs, rewards):
        left = []
        right = []
        for i in range(len(rewards)):
            gamma = 1
            disc_reward = 0 
            log_prob = sum(log_probs[i])
            for j in range(len(rewards[i])): # we may need to do this sum with torch to allow for backpropagtion?
                disc_reward += gamma * rewards[i][j] 
                gamma *= self.gamma

            left.append(disc_reward)
            right.append(log_prob)

        disc_reward = torch.tensor(left)
        log_probs = torch.stack(right)

        return -torch.mean(disc_reward * log_probs)

    def rollout(self):
        """
        Rollout to calculate loss
        """
        log_probs = []
        rewards = []

        env = self.env # ()
        for episode in range(self.rollout_length):
            obs, _ = env.reset()
            ep_log_probs = []
            ep_rewards = []
            while True:
                team_action, team_log_prob = self.team_policy.get_actions(obs[0])
                action = {}
                for i in range(len(team_action)):
                    action[i] = team_action[i]
                action[i+1], adv_log_prob = self.adv_policy.get_action(torch.tensor(obs[len(obs) - 1]).float())
                action[i+1] = action[i+1].item()
                obs, reward, done, trunc, _ = env.step(action) 
                
                ep_log_probs.append(torch.sum(team_log_prob))
                ep_rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break # >= 2 comes from 2 terminal states in treasure hunt
            
            log_probs.append(ep_log_probs)
            rewards.append(ep_rewards)

        return self.calculate_loss(log_probs, rewards)
    
    def step(self):
        reward_vec, init_data = self.find_lambda()
        for i in range(self.rollout_length):
            self.adv_optimizer.zero_grad()
            policy_features = self.adv_policy.forward_init(init_data[i])
            loss = -torch.dot(self.lambda_network.forward(policy_features), reward_vec[i])
            loss.backward()
            self.adv_optimizer.step()
            
            self.team_optimizer.zero_grad()
            team_loss = self.rollout()
            team_loss.backward()
            self.team_optimizer.step()

        
            
        
        
        