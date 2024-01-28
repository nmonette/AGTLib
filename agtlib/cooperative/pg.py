from typing import Iterable

import torch
import torch.nn as nn
import numpy as np

from .base import RLBase, PolicyNetwork
from gymnasium import register

class SoftmaxPolicy(nn.Module):
    def __init__(self, n_agents, n_actions, param_dims, lr= 0.01):
        super(SoftmaxPolicy, self).__init__()
        self.lr = lr

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.param_dims = param_dims

        # empty = torch.empty(*param_dims)
        # print(param_dims)
        empty = torch.zeros(param_dims)
        # print(empty)
        # nn.init.orthogonal_(empty)
        self.params = nn.Parameter(nn.Softmax()(empty), requires_grad=True)
        print("Running...")
        

    def forward(self, x):
        # return 0
        # print(x)
        ret = self.params[x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], :, :]
        # print(ret.shape)
        return ret

    def get_actions(self, x):
        dist = torch.distributions.Categorical(self.forward(x)) # make categorical distribution and then decode the action index
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # print(action[0])
        return action, log_prob
    
    def step(self, loss):
        loss.backward(inputs=(self.params,)) # inputs=(self.params,)

        x = self.params - self.lr * self.params.grad
        self.params.grad.zero_()
        self.params.data = nn.Softmax()(x)

class MAPolicyNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int, action_map: [[int, ]], hl_dims: Iterable[int, ] = [64, 128]) -> None:
        """
        Parameters
        ----------
        obs_size: int
            The length of the flattened observation of the agent(s). 
        action_size: int
            The cardinality of the action space of a single agent. 
        hl_dims: Iterable(int), optional
            An iterable such that the ith element represents the width of the ith hidden layer. 
            Defaults to `[64,128]`. Note that this does not include the input or output layers.
        """
        super(MAPolicyNetwork, self).__init__()
        prev_dim = obs_size 
        hl_dims.append(action_size)
        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]

        self.action_map = action_map
        
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
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        for i in range(len(self.layers) - 1):
            x = torch.nn.ReLU()(self.layers[i](x))
        return self.layers[-1](x)

    def get_actions(self, x: torch.Tensor) -> int:
        """
        Samples an action from the current policy and returns it as an integer index.
        Parameters
        ----------
        x: torch.Tensor
            The flattened observation of the agent(s).
        Returns
        -------
        int
            The integer index of the action samples from the policy.
        float
            The log probability of the returned action with reference to the current policy.
        """

        dist = torch.distributions.Categorical(logits=self.forward(x))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return self.action_map[action], log_prob
        
class GDmax:
    def __init__(self, obs_size, action_size, env, param_dims, hl_dims=[64,128], team_size: int = 2, lr: float = 0.01, gamma:float = 0.9, n_rollouts:int = 100):
        self.obs_size = obs_size
        self.action_size = action_size
        self.team_size = team_size
        self.env = env()

        self.lr = lr
        self.gamma = gamma
        self.n_rollouts = n_rollouts
        self.num_steps = n_rollouts * 10
        
        self.adv_policy = PolicyNetwork(obs_size, action_size, hl_dims)
        self.adv_optimizer = torch.optim.Adam(self.adv_policy.parameters(), lr=lr)
        self.team_policy = SoftmaxPolicy(2, 4, param_dims) 

        self.episode_avg_adv_rewards = []
        self.adv_loss = []
        self.episode_avg_team_rewards = []

    def rollout(self, adversary=True):
        """
        Rollout to calculate loss
        """
        log_probs = []
        rewards = []

        env = self.env # ()
        for episode in range(self.n_rollouts):
            obs, _ = env.reset()
            ep_log_probs = []
            ep_rewards = []
            while True:
                team_action, team_log_prob = self.team_policy.get_actions(obs[0])
                action = {}
                for i in range(self.team_size):
                    action[i] = team_action[i]
                action[i+1], adv_log_prob = self.adv_policy.get_action(torch.tensor(obs[0]).float())
                obs, reward, done, trunc, _ = env.step(action) 
                if adversary:
                    ep_log_probs.append(adv_log_prob)
                    ep_rewards.append(reward[len(reward) - 1])
                else:
                    ep_log_probs.append(torch.sum(team_log_prob))
                    ep_rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break # >= 2 comes from 2 terminal states in treasure hunt
            
            log_probs.append(ep_log_probs)
            rewards.append(ep_rewards)

        return self.calculate_loss(log_probs, rewards)
        
    def get_utility(self, calc_logs=True):
        """
        Finds utility with both strategies fixed, averaged across 
        """
        for episode in range(self.n_rollouts):
            obs, _ = self.env.reset()
            log_prob_rewards = []
            rewards = []

            while True:
                team_action, team_log_prob = self.team_policy.get_actions(obs[0])
                action = {}
                
                for i in range(self.team_size):
                    action[i] = team_action[i]
                action[i+1], adv_log_prob = self.adv_policy.get_action(torch.tensor(obs[0]).float())
                obs, reward, done, trunc, _ = self.env.step(action) 
                
                if calc_logs:
                    log_prob_rewards.append(reward[0] * (adv_log_prob.detach() + torch.sum(team_log_prob)))
                rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break # >= 2 comes from 2 terminal states in treasure hunt

            if calc_logs:
                # avg_utility = torch.mean(torch.tensor(log_rewards), requires_grad = True)
                avg_utility = sum(log_prob_rewards) / len(log_prob_rewards)
                # expected_utility = self.team_policy.params.flatten().T @ self.reward_table.flatten().float()
                return avg_utility
            else:
                return -torch.mean(torch.tensor(rewards).float()).item(), torch.mean(torch.tensor(rewards).float()).item()
        
            


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

    def step(self):
        for _ in range(self.n_rollouts * 2): # self.num_steps
            total_loss = self.rollout()

            self.adv_optimizer.zero_grad()
            total_loss.backward()
            self.adv_optimizer.step()
            
        team_loss = self.rollout(adversary=False) # ray.get(self.rollout.remote(self, adversary=False))
        
        self.team_policy.step(team_loss)
        adv_utility, team_utility = self.get_utility(calc_logs=False)

        self.episode_avg_adv_rewards.append(adv_utility)
        self.episode_avg_team_rewards.append(team_utility)

class NGDmax(GDmax):
    """
    Neural GDMax, i.e. the team plays
    with a neural network instead of 
    direct parameteriation.
    """
    def __init__(self, obs_size, action_size, env, param_dims, hl_dims=[64,128], team_size: int = 2, lr: float = 0.01, gamma:float = 0.9, n_rollouts:int = 100):
        super().__init__(obs_size, action_size, env, param_dims, hl_dims, team_size, lr, gamma, n_rollouts)
        self.team_policy = MAPolicyNetwork(15, 16, [(i,j) for i in range(4) for j in range(4)])
        self.team_optimizer = torch.optim.Adam(self.team_policy.parameters(), lr=lr)

    def step(self):
        for _ in range(self.n_rollouts * 2): # self.num_steps
            adv_loss = self.rollout()

            self.adv_optimizer.zero_grad()
            adv_loss.backward()
            self.adv_optimizer.step()
            
        team_loss = self.rollout(adversary=False) # ray.get(self.rollout.remote(self, adversary=False))
        
        self.team_optimizer.zero_grad()
        team_loss.backward()
        self.team_optimizer.step()
        adv_utility, team_utility = self.get_utility(calc_logs=False)

        self.episode_avg_adv_rewards.append(adv_utility)
        self.episode_avg_team_rewards.append(team_utility)

class LGDmax:
    """
    GDMax optimization algorithm
    that is working on 
    """