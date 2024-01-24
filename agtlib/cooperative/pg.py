import torch
import torch.nn as nn
import numpy as np

from .base import RLBase, PolicyNetwork

class SoftmaxPolicy(nn.Module):
    def __init__(self, n_agents, n_actions, param_dims, lr= 0.01):
        super(SoftmaxPolicy, self).__init__()
        self.lr = lr

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.param_dims = param_dims

        empty = torch.empty(*param_dims)
        nn.init.orthogonal_(empty)
        self.params = nn.Parameter(nn.Softmax()(empty), requires_grad=True)
        

    def forward(self, x):
        return self.params[*x, :, :]

    def get_actions(self, x):
        dist = torch.distributions.Categorical(self.forward(x)) # make categorical distribution and then decode the action index
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def step(self, utility):
        loss = -utility
        loss.backward(inputs=(self.params,)) # inputs=(self.params,)

        x = self.params - self.lr * self.params.grad
        self.params.grad.zero_()
        self.params.data = nn.Softmax()(x)


        
class GDmax:
    def __init__(self, obs_size, action_size, env, param_dims, hl_dims=[64,128], team_size: int = 2, lr: float = 0.01, gamma:float = 0.99, n_rollouts:int = 100):
        self.obs_size = obs_size
        self.action_size = action_size
        self.team_size = team_size
        self.env = env

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

    def rollout(self):
        """
        Rollout to calculate the adversary's loss
        """
        adv_log_probs = []
        adv_rewards = []

        for episode in range(self.n_rollouts):
            obs, _ = self.env.reset()
            ep_log_probs = []
            ep_adv_rewards = []
            while True:
                team_action, _ = self.team_policy.get_actions(obs[0])
                action = {}
                for i in range(self.team_size):
                    action[i] = team_action[i]
                action[i+1], log_prob = self.adv_policy.get_action(torch.tensor(obs[0]).float())
                obs, reward, done, trunc, _ = self.env.step(action) 

                ep_log_probs.append(log_prob)
                ep_adv_rewards.append(reward[len(reward) - 1])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break # >= 2 comes from 2 terminal states in treasure hunt
            
            adv_log_probs.append(ep_log_probs)
            adv_rewards.append(ep_adv_rewards)

            return adv_log_probs, adv_rewards
        
    def get_utility(self, calc_avg=True):
        """
        Finds utility with both strategies fixed, averaged across 
        """
        for episode in range(self.num_steps):
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
                
                if calc_avg:
                    log_prob_rewards.append(reward[0] * (adv_log_prob.detach() + torch.sum(team_log_prob)))
                rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break # >= 2 comes from 2 terminal states in treasure hunt
            self.episode_avg_adv_rewards.append(-torch.mean(torch.tensor(rewards)).item())
            self.episode_avg_team_rewards.append(torch.mean(torch.tensor(rewards)).item())
            if calc_avg:
                # avg_utility = torch.mean(torch.tensor(log_rewards), requires_grad = True)
                avg_utility = sum(log_prob_rewards) / len(log_prob_rewards)
                # expected_utility = self.team_policy.params.flatten().T @ self.reward_table.flatten().float()
                return avg_utility
            else:
                return rewards
        
            


    def calculate_loss(self, log_probs, rewards):
        left = []
        right = []
        for i in range(len(rewards)):
            gamma = 1
            disc_reward = 0
            log_prob = 0
            for j in range(len(rewards[i])):
                disc_reward += gamma * rewards[i][j] 
                log_prob += log_probs[i][j]
                gamma *= self.gamma

            left.append(disc_reward)
            right.append(log_prob)

        disc_reward = torch.tensor(left)
        log_probs = torch.tensor(right, requires_grad=True)

        return torch.mean(disc_reward * log_probs)

    def step(self):
        for _ in range(self.num_steps):
            adv_log_probs, adv_rewards = self.rollout()
            adv_loss = self.calculate_loss(adv_log_probs, adv_rewards)

            self.adv_optimizer.zero_grad()
            adv_loss.backward()
            self.adv_optimizer.step()
        
        self.team_policy.step(self.get_utility())
