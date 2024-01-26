from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import numpy as np
import ray

from .base import RLBase, PolicyNetwork
from multigrid.multigrid.envs.team_empty import TeamEmptyEnv
import multigrid
from gymnasium import register

CONFIGURATIONS = {
        'MultiGrid-Empty-8x8-Team': (TeamEmptyEnv, {'size': 5, "agents": 3, "allow_agent_overlap":True, "max_steps":20}),
        'MultiGrid-Empty-6x6-Team': (TeamEmptyEnv, {'size': 8, "agents": 3, "allow_agent_overlap":True, "max_steps":60}),
        'MultiGrid-Empty-4x4-Team': (TeamEmptyEnv, {'size': 6, "agents": 3, "allow_agent_overlap":True, "max_steps":40}),
        'MultiGrid-Empty-3x3-Team': (TeamEmptyEnv, {'size': 5, "agents": 3, "allow_agent_overlap":True, "max_steps":20})
    }

for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)

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
    
    def step(self, loss):
        loss.backward(inputs=(self.params,)) # inputs=(self.params,)

        x = self.params - self.lr * self.params.grad
        self.params.grad.zero_()
        self.params.data = nn.Softmax()(x)


        
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

    # @ray.remote(num_returns=1)
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
        for _ in range(self.n_rollouts): # self.num_steps
            # NON REMOTE: (WORKING)
            # adv_log_probs, adv_rewards = self.rollout()
            # adv_loss = self.calculate_loss(adv_log_probs, adv_rewards)

            # REMOTE: (NOT WORKING)
            # adv_loss = []
            # for i in range(32):
            #    adv_loss.append(self.rollout.remote(self))
            
            # adv_loss = ray.get(adv_loss)
            # total_loss = torch.mean(torch.stack(adv_loss))

            total_loss = self.rollout()

            self.adv_optimizer.zero_grad()
            # adv_loss.backward()
            total_loss.backward()
            self.adv_optimizer.step()

        # team_log_probs, team_rewards = self.rollout(adversary=False)
        # team_loss = self.calculate_loss(team_log_probs, team_rewards)
            
        team_loss = self.rollout(adversary=False) # ray.get(self.rollout.remote(self, adversary=False))
        
        self.team_policy.step(team_loss)
        adv_utility, team_utility = self.get_utility(calc_logs=False)

        self.episode_avg_adv_rewards.append(adv_utility)
        self.episode_avg_team_rewards.append(team_utility)
