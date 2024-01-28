from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import numpy as np

from .base import RLBase, PolicyNetwork
from ..utils.rollout import MCBuffer
from ..utils.stable_baselines.vec_env.subproc_vec_env import SubprocVecEnv
from multigrid.multigrid.envs.team_empty import TeamEmptyEnv
import multigrid
from gymnasium import register

import warnings

class SoftmaxPolicy(nn.Module):
    def __init__(self, n_agents, n_actions, param_dims, action_map, lr= 0.01,):
        super(SoftmaxPolicy, self).__init__()
        self.lr = lr

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.param_dims = param_dims
        self.action_map = action_map

        empty = torch.empty(*param_dims)
        nn.init.orthogonal_(empty)
        self.params = nn.Parameter(nn.Softmax()(empty), requires_grad=True)
        

    def forward(self, x):
        return torch.stack([self.params[*x[i], :] for i in range(len(x))])
         

    def get_actions(self, x):
        dist = torch.distributions.Categorical(self.forward(x)) # make categorical distribution and then decode the action index
        action = dist.sample()
        log_prob = dist.log_prob(action)
        actions = [self.action_map[i] for i in action]
        return actions, log_prob
    
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
        self.env = env

        self.lr = lr
        self.gamma = gamma
        self.n_rollouts = n_rollouts
        self.num_steps = n_rollouts * 10

        self.rollout_env = SubprocVecEnv([self.env for _ in range(5)])
        
        self.adv_policy = PolicyNetwork(obs_size, action_size, hl_dims)
        self.adv_optimizer = torch.optim.Adam(self.adv_policy.parameters(), lr=lr)
        self.team_policy = SoftmaxPolicy(2, 4, param_dims,  [(i,j) for i in range(action_size) for j in range(action_size)]) 

        self.episode_avg_adv_rewards = []
        self.adv_loss = []
        self.episode_avg_team_rewards = []

    def rollout(self, env, adversary=True, n_envs = 1):
        """
        Rollout to calculate loss
        """
        complete = False
        buffer = MCBuffer(self.n_rollouts, n_envs, 2 if adversary else 0)
        obs = env.reset()
        while not complete:
            team_obs = torch.from_numpy(obs[0]).int()
            adv_obs = torch.from_numpy(obs[len(obs)-1]).float()

            team_action, team_log_prob = self.team_policy.get_actions(team_obs)
            adv_action, adv_log_prob = self.adv_policy.get_action(adv_obs)

            action = []
            for i in range(n_envs):
                action_dict = {}
                for j in range(2): # 2 is the team length for now -- hardcoded
                    action_dict[j] = team_action[i][j]
                action_dict[len(team_action[i])] = adv_action[i].int().item()
                action.append(action_dict)
            
            env.step_async(action)
            obs, reward, done, _ = env.step_wait() 
            
            done = [(list(i.values()).count(True) >= 2) for i in done]
            if adversary:
                complete = buffer.add(reward, adv_log_prob, done)
            else:
                complete = buffer.add(reward, team_log_prob, done)

        return self.calculate_loss(*buffer.get_data())
        
    def get_utility(self, calc_logs=True):
        """
        Finds utility with both strategies fixed, averaged across 
        """
        env = self.env()
        for episode in range(self.n_rollouts):
            obs, _ = env.reset()
            log_prob_rewards = []
            rewards = []

            while True:
                team_action, team_log_prob = self.team_policy.get_actions([obs[0]])
                action = {}
                for i in range(self.team_size):
                    action[i] = team_action[0][i]
                action[i+1], adv_log_prob = self.adv_policy.get_action(torch.tensor(obs[0]).float())
                obs, reward, done, trunc, _ = env.step(action) 
                
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

        x = torch.tensor(left)
        y = torch.stack(right)

        return -torch.mean(x * y)

    def step(self, n_envs):
        rollout_envs = self.rollout_env
        for _ in range(100): # self.num_steps
            total_loss = self.rollout(rollout_envs, n_envs)

            self.adv_optimizer.zero_grad()
            # adv_loss.backward()
            total_loss.backward()
            self.adv_optimizer.step()

        team_loss = self.rollout(rollout_envs, n_envs, adversary=False) # ray.get(self.rollout.remote(self, adversary=False))
        self.team_policy.step(team_loss)
        adv_utility, team_utility = self.get_utility(calc_logs=False)

        self.episode_avg_adv_rewards.append(adv_utility)
        self.episode_avg_team_rewards.append(team_utility)
