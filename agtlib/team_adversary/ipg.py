import torch
import torch.nn as nn

from agtlib.utils.projection import projection_simplex_truncated
from agtlib.common.base import IndependentDirectPolicy

class TruncDirectPolicy(nn.Module):
    def __init__(self, n_actions, param_dims, lr=0.01, eps=0.1):
        super(TruncDirectPolicy, self).__init__()
        self.lr = lr
        self.eps = eps

        self.n_actions = n_actions
        self.param_dims = param_dims

        uniform = torch.ones(*param_dims) * (1 / n_actions)
        self.params = nn.Parameter(uniform, requires_grad=True)
        
    def forward(self, x):
        return self.params[*x.int(), :]
    
    def get_action(self, x):
        dist = torch.distributions.Categorical(logits=self.__call__(x)) # make categorical distribution and then decode the action index
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob
    
    def step(self, loss):
        loss.backward(inputs=(self.params,)) # inputs=(self.params,)

        self.params.data = projection_simplex_truncated(self.params - self.lr * self.params.grad, self.eps)
        self.params.grad.zero_()

class IPGDmax:
    def __init__(self, env, num_actions, num_agents, team_lr, adv_lr, gamma, param_dims, br, eps, num_rollouts, nu):
        self.lr = adv_lr
        self.gamma = gamma
        self.nu = nu # lambda coefficient for regularizing reward

        self.br = br
        self.eps = eps
        self.param_dims = param_dims

        self.num_agents = num_agents
        self.num_rollouts = num_rollouts

        self.team_args = (num_agents - 1, num_actions, param_dims, team_lr)
        self.adv_args = (num_actions, param_dims, adv_lr, eps)

        self.team_policy = IndependentDirectPolicy(*self.team_args)
        self.adv_policy = TruncDirectPolicy(*self.adv_args)

        self.env = env

        self.nash_gap = []

    def update_adv(self, policy):
        log_probs = []
        lambda_ = torch.zeros(self.param_dims)
        rewards = []
        actions = []
        states = []
        dones = []

        for episode in range(self.num_rollouts):
            episode_log_probs = []
            episode_lambda = torch.zeros(self.param_dims)
            episode_rewards = []
            episode_actions = []
            episode_states = []

            obs, _ = self.env.reset()
            gamma = 1
            t = 0
            while True:
                team_obs1 = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
                team_obs2 = torch.tensor(obs[1], device="cpu", dtype=torch.float32)
                adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
                team_action, _ = self.team_policy.get_action([team_obs1, team_obs2])
                action = {}
                for i in range(len(team_action)):
                    action[i] = team_action[i]
                adv_action, adv_log_prob = policy.get_action(adv_obs)
                action[i+1] = adv_action.item()
                obs, reward, done, trunc, _ = self.env.step(action) 

                episode_log_probs.append(adv_log_prob)
                episode_rewards.append(reward[len(obs) - 1])
                episode_actions.append(adv_action)
                episode_states.append(adv_obs.int())
                step_lambda = torch.zeros_like(episode_lambda, device="mps")
                step_lambda[*obs[len(obs) - 1].astype(int), adv_action] = gamma
                gamma *= self.gamma

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break

                t += 1

            log_probs.append(episode_log_probs)
            lambda_ += episode_lambda
            rewards.append(torch.tensor(episode_rewards, device="cpu"))
            actions.append(episode_actions)
            states.append(episode_states)
            dones.append(t)
            
        
        lambda_ /= self.num_rollouts

        returns = []
        for i in range(len(dones)):
            start_return = (rewards[i][-1] - self.nu * lambda_[*states[i][-1], actions[i][-1]]) * sum(log_probs[i])
            for j in range(2, dones[i]+1):
                start_return += (rewards[i][-j] - self.nu * lambda_[*states[i][-j], actions[i][-j]]) * sum(log_probs[i][:-j])
                
            returns.append(start_return)

        loss = -torch.stack(returns).mean()
        policy.step(loss)

    def update_team(self):
        log_probs = []
        rewards = []
        env = self.env

        obs, _ = env.reset()
        while True:
            team_obs1 = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
            team_obs2 = torch.tensor(obs[1], device="cpu", dtype=torch.float32)
            adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
            team_action, team_log_prob = self.team_policy.get_action([team_obs1, team_obs2])
            action = {}
            for i in range(len(team_action)):
                action[i] = team_action[i]
            adv_action, _ = self.adv_policy.get_action(adv_obs)
            action[i+1] = adv_action.item()
            obs, reward, done, trunc, _ = env.step(action) 
            log_probs.append(team_log_prob)
            rewards.append(reward[0])

            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

        returns = [rewards[-1]]
        for i in range(2, len(rewards)+1):
            returns.append(self.gamma * returns[-1] + rewards[-i])

        log_prob_data = torch.stack(log_probs)
        return_data = torch.tensor(returns, requires_grad=False, dtype=torch.float32, device="cpu").flip(-1)

        loss1 = -torch.dot(log_prob_data[:, 0].flatten(), return_data) / len(returns)
        loss2 = -torch.dot(log_prob_data[:, 1].flatten(), return_data.clone()) / len(returns)

        self.team_policy.step([loss1, loss2])

    def single_update(self, policy, policy_idx):
        log_probs = []
        rewards = []
        env = self.env

        obs, _ = env.reset()
        while True:
            team_obs1 = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
            team_obs2 = torch.tensor(obs[1], device="cpu", dtype=torch.float32)
            adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
            team_action, team_log_prob = self.team_policy.get_action([team_obs1, team_obs2])
            action = {}
            for i in range(len(team_action)):
                action[i] = team_action[i]
            adv_action, adv_log_prob = self.adv_policy.get_action(adv_obs)
            action[i+1] = adv_action.item()
            action[policy_idx], policy_log_prob = policy.get_action(torch.tensor(obs[policy_idx], device="cpu", dtype=torch.float32))
            obs, reward, done, trunc, _ = env.step(action) 
            
            log_probs.append(policy_log_prob)
            rewards.append(reward[policy_idx])

            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

        returns = [rewards[-1]]
        for i in range(2, len(rewards)+1):
            returns.append(self.gamma * returns[-1] + rewards[-i])

        log_prob_data = torch.stack(log_probs)
        return_data = torch.tensor(returns, requires_grad=False, dtype=torch.float32, device="cpu").flip(-1)

        loss = -torch.dot(log_prob_data, return_data) / len(returns)

        policy.step(loss)


    def get_utility(self, team_policy=None, adv_policy=None):
        team_rewards = []
        adv_rewards = []
        if team_policy is None:
            team_policy = self.team_policy
        if adv_policy is None:
            adv_policy = self.adv_policy

        for i in range(self.num_rollouts):
            env = self.env
            obs, _ = env.reset()
            temp_adv_rewards = []
            temp_team_rewards = []
            while True:
                team_obs1 = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
                team_obs2 = torch.tensor(obs[1], device="cpu", dtype=torch.float32)
                adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
                team_action, team_log_prob = team_policy.get_action([team_obs1, team_obs2])
                action = {}
                for i in range(len(team_action)):
                    action[i] = team_action[i]
                adv_action, adv_log_prob = adv_policy.get_action(adv_obs)
                action[i+1] = adv_action.item()
                
                obs, reward, done, trunc, _ = env.step(action)

                temp_adv_rewards.append(reward[len(reward) - 1])
                temp_team_rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    adv_rewards.append(sum(temp_adv_rewards))
                    team_rewards.append(sum(temp_team_rewards))
                    break
        
        return torch.mean(torch.tensor(adv_rewards), dtype=torch.float32), torch.mean(torch.tensor(team_rewards, dtype=torch.float32))
    
    def get_adv_br(self):
        temp_adv = TruncDirectPolicy(*self.adv_args)
        temp_adv.load_state_dict(self.adv_policy.state_dict())

        for _ in range(self.br):
            self.update_adv(temp_adv)

        return self.get_utility(adv_policy=temp_adv)[0]
    

    def get_team_br(self, policy_idx):
        temp_team = IndependentDirectPolicy(*self.team_args)
        temp_team.load_state_dict(self.team_policy.state_dict())

        for _ in range(self.br):
            self.single_update(temp_team.policies[policy_idx], policy_idx)

        return self.get_utility(team_policy=temp_team)[1]
    
    def step(self):
        """
        GDmax training step with no metrics.
        """
        for _ in range(self.br):
            self.update_adv(self.adv_policy)

        self.update_team()

    def step_with_gap(self):
        for _ in range(self.br):
            self.update_adv(self.adv_policy)

        self.update_team()

        adv_base, team_base = self.get_utility()

        adv_br = self.get_adv_br()
        team_br = torch.zeros((len(self.team_policy.policies)))
        
        for i in range(len(team_br)):
            team_br[i] = self.get_team_br(i)

        team_diff = torch.max(team_br - team_base)

        self.nash_gap.append(max(adv_br.item() - adv_base.item(), team_diff.item()))



        



                