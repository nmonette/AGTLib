import torch

from .pg import SoftmaxPolicy
from .base import PolicyNetwork

class GDmax:
    def __init__(self, obs_size, action_size, env, param_dims, hl_dims=[64,128], lr: float = 0.01, gamma:float = 0.9, rollout_length:int = 50):
        self.obs_size = obs_size
        self.action_size = action_size
        self.env = env()

        self.lr = lr
        self.gamma = gamma
        self.rollout_length = rollout_length
        
        self.adv_policy = PolicyNetwork(obs_size, action_size, hl_dims)
        self.adv_optimizer = torch.optim.Adam(self.adv_policy.parameters(), lr=lr, maximize=True)
        self.param_dims = param_dims
        if param_dims is not None:
            self.team_policy = SoftmaxPolicy(2, 4, param_dims, lr, [(i,j) for i in range(self.action_size) for j in range(self.action_size)]) 

        self.nash_gap = []

    def update(self, adversary=True, team_policy=None):
        log_probs = []
        rewards = []

        if team_policy is None:
            team_policy = self.team_policy

        env = self.env
        obs, _ = env.reset()
        while True:
            team_action, team_log_prob = team_policy.get_actions(obs[0])
            action = {}
            for i in range(len(team_action)):
                action[i] = team_action[i]
            action[i+1], adv_log_prob = self.adv_policy.get_action(torch.tensor(obs[len(obs) - 1]).float())
            action[i+1] = action[i+1].item()
            obs, reward, done, trunc, _ = env.step(action) 
            if adversary:
                log_probs.append(adv_log_prob)
                rewards.append(reward[len(reward) - 1])
            else:
                log_probs.append(torch.sum(team_log_prob))
                rewards.append(reward[0])

            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

        returns = [rewards[-1]]
        for i in range(2, len(rewards)+1):
            returns.append(self.gamma * returns[-1] + rewards[-i])

        log_probs = torch.stack(log_probs)
        returns = torch.tensor(returns, dtype=torch.float32).flip(-1)

        loss = torch.dot(log_probs, returns)

        if adversary:
            self.adv_optimizer.zero_grad()
            loss.backward()
            self.adv_optimizer.step()

        else:
            team_policy.step(loss)

    def get_utility(self, team_policy=None, adversary=True):
        team_rewards = []
        adv_rewards = []
        if team_policy is None:
            team_policy = self.team_policy

        for i in range(self.rollout_length):
            env = self.env
            obs, _ = env.reset()
            while True:
                team_action, team_log_prob = team_policy.get_actions(obs[0])
                action = {}
                for i in range(len(team_action)):
                    action[i] = team_action[i]
                action[i+1], adv_log_prob = self.adv_policy.get_action(torch.tensor(obs[len(obs) - 1]).float())
                action[i+1] = action[i+1].item()
                
                obs, reward, done, trunc, _ = env.step(action)

                adv_rewards.append(reward[len(reward) - 1])
                team_rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break
        
        return -torch.mean(torch.tensor(team_rewards), dtype=float), torch.mean(torch.tensor(team_rewards, dtype=float))
    
    def get_adv_gap(self):
        temp_team = SoftmaxPolicy(2, self.action_size, self.param_dims, self.lr, [(i,j) for i in range(self.action_size) for j in range(self.action_size)])
        temp_team.load_state_dict(self.team_policy.state_dict())

        for i in range(self.rollout_length):
            self.update(adversary=False, team_policy=temp_team)

        return self.get_utility(team_policy=temp_team)[0]

    def step(self):
        for i in range(self.rollout_length):
            self.update()

        self.update(adversary=False)

    def step_with_gap(self):
        base_adv, base_team = self.get_utility()
        gap_adv = self.get_adv_gap()

        for i in range(self.rollout_length):
            self.update()

        gap_team = self.get_utility()[1]

        self.update(adversary=False)
        
        self.nash_gap.append(max(gap_adv - base_adv, gap_team - base_team).item())