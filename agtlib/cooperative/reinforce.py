import torch

from .pg import SoftmaxPolicy, MAPolicyNetwork
from .base import PolicyNetwork
from agtlib.utils.stable_baselines.vec_env.subproc_vec_env import SubprocVecEnv


class GDmax:
    def __init__(self, obs_size, action_size, env, param_dims, hl_dims=[64,128], lr: float = 0.01, gamma:float = 0.9, rollout_length:int = 50):
        self.obs_size = obs_size
        self.action_size = action_size
        self.env = env() # used to be env()

        self.lr = lr
        self.gamma = gamma
        self.rollout_length = rollout_length
        
        self.adv_policy = PolicyNetwork(obs_size, action_size, hl_dims)
        # self.adv_policy.load_state_dict(torch.load("/Users/phillip/projects/AGTLib/output/experiment-40/end-3x3-adv-policy-n-reinforce.pt"))
        self.adv_optimizer = torch.optim.Adam(self.adv_policy.parameters(), lr=lr, maximize=True)
        self.param_dims = param_dims
        if param_dims is not None:
            self.team_policy = SoftmaxPolicy(2, 4, param_dims, lr, [(i,j) for i in range(self.action_size) for j in range(self.action_size)]) 

        self.nash_gap = []
        self.team_utility = []

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
                log_probs.append(team_log_prob)
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
            self.adv_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.adv_optimizer.step()

        else:
            team_policy.step(loss)

    def get_utility(self, team_policy=None, adv_policy=None):
        team_rewards = []
        adv_rewards = []
        if team_policy is None:
            team_policy = self.team_policy
        if adv_policy is None:
            adv_policy = self.adv_policy

        for i in range(self.rollout_length):
            env = self.env
            obs, _ = env.reset()
            while True:
                team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
                adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
                team_action, team_log_prob = team_policy.get_actions(team_obs)
                action = {}
                for i in range(len(team_action)):
                    action[i] = team_action[i]
                action[i+1], adv_log_prob = adv_policy.get_action(adv_obs)
                action[i+1] = action[i+1].item()
                
                obs, reward, done, trunc, _ = env.step(action)

                adv_rewards.append(reward[len(reward) - 1])
                team_rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break
        
        return -torch.mean(torch.tensor(team_rewards), dtype=torch.float32), torch.mean(torch.tensor(team_rewards, dtype=torch.float32))
    
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
        # base_adv, base_team = self.get_utility()
        gap_adv = self.get_adv_gap()

        for i in range(self.rollout_length):
            self.update()

        gap_team = self.get_utility()[1]

        self.update(adversary=False)
        
        # self.nash_gap.append(max(gap_adv - base_adv, gap_team - base_team).item())
        self.nash_gap.append(gap_adv + gap_team)

class NGDmax(GDmax):

    def __init__(self, obs_size, action_size, env, hl_dims=[64,128], lr: float = 0.01, gamma:float = 0.9, rollout_length:int = 50, batch_size:int =32, epochs=100, seg_length = 12, num_threads = 10):
        super().__init__(obs_size, action_size, env, None, hl_dims, lr, gamma, rollout_length)
        self.team_policy = MAPolicyNetwork(15, 16, [(i,j) for i in range(4) for j in range(4)])
        # self.team_policy.load_state_dict(torch.load("/Users/phillip/projects/AGTLib/output/experiment-40/end-3x3-team-policy-n-reinforce.pt"))
        self.team_optimizer = torch.optim.Adam(self.team_policy.parameters(), lr=lr, maximize=True)

        self.batch_size = batch_size
        self.epochs = epochs

        self.seg_length = seg_length
        self.num_threads = num_threads

    def update(self, adversary=True, team_policy=None, team_optimizer=None, adv_policy=None, adv_optimizer=None, rollout_length=None):
        if rollout_length is None:
            rollout_length = self.rollout_length

        if team_policy is None:
            team_policy = self.team_policy
        
        if team_optimizer is None:
            team_optimizer = self.team_optimizer

        if adv_policy is None:
            adv_policy = self.adv_policy
        
        if adv_optimizer is None:
            adv_optimizer = self.adv_optimizer

        log_probs = []
        return_data = []

        env = self.env

        for episode in range(rollout_length):
            rewards = []
            obs, _ = env.reset()
            while True:
                team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
                adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
                team_action, team_log_prob = team_policy.get_actions(team_obs)
                action = {}
                for i in range(len(team_action)):
                    action[i] = team_action[i]
                action[i+1], adv_log_prob = adv_policy.get_action(adv_obs)
                action[i+1] = action[i+1].item()
                obs, reward, done, trunc, _ = env.step(action) 
                if adversary:
                    log_probs.append(adv_log_prob)
                    rewards.append(reward[len(reward) - 1])
                else:
                    log_probs.append(team_log_prob)
                    rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break

            returns = [rewards[-1]]
            for i in range(2, len(rewards)+1):
                returns.append(self.gamma * returns[-1] + rewards[-i])

            return_data.extend(returns)

        perm  = torch.randperm(len(log_probs))
        log_probs = torch.stack(log_probs)[perm]
        returns = torch.tensor(return_data, device="cpu")[perm]

        policy = adv_policy if adversary else team_policy
        optimizer = adv_optimizer if adversary else team_optimizer

        policy = policy.to("mps")

        for epoch in range(self.epochs):
            for batch in range(0, len(log_probs), self.batch_size):
                batch_log_probs = log_probs[batch:batch+self.batch_size].to("mps")
                batch_returns = returns[batch:batch+self.batch_size].to("mps")
                loss = torch.dot(batch_log_probs, batch_returns)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        policy = policy.to("cpu")

    def get_adv_br(self):
        temp_adv = PolicyNetwork(self.obs_size, self.action_size, hl_dims=[64,128])
        temp_adv.load_state_dict(self.adv_policy.state_dict())
        temp_optimizer = torch.optim.Adam(temp_adv.parameters(), lr=self.lr, maximize=True)

        self.update(adversary=True, adv_policy=temp_adv, adv_optimizer=temp_optimizer)

        return self.get_utility(adv_policy=None)

    def get_team_br(self):
        temp_team = MAPolicyNetwork(self.obs_size, self.action_size*self.action_size, [(i,j) for i in range(4) for j in range(4)], hl_dims=[64,128])
        temp_team.load_state_dict(self.team_policy.state_dict())
        temp_optimizer = torch.optim.Adam(temp_team.parameters(), lr=self.lr, maximize=True)

        self.update(adversary=False, team_policy=temp_team, team_optimizer=temp_optimizer)

        return self.get_utility(team_policy=temp_team)[1]

    def step(self):
        self.update()

        self.update(adversary=False, rollout_length=1)
    
    def step_with_gap(self):
        self.update()
        
        self.update(adversary=False, rollout_length=1)

        adv_base, team_base = self.get_utility()

        adv_br, team_adv_br = self.get_adv_br()

        self.team_utility.append(team_adv_br)
        self.nash_gap.append(max(adv_br.item() - adv_base.item(), self.get_team_br().item() - team_base.item()))

        # team plays some x, assume adversary plays best response, then print utility of team when adversary is giving best response

        # one treasure in top right, one treasure in bottom left, other two corners put the team, put the adversary in the center
        # make it so 