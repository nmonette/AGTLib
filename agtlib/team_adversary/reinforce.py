import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv


from .q import TabularQ
from agtlib.common.base import PolicyNetwork, SELUPolicy, SoftmaxPolicy, MAPolicyNetwork, SELUMAPolicy
# from agtlib.utils.stable_baselines.vec_env.subproc_vec_env import SubprocVecEnv

from agtlib.utils.rollout import GDmaxRollout


class GDmax:
    """
    One of the 'Algorithm' classes that can be used in the agtlib.runners.gdmax_experiments.train
    function. Operates by learning an adversary's best response policy, and then doing
    a single gradient step for the team playing against the adversary. 

    This version uses a SELUPolicy network for the adversary and SoftmaxPolicy for the team. 
    """
    def __init__(self, obs_size, action_size, env, param_dims, hl_dims=[64,128], lr: float = 0.01, gamma:float = 0.9, rollout_length:int = 50):
        self.obs_size = obs_size
        self.action_size = action_size
        self.env = env() # used to be env()

        self.lr = lr
        self.gamma = gamma
        self.rollout_length = rollout_length
        self.hl_dims = hl_dims
        
        self.adv_policy = SELUPolicy(obs_size - 4, action_size, hl_dims.copy())
        # self.adv_policy.load_state_dict(torch.load("/Users/phillip/projects/AGTLib/output/experiment-40/end-3x3-adv-policy-n-reinforce.pt"))
        self.adv_optimizer = torch.optim.Adam(self.adv_policy.parameters(), lr=lr, maximize=False)
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
        returns = torch.tensor(returns).flip(-1)

        loss = -torch.dot(log_probs, returns)

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
                team_translated = team_policy.action_map[team_action]
                action = {}
                for i in range(len(team_translated)):
                    action[i] = team_translated[i]
                adv_action, adv_log_prob = adv_policy.get_action(adv_obs)
                action[i+1] = adv_action.item()
                
                obs, reward, done, trunc, _ = env.step(action)

                adv_rewards.append(reward[len(reward) - 1])
                team_rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break
        
        return torch.mean(torch.tensor(adv_rewards), dtype=torch.float32), torch.mean(torch.tensor(team_rewards, dtype=torch.float32))
    
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

    def __init__(self, obs_size, action_size, env, hl_dims=[64,128], lr: float = 0.01, gamma:float = 0.9, rollout_length:int = 50, br_length: int = 100):
        self.obs_size = obs_size
        self.action_size = action_size
        self.env = env() # used to be env()
        self.lr = lr

        self.gamma = gamma
        self.rollout_length = rollout_length
        self.br_length = br_length
        self.hl_dims = hl_dims
                
        self.nash_gap = []
        self.team_utility = []

        self.adv_policy = SELUPolicy(obs_size - 4, action_size, hl_dims.copy())
        self.adv_optimizer = torch.optim.Adam(self.adv_policy.parameters(), lr=lr, maximize=False)

        self.team_policy = SELUMAPolicy(obs_size, action_size * action_size, [(i,j) for i in range(4) for j in range(4)], hl_dims=hl_dims.copy())
        self.team_optimizer = torch.optim.Adam(self.team_policy.parameters(), lr=lr, maximize=False)

    def update(self, adversary=True, team_policy=None, team_optimizer=None, adv_policy=None, adv_optimizer=None, rollout_length=None):
        if rollout_length is None:
            rollout_length = self.rollout_length

        if team_policy is None:
            team_policy = self.team_policy
        
        if team_optimizer is None:
            team_optimizer = self.team_optimizer

        if adv_policy is None:
            adv_policy = self.adv_policy
        
        if adv_optimizer is None and adversary:
            adv_optimizer = self.adv_optimizer

        log_probs = []
        rewards = []
        env = self.env

        obs, _ = env.reset()
        while True:
            team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
            adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
            team_action, team_log_prob = team_policy.get_actions(team_obs)
            team_translated = team_policy.action_map[team_action]
            action = {}
            for i in range(len(team_translated)):
                action[i] = team_translated[i]
            adv_action, adv_log_prob = adv_policy.get_action(adv_obs)
            action[i+1] = adv_action.item()
            obs, reward, done, trunc, _ = env.step(action) 
            if adversary:
                log_probs.append(adv_log_prob)
                # obs_data.append(adv_obs)
                # action_data.append(adv_action)
                rewards.append(reward[len(reward) - 1])
            else:
                log_probs.append(team_log_prob)
                # obs_data.append(team_obs)
                # action_data.append(team_action)
                rewards.append(reward[0])

            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

        returns = [rewards[-1]]
        for i in range(2, len(rewards)+1):
            returns.append(self.gamma * returns[-1] + rewards[-i])

        log_prob_data = torch.stack(log_probs)
        return_data = torch.tensor(returns, requires_grad=True, dtype=torch.float32).flip(-1)

        policy = adv_policy if adversary else team_policy
        optimizer = adv_optimizer if adversary else team_optimizer

        # policy = policy.to("mps")

        loss = -torch.dot(log_prob_data, return_data)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        return torch.mean(return_data)
    
        # policy = policy.to("cpu")

    def get_adv_br(self):
        temp_adv = SELUPolicy(self.obs_size - 4, self.action_size, hl_dims=self.hl_dims.copy())
        temp_adv.load_state_dict(self.adv_policy.state_dict())
        temp_optimizer = torch.optim.Adam(temp_adv.parameters(), lr=self.lr, maximize=False)

        for i in range(self.br_length):
            self.update(adversary=True, adv_policy=temp_adv, adv_optimizer=temp_optimizer)

        return self.get_utility(adv_policy=temp_adv)[0]

    def get_team_br(self):
        temp_team = SELUMAPolicy(self.obs_size, self.action_size*self.action_size, [(i,j) for i in range(4) for j in range(4)], hl_dims=self.hl_dims.copy())
        temp_team.load_state_dict(self.team_policy.state_dict())
        temp_optimizer = torch.optim.Adam(temp_team.parameters(), lr=self.lr, maximize=False)

        for i in range(self.br_length):
            self.update(adversary=False, team_policy=temp_team, team_optimizer=temp_optimizer)

        return self.get_utility(team_policy=temp_team)[1]

    def step(self):
        for i in range(self.br_length):
            self.update()

        self.update(adversary=False)
    
    def step_with_gap(self):
        for i in range(self.br_length):
            self.update()

        self.update(adversary=False)

        adv_base, team_base = self.get_utility()

        adv_br = self.get_adv_br()

        self.nash_gap.append(max(adv_br.item() - adv_base.item(), self.get_team_br().item() - team_base.item()))

        # team plays some x, assume adversary plays best response, then print utility of team when adversary is giving best response

        # one treasure in top right, one treasure in bottom left, other two corners put the team, put the adversary in the center
        # make it so 


class QGDmax(NGDmax):
    def __init__(self, qtable, obs_size, action_size, env, eps_decay=0.005, min_eps=0.05, max_eps = 1, hl_dims=[64,128], lr: float = 0.01, gamma:float = 0.9, rollout_length:int = 50, br_length: int = 100):
        super().__init__(obs_size, action_size, env, hl_dims, lr, gamma, rollout_length, br_length)
        self.qpolicy = TabularQ(qtable, eps_decay, min_eps, max_eps, lr, gamma, rollout_length, env)
        self.adv_policy = self.qpolicy
        self.q_args = (qtable, eps_decay, min_eps, max_eps, lr, gamma, rollout_length, env)
    
    def get_adv_br(self):
        temp_adv = TabularQ(*self.q_args)
        temp_adv.train(self.team_policy)

        return self.get_utility(adv_policy=temp_adv)[0]

    def get_team_br(self):
        temp_team = SELUMAPolicy(self.obs_size, self.action_size*self.action_size, [(i,j) for i in range(4) for j in range(4)], hl_dims=self.hl_dims.copy())
        temp_team.load_state_dict(self.team_policy.state_dict())
        temp_optimizer = torch.optim.Adam(temp_team.parameters(), lr=self.lr, maximize=False)

        for i in range(100):
            self.update(adversary=False, team_policy=temp_team, team_optimizer=temp_optimizer)

        return self.get_utility(team_policy=temp_team)[1]
        
    
    def step(self):
        """
        GDmax training step with no metrics.
        """
        self.qpolicy.train(self.team_policy)

        self.update(adversary=False)

    def step_with_gap(self):
        """
        GDmax training step with the nash-gap metric.
        """
        self.qpolicy.train(self.team_policy)

        self.update(adversary=False)

        adv_base, team_base = self.get_utility()

        adv_br = self.get_adv_br()
        team_br = self.get_team_br()

        self.nash_gap.append(max(adv_br.item() - adv_base.item(), team_br.item() - team_base.item()))


class PGDmax(NGDmax):
    """
    GDmax but the adversary learns the best response with PPO. 
    """
    def __init__(self, obs_size, action_size, env, hl_dims=[64,128], lr: float = 0.01, gamma:float = 0.9, rollout_length:int = 50, br_length: int = 100):
        super().__init__(obs_size, action_size, env, hl_dims, lr, gamma, rollout_length, br_length)
        
        self.ppo_args = dict(policy="MlpPolicy", env=SubprocVecEnv([env for _ in range(10)]), gdmax=True, monitor_wrapper=False)

        self.ppo = PPO(**self.ppo_args)
        self.adv_policy = self.ppo.policy

    def update(self, adversary=True, team_policy=None, team_optimizer=None, adv_policy=None, adv_optimizer=None, rollout_length=None):
        if rollout_length is None:
            rollout_length = self.rollout_length

        if team_policy is None:
            team_policy = self.team_policy
        
        if team_optimizer is None:
            team_optimizer = self.team_optimizer

        if adv_policy is None:
            adv_policy = self.adv_policy
        
        if adv_optimizer is None and adversary:
            adv_optimizer = self.adv_optimizer

        log_probs = []
        rewards = []
        env = self.env

        obs, _ = env.reset()
        while True:
            team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
            adv_obs = obs_as_tensor(obs[len(obs) - 1], torch.device("cpu")).reshape(-1, 8) #  torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
            team_action, team_log_prob = team_policy.get_actions(team_obs)
            team_translated = team_policy.action_map[team_action]
            action = {}
            for i in range(len(team_translated)):
                action[i] = team_translated[i]
            adv_action, _1, _2 = adv_policy(adv_obs)
            action[i+1] = adv_action.item()
            obs, reward, done, trunc, _ = env.step(action) 
            
            log_probs.append(team_log_prob)
            # obs_data.append(team_obs)
            # action_data.append(team_action)
            rewards.append(reward[0])

            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

        returns = [rewards[-1]]
        for i in range(2, len(rewards)+1):
            returns.append(self.gamma * returns[-1] + rewards[-i])

        log_prob_data = torch.stack(log_probs)
        return_data = torch.tensor(returns, requires_grad=True, dtype=torch.float32).flip(-1)

        policy = team_policy
        optimizer = team_optimizer

        # policy = policy.to("mps")

        loss = -torch.dot(log_prob_data, return_data)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        return torch.mean(return_data)
    
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
                adv_obs = obs_as_tensor(obs[len(obs) - 1], torch.device("cpu")).reshape(-1, 8) #  torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
                team_action, team_log_prob = team_policy.get_actions(team_obs)
                team_translated = team_policy.action_map[team_action]
                action = {}
                for i in range(len(team_translated)):
                    action[i] = team_translated[i]
                adv_action, _, _ = adv_policy(adv_obs)
                action[i+1] = adv_action.item()
                
                obs, reward, done, trunc, _ = env.step(action)

                adv_rewards.append(reward[len(reward) - 1])
                team_rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break
        
        return torch.mean(torch.tensor(adv_rewards), dtype=torch.float32), torch.mean(torch.tensor(team_rewards, dtype=torch.float32))
        
    def get_adv_br(self):
        temp_adv = PPO(**self.ppo_args)
        temp_adv.learn(50000, opponent_policy=self.team_policy)

        return self.get_utility(adv_policy=temp_adv.policy)[0]

    def get_team_br(self):
        temp_team = SELUMAPolicy(self.obs_size, self.action_size*self.action_size, [(i,j) for i in range(4) for j in range(4)], hl_dims=self.hl_dims.copy())
        temp_team.load_state_dict(self.team_policy.state_dict())
        temp_optimizer = torch.optim.Adam(temp_team.parameters(), lr=self.lr, maximize=False)

        for i in range(100):
            self.update(adversary=False, team_policy=temp_team, team_optimizer=temp_optimizer)

        return self.get_utility(team_policy=temp_team)[1]

    def step(self):
        self.ppo.learn(total_timesteps=50000, opponent_policy=self.team_policy)

        self.update(adversary=False)

    def step_with_gap(self):
        """
        GDmax training step with the nash-gap metric.
        """
        self.ppo.learn(total_timesteps=50000, opponent_policy=self.team_policy)

        self.update(adversary=False)

        adv_base, team_base = self.get_utility()

        adv_br = self.get_adv_br()
        team_br = self.get_team_br()

        self.nash_gap.append(max(adv_br.item() - adv_base.item(), team_br.item() - team_base.item()))