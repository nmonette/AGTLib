from typing import Tuple

import torch
import gymnasium as gym

class TabularQ:
    """
    Policy for Tabular Q-learning (SARSA). Uses an epsilon-greedy policy. 
    """
    def __init__(self, table: torch.Tensor, eps_decay: float, min_eps: float, max_eps: float, lr: float, gamma: float, rollout_length: int, env:gym.Env, inti_epsilon: float=1):
        """
        Parameters
        ----------
        table: torch.Tensor
            The Q-table to be learned. Can pass in a pre-learned one as well for evaluation purposes. 
        eps_decay: float
            The decay rate of decay for $\epsilon$.
        min_eps: float
            The minimum value for $\epsilon$$
        max_eps: float
            The minimum value for $\epsilon$$
        lr: float
            The learning rate
        gamma: float
            The discount factor
        rollout_length: int
            The number of rollout episodes done per training cycle. 
        env: gym.Env
            The environment to be trained on
        init_epsilon: float, optional
            The initial epsilon to be used. Defaults to 1 but can be set to 0 for evaluation. 
        """
        
        self.table = table

        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.gamma = gamma

        self.epsilon = inti_epsilon

        self.rollout_length = rollout_length

        self.env = env()


        self.dist = torch.distributions.Uniform(0, 1)

    def get_action(self, obs) -> Tuple[torch.Tensor, None]:
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
        None
            Returns an extra item to fit the return signature of the rest of the library. 
            Typically would be reserved for the log-prob of the action. 
        """
        obs = obs.to(torch.int)
        if self.epsilon < self.dist.sample():
            action = torch.argmin(self.table[*obs, :])
        else:
            action = torch.randint(0, self.table.shape[-1], (1, ))
        return action, None
        
    
    def train(self, opponent_policy):
        """
        Learns the Q-table with the opponent's policy fixed. 

        opponent_policy: torch.nn.Module
            a policy that has `get_actions` and `action_map`
        """
        reward_means = []
        for episode in range(self.rollout_length):
            self.epsilon = self.min_eps + (self.max_eps - self.min_eps)*torch.exp(torch.tensor(-self.eps_decay*episode))
            # Reset the environment
            obs, _ = self.env.reset()
            done = False
            rewards = []
            while True:
                team_obs = [torch.tensor(obs[0], device="cpu", dtype=torch.float32), torch.tensor(obs[1], device="cpu", dtype=torch.float32)]
                adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.int)
                team_translated, _ = opponent_policy.get_actions(team_obs)
                adv_action, _ = self.get_action(adv_obs)
                adv_action = adv_action.item()
                # team_translated = opponent_policy.action_map[team_action]
                action = {}
                for i in range(len(team_translated)):
                    action[i] = team_translated[i]
                action[len(action)] = adv_action
                new_obs, reward, trunc, done, _ = self.env.step(action)

                reward = reward[len(reward) - 1]

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break
                
                new_adv_obs = torch.tensor(new_obs[len(new_obs) - 1], device="cpu", dtype=torch.int)
                prev = self.table[*adv_obs, adv_action]
                self.table[*adv_obs, adv_action] = prev + self.lr * (reward + self.gamma * max(self.table[*new_adv_obs, v] - self.table[*adv_obs, adv_action] for v in range(self.table.shape[-1])))
                obs = new_obs