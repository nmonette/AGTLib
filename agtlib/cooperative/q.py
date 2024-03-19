import torch

class TabularQ:
    def __init__(self, table, eps_decay, min_eps, max_eps, lr, gamma, rollout_length, max_steps, env):
        self.table = table

        self.min_eps = min_eps
        self.max_eps = max_eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.gamma = gamma

        self.epsilon = 1

        self.rollout_length = rollout_length

        self.env = env()

        self.max_steps = max_steps

    def get_action(self, obs):
        obs = obs.to(torch.int)
        if torch.distributions.Bernoulli(1 - self.epsilon):
            action = torch.argmin(self.table[*obs, :])
        else:
            action = torch.randint(0, self.table.shape[-1])
        return action, None
    
    def train(self, opponent_policy):
        
        for episode in range(self.rollout_length):
            self.epsilon = self.min_eps + (self.max_eps - self.min_eps)*torch.exp(torch.tensor(-self.eps_decay*episode))
            # Reset the environment
            obs, _ = self.env.reset()
            done = False

            for t in range(self.max_steps):
                team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
                adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
                team_action, _ = opponent_policy.get_actions(team_obs)
                adv_action, _ = self.get_action(torch.tensor(adv_obs))
                adv_action = adv_action.item()
                team_translated = opponent_policy.action_map[team_action]
                action = {}
                for i in range(len(team_translated)):
                    action[i] = team_translated[i]
                action[len(action)] = adv_action
                new_obs, reward, trunc, done, _ = self.env.step(action)

                reward = reward[len(reward) - 1]

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break

                new_adv_obs = adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.int)
                prev = self.table[adv_obs][adv_action]
                self.table[adv_obs][adv_action] = prev + self.lr * (reward + self.gamma * max(self.table[*new_adv_obs, v] - self.table[*adv_obs, adv_action] for v in range(self.table.shape[-1])))

                obs = new_obs