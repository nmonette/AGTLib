import torch

class TabularQ:
    def __init__(self, table, eps, eps_decay, lr, gamma):
        self.table = self.table

        self.eps = eps
        self.eps_decay = eps_decay
        self.lr = lr
        self.gamma = gamma

        self.current_pair = None
        self.prev_state = None

    def get_actions(self, obs):
        if torch.distributions.Bernoulli(self.epsilon)
            action = torch.argmin(self.table[*obs])
            self.current_pair = (obs, action)
        return action
    
    def update(self, reward):
        prev = self.table[*self.current_pair[0], self.current_pair[1]] 
        self.table[*self.current_pair[0], self.current_pair[1]] = prev + self.lr * (reward + self.gamma * max())