import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_size, action_size, hidden_units):
        self.hl1 = nn.Dense(...)