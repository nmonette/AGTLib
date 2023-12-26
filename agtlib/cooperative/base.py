"""
## Base Classes for Cooperative RL Algorithms
Some utility objects that will be used, as well as 
base classes to inherit from.
"""

from typing import Iterable, DefaultDict
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam


class PolicyNetwork(nn.Module):
    """
    Feedforward neural network that serves
    as a function approximation of the 
    policy gradient.
    """
    def __init__(self, obs_size: int, action_size: int, hl_dims: Iterable[int, ] = [64, 128]) -> None:
        """
        Parameters
        ----------
        obs_size: int
            The length of the flattened observation of the agent(s). 
        action_size: int
            The cardinality of the action space of a single agent. 
        hl_dims: Iterable(int), optional
            An iterable such that the ith element represents the width of the ith hidden layer. 
            Defaults to `[64,128]`. Note that this does not include the input or output layers.
        """
        super().__init__(self)
        prev_dim = obs_size 
        hl_dims.append(action_size)
        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the neural network.
        Parameters
        ----------
        x: torch.Tensor
            The flattened observation of the agent(s).

        Returns
        -------
        torch.Tensor
            Probability vector containing the action-probabilities.
        """
        if len(x.shape) != 1:
            raise ValueError("Parameter 'x' is not a flattened tensor")
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        return self.layers[-1](x)

    @torch.no_grad()
    def get_action(self, x: torch.Tensor) -> int:
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
        float
            The log probability of the returned action with reference to the current policy.
        """
        if len(x.shape) != 1:
            raise ValueError("Parameter 'x' is not a flattened tensor")

        dist = torch.distributions.Categorical(self.policy.forward(x))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Feedforward neural network that serves
    as a function approximation of the 
    value function.
    """
    def __init__(self, obs_size: int, hl_dims: Iterable[int, ] = [64, 128]) -> None:
        """
        Parameters
        ----------
        obs_size: int
            The length of the flattened observation of the agent(s). 
        hl_dims: Iterable(int), optional
            An iterable such that the ith element represents the width of the ith hidden layer. 
            Defaults to `[64,128]`. Note that this does not include the input or output layers.
        """
        super().__init__(self)
        prev_dim = obs_size 
        hl_dims.append(1)

        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the neural network.
        Parameters
        ----------
        x: torch.Tensor
            The flattened observation of the agent(s).

        Returns
        -------
        torch.Tensor
            Probability vector containing the action-probabilities.
        """
        if len(x.shape) != 1:
            raise ValueError("Parameter 'x' is not a flattened tensor")
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        return nn.Softmax(-1)(self.layers[-1](x))

class LinearValue(nn.Module):
    """
    Linear function approximation
    of the value function.
    """
    def __init__(self, obs_size: int, num_features: int) -> None:
        """
        Parameters
        ----------
        obs_size: int
            The length of the flattened observation of the agent(s). 
        num_features: int
            The number of features that the linear regression will process. 
        """
        super().__init__(self)
        self.feature_mapping = nn.Linear(obs_size, num_features)
        self.line = nn.Linear(num_features, 1)

    def forward(self, x) -> torch.Tensor:
        """
        Performs the prediction step of the linear regression.
        Parameters
        ----------
        x: torch.Tensor
            The flattened observation of the agent(s).
            Alternatively, can be the output of the vector-valued
            feature mapping of the observation.

        Returns
        -------
        torch.Tensor
            Tensor containing the approximate value of the given state. 
        """
        if len(x.shape) != 1:
            raise ValueError("Parameter 'x' is not a flattened tensor")
        return self.line(self.feature_mapping(x))

class QNetwork(nn.Module):
    """
    Feedforward neural network that serves
    as a function approximation of the 
    Q (state-action) function.
    """
    def __init__(self, obs_size: int, hl_dims: Iterable[int, ] = [64, 128]) -> None:
        """
        Parameters
        ----------
        obs_size: int
            The length of the flattened observation of the agent(s). 
        hl_dims: Iterable(int), optional
            An iterable such that the ith element represents the width of the ith hidden layer. 
            Defaults to `[64,128]`. Note that this does not include the input or output layers.
        """
        super().__init__(self)
        prev_dim = obs_size + 1 # + 1 for the action's value
        hl_dims.append(1)

        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the neural network.
        Parameters
        ----------
        x: torch.Tensor
            The flattened observation of the agent(s).

        Returns
        -------
        torch.Tensor
            Probability vector containing the action-probabilities.
        """
        if len(x.shape) != 1:
            raise ValueError("Parameter 'x' is not a flattened tensor")
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        return nn.Softmax(-1)(self.layers[-1](x))

class LinearQ(nn.Module):
    """
    Linear function approximation 
    of the Q (state-action) function.
    """
    def __init__(self, obs_size: int, num_features: int) -> None:
        """
        Parameters
        ----------
        obs_size: int
            The length of the flattened observation of the agent(s). 
        num_features: int
            The number of features that the linear regression will process. 
        """
        super().__init__(self)
        self.feature_mapping = nn.Linear(obs_size, num_features)
        self.line = nn.Linear(num_features, 1)

    def forward(self, x) -> torch.Tensor:
        """
        Performs the prediction step of the linear regression.
        Parameters
        ----------
        x: torch.Tensor
            The flattened observation of the agent(s).
            Alternatively, can be the output of the vector-valued
            feature mapping of the observation.
            
        Returns
        -------
        torch.Tensor
            Tensor containing the approximate value of the given state. 
        """
        if len(x.shape) != 1:
            raise ValueError("Parameter 'x' is not a flattened tensor")
        return self.line(self.feature_mapping(x))

class RLBase(ABC):
    """
    Base class for RL models to override.
    """
    def __init__(self, action_size: int, obs_size: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, value_type: str = "V", gamma: float = 0.99) -> None:
        """
        Parameters
        ----------
        action_size: int
            The cardinality of the action space of a single agent.
        obs_size: int
            The length of the flattened observation of the agent(s). 
        v_obs_size: int, optional, keyword only
            The length of the flattened observation size of the agent(s).
            Only specify if different from policy `obs_size` (e.g. the 
            policy network uses locallized observations, but the value 
            network uses the joint observation of multiple agents).
        policy_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents the width of the ith hidden layer
            of the policy network. Defaults to `[64,128]`. Note that this does not include the 
            input or output layers.
        value_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents the width of the ith hidden layer
            of the value network. Defaults to `[64,128]`. Note that this does not include the 
            input or output layers. Must be length 1 if linear_value is enabled, with the value 
            equal to the number of linear features.
        linear_value: bool, optional, keyword only
            Indicates whether or not the value function is to be approximated linearly. Defaults 
            to `False`.
        value_type: str, optional, keyword only
            Indicates which value function to use (i.e. Q or V). Only accepts arguments in `{"Q","V"}.
            Defaults to `"V"`.
        gamma: float, optional, keyword only
            The discount factor to be used in the calculation of expectations. Must be in the range
            `[0,1]` for (finite time horizons). Defaults to `0.99`.
        """
        self.action_size = action_size  
        self.obs_size = obs_size

        if v_obs_size is None:
            self.v_obs_size = obs_size
        else:
            self.v_obs_size = v_obs_size

        if not isinstance(policy_hl_dims, Iterable):
            raise TypeError("Parameter 'policy_hl_dims' is not type iterable")
        else:
            self.policy_hl_dims = policy_hl_dims

        if linear_value and len(value_hl_dims) != 1:
            raise ValueError("Parameter 'value_hl_dims' must have length 1 when 'linear_value' is True")
        else:
            self.value_hl_dims = value_hl_dims
        
        self.linear_value = linear_value
        
        if value_type not in {"Q", "V"}:
            raise ValueError("Parameter 'value_type' not 'Q' or 'V'")
        else:
            self.value_type = value_type

        if not (0 <= gamma <= 1):
            raise ValueError("Parameter 'gamma' is not in the range [0,1]")
        else:
            self.gamma = gamma

        if self.value_type == 'V' and self.linear_value:
            self.value = LinearValue(v_obs_size, value_hl_dims)
        elif self.value_type == 'V':
            self.value = ValueNetwork(v_obs_size, value_hl_dims)
        elif self.value_type == 'Q' and self.linear_value:
            self.value = LinearQ(v_obs_size, value_hl_dims)
        else:
            self.value_type = QNetwork(v_obs_size, value_hl_dims)

        self.policy = PolicyNetwork(obs_size, action_size, policy_hl_dims)
        
    @abstractmethod
    def step(self, utility) -> None:
        """
        Updates weights of policy and value networks. 

        Parameters
        ----------
        utility: int, float or other scalar representation
            The agent's utility at the current time step
        """
        pass

    ## TO DO: add util methods to this class as necessary

