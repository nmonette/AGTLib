"""
## Base Classes for Cooperative RL Algorithms
Some utility objects that will be used, as well as 
base classes to inherit from.
"""

import numpy as np
import torch
import torch.nn as nn

from typing import Iterable

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
        hl_dims: iterable(int), optional
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
        elif isinstance(x, torch.Tensor):
            raise TypeError("Parameter 'x' is not type torch.Tensor")
        for i in range(len(self.layers) - 1):
            x = torch.relu(self.layers[i](x))
        return self.layers[-1](x)


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
        hl_dims: iterable(int), optional
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
        elif isinstance(x, torch.Tensor):
            raise TypeError("Parameter 'x' is not type torch.Tensor")
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
        elif isinstance(x, torch.Tensor):
            raise TypeError("Parameter 'x' is not type torch.Tensor")
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
        hl_dims: iterable(int), optional
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
        elif isinstance(x, torch.Tensor):
            raise TypeError("Parameter 'x' is not type torch.Tensor")
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
        elif isinstance(x, torch.Tensor):
            raise TypeError("Parameter 'x' is not type torch.Tensor")
        return self.line(self.feature_mapping(x))

class RLBase:
    """
    Base class for RL models to override.
    """
    def __init__(self, action_size: int, obs_size: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, value_type: str = "V") -> None:
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
        policy_hl_dims: iterable(int), optional, keyword only
            An iterable such that the ith element represents the width of the ith hidden layer
            of the policy network. Defaults to `[64,128]`. Note that this does not include the 
            input or output layers.
        value_hl_dims: iterable(int), optional, keyword only
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
        """
        if not isinstance(action_size, int):
            raise TypeError("Parameter 'action_size' is not type integer")
        elif action_size < 1:
            raise ValueError("Parameter 'action_size' is not type integer > 1")
        else:
            self.action_size = action_size
        
        if not isinstance(obs_size, int):
            raise TypeError("Parameter 'obs_size' is not type integer")
        elif obs_size < 1:
            raise ValueError("Parameter 'obs_size' is not an integer > 1")
        else:
            self.obs_size = obs_size

        if v_obs_size is None:
            self.v_obs_size = obs_size
        elif not isinstance(v_obs_size, int):
            raise TypeError("Parameter 'v_obs_size' is not type integer")
        elif v_obs_size < 1:
            raise ValueError("Parameter 'v_obs_size' is not an integer > 1")
        else:
            self.v_obs_size = v_obs_size

        if not isinstance(policy_hl_dims, Iterable):
            raise TypeError("Parameter 'policy_hl_dims' is not type iterable")
        elif not all([isinstance(i, int) for i in range(policy_hl_dims)]):
            raise TypeError("Parameter 'policy_hl_dims' does not contain all integers")
        elif all([i > 0 for i in range(policy_hl_dims)]):
            raise ValueError("Parameter 'policy_hl_dims' does not contain all positive integers")
        else:
            self.policy_hl_dims = policy_hl_dims

        if not isinstance(value_hl_dims, Iterable):
            raise TypeError("Parameter 'value_hl_dims' is not type iterable")
        elif not all([isinstance(i, int) for i in range(value_hl_dims)]):
            raise TypeError("Parameter 'value_hl_dims' does not contain all integers")
        elif linear_value and len(value_hl_dims) != 1:
            raise ValueError("Parameter 'value_hl_dims' must have length 1 when 'linear_value' is True")
        elif all([i > 0 for i in range(value_hl_dims)]):
            raise ValueError("Parameter 'value_hl_dims' does not contain all positive integers")
        else:
            self.value_hl_dims = value_hl_dims
        
        if not isinstance(linear_value, bool):
            raise TypeError("Parameter 'linear_value' is not type bool")
        else:
            self.linear_value = linear_value
        
        if not isinstance(value_type, str):
            raise TypeError("Parameter 'value_type' is not type str")
        elif value_type not in {"Q", "V"}:
            raise ValueError("Parameter 'value_type' not 'Q' or 'V'")
        else:
            self.value_type = value_type

        if self.value_type == 'V' and self.linear_value:
            self.value = LinearValue(v_obs_size, value_hl_dims)
        elif self.value_type == 'V':
            self.value = ValueNetwork(v_obs_size, value_hl_dims)
        elif self.value_type == 'Q' and self.linear_value:
            self.value = LinearQ(v_obs_size, value_hl_dims)
        else:
            self.value_type = QNetwork(v_obs_size, value_hl_dims)

        self.policy = PolicyNetwork(obs_size, action_size, policy_hl_dims)
        

    def step(self, utility: int) -> None:
        """
        Updates weights of policy and value networks. 

        Parameters
        ----------
        utility: int
            Observed utility at the current time step.
        """
        pass

    def get_action(self) -> int:
        """
        Samples an action from the current policy and returns it as an integer index.

        Returns
        -------
        int
            The integer index of the action samples from the policy.
        float
            Log probability of the returned action with reference to the current policy.
        """
        pass

