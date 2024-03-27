"""
## Base Classes for Cooperative RL Algorithms
Some utility objects that will be used, as well as 
base classes to inherit from.
"""

from typing import Iterable, DefaultDict, List
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
        super(PolicyNetwork, self).__init__()
        prev_dim = obs_size 
        hl_dims.append(action_size)
        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]

        self.relu = torch.nn.ReLU()
        
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
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
        return self.layers[-1](x)

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

        dist = torch.distributions.Categorical(logits=self.__call__(x))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def evaluate_actions(self, obs: torch.tensor, actions: torch.tensor):
        """
        Passes observation through the network and then returns
        the log probability of taking that action at the current state
        of the network. 
        Parameters
        ----------
        obs: torch.Tensor
            The flattened observation(s).
        actions: torch.Tensor
            The actions to evaluate. 
        """
        dist = torch.distributions.Categorical(logits=(self.__call__(obs)))
        return dist.log_prob(actions)
    
class SELUPolicy(nn.Module):
    """
    Feedforward neural network that serves
    as a function approximation of the 
    policy gradient. Uses SELU Activation 
    Functions. 
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
        super(SELUPolicy, self).__init__()
        prev_dim = obs_size 
        hl_dims.append(action_size)
        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]

        self.selu = torch.nn.SELU()
        
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
        dim = 4
        obs_space = np.array([dim,dim,dim,dim,2,dim,dim,2])
        obs = x.reshape(-1, len(obs_space))
        obs = torch.cat(
            [
                torch.nn.functional.one_hot(obs_.long(), num_classes=int(obs_space[idx])).float()
                for idx, obs_ in enumerate(torch.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(obs_space))
        for i in range(len(self.layers) - 1):
            x = self.selu(self.layers[i](x))
        return self.layers[-1](x)

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

        dist = torch.distributions.Categorical(logits=self.__call__(x))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob
    
    def evaluate_actions(self, obs: torch.tensor, actions: torch.tensor):
        """
        Passes observation through the network and then returns
        the log probability of taking that action at the current state
        of the network. 
        Parameters
        ----------
        obs: torch.Tensor
            The flattened observation(s).
        actions: torch.Tensor
            The actions to evaluate. 
        """
        dist = torch.distributions.Categorical(logits=(self.__call__(obs)))
        return dist.log_prob(actions)


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
        super(ValueNetwork, self).__init__()
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

        for i in range(len(self.layers) - 1):
            x = torch.nn.ReLU()(self.layers[i](x))
        return self.layers[-1](x)

class LinearValue(nn.Module):
    """
    Linear function approximation
    of the value function.
    """
    def __init__(self, obs_size: int, n_features: int) -> None:
        """
        Parameters
        ----------
        obs_size: int
            The length of the flattened observation of the agent(s). 
        n_features: int
            The number of features that the linear regression will process. 
        """
        super(LinearValue, self).__init__()
        self.feature_mapping = nn.Linear(obs_size, n_features)
        self.line = nn.Linear(n_features, 1)

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
        super(QNetwork, self).__init__()
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
        for i in range(len(self.layers) - 1):
            x = torch.nn.ReLU()(self.layers[i](x))
        return self.layers[-1](x)

class LinearQ(nn.Module):
    """
    Linear function approximation 
    of the Q (state-action) function.
    """
    def __init__(self, obs_size: int, n_features: int) -> None:
        """
        Parameters
        ----------
        obs_size: int
            The length of the flattened observation of the agent(s). 
        n_features: int
            The number of features that the linear regression will process. 
        """
        super(LinearQ, self).__init__()
        self.feature_mapping = nn.Linear(obs_size, n_features)
        self.line = nn.Linear(n_features, 1)

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
        return self.line(self.feature_mapping(x))
    
class ActorCritic(nn.Module):
    """
    Housing of the policy and value functions 
    in the same place. 
    """
    def __init__(self, policy, value):
        super(ActorCritic, self).__init__()
        self.policy = policy
        self.value = value
    
    def forward(self, x):
        return self.policy.__call__(x), self.value.__call__(x)
    
class ActorCriticCTDE(nn.Module):
    """
    Housing of the policy and value functions 
    in the same place. Used for CTDE models.
    """
    def __init__(self, policies, value):
        super(ActorCritic, self).__init__()
        self.policies = policies
        self.value = value
    
    def forward(self, x):
        return *[policy.__call__(x) for policy in range(len(self.policies))], self.value.__call__(x)
    
class SoftmaxPolicy(nn.Module):
    """
    Policy that uses a softmax parameterization, i.e. 
    $$ \pi(a | s) = \frac{e^{\theta_{a, s}}}{\sum_{a'} \theta_{a', s}}
    """
    def __init__(self, n_agents, n_actions, param_dims, lr= 0.01, action_map=None):
        super(SoftmaxPolicy, self).__init__()
        self.lr = lr

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.param_dims = param_dims

        self.action_map = action_map

        empty = torch.empty(*param_dims)
        nn.init.orthogonal_(empty)
        self.params = nn.Parameter(nn.Softmax()(empty), requires_grad=True)
        

    def forward(self, x):
        # [dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16]
        return self.params[*x, :]

    def get_actions(self, x):
        dist = torch.distributions.Categorical(self.__call__(x)) # make categorical distribution and then decode the action index
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if self.action_map is not None:
            return self.action_map[action], log_prob
        else:
            return action, log_prob
    
    def step(self, loss):
        loss.backward(inputs=(self.params,)) # inputs=(self.params,)

        x = self.params + self.lr * self.params.grad
        self.params.grad.zero_()
        self.params.data = nn.Softmax()(x)

class MAPolicyNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int, action_map: List[List[int, ]], hl_dims: Iterable[int, ] = [64, 128]) -> None:
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
        super(MAPolicyNetwork, self).__init__()
        prev_dim = obs_size 
        hl_dims.append(action_size)
        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]

        self.action_map = action_map
        # self.lookup_table = torch.cumsum(torch.ones((len(action_map), ), dtype=int), 0).reshape((int(np.sqrt(action_size)), int(np.sqrt(action_size)))) - 1
        self.relu = torch.nn.ReLU()
        
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
        # if isinstance(x, np.ndarray):
        #     x = torch.tensor(x, dtype=torch.float)
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
        return self.layers[-1](x)

    def get_actions(self, x: torch.Tensor) -> int:
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

        dist = torch.distributions.Categorical(logits=self.__call__(x))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob # self.action_map[action], log_prob
    
    def evaluate_actions(self, obs: torch.tensor, actions: torch.tensor):
        """
        Passes observation through the network and then returns
        the log probability of taking that action at the current state
        of the network. 
        Parameters
        ----------
        obs: torch.Tensor
            The flattened observation(s).
        actions: torch.Tensor
            The actions to evaluate. 
        """
        dist = torch.distributions.Categorical(logits=(self.__call__(obs)))
        return dist.log_prob(actions)
    
class SELUMAPolicy(nn.Module):
    def __init__(self, obs_size: int, action_size: int, action_map: List[List[int, ]], hl_dims: Iterable[int, ] = [64, 128]) -> None:
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
        super(SELUMAPolicy, self).__init__()
        prev_dim = obs_size 
        hl_dims.append(action_size)
        self.layers = nn.ModuleList()
        for i in range(len(hl_dims)):
            self.layers.append(nn.Linear(prev_dim, hl_dims[i]))
            prev_dim = hl_dims[i]

        self.action_map = action_map
        # self.lookup_table = torch.cumsum(torch.ones((len(action_map), ), dtype=int), 0).reshape((int(np.sqrt(action_size)), int(np.sqrt(action_size)))) - 1
        self.selu = torch.nn.SELU()
        
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
        # if isinstance(x, np.ndarray):
        #     x = torch.tensor(x, dtype=torch.float)
        dim = 4
        obs_space = np.array([dim,dim,2, dim,dim,2,dim,dim,2,dim,dim,2])
        obs = x.reshape(-1, len(obs_space))
        obs = torch.cat(
            [
                torch.nn.functional.one_hot(obs_.long(), num_classes=int(obs_space[idx])).float()
                for idx, obs_ in enumerate(torch.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(obs_space))
        for i in range(len(self.layers) - 1):
            x = self.selu(self.layers[i](x))
        return self.layers[-1](x)

    def get_actions(self, x: torch.Tensor) -> int:
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

        dist = torch.distributions.Categorical(logits=self.__call__(x))
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob # self.action_map[action], log_prob
    
    def evaluate_actions(self, obs: torch.tensor, actions: torch.tensor):
        """
        Passes observation through the network and then returns
        the log probability of taking that action at the current state
        of the network. 
        Parameters
        ----------
        obs: torch.Tensor
            The flattened observation(s).
        actions: torch.Tensor
            The actions to evaluate. 
        """
        dist = torch.distributions.Categorical(logits=(self.__call__(obs)))
        return dist.log_prob(actions)


class RLBase(ABC):
    """
    Base class for RL models to override.
    """
    def __init__(self, action_size: int, obs_size: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, value_type: str = "V", gamma: float = 0.99, \
                 n_policies: int = 1) -> None:
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
        n_policies: int, optional, keyword only
            The amount of policy networks to instantiate. Defaults to `1`.
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
            self.value = LinearValue(self.v_obs_size, self.value_hl_dims)
        elif self.value_type == 'V':
            self.value = ValueNetwork(self.v_obs_size, self.value_hl_dims)
        elif self.value_type == 'Q' and self.linear_value:
            self.value = LinearQ(self.v_obs_size, self.value_hl_dims)
        else:
            self.value = QNetwork(self.v_obs_size, self.value_hl_dims)

        if n_policies == 1:
            self.policy = PolicyNetwork(self.obs_size, self.action_size, self.policy_hl_dims)
        else:
            self.policy = [PolicyNetwork(self.obs_size, self.action_size, self.policy_hl_dims) for _ in range(n_policies)]
        
    @abstractmethod
    def train(self, utility) -> None:
        """
        Updates weights of policy and value networks. 

        Parameters
        ----------
        utility: int, float or other scalar representation
            The agent's utility at the current time step
        """
        pass

    ## TO DO: add util methods to this class as necessary

