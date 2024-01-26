from typing import Iterable, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from .base import RLBase, PolicyNetwork, ValueNetwork, ActorCritic
from ..utils.rollout import RolloutBuffer, RolloutManager

from agtlib.utils.stable_baselines.vec_env.base_vec_env import VecEnv
from agtlib.utils.stable_baselines.vec_env.subproc_vec_env import SubprocVecEnv
from agtlib.utils.stable_baselines.monitor import Monitor

# [(i.j,k) for i in range(4) for j in range(4) for k in range(4)][0]

class PPO(RLBase):
    """
    Base Implementation of Proximal Policy Optimization. Inspired by the OpenAI stable baselines 
    implementation: https://github.com/DLR-RM/stable-baselines3, as well as the seminal paper:
    https://arxiv.org/abs/1707.06347.Implementation. Also
    used the https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ blog post as a reference.
    """
    def __init__(self, action_size: int, obs_size: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    gae_lambda: float = 0.95, clip_range: float = 0.2, clip_range_vf: float = 0.2,
                    vf_coef: float = 0.5, ent_coef: float = 0.0, target_kl: float = None, normalize_advantage: bool = True, 
                    verbose: int = 0, max_grad_norm: float = 0.5):
        '''
        Initialize PPO parameters.
        Parameters
        -------
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
            An iterable such that the ith element represents 
            the width of the ith hidden layer
            of the policy network. Defaults to `[64,128]`.
            Note that this does not include the 
            input or output layers.
        value_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents the 
            width of the ith hidden layer of the value network. 
            Defaults to `[64,128]`. Note that this does not include 
            the input or output layers. Must be length 1 if linear_value 
            is enabled, with the value equal to the number 
            of linear features.
        linear_value: bool, optional, keyword only
            Indicates whether or not the value function is to be 
            approximated linearly. Defaults to `False`.
        gamma: float, optional, keyword only
            The dicount factor. Defaults to `0.99`.
        gae_lambda: float, optional, keyword only
            Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Defaults to `0.95`.
        clip_range: float, optional, keyword only
            The clip range of the ratio for the policy Defaults to `0.2`,
            meaning the value is restricted to 1.2 and 0.8.
        clip_range_vf: float, optional, keyword only
            The clip range for the value function. Defaults to `0.2`, 
            meaning the value is restricted to 1.2 and 0.8.
        vf_coef: float, optional, keyword only
            Coefficient for the Value function in the calculation of loss.
            Defaults to `0.5`.__async_iterable 
        ent_coef: float, optional, keyword only
            Coefficient for the entropy in the calculation of loss. 
            Defaults to `0.0`.
        target_kl: float, optional, keyword only
            KL divergence between old and new policies
            that stops training when reached.
            Defaults to `None`. 
        normalize_advantage: bool, optional, keyword only
            Boolean for determining whether to normalize the advantage. 
            Defaults to `True`.
        max_grad_norm: float, optional, keyword
            Maximum value for the clipping of the gradient. Defaults to `0.5`.
        '''
        super().__init__(action_size, obs_size, v_obs_size=v_obs_size, policy_hl_dims=policy_hl_dims, value_hl_dims=value_hl_dims, 
                         linear_value=linear_value, gamma=gamma)
        
        if not (0 <= gae_lambda <= 1):
            raise ValueError("Parameter 'gae_lambda' is not in the range `[0,1]`")
        else:
            self.gae_lambda = gae_lambda
        
        if not (0 <= clip_range < 1):
            raise ValueError("Parameter 'clip_range' is not in the range `[0,1)`")
        else:
            self.clip_range = clip_range

        if not (0 <= clip_range_vf < 1):
            raise ValueError("Parameter 'clip_range' is not in the range `[0,1)`")
        else:
            self.clip_range_vf = clip_range_vf

        if not (0 <= vf_coef < 1):
            raise ValueError("Parameter 'vf_coef' is not in the range `[0,1)`")
        else:
            self.vf_coef = vf_coef

        if not (0 <= ent_coef < 1):
            raise ValueError("Parameter 'ent_coef' is not in the range `[0,1)`")
        else:
            self.ent_coef = ent_coef
        
        if not (0 <= ent_coef < 1):
            raise ValueError("Parameter 'ent_coef' is not in the range `[0,1)`")
        else:
            self.ent_coef = ent_coef
        

        self.normalize_advantage = True # TODO write if else
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.verbose = verbose
        self.max_grad_norm = max_grad_norm

        policy_params = nn.utils.parameters_to_vector(self.policy.parameters())
        value_params = nn.utils.parameters_to_vector(self.value.parameters())
        self.actor_critic = ActorCritic(self.policy, self.value)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), eps=1e-5) # can make it so that there are other compatible optimizers in the future

    def preprocess(self, obs: torch.Tensor):
        '''
        Preprocesses observations.
        Parameters
        -------
        obs: np.ndarray
            Array containing the flattened observation at each time step.

        Returns
        -------
        torch.Tensor
            Tensor that contains the output of actor_extractor
        torch.Tensor
            Tensor that contains the output of critic_extractor
        '''
        return nn.Flatten()(obs).float()
        

    def _evaluate_actions(self, obs: torch.Tensor, actions: np.ndarray):
        """
        Evaluates analysis of the rollout data that is necessary for 
        calculation of the PPO loss. 
        Parameters
        ----------
        obs: np.ndarray
            Array containing the flattened observation at each time step.
        actions: np.ndarray
            Array containing the integer index of each action at each time step.

        Returns
        -------
        torch.Tensor
            Tensor containing the values for each state at each time step.
        torch.Tensor
            Tensor containing the log probabilities for each action at each time step.
        torch.Tensor
            Tensor containing the entropy of the action distribution at each time step.
        """
        obs = self.preprocess(obs)
        values = self.value(obs)
        action_logits = self.policy(obs)

        dist = torch.distributions.Categorical(logits=action_logits)
        
        log_prob = dist.log_prob(actions) #
        entropy = dist.entropy()

        return values, log_prob, entropy 

    def train(self, buffer: RolloutBuffer, n_epochs: int = 1, batch_size: int = 32) -> None:
        """
        Performs a training update with rollout data.
        Intended to be performed in parallel with 
        other agent. 
        Parameter
        ---------
        buffer: RolloutBuffer
            Data collected from the rollout.
        n_epochs: int, optional
            Number of passes for the model to go through 
            the rollout data. Defaults to `1`.
        batch_size: int, optional
            Size of each minibatch for the model to train
            on. Defaults to `32`.
        """
        continue_training = True

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        
        for epoch in range(n_epochs):
            approx_kl_divs = []
            for data in buffer.get_data(batch_size):
                actions = data.action_buffer

                values, log_prob, entropy = self._evaluate_actions(data.obs_buffer, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = data.adv_buffer
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - data.log_prob_buffer)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = data.value_buffer + torch.clamp(
                        values - data.value_buffer, -self.clip_range_vf, self.clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(data.return_buffer, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (policy_loss.to(torch.float32) + self.ent_coef * entropy_loss + self.vf_coef * value_loss.to(torch.float32)).float()
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - data.log_prob_buffer
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    print(f"Early Stopping because KL Divergence is {approx_kl_div}.")
                    break

                # Optimization step
                self.optimizer.zero_grad() # self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step() # self.policy.optimizer.step()
            
            if not continue_training:
                break

def train_ppo():
    ppo = PPO(2, 4, policy_hl_dims=[16], value_hl_dims=[16])

    def create_env():
        env = Monitor(gym.make("CartPole-v1"))
        # env = SingleAgentEnvWrapper(env)
        return env

    multi_env = SubprocVecEnv([create_env for _ in range(10)])
    next_obs = multi_env.reset()
    for epoch in range(100):
        rollout = RolloutManager(5, multi_env, [ppo.policy], [ppo.value], "PPO", n_envs=10)
        buffer, next_obs = rollout.rollout(next_obs)
        buffer = buffer[0]

        ppo.train(buffer, 5, 64)

    x = [torch.tensor(y) for y in multi_env.env_method("get_episode_rewards")]
    print("mean reward:", torch.mean(torch.cat(x)).item())

    multi_env.close()

    env = gym.make("CartPole-v1", render_mode="human")
    for demo in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            obs, reward, done, trunc, _ = env.step(ppo.policy.get_action(torch.from_numpy(obs))[0].item())
    
            if done or trunc:
                break


# class MAPPO(PPO):
#     ## TO DO: fix this, code was written for sake of explanation
#     def __init__(self, teams: [int, ], env: gym.Env, ctde: bool = False):
#         self.policy_groups = teams

#         self.policies = [PolicyNetwork(...) for i in range(len(set(teams)))]

#         self.rollout = RolloutManager(self.policies...)

#     def train(self):
#         for i in range(epochs):
#             data = self.rollout.rollout()
#             for j in range(len()):
#                 self.policies[j].train(data[j])
            
class PPO_CTDE(PPO):
    """Housing for each multi-policy, single-value unit in a MAPPO-CTDE 
    model."""
    def __init__(self, action_size: int, obs_size: int, n_agents: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    gae_lambda: float = 0.95, clip_range: float = 0.2, clip_range_vf: float = 0.2,
                    vf_coef: float = 0.5, ent_coef: float = 0.0, target_kl: float = None, normalize_advantage: bool = True, 
                    verbose: int = 0, max_grad_norm: float = 0.5):
        '''
        Initialize PPO parameters.
        Parameters
        -------
        action_size: int
            The cardinality of the action space of a single agent.
        obs_size: int
            The length of the flattened observation of the agent(s). 
        n_agents: int
            The number of agents in the environment. Agent groups (e.g. 
            teams) are not needed to be specified as per the independent
            nature of the model.
        v_obs_size: int, optional, keyword only
            The length of the flattened observation size of the agent(s).
            Only specify if different from policy `obs_size` (e.g. the 
            policy network uses locallized observations, but the value 
            network uses the joint observation of multiple agents).
        policy_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents 
            the width of the ith hidden layer
            of the policy network. Defaults to `[64,128]`.
            Note that this does not include the 
            input or output layers.
        value_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents the 
            width of the ith hidden layer of the value network. 
            Defaults to `[64,128]`. Note that this does not include 
            the input or output layers. Must be length 1 if linear_value 
            is enabled, with the value equal to the number 
            of linear features.
        linear_value: bool, optional, keyword only
            Indicates whether or not the value function is to be 
            approximated linearly. Defaults to `False`.
        gamma: float, optional, keyword only
            The dicount factor. Defaults to `0.99`.
        gae_lambda: float, optional, keyword only
            Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Defaults to `0.95`.
        clip_range: float, optional, keyword only
            The clip range of the ratio for the policy Defaults to `0.2`,
            meaning the value is restricted to 1.2 and 0.8.
        clip_range_vf: float, optional, keyword only
            The clip range for the value function. Defaults to `0.2`, 
            meaning the value is restricted to 1.2 and 0.8.
        vf_coef: float, optional, keyword only
            Coefficient for the Value function in the calculation of loss.
            Defaults to `0.5`.__async_iterable 
        ent_coef: float, optional, keyword only
            Coefficient for the entropy in the calculation of loss. 
            Defaults to `0.0`.
        target_kl: float, optional, keyword only
            KL divergence between old and new policies
            that stops training when reached.
            Defaults to `None`. 
        normalize_advantage: bool, optional, keyword only
            Boolean for determining whether to normalize the advantage. 
            Defaults to `True`.
        max_grad_norm: float, optional, keyword
            Maximum value for the clipping of the gradient. Defaults to `0.5`.
        '''
        super().__init__(action_size, obs_size, 
                        v_obs_size=v_obs_size, 
                        policy_hl_dims=policy_hl_dims,
                        value_hl_dims=value_hl_dims,
                        linear_value=linear_value,
                        gamma=gamma, 
                        gae_lambda=gae_lambda,
                        clip_range=clip_range,
                        clip_range_vf=clip_range_vf,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        target_kl=target_kl,
                        normalize_advantage=normalize_advantage,
                        verbose=verbose,
                        max_grad_norm=max_grad_norm, 
                        n_policies=n_agents)
        
        self.n_agents = n_agents
        

            
class MAPPO_CTDE:
    """
    Multi-Agent PPO with centralized training and decentralized 
    execution. Has a shared value function and decentralized 
    policy functions for each policy group.
    """
    pass

class MAPPO:
    """
    Fully centralized Multi-Agent PPO.
    """
    def __init__(self, action_size: int, obs_size: int, n_agents: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    gae_lambda: float = 0.95, clip_range: float = 0.2, clip_range_vf: float = 0.2,
                    vf_coef: float = 0.5, ent_coef: float = 0.0, target_kl: float = None, normalize_advantage: bool = True, 
                    verbose: int = 0, max_grad_norm: float = 0.5, policy_groups):
        '''
        Initialize PPO parameters.
        Parameters
        -------
        action_size: int
            The cardinality of the action space of a single agent.
        obs_size: int
            The length of the flattened observation a single agent. 
        n_agents: int
            The number of agents in the environment. 
        v_obs_size: int, optional, keyword only
            The length of the flattened observation size of the agent(s).
            Only specify if different from policy `obs_size` (e.g. the 
            policy network uses locallized observations, but the value 
            network uses the joint observation of multiple agents).
        policy_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents 
            the width of the ith hidden layer
            of the policy network. Defaults to `[64,128]`.
            Note that this does not include the 
            input or output layers.
        value_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents the 
            width of the ith hidden layer of the value network. 
            Defaults to `[64,128]`. Note that this does not include 
            the input or output layers. Must be length 1 if linear_value 
            is enabled, with the value equal to the number 
            of linear features.
        linear_value: bool, optional, keyword only
            Indicates whether or not the value function is to be 
            approximated linearly. Defaults to `False`.
        gamma: float, optional, keyword only
            The dicount factor. Defaults to `0.99`.
        gae_lambda: float, optional, keyword only
            Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Defaults to `0.95`.
        clip_range: float, optional, keyword only
            The clip range of the ratio for the policy Defaults to `0.2`,
            meaning the value is restricted to 1.2 and 0.8.
        clip_range_vf: float, optional, keyword only
            The clip range for the value function. Defaults to `0.2`, 
            meaning the value is restricted to 1.2 and 0.8.
        vf_coef: float, optional, keyword only
            Coefficient for the Value function in the calculation of loss.
            Defaults to `0.5`.__async_iterable 
        ent_coef: float, optional, keyword only
            Coefficient for the entropy in the calculation of loss. 
            Defaults to `0.0`.
        target_kl: float, optional, keyword only
            KL divergence between old and new policies
            that stops training when reached.
            Defaults to `None`. 
        normalize_advantage: bool, optional, keyword only
            Boolean for determining whether to normalize the advantage. 
            Defaults to `True`.
        max_grad_norm: float, optional, keyword
            Maximum value for the clipping of the gradient. Defaults to `0.5`.
        policy_groups: Iterable(int), optional
            An iterable such that the ith element represents the integer index
            policy of the ith agent. Used to specify which agents share which
            policy. Defaults to `None`. If None specified, assumes all agents
            share the same policy.
        '''
        self.ppo = self.ppo = [PPO(action_size, obs_size, 
                        v_obs_size=v_obs_size, 
                        policy_hl_dims=policy_hl_dims,
                        value_hl_dims=value_hl_dims,
                        linear_value=linear_value,
                        gamma=gamma, 
                        gae_lambda=gae_lambda,
                        clip_range=clip_range,
                        clip_range_vf=clip_range_vf,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        target_kl=target_kl,
                        normalize_advantage=normalize_advantage,
                        verbose=verbose,
                        max_grad_norm=max_grad_norm) 
                        for _ in range(len(set(policy_groups)))]
        
        self.policy_groups = policy_groups
        self.gamma = gamma
        self.gae_lambda = gae_lambda
    
    def train(self, make_env: Callable[[None], gym.Env], *, n_envs: int = 1, rollout_length: int = 100, batch_size: int = 32, n_epochs: int = 10, n_updates: int = 1000):
        """
        Function to handle training of the models. 
        Parameters
        ----------
        make_env: Callable() -> gym.Env
            Callable that returns a gym.Env. 
        n_envs: int, optional
            Number of environments to run in parallel during rollouts. 
            Defaults to `1`. 
        rollout_length: int, optional
            Length (per environment) of the "fixed-length trajectory segments" as per the 
            seminal PPO paper (https://arxiv.org/pdf/1707.06347.pdf).
        batch_size: int, optional
            Length of the minibatches of data that is processed at a time per training update.
            Defaults to `32`.
        n_epochs: int, optional
            Number of epochs (i.e. passes through a batch) per training update.
            Defaults to `10`.
        n_updates: int, optional
            Number of training updates. The real number of updates is equal to 
            `n_updates * n_epochs * (rollout_length * n_envs // batch_size)`.
            Defaults to 1000.
        """
        policies = []
        values = []
        for ppo in self.ppo:
            policies.append(ppo.policy)
            values.append(ppo.value)

        env = SubprocVecEnv([make_env for _ in range(n_envs)])
        next_obs = env.reset()

        for update in range(n_updates):
            rm = RolloutManager(rollout_length, env, policies, values, gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=n_envs, policy_groups=self.policy_groups)
            buffers, next_obs = rm.rollout(next_obs)

            for i in range(len(self.ppo)): # do we want to parallelize this... can we even do that?
                self.ppo[i].train(buffers[i], n_epochs, batch_size)

class IPPO:
    """
    Fully independent Multi-agent PPO. Only supports 
    homogeneous action spaces. 
    """
    def __init__(self, action_size: int, obs_size: int, n_agents: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    gae_lambda: float = 0.95, clip_range: float = 0.2, clip_range_vf: float = 0.2,
                    vf_coef: float = 0.5, ent_coef: float = 0.0, target_kl: float = None, normalize_advantage: bool = True, 
                    verbose: int = 0, max_grad_norm: float = 0.5):
        '''
        Initialize PPO parameters.
        Parameters
        -------
        action_size: int
            The cardinality of the action space of a single agent.
        obs_size: int
            The length of the flattened observation a single agent. 
        n_agents: int
            The number of agents in the environment. Agent groups (e.g. 
            teams) are not needed to be specified as per the independent
            nature of the model.
        v_obs_size: int, optional, keyword only
            The length of the flattened observation size of the agent(s).
            Only specify if different from policy `obs_size` (e.g. the 
            policy network uses locallized observations, but the value 
            network uses the joint observation of multiple agents).
        policy_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents 
            the width of the ith hidden layer
            of the policy network. Defaults to `[64,128]`.
            Note that this does not include the 
            input or output layers.
        value_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents the 
            width of the ith hidden layer of the value network. 
            Defaults to `[64,128]`. Note that this does not include 
            the input or output layers. Must be length 1 if linear_value 
            is enabled, with the value equal to the number 
            of linear features.
        linear_value: bool, optional, keyword only
            Indicates whether or not the value function is to be 
            approximated linearly. Defaults to `False`.
        gamma: float, optional, keyword only
            The dicount factor. Defaults to `0.99`.
        gae_lambda: float, optional, keyword only
            Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Defaults to `0.95`.
        clip_range: float, optional, keyword only
            The clip range of the ratio for the policy Defaults to `0.2`,
            meaning the value is restricted to 1.2 and 0.8.
        clip_range_vf: float, optional, keyword only
            The clip range for the value function. Defaults to `0.2`, 
            meaning the value is restricted to 1.2 and 0.8.
        vf_coef: float, optional, keyword only
            Coefficient for the Value function in the calculation of loss.
            Defaults to `0.5`.__async_iterable 
        ent_coef: float, optional, keyword only
            Coefficient for the entropy in the calculation of loss. 
            Defaults to `0.0`.
        target_kl: float, optional, keyword only
            KL divergence between old and new policies
            that stops training when reached.
            Defaults to `None`. 
        normalize_advantage: bool, optional, keyword only
            Boolean for determining whether to normalize the advantage. 
            Defaults to `True`.
        max_grad_norm: float, optional, keyword
            Maximum value for the clipping of the gradient. Defaults to `0.5`.
        policy_groups: Iterable(int), optional
            An iterable such that the ith element represents the integer index
            policy of the ith agent. Used to specify which agents share which
            policy. Defaults to `None`. Assumes that if none is specified,
            there is no shared policy.
        '''
        self.ppo = [PPO(action_size, obs_size, 
                        v_obs_size=v_obs_size, 
                        policy_hl_dims=policy_hl_dims,
                        value_hl_dims=value_hl_dims,
                        linear_value=linear_value,
                        gamma=gamma, 
                        gae_lambda=gae_lambda,
                        clip_range=clip_range,
                        clip_range_vf=clip_range_vf,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        target_kl=target_kl,
                        normalize_advantage=normalize_advantage,
                        verbose=verbose,
                        max_grad_norm=max_grad_norm) 
                        for _ in range(n_agents)]
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # self.n_agents = n_agents
        
    def train(self, make_env: Callable[[None], gym.Env], *, n_envs: int = 1, rollout_length: int = 100, batch_size: int = 32, n_epochs: int = 10, n_updates: int = 1000):
        """
        Function to handle training of the models. 
        Parameters
        ----------
        make_env: Callable() -> gym.Env
            Callable that returns a gym.Env. 
        n_envs: int, optional
            Number of environments to run in parallel during rollouts. 
            Defaults to `1`. 
        rollout_length: int, optional
            Length (per environment) of the "fixed-length trajectory segments" as per the 
            seminal PPO paper (https://arxiv.org/pdf/1707.06347.pdf).
        batch_size: int, optional
            Length of the minibatches of data that is processed at a time per training update.
            Defaults to `32`.
        n_epochs: int, optional
            Number of epochs (i.e. passes through a batch) per training update.
            Defaults to `10`.
        n_updates: int, optional
            Number of training updates. The real number of updates is equal to 
            `n_updates * n_epochs * (rollout_length * n_envs // batch_size)`.
            Defaults to 1000.
        """
        # make_env_new = lambda: Monitor(make_env())
        
        policies = []
        values = []
        rewards = []
        for ppo in self.ppo:
            policies.append(ppo.policy)
            values.append(ppo.value)

        env = SubprocVecEnv([make_env for _ in range(n_envs)])
        next_obs = env.reset()

        for update in range(n_updates):
            rm = RolloutManager(rollout_length, env, policies, values, "IPPO", gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=n_envs)
            buffers, next_obs = rm.rollout(next_obs)

            for i in range(len(self.ppo)): # do we want to parallelize this... can we even do that?
                self.ppo[i].train(buffers[i], n_epochs, batch_size)

        

        ## TODO: write monitor class so that the code below actually works

        # rewards = env.env_method("get_episode_rewards")
        # mean_rewards = defaultdict(int)
        # count = 0
        # for i in range(len(rewards)):
        #     for j in rewards[i]:
        #         for k in range(len(self.ppo)):
        #             mean_rewards[k] += j[k]
        #         count += 1
        
        # for i in range(len(mean_rewards)):
        #     mean_rewards[i] = mean_rewards[i] / count

        # print("mean rewards:")
        # for i in range(len(self.ppo)):
        #     print(i, ': ', mean_rewards[i])
    
class advPPO:
    """
    Fully centralized Multi-Agent PPO playing against an adversary playing PPO.
    """
    def __init__(self, action_size: int, obs_size: int, n_agents: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    gae_lambda: float = 0.95, clip_range: float = 0.2, clip_range_vf: float = 0.2,
                    vf_coef: float = 0.5, ent_coef: float = 0.0, target_kl: float = None, normalize_advantage: bool = True, 
                    verbose: int = 0, max_grad_norm: float = 0.5):
        '''
        Initialize PPO parameters.
        Parameters
        -------
        action_size: int
            The cardinality of the action space of a single agent.
        obs_size: int
            The length of the flattened observation a single agent. 
        n_agents: int
            The number of agents in the environment. 
        v_obs_size: int, optional, keyword only
            The length of the flattened observation size of the agent(s).
            Only specify if different from policy `obs_size` (e.g. the 
            policy network uses locallized observations, but the value 
            network uses the joint observation of multiple agents).
        policy_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents 
            the width of the ith hidden layer
            of the policy network. Defaults to `[64,128]`.
            Note that this does not include the 
            input or output layers.
        value_hl_dims: Iterable(int), optional, keyword only
            An iterable such that the ith element represents the 
            width of the ith hidden layer of the value network. 
            Defaults to `[64,128]`. Note that this does not include 
            the input or output layers. Must be length 1 if linear_value 
            is enabled, with the value equal to the number 
            of linear features.
        linear_value: bool, optional, keyword only
            Indicates whether or not the value function is to be 
            approximated linearly. Defaults to `False`.
        gamma: float, optional, keyword only
            The dicount factor. Defaults to `0.99`.
        gae_lambda: float, optional, keyword only
            Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Defaults to `0.95`.
        clip_range: float, optional, keyword only
            The clip range of the ratio for the policy Defaults to `0.2`,
            meaning the value is restricted to 1.2 and 0.8.
        clip_range_vf: float, optional, keyword only
            The clip range for the value function. Defaults to `0.2`, 
            meaning the value is restricted to 1.2 and 0.8.
        vf_coef: float, optional, keyword only
            Coefficient for the Value function in the calculation of loss.
            Defaults to `0.5`.__async_iterable 
        ent_coef: float, optional, keyword only
            Coefficient for the entropy in the calculation of loss. 
            Defaults to `0.0`.
        target_kl: float, optional, keyword only
            KL divergence between old and new policies
            that stops training when reached.
            Defaults to `None`. 
        normalize_advantage: bool, optional, keyword only
            Boolean for determining whether to normalize the advantage. 
            Defaults to `True`.
        max_grad_norm: float, optional, keyword
            Maximum value for the clipping of the gradient. Defaults to `0.5`.
        '''
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_agents = n_agents
        
        self.team_action_map = [(i,j) for i in range(action_size) for j in range(action_size)] # dim 2 for multigrid
        self.team_ppo = PPO(action_size**(n_agents-1), obs_size, 
                        v_obs_size=v_obs_size, 
                        policy_hl_dims=policy_hl_dims,
                        value_hl_dims=value_hl_dims,
                        linear_value=linear_value,
                        gamma=gamma, 
                        gae_lambda=gae_lambda,
                        clip_range=clip_range,
                        clip_range_vf=clip_range_vf,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        target_kl=target_kl,
                        normalize_advantage=normalize_advantage,
                        verbose=verbose,
                        max_grad_norm=max_grad_norm) 
        self.adv_ppo = PPO(action_size, obs_size, 
                        v_obs_size=v_obs_size, 
                        policy_hl_dims=policy_hl_dims,
                        value_hl_dims=value_hl_dims,
                        linear_value=linear_value,
                        gamma=gamma, 
                        gae_lambda=gae_lambda,
                        clip_range=clip_range,
                        clip_range_vf=clip_range_vf,
                        vf_coef=vf_coef,
                        ent_coef=ent_coef,
                        target_kl=target_kl,
                        normalize_advantage=normalize_advantage,
                        verbose=verbose,
                        max_grad_norm=max_grad_norm) 
        
        self.episode_avg_adv_rewards = []
        self.episode_avg_team_rewards = []
        
    def get_utility(self, env, n_episodes):
        rewards = []

        env = env()

        for episode in range(n_episodes):
            obs, _ = env.reset()
            while True:
                team_action, team_log_prob = self.team_ppo.policy.get_action(torch.tensor(obs[0]).float())
                team_action = self.team_action_map[team_action]
                action = {}
                
                for i in range(self.team_size):
                    action[i] = team_action[i]
                action[i+1], adv_log_prob = self.adv_ppo.policy.get_action(torch.tensor(obs[0]).float())
                obs, reward, done, trunc, _ = self.env.step(action) 
                
                rewards.append(reward[0])

                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break # >= 2 comes from 2 terminal states in treasure hunt

        return -torch.mean(torch.tensor(rewards).float()).item(), torch.mean(torch.tensor(rewards).float()).item()

    def step(self, make_env: Callable[[None], gym.Env], *, n_envs: int = 1, rollout_length: int = 100, batch_size: int = 32, n_epochs: int = 10, n_updates: int = 1000):
        rewards = []
        policies = [self.team_ppo.policy, self.adv_ppo.policy]
        values = [self.team_ppo.value, self.adv_ppo.value]
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
        next_obs = env.reset()

        rm = RolloutManager(rollout_length, env, policies, values, "advPPO", gamma=self.gamma, gae_lambda=self.gae_lambda, n_envs=n_envs)

        for i in range(rollout_length):
            buffers, next_obs = rm.rollout(next_obs, self.team_action_map)
            self.adv_ppo.train(buffers[1], 100)

        buffers, next_obs = rm.rollout(next_obs, self.team_action_map)
        self.team_ppo.train(buffers[0], 100)

        adv_utility, team_utility = self.get_utility(make_env, rollout_length)
        self.episode_avg_adv_rewards.append(adv_utility)
        self.episode_avg_team_rewards.append(team_utility) 

        
                








        
        
        
