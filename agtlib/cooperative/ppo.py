from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from .base import RLBase, PolicyNetwork, ValueNetwork
from ..utils.rollout import RolloutBuffer, RolloutManager

class PPO(RLBase):
    """
    Base Implementation of Proximal Policy Optimization. Inspired by the OpenAI stable baselines 
    implementation: https://github.com/DLR-RM/stable-baselines3, as well as the seminal paper:
    https://arxiv.org/abs/1707.06347.Implementation. Also
    used the https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ blog post as a reference.
    """
    def __init__(self, action_size: int, obs_size: int, *, v_obs_size: int = None, policy_hl_dims: Iterable[int, ] = [64,128], \
                 value_hl_dims: Iterable[int, ] = [64, 128], linear_value: bool = False, gamma: float = 0.99, \
                    gae_lambda: float = 0.95, clip_range: float = 0.2, clip_range_vf: float = 0.2, action_latent_dim: int = 100,
                    vf_coef: float = 0.5, ent_coef: float = 0.0, target_kl: float = None, normalize_advantage: bool = True, 
                    verbose: int = 0): # write docstrings for parameters
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

        self.normalize_advantage = True # TODO write if else
        self.ent_coef = ent_coef
        self.target_kl = target_kl
        self.verbose = verbose

        # self.actor_extractor = PolicyNetwork(obs_size, action_latent_dim)
        # self.critic_extractor = ValueNetwork(obs_size)

        # self.latent_net = nn.Linear(action_size)

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

    def train(self, buffer: RolloutBuffer, num_epochs: int = 1, batch_size: int = 32) -> None:
        """
        Performs a training update with rollout data.
        Intended to be performed in parallel with 
        other agent. 
        Parameter
        ---------
        buffer: RolloutBuffer
            Data collected from the rollout.
        num_epochs: int, optional
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
        
        for epoch in range(num_epochs):
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

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

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
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            
            if not continue_training:
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

class IPPO:
    pass

