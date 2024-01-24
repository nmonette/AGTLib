from time import sleep

import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from ..cooperative.pg import GDmax, SoftmaxPolicy
from ..cooperative.base import PolicyNetwork
from ..utils.env import MultiGridWrapper


def grid_experiment_3x3(env1):

    dim = 3
    
    gdm = GDmax(15,4, env1, param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 2, 4], n_rollouts=100)
    
    for i in range(1000):
        gdm.step()

    team = gdm.team_policy
    torch.save(team.state_dict(), f"{dim}x{dim}-team-policy.pt")
    adv = gdm.adv_policy
    torch.save(adv.state_dict(), f"{dim}x{dim}-adv-policy.pt")

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title("Adversary Mean Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Mean Reward")
    ax1.plot(gdm.episode_avg_adv_rewards)
    ax2.set_title("Team Mean Episode Rewards")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean Reward")
    ax2.plot(gdm.episode_avg_team_rewards)

    fig.savefig("experiment_rewards.png")

    
    """
    team = SoftmaxPolicy(2, 4, [dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 2, 4])
    adv = PolicyNetwork(15, 4)

    team.load_state_dict(torch.load(f"{dim}x{dim}-team-policy.pt"))
    adv.load_state_dict(torch.load(f"{dim}x{dim}-adv-policy.pt"))
    """

    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    for i in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[0]).float())
            action = {i: team_action[i] for i in range(len(team_action))}
            action[len(action)] = adv_action

            obs, reward, trunc, done, _ = env.step(action)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break