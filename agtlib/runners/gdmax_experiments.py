from time import sleep, time

import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import ray

from ..cooperative.pg import GDmax, SoftmaxPolicy
from ..cooperative.base import PolicyNetwork
from ..cooperative.ppo import advPPO
from ..utils.env import MultiGridWrapper

ray.init()

def grid_experiment_3x3(env1):
    dim = 3
    
    gdm = GDmax(15,4, lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, size=5)), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 2, 4], n_rollouts=10)

    for i in range(1000):
        x = time()
        gdm.step()
        print(f"iteration {i} done in {time() - x}s")

    team = gdm.team_policy
    torch.save(team.state_dict(), f"{dim}x{dim}-team-policy.pt")
    adv = gdm.adv_policy
    torch.save(adv.state_dict(), f"{dim}x{dim}-adv-policy.pt")

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle("GDMax with Adversarial TMG")
    ax1.set_title("Adversary Mean Episode Rewards")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Mean Reward")
    ax1.plot(gdm.episode_avg_adv_rewards)
    ax2.set_title("Team Mean Episode Rewards")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Mean Reward")
    ax2.plot(gdm.episode_avg_team_rewards)

    fig.savefig("gdmax_experiment_rewards.png")
    

    """
    ppo = advPPO(4, 15, 3)
    
    def make_env():
        return MultiGridWrapper(gym.make("MultiGrid-Empty-8x8-Team", agents=3, size=5))

    for i in range(10):
        x = time()
        ppo.step(make_env, n_envs=32)
        print(f"iteration {i} done in {time() - x}s")

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle("PPO with Adversarial TMG")
    ax1.set_title("Adversary Mean Episode Rewards")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Mean Reward")
    ax1.plot(ppo.episode_avg_adv_rewards)
    ax2.set_title("Team Mean Episode Rewards")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel("Mean Reward")
    ax2.plot(ppo.episode_avg_team_rewards)

    team = ppo.team_ppo.policy
    adv = ppo.adv_ppo.policy

    """
    
    """ 
    team = SoftmaxPolicy(2, 4, [dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 2, 4])
    adv = PolicyNetwork(15, 4)

    team.load_state_dict(torch.load(f"{dim}x{dim}-team-policy.pt"))
    adv.load_state_dict(torch.load(f"{dim}x{dim}-adv-policy.pt"))
    """ 

    
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[0]).float())
            # print(torch.nn.Softmax()(adv.forward(torch.tensor(obs[0]).float())))
            action = {i: team_action[i] for i in range(len(team_action))}
            action[len(action)] = adv_action

            obs, reward, trunc, done, _ = env.step(action)
            print(reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break