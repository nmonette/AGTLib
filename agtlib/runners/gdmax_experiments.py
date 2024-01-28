from time import sleep, time

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

from ..cooperative.base import PolicyNetwork
from ..cooperative.pg import GDmax as GDMax
from ..cooperative.pg import MAPolicyNetwork, NGDmax, SoftmaxPolicy
from ..cooperative.pg_parallel import GDmax as PGDMax
from ..cooperative.ppo import advPPO
from ..utils.env import MultiGridWrapper

# ray.init()

def grid_experiment_3x3(env1):
    dim = 3
    
    gdm = PGDMax(15,4, lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], n_rollouts=10, lr=0.1)
    # gdm = GDMax(15,4, lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 4,4], n_rollouts=50, lr=0.1)
    # gdm = NGDmax(15,4, lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], n_rollouts=50, lr=0.1)
    for i in range(100):
        x = time()
        gdm.step(4) # 4
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
        return MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, size=5, max_episode_steps=12))

    for i in range(10):
        x = time()
        ppo.step(make_env, n_envs=4)
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
    team = MAPolicyNetwork(15, 16, [(i,j) for i in range(4) for j in range(4)]) # SoftmaxPolicy(2, 4, [dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 2, 4])
    adv = PolicyNetwork(15, 4)

    team.load_state_dict(torch.load(f"{dim}x{dim}-team-policy.pt"))
    adv.load_state_dict(torch.load(f"{dim}x{dim}-adv-policy.pt"))

    
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
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break