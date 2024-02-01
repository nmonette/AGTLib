from time import sleep, time

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
# from pettingzoo.mpe import simple_adversary_v3

from ..cooperative.base import PolicyNetwork
from ..cooperative.pg import GDmax, LGDmax
from ..cooperative.pg import MAPolicyNetwork
from ..cooperative.lambda_pg import NLGDmax, TwoHeadPolicy
from ..utils.env import MultiGridWrapper

# ray.init()

def grid_experiment_3x3(env1):
    dim = 3
    # lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True)
    # gdm = PGDMax(15,4, lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], n_rollouts=10, lr=0.1)

    gdm = GDmax(15,4, lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 4,4], n_rollouts=50, lr=0.1)
    time_taken_sum = 0
    # gdm = NGDmax(15,4, lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, size=5, max_episode_steps=12)), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], n_rollouts=50, lr=0.01)
    for i in range(100):
        x = time()
        gdm.step() # 4
        print(f"Iteration {i} done in {time() - x}s\t", end="")
        time_taken_sum += time() - x
        print("Estimated time remaining: ", (10000 - i) * (time_taken_sum / (i+1)))
        if i % 1000 == 0:
            team = gdm.team_policy
            torch.save(team.state_dict(), f"{dim}x{dim}-team-policy-step{i+1}.pt")
            adv = gdm.adv_policy
            torch.save(adv.state_dict(), f"{dim}x{dim}-adv-policy-step{i+1}.pt")
    team = gdm.team_policy
    torch.save(team.state_dict(), f"{dim}x{dim}-team-policy-final.pt")
    adv = gdm.adv_policy
    torch.save(adv.state_dict(), f"{dim}x{dim}-adv-policy-final.pt")

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

    team.load_state_dict(torch.load(f"{dim}x{dim}-team-policy-final.pt"))
    adv.load_state_dict(torch.load(f"{dim}x{dim}-adv-policy-final.pt"))

    
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
            # print(torch.nn.Softmax()(adv.forward(torch.tensor(obs[0]).float())))
            action = {i: team_action[i] for i in range(len(team_action))}
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

# def mpe_experiment():
    # env = lambda: PettingZooWrapper(simple_adversary_v3.parallel_env(N=2, continuous_actions=False, max_cycles=12))
    # gdm = NGDmax(10,5, env, param_dims=None, n_rollouts=50, lr=0.1, adv_obs_size=8)

    # for i in range(100):
    #     x = time()
    #     gdm.step() # 4
    #     print(f"iteration {i} done in {time() - x}s")
    #     if i % 1000 == 0:
    #         team = gdm.team_policy
    #         torch.save(team.state_dict(), f"mpe-team-policy-step{i+1}.pt")
    #         adv = gdm.adv_policy
    #         torch.save(adv.state_dict(), f"mpe-adv-policy-step{i+1}.pt")
            
            
    # team = gdm.team_policy
    # torch.save(team.state_dict(), f"mpe-team-policy-final.pt")
    # adv = gdm.adv_policy
    # torch.save(adv.state_dict(), f"mpe-adv-policy-final.pt")

    # fig, (ax1, ax2) = plt.subplots(1, 2)

    # fig.suptitle("GDMax with MPE")
    # ax1.set_title("Adversary Mean Episode Rewards")
    # ax1.set_xlabel("Iterations")
    # ax1.set_ylabel("Mean Reward")
    # ax1.plot(gdm.episode_avg_adv_rewards)
    # ax2.set_title("Team Mean Episode Rewards")
    # ax2.set_xlabel("Iterations")
    # ax2.set_ylabel("Mean Reward")
    # ax2.plot(gdm.episode_avg_team_rewards)

    # fig.savefig("gdmax_experiment_rewards_mpe.png")
    # team = MAPolicyNetwork(10, 25, [(i,j) for i in range(5) for j in range(5)])
    # adv = PolicyNetwork(8, 5)

    # team.load_state_dict(torch.load(f"mpe-team-policy-final.pt"))
    # adv.load_state_dict(torch.load(f"mpe-adv-policy-final.pt"))

    
    # env = PettingZooWrapper(simple_adversary_v3.parallel_env(N=2, continuous_actions=False, render_mode="rgb_array"))
    # obs, _ = env.reset()
    # env = RecordVideo(env, "mpe_video", lambda v: True)
    # env.start_video_recorder()
    # for episode in range(4):
    #     obs, _ = env.reset()
    #     print("reset")
    #     # env.render()
    #     while True:
    #         team_action, _ = team.get_actions(obs[0])
    #         adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
    #         # print(torch.nn.Softmax()(adv.forward(torch.tensor(obs[0]).float())))
    #         action = {i: team_action[i] for i in range(len(team_action))}
    #         action[len(action)] = adv_action.item()
    #         obs, reward, trunc, done, _ = env.step(action)
    #         env.render()
    #         # print(trunc, done)
    #         # print(action, reward)
    #         # sleep(0.5)
    #         if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
    #             break

    # env.close_video_recorder()


def lgdmax_grid_experiment():
    num_states = torch.prod(torch.tensor([3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2])).item()
    table = torch.from_numpy(np.load("3x3-3-agents-table.npy")).reshape(num_states, 4, 16).double()
    gdm = LGDmax(15, 4, num_states, [(i,j) for i in range(4) for j in range(4)], [3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 16], [3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 4],
                 table, env=lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, size=5, max_episode_steps=12)))

    for i in range(100):
        x = time()
        gdm.step() # 4
        print(f"iteration {i} done in {time() - x}s")
    
    team = gdm.team_policy
    torch.save(team.state_dict(), "3x3-team-policy-final-lambda.pt")
    adv = gdm.adv_policy
    torch.save(adv.state_dict(), "3x3-adv-policy-final-lambda.pt")

    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
            adv_action = adv_action.item()
            # print(torch.nn.Softmax()(adv.forward(torch.tensor(obs[0]).float())))
            action = {i: team_action[i] for i in range(len(team_action))}
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

def nlgdmax_grid_experiment():
    gdm = NLGDmax(15, 4, [(i,j) for i in range(4) for j in range(4)], env=lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, size=5, max_episode_steps=12)),lr=0.1)

    # mkdir -p output
    import os
    os.makedirs("output", exist_ok=True)

    def save(iteration="end"):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        fig.suptitle("NLGDmax with Adversarial TMG")
        ax1.set_title("Adversary Mean Episode Rewards")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Mean Reward")
        ax1.plot([-i for i in gdm.reward])
        ax2.set_title("Team Mean Episode Rewards")
        ax2.set_xlabel("Iterations")
        ax2.set_ylabel("Mean Reward")
        ax2.plot(gdm.reward)

        fig.savefig("output/" + str(iteration) + "-lgdmax_experiment_rewards.png")
        
        team = gdm.team_policy
        torch.save(team.state_dict(), "output/" + str(iteration) + "-3x3-team-policy-final-nlambda.pt")
        adv = gdm.adv_policy
        torch.save(adv.state_dict(), "output/" + str(iteration) + "-3x3-adv-policy-final-nlambda.pt")

    time_taken_sum = 0
    for i in range(100):
        x = time()
        gdm.step() # 4
        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (10000 - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        
        if i % 500 == 0:
            # Save progress
            print("Saving progress...")
            save(i)
            
    save()

    # env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    # for episode in range(100):
    #     obs, _ = env.reset()
    #     env.render()
    #     while True:
    #         team_action, _ = team.get_actions(obs[0])
    #         adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
    #         adv_action = adv_action.item()
    #         # print(torch.nn.Softmax()(adv.forward(torch.tensor(obs[0]).float())))
    #         action = {i: team_action[i] for i in range(len(team_action))}
    #         action[len(action)] = adv_action
    #         obs, reward, trunc, done, _ = env.step(action)
    #         print(action, reward)
    #         sleep(0.5)
    #         if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
    #             break

def test_lgdmax_weights():
    team = MAPolicyNetwork(15, 4*4, [(i,j) for i in range(4) for j in range(4)])
    adv = TwoHeadPolicy(15, 4, fm_dim1=64, fm_dim2=128)

    team.load_state_dict(torch.load("3x3-team-policy-final-nlambda.pt"))
    adv.load_state_dict(torch.load("3x3-adv-policy-final-nlambda.pt"))

    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
            adv_action = adv_action.item()
            action = {i: team_action[i] for i in range(len(team_action))}
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break