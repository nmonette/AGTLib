from time import sleep, time
# mkdir -p output
import os
os.makedirs("output", exist_ok=True)

import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import numpy as np
# from pettingzoo.mpe import simple_adversary_v3

from ..common.base import PolicyNetwork
from ..team_adversary.pg import GDmax, LGDmax, NGDmax
from ..team_adversary.reinforce import GDmax as REINFORCE, NGDmax as NREINFORCE, QGDmax as QREINFORCE
from ..team_adversary.pg import MAPolicyNetwork, SoftmaxPolicy
from ..team_adversary.lambda_pg import NLGDmax, TwoHeadPolicy
from ..utils.env import MultiGridWrapper

# ray.init()

def test_reinforce():
    dim = 3
    team = SoftmaxPolicy(2, 4, [dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], 0.01, [(i,j) for i in range(4) for j in range(4)]) 
    adv = PolicyNetwork(15, 4)

    team.load_state_dict(torch.load("./output/8000-3x3-team-policy-reinforce.pt"))
    adv.load_state_dict(torch.load("./output/8000-3x3-adv-policy-reinforce.pt"))
    
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human", disable_env_checker=True))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
            adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
            team_action, _ = team.get_actions(team_obs)
            adv_action, _ = adv.get_action(adv_obs)
            adv_action = adv_action.item()
            team_translated = team.action_map[team_action]
            action = {}
            for i in range(len(team_translated)):
                action[i] = team_translated[i]
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

def test_n_reinforce():
    team = MAPolicyNetwork(15, 16, [(i,j) for i in range(4) for j in range(4)]) 
    adv = PolicyNetwork(15, 4)

    team.load_state_dict(torch.load(f"output/experiment-207/500-3x3-team-policy-n-reinforce.pt"))
    adv.load_state_dict(torch.load(f"output/experiment-207/500-3x3-adv-policy-n-reinforce.pt"))
    
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, allow_agent_overlap=True, render_mode="human", disable_env_checker=True))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
            adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
            team_action, _ = team.get_actions(team_obs)
            adv_action, _ = adv.get_action(adv_obs)
            adv_action = adv_action.item()
            team_translated = team.action_map[team_action]
            action = {}
            for i in range(len(team_translated)):
                action[i] = team_translated[i]
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

def q_reinforce_experiment():
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}")
    def save(iteration="end"):
        # plt.xlabel("Iterations")
        # plt.ylabel("Nash Gap")
        # plt.plot(range(0, len(gdm.nash_gap)), gdm.nash_gap)
        # plt.savefig(f"output/experiment-{experiment_num}/"+ str(iteration) + "-n-reinforce_experiment-nashgap.png")
        # plt.close()

        # plt.xlabel("Iterations")
        # plt.ylabel("Team Utility against ADV BR")
        # plt.plot(range(0, len(gdm.nash_gap)), gdm.team_utility)
        # plt.savefig(f"output/experiment-{experiment_num}/"+ str(iteration) + "-n-reinforce_experiment-team-rewards.png")
        # plt.close()

        team = gdm.team_policy
        torch.save(team.state_dict(), f"output/experiment-{experiment_num}/" + str(iteration) + "-3x3-team-policy-n-reinforce.pt")

        adv = gdm.qpolicy.table
        torch.save(adv, f"output/experiment-{experiment_num}/" + str(iteration) + "-3x3-adv-qpolicy-n-reinforce.pt")

    dim = 3
    # lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True)
    qtable = torch.zeros((dim, dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 4))
    gdm = QREINFORCE(qtable, 15,4, lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, disable_env_checker=True)), rollout_length=100, lr=0.1, gamma=1, batch_size=64, epochs=50, br_thresh=1e-8)

    PROFILING_MODE = True

    time_taken_sum = 0
    iterations = 1000
    for i in range(iterations):
        x = time()
        if i % 50 == 0 and not PROFILING_MODE:
            gdm.step_with_gap()
            print(f"Nash Gap: {gdm.nash_gap[-1]:.6f}")
        else:
            gdm.step() # 4

        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (iterations - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        if i % 500 == 0 and not PROFILING_MODE:
            # Save progress
            print("Saving progress...")
            save(i)
            
    save()

    team = gdm.team_policy
    adv = gdm.qpolicy
    
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human", disable_env_checker=True))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
            adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32)
            team_action, _ = team.get_actions(team_obs)
            adv_action, _ = adv.get_action(adv_obs)
            adv_action = adv_action.item()
            team_translated = team.action_map[team_action]
            action = {}
            for i in range(len(team_translated)):
                action[i] = team_translated[i]
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break



def n_reinforce_experiment():
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}")
    def save(iteration="end"):
        plt.xlabel("Iterations")
        plt.ylabel("Nash Gap")
        plt.plot(range(0, len(gdm.nash_gap)), gdm.nash_gap)
        plt.savefig(f"output/experiment-{experiment_num}/"+ str(iteration) + "-n-reinforce_experiment-nashgap.png")
        plt.close()

        plt.xlabel("Iterations")
        plt.ylabel("Team Utility against ADV BR")
        plt.plot(range(0, len(gdm.nash_gap)), gdm.team_utility)
        plt.savefig(f"output/experiment-{experiment_num}/"+ str(iteration) + "-n-reinforce_experiment-team-rewards.png")
        plt.close()

        team = gdm.team_policy
        torch.save(team.state_dict(), f"output/experiment-{experiment_num}/" + str(iteration) + "-3x3-team-policy-n-reinforce.pt")
        adv = gdm.adv_policy
        torch.save(adv.state_dict(), f"output/experiment-{experiment_num}/" + str(iteration) + "-3x3-adv-policy-n-reinforce.pt")
    
    dim = 3
    # lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True)
    gdm = NREINFORCE(15,4, lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, disable_env_checker=True)), rollout_length=50, lr=0.1, gamma=1, batch_size=64, epochs=50, br_thresh=1e-8)

    PROFILING_MODE = False

    time_taken_sum = 0
    iterations = 1000
    for i in range(iterations):
        x = time()
        if i % 50 == 0 and not PROFILING_MODE:
            gdm.step_with_gap()
            print(f"Nash Gap: {gdm.nash_gap[-1]:.6f}")
        else:
            gdm.step() # 4

        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (iterations - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        if i % 500 == 0 and not PROFILING_MODE:
            # Save progress
            print("Saving progress...")
            save(i)
            
    save()
    
    # team = MAPolicyNetwork(15, 16, [(i,j) for i in range(4) for j in range(4)]) 
    # team = SoftmaxPolicy(2, 4, [dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], 0.01, [(i,j) for i in range(4) for j in range(4)])
    # adv = PolicyNetwork(15, 4)

    # team.load_state_dict(torch.load(f"output/experiment-125/end-3x3-team-policy-n-reinforce.pt"))
    # adv.load_state_dict(torch.load(f"output/experiment-125/end-3x3-adv-policy-n-reinforce.pt"))

    team = gdm.team_policy
    adv = gdm.adv_policy
    
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human", disable_env_checker=True))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
            adv_action = adv_action.item()
            # print(torch.nn.Softmax()(adv.__call__(torch.tensor(obs[0]).float())))
            team_translated = team.action_map[team_action]
            action = {}
            for i in range(len(team_translated)):
                action[i] = team_translated[i]
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

def reinforce_experiment():
    def save(iteration="end"):
        plt.xlabel("Iterations")
        plt.ylabel("Nash Gap")
        plt.plot(gdm.nash_gap)
        plt.savefig("output/" + str(iteration) + "-reinforce_experiment_rewards.png")
        
        team = gdm.team_policy
        torch.save(team.state_dict(), "output/" + str(iteration) + "-3x3-team-policy-reinforce.pt")
        adv = gdm.adv_policy
        torch.save(adv.state_dict(), "output/" + str(iteration) + "-3x3-adv-policy-reinforce.pt")
    
    dim = 3
     # lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True)
    gdm = REINFORCE(15,4, lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3)), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], rollout_length=50, lr=0.001)
    time_taken_sum = 0
    time_taken_sum = 0
    iterations = 100
    for i in range(iterations):
        x = time()
        if i % 100 == 0:
            gdm.step_with_gap()
            print("Nash Gap:", gdm.nash_gap[-1])
        else:
            gdm.step() # 4

        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (iterations - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        if i % 500 == 0:
            # Save progress
            print("Saving progress...")
            save(i)
            
    save()
    
    # team = MAPolicyNetwork(15, 16, [(i,j) for i in range(4) for j in range(4)]) 
    # team = SoftmaxPolicy(2, 4, [dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], 0.01, [(i,j) for i in range(4) for j in range(4)])
    # adv = PolicyNetwork(15, 4)

    # team.load_state_dict(torch.load(f"{dim}x{dim}-team-policy-final.pt"))
    # adv.load_state_dict(torch.load(f"{dim}x{dim}-adv-policy-final.pt"))

    team = gdm.team_policy
    adv = gdm.adv_policy
    
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
            adv_action = adv_action.item()
            # print(torch.nn.Softmax()(adv.__call__(torch.tensor(obs[0]).float())))
            action = {i: team_action[i] for i in range(len(team_action))}
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

def ngdmax_experiment():
    dim = 3
    gdm = NGDmax(15,4, lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, size=5, max_episode_steps=12)), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], n_rollouts=50, lr=0.01)
    time_taken_sum = 0
    iterations = 100
    for i in range(iterations):
        x = time()
        gdm.step() # 4
        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (iterations - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        if i % 1000 == 0:
            team = gdm.team_policy
            torch.save(team.state_dict(), f"{dim}x{dim}-team-policy-step{i+1}.pt")
            adv = gdm.adv_policy
            torch.save(adv.state_dict(), f"{dim}x{dim}-adv-policy-step{i+1}.pt")

    team = gdm.team_policy
    torch.save(team.state_dict(), f"{dim}x{dim}-team-policy-final.pt")
    adv = gdm.adv_policy
    torch.save(adv.state_dict(), f"{dim}x{dim}-adv-policy-final.pt")

def gdmax_experiment():
    def save(iteration="end"):
        plt.xlabel("Iterations")
        plt.ylabel("Nash Gap")
        plt.plot(gdm.nash_gap)
        plt.savefig("output/" + str(iteration) + "-lgdmax_experiment_rewards.png")
        
        team = gdm.team_policy
        torch.save(team.state_dict(), "output/" + str(iteration) + "-3x3-team-policy-nlambda.pt")
        adv = gdm.adv_policy
        torch.save(adv.state_dict(), "output/" + str(iteration) + "-3x3-adv-policy-nlambda.pt")
    
    dim = 3
    gdm = GDmax(15,4, lambda: gym.make("TreasureHunt-3x3-Team", disable_env_checker=True), param_dims=[dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], n_rollouts=50, lr=0.01)
    time_taken_sum = 0
    time_taken_sum = 0
    iterations = 10000
    for i in range(iterations):
        x = time()
        gdm.step()
        if i % 20 == 0:
            gdm.step_with_gap()
            print("Nash Gap:", gdm.nash_gap[-1])
        else:
            gdm.step() # 4

        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (iterations - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        if i % 500 == 0:
            # Save progress
            print("Saving progress...")
            save(i)
            
    save()
    
    # team = MAPolicyNetwork(15, 16, [(i,j) for i in range(4) for j in range(4)]) 
    # team = SoftmaxPolicy(2, 4, [dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 16], 0.01, [(i,j) for i in range(4) for j in range(4)])
    # adv = PolicyNetwork(15, 4)

    # team.load_state_dict(torch.load(f"{dim}x{dim}-team-policy-final.pt"))
    # adv.load_state_dict(torch.load(f"{dim}x{dim}-adv-policy-final.pt"))

    team = gdm.team_policy
    adv = gdm.adv_policy
    
    env = MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, render_mode="human"))
    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            team_action, _ = team.get_actions(obs[0])
            adv_action, _ = adv.get_action(torch.tensor(obs[len(obs)-1]).float())
            # print(torch.nn.Softmax()(adv.__call__(torch.tensor(obs[0]).float())))
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
    #         # print(torch.nn.Softmax()(adv.__call__(torch.tensor(obs[0]).float())))
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
            # print(torch.nn.Softmax()(adv.__call__(torch.tensor(obs[0]).float())))
            action = {i: team_action[i] for i in range(len(team_action))}
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(0.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

def nlgdmax_grid_experiment():
    gdm = NLGDmax(15, 4, [(i,j) for i in range(4) for j in range(4)], env=lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, size=5, max_episode_steps=12)),lr=0.01)


    def save(iteration="end"):
        plt.xlabel("Iterations")
        plt.ylabel("Nash Gap")
        plt.plot(gdm.nash_gap)
        plt.savefig("output/" + str(iteration) + "-lgdmax_experiment_rewards.png")
        
        team = gdm.team_policy
        torch.save(team.state_dict(), "output/" + str(iteration) + "-3x3-team-policy-nlambda.pt")
        adv = gdm.adv_policy
        torch.save(adv.state_dict(), "output/" + str(iteration) + "-3x3-adv-policy-nlambda.pt")

    time_taken_sum = 0
    iterations = 10000
    nash_gap = []
    for i in range(iterations):
        x = time()
        if i % 20 == 0:
            gdm.step_with_gap()
            print("Nash Gap:", gdm.nash_gap[-1])
        else:
            gdm.step() 
        
        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (iterations - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        
        if i % 500 == 0:
            # Save progress
            print("Saving progress...")
            save(i)
            
    save()

    team = gdm.team_policy
    adv = gdm.adv_policy

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