from time import sleep, time
import os

from agtlib.utils.env import MultiGridWrapper, DecentralizedMGWrapper

import torch
import gymnasium as gym
import matplotlib.pyplot as plt

def eval(team, adv):
    env = DecentralizedMGWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, disable_env_checker=True, render_mode="human"))

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

def train(alg, args):
    experiment_num = len(list(os.walk('./output')))
    os.makedirs(f"output/experiment-{experiment_num}")
    def save(iteration="end"):
        if args.nash_gap:
            plt.xlabel("Iterations")
            plt.ylabel("Nash Gap")
            plt.plot(range(0, len(alg.nash_gap)), alg.nash_gap)
            plt.savefig(f"output/experiment-{experiment_num}/"+ str(iteration) + "-n-reinforce_experiment-nashgap.png")
            plt.close()

            # plt.xlabel("Iterations")
            # plt.ylabel("Team Utility against ADV BR")
            # plt.plot(range(0, len(alg.nash_gap)), alg.team_utility)
            # plt.savefig(f"output/experiment-{experiment_num}/"+ str(iteration) + "-n-reinforce_experiment-team-rewards.png")
            # plt.close()

        team = alg.team_policy
        torch.save(team.state_dict(), f"output/experiment-{experiment_num}/" + str(iteration) + "-3x3-team-policy-n-reinforce.pt")
        adv = alg.adv_policy if args.algorithm != "QREINFORCE" else alg
        torch.save(adv.state_dict() if args.algorithm != "QREINFORCE" else adv.qpolicy.table, f"output/experiment-{experiment_num}/" + str(iteration) + "-3x3-adv-policy-n-reinforce.pt")
    
    time_taken_sum = 0
    for i in range(args.iters):
        x = time()
        if args.nash_gap and i % 50 == 0: 
            alg.step_with_gap()
            print(f"Nash Gap: {alg.nash_gap[-1]:.6f}")
        else:
            alg.step() 

        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (args.iters - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        if i % 500 == 0 and not args.disable_save:
            # Save progress
            print("Saving progress...")
            save(i)
    
    if not args.disable_save:      
        save()