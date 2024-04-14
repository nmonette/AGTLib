from time import sleep, time
import os

from agtlib.utils.env import MultiGridWrapper, DecentralizedMGWrapper, IndepdendentTeamWrapper

import torch
import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo
import matplotlib.pyplot as plt

def eval(team, adv, args):
    
    if False and not args.disable_save:
        experiment_num = len(list(os.walk('./output')))
        os.makedirs(f"output/experiment-{experiment_num}")
        env = RecordVideo(DecentralizedMGWrapper(gym.make(args.env, agents=3, size = args.dim + 2, disable_env_checker=True, render_mode="rgb_array")), f"output/experiment-{experiment_num}", step_trigger= lambda i: True, name_prefix="demo", disable_logger=True)
        for episode in range(10):
            obs, _ = env.reset()
            env.render()
            while True:
                team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
                if args.algorithm == "PREINFORCE":
                    adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32).reshape(-1, len(obs[len(obs) - 1]))
                else:
                    adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32) 
                team_action = team.get_actions(team_obs)[0]
                adv_action = adv.get_action(adv_obs)[0]
                adv_action = adv_action.item()
                team_translated = team.action_map[team_action]
                action = {}
                for i in range(len(team_translated)):
                    action[i] = team_translated[i]
                action[len(action)] = adv_action
                sleep(0.5)
                obs, reward, trunc, done, _ = env.step(action)
                if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                    break
    if args.algorithm != "TQREINFORCE":
        env = DecentralizedMGWrapper(gym.make(args.env, agents=3, size = args.dim + 2, disable_env_checker=True, render_mode="human"))
    else:
        env = MultiGridWrapper(gym.make(args.env, agents=3, size = args.dim + 2, disable_env_checker=True, render_mode="human"))# IndepdendentTeamWrapper(gym.make(args.env, agents=3, size = args.dim + 2, disable_env_checker=True, render_mode="human"))

    for episode in range(100):
        obs, _ = env.reset()
        env.render()
        while True:
            
            if args.algorithm == "PREINFORCE":
                adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32).reshape(-1, len(obs[len(obs) - 1]))
            else:
                adv_obs = torch.tensor(obs[len(obs) - 1], device="cpu", dtype=torch.float32) 
            if args.algorithm == "TQREINFORCE":
                team_obs1 = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
                team_obs2 = torch.tensor(obs[1], device="cpu", dtype=torch.float32)
                team_translated = team.get_actions([team_obs1, team_obs2])[0]
            else:
                team_obs = torch.tensor(obs[0], device="cpu", dtype=torch.float32)
                team_action = team.get_actions(team_obs)[0]
                team_translated = team.action_map[team_action]
            adv_action = adv.get_action(adv_obs)[0]
            adv_action = adv_action.item()
            
            action = {}
            for i in range(len(team_translated)):
                action[i] = team_translated[i]
            action[len(action)] = adv_action
            obs, reward, trunc, done, _ = env.step(action)
            print(action, reward)
            sleep(1.5)
            if list(trunc.values()).count(True) >= 2 or list(done.values()).count(True) >= 2:
                break

def train(alg, args):
    if not args.disable_save:
        experiment_num = len(list(os.walk('./output')))
        os.makedirs(f"output/experiment-{experiment_num}")
    def save(iteration="end"):
        if args.nash_gap:
            plt.xlabel("Iterations")
            plt.ylabel("Nash Gap")
            plt.plot(range(0, len(alg.nash_gap)), alg.nash_gap)
            plt.savefig(f"output/experiment-{experiment_num}/"+ str(iteration) + "-nashgap.png")
            plt.close()

            # plt.xlabel("Iterations")
            # plt.ylabel("Team Utility against ADV BR")
            # plt.plot(range(0, len(alg.nash_gap)), alg.team_utility)
            # plt.savefig(f"output/experiment-{experiment_num}/"+ str(iteration) + "-team-rewards.png")
            # plt.close()

        team = alg.team_policy
        torch.save(team.state_dict(), f"output/experiment-{experiment_num}/" + str(iteration) + "-team-policy.pt")
        adv = alg.adv_policy if args.algorithm not in ["QREINFORCE", "TQREINFORCE"] else alg
        torch.save(adv.state_dict() if args.algorithm not in ["QREINFORCE", "TQREINFORCE"] else adv.qpolicy.table, f"output/experiment-{experiment_num}/" + str(iteration) + "-adv-policy.pt")
    
    time_taken_sum = 0
    for i in range(1, args.iters + 1):
        x = time()
        if args.nash_gap and (i % args.metric_interval == 0 or i == 1): 
            alg.step_with_gap()
            print(f"Nash Gap: {alg.nash_gap[-1]:.6f}")
        else:
            alg.step() 

        print(f"Iteration {i} done in {time() - x:.2f}s\t", end="")
        time_taken_sum += time() - x
        time_remaining = (args.iters - i) * (time_taken_sum / (i+1))
        print(f"Estimated time remaining: {time_remaining // 3600}h {time_remaining % 3600 // 60}m {time_remaining % 60:.2f}s")
        if i % args.save_interval == 0  and i > 0 and not args.disable_save:
            # Save progress
            print("Saving progress...")
            save(i)
    
    if not args.disable_save:    
        print("Saving progress...")  
        save()