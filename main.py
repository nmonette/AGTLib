import sys

from agtlib.runners.parse_args import parse_args
from agtlib.runners.gdmax_experiments import eval, train
from agtlib.team_adversary.reinforce import GDmax, NGDmax as NREINFORCE, QGDmax as QREINFORCE, PGDmax as PREINFORCE
from agtlib.common.base import SELUPolicy, SELUMAPolicy
from agtlib.team_adversary.q import TabularQ
from agtlib.utils.env import MultiGridWrapper, DecentralizedMGWrapper
from agtlib.cooperative.ppo import train_ppo

from stable_baselines3 import PPO
import gymnasium as gym
import torch
import multigrid.multigrid.envs

def main(cmd_args=sys.argv[1:]):
    args = parse_args(cmd_args)
    
    if args.algorithm == "NREINFORCE":
        if args.eval:
            team = SELUMAPolicy(12, 16, [(i,j) for i in range(4) for j in range(4)], args.net_arch) 
            adv = SELUPolicy(8, 4)

            team.load_state_dict(torch.load(args.team))
            adv.load_state_dict(torch.load(args.adv))

            eval(team, adv, args)
   
        else:
            alg = NREINFORCE(12,4, lambda: DecentralizedMGWrapper(gym.make(args.env, agents=3, disable_env_checker=True, size=args.dim + 2)), rollout_length=args.rollout_length, lr=args.lr, gamma=args.gamma, br_length=args.br_length)
            if args.team is not None:
                alg.team_policy.load_state_dict(torch.load(args.team))
            if args.adv is not None:
                alg.adv_policy.load_state_dict(torch.load(args.adv))

            train(alg, args)

            if not args.disable_eval:
                eval(alg.team_policy, alg.adv_policy, args)

    elif args.algorithm == "QREINFORCE":
        dim = args.dim
        if args.eval:
            team = SELUMAPolicy(12, 16, [(i,j) for i in range(4) for j in range(4)], hl_dims=args.net_arch) 
            team.load_state_dict(torch.load(args.team))

            qtable = torch.load(args.adv)
            adv = TabularQ(qtable, 0.005, 0.05, 1, args.lr, args.gamma, args.rollout_length, lambda: None, 0)

            eval(team, adv, args)

        else:
            if args.adv is not None:
                qtable = torch.load(args.adv)
            else:
                qtable = torch.zeros((dim, dim, dim, dim, 2, dim ,dim, 2, 4))

            
            alg = QREINFORCE(qtable, 12, 4, lambda: DecentralizedMGWrapper(gym.make(args.env,  agents=3, size = dim + 2, disable_env_checker=True)), rollout_length=args.rollout_length, lr=args.lr, gamma=args.gamma, hl_dims=args.net_arch, br_length=args.br_length)
            if args.team is not None:
                alg.team_policy.load_state_dict(torch.load(args.team))
            
            train(alg, args)

            if not args.disable_eval:
                eval_adv = TabularQ(alg.qpolicy.table, 0.005, 0.05, 1, args.lr, args.gamma, args.rollout_length, lambda: None, 0)
                eval(alg.team_policy, eval_adv, args)

    elif args.algorithm == "PREINFORCE":
        if args.eval:
            team = SELUMAPolicy(12, 16, [(i,j) for i in range(4) for j in range(4)], hl_dims=args.net_arch) 
            team.load_state_dict(torch.load(args.team))

            adv = PPO(policy="MlpPolicy", env=DecentralizedMGWrapper(gym.make(args.env,  agents=3, size = args.dim + 2, disable_env_checker=True)), gdmax=True, monitor_wrapper=False).policy
            adv.load_state_dict(torch.load(args.adv))

            eval(team, adv, args)

        else:
            alg = PREINFORCE(12,4, lambda: DecentralizedMGWrapper(gym.make(args.env, agents=3, disable_env_checker=True, size=args.dim + 2)), rollout_length=args.rollout_length, lr=args.lr, gamma=args.gamma, br_length=args.br_length, hl_dims=args.net_arch)
            if args.team is not None:
                alg.team_policy.load_state_dict(torch.load(args.team))
            if args.adv is not None:
                alg.adv_policy.load_state_dict(torch.load(args.adv))

            train(alg, args)

            if not args.disable_eval:
                eval(alg.team_policy, alg.adv_policy, args)
    else:
        raise NotImplemented
if __name__ == "__main__":
    main()