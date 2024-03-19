import sys

from agtlib.runners.parse_args import parse_args
from agtlib.runners.gdmax_experiments import eval, train
from agtlib.cooperative.reinforce import GDmax, NGDmax as NREINFORCE, QGDmax as QREINFORCE
from agtlib.cooperative.pg import NGDmax, SELUMAPolicy
from agtlib.cooperative.base import SELUPolicy
from agtlib.cooperative.q import TabularQ

from agtlib.utils.env import MultiGridWrapper

import gymnasium as gym
import torch

def main(cmd_args=sys.argv[1:]):
    args = parse_args(cmd_args)

    if args.algorithm == "NREINFORCE":
        if args.eval:
            team = SELUMAPolicy(15, 16, [(i,j) for i in range(4) for j in range(4)], args.net_arch) 
            adv = SELUPolicy(15, 4)
            team.load_state_dict(torch.load(args.team))
            adv.load_state_dict(torch.load(args.adv))

            eval(team, adv)
   
        else:
            alg = NREINFORCE(15,4, lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, disable_env_checker=True)), rollout_length=args.rollout_length, lr=args.lr, gamma=args.gamma)
            train(alg, args)

    elif args.algorithm == "QREINFORCE":
        dim = 3
        if args.eval:
            team = SELUMAPolicy(15, 16, [(i,j) for i in range(4) for j in range(4)], args.net_arch) 
            qtable = torch.load(args.adv)
            adv = TabularQ(qtable, 0.005, 0.05, 1, args.lr, args.gamma, args.rollout_length, 12, lambda: None)
            team.load_state_dict(torch.load(args.team))

            eval(team, adv)

        else:
            qtable = torch.zeros((dim, dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2, 4))
            alg = QREINFORCE(qtable, 15,4, lambda: MultiGridWrapper(gym.make("MultiGrid-Empty-3x3-Team", agents=3, disable_env_checker=True)), rollout_length=100, lr=0.1, gamma=1)
            train(alg, args)

    else:
        raise NotImplemented
if __name__ == "__main__":
    main()
