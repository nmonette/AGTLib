import sys
import argparse

def csv(arg, typecast):
    return [typecast(i) for i in arg.split(',')]

def parse_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a", "--algorithm", help="Algorithm name", default="NREINFORCE", choices=["NREINFORCE", "QREINFORCE"]
    )
    parser.add_argument(
        "-l", "--rollout-length", help="Number of rollout episodes", default=50, type=int, dest="rollout_length"
    )
    parser.add_argument(
        "--lr", help="Learning rate", default = 0.1, type=float
    )
    parser.add_argument(
        "-g", "--gamma", help="Discount Factor", default = 0.99, type=float
    )
    parser.add_argument(
        "-na", "--net-arch", help="Network architecture (comma separated)", default=[64,128], dest="net_arch", type=lambda v: csv(v, int)
    )
    parser.add_argument(
        "-e", "--eval", help="Display environment with given policies", action="store_true"
    )
    parser.add_argument(
        "-team", "--team-path", help="Path to team policy", default=None, dest="team"
    )
    parser.add_argument(
        "-adv", "--adv-path", help="Path to adversarial policy", default=None, dest="adv"
    )
    parser.add_argument(
        "-i", "--iters", help="Number of training iterations", default=1000, type=int
    )
    parser.add_argument(
       "-ng","--nash-gap", help="Measure Nash-Gap", action="store_true", dest="nash_gap"
    )
    parser.add_argument(
        "-ds", "--disable-save", help="Save model parameters", action="store_true", dest="disable_save"
    )

    args, _ = parser.parse_known_args(cmd_args)

    return args
    



