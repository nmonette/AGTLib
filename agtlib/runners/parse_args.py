import sys
import argparse

def csv(arg, typecast):
    return [typecast(i) for i in arg.split(',')]

def parse_args(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a", "--algorithm", help="Algorithm name", default="NREINFORCE", choices=["NREINFORCE", "QREINFORCE", "PREINFORCE", "TQREINFORCE"]
    )
    parser.add_argument(
        "-l", "--rollout-length", help="Number of rollout episodes", default=50, type=int, dest="rollout_length"
    )
    parser.add_argument(
        "-lr", help="Learning rate", default = 0.1, type=float
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
        "-de", "--disable-eval", help="Disable post-training evaluation", action="store_true", dest="disable_eval"
    )
    parser.add_argument(
        "-team", "--team-path", help="Path to team policy (eval or warmstart)", default=None, dest="team"
    )
    parser.add_argument(
        "-adv", "--adv-path", help="Path to adversarial policy (eval or warmstart)", default=None, dest="adv"
    )
    parser.add_argument(
        "-i", "--iters", help="Number of training iterations", default=1000, type=int
    )
    parser.add_argument(
       "-ng","--nash-gap", help="Measure Nash-Gap", action="store_true", dest="nash_gap"
    )
    parser.add_argument(
        "-ds", "--disable-save", help="Disable checkpoint and eval video saving", action="store_true", dest="disable_save"
    )
    parser.add_argument(
        "-br", "--br-length", help="Number of updates to find best respond", type=int, default=100, dest="br_length"
    )
    parser.add_argument(
        "-mi", "--metric-interval", help="Number of iterations between metric collection", type=int, default=50, dest="metric_interval"
    )
    parser.add_argument(
        "-si", "--save-interval", help="Number of iterations between model saves", type=int, default=500, dest="save_interval"
    )
    parser.add_argument(
        "-dim", "--grid-dimension", help="Grid dimension", type=int, default=3, dest="dim"
    )
    parser.add_argument(
        "-f", "--fix-grid", help="Fix grid configuration", action="store_const", const="MultiGrid-Empty-3x3-TeamCoop", default= "MultiGrid-Empty-3x3-Team", dest="env"
    ) # const = "MultiGrid-Empty-3x3-TeamWins"

    args, _ = parser.parse_known_args(cmd_args)

    return args
    



