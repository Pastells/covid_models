import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", type=int, default=1, help="seed for the automatic configuration"
)
parser.add_argument(
    "--timeout",
    type=int,
    default=1200,
    help="timeout for the automatic configuration",
)
parser.add_argument(
    "--data", type=str, default="../data/italy_i.csv", help="file with time series"
)
parser.add_argument(
    "--beta", type=float, default=0.5, help="parameter: ratio of infection [0.05,1]"
)

args = parser.parse_args()
cost = args.beta ** 2
sys.stdout.write(f"GGA SUCCESS {cost}\n")
