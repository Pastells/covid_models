import argparse
import os.path
import sys

import pandas
from matplotlib import pyplot as plt

_base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(os.path.join(_base_dir, "..")))

from models.utils import utils, config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--evolution", required=True)
    parser.add_argument("--day-min", default=config.DAY_MIN, type=int)
    parser.add_argument("--day-max", default=config.DAY_MAX, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    reported_data = utils.get_time_series(args)

    simulation_data = pandas.read_csv(args.evolution, index_col=0, header=[0, 1])

    predicted_infected_mean = simulation_data["infected"].mean(axis=1)
    predicted_infected_std = simulation_data["infected"].std(axis=1)
    days = simulation_data.index

    plt.plot(days, reported_data[:, 0])
    plt.errorbar(
        days, predicted_infected_mean, yerr=predicted_infected_std, marker="o", ls=""
    )
    plt.show()


if __name__ == "__main__":
    main()
