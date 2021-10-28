import argparse
import os.path
import sys

import numpy
import pandas
from matplotlib import pyplot as plt

_base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.realpath(os.path.join(_base_dir, "..")))

from models.utils import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--evolution", required=True)
    parser.add_argument("--day-min", default=config.DAY_MIN, type=int)
    parser.add_argument("--day-max", default=config.DAY_MAX, type=int)

    return parser.parse_args()


def main():
    args = parse_args()

    simulation_data = pandas.read_csv(args.evolution, index_col=0, header=[0, 1])
    compartiments = simulation_data.columns.levels[0]

    mean_simulation = simulation_data.mean(axis=1, level=0)
    days = simulation_data.index

    plt.errorbar(
        days,
        mean_simulation["infected"],
        yerr=simulation_data["infected"].std(axis=1),
        marker="o",
        ls="",
        label="Mean infected",
    )
    plt.stackplot(
        days,
        *[mean_simulation[c] for c in compartiments],
        labels=compartiments,
        alpha=0.4
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
