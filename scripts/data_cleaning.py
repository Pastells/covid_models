"""Create train/test cost comparison plot"""

import sys, os
from copy import deepcopy
from functools import reduce
from itertools import compress
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(sys.path[0], os.pardir, "models")))
from utils.utils import get_time_series

N_GENERATIONS = 500
N_SEEDS = 20


# --------------------------------------------------------


def main():
    costs_df, costs_df_vec = get_dataframe(args.cost_file)

    print(f"Average over {len(costs_df_vec)} seeds")

    results_df, results_df_vec = get_dataframe(args.results_file)
    results_df_vec = reduce(lambda x, y: pd.merge(x, y, how="outer"), results_df_vec)
    print(results_df)
    exit()

    data = get_data()

    if args.day_mid is False:
        # cost for average
        cost = compute_cost(data, results_df)
        costs_df["test_av_curve"] = cost

    else:
        costs_df2, costs_df2_vec = get_dataframe(args.cost_file2)
        print(f"Average 2 over {len(costs_df_vec)} seeds")
        results_df2, results_df2_vec = get_dataframe(args.results_file2)

        cost = compute_cost(data, results_df2)

        # del costs_df["test_av_cost"]
        costs_df["train_without_val"] = costs_df2["train_cost"]
        costs_df["test_of_best_val"] = val_test(results_df_vec)
        costs_df["test_av_curve"] = cost

    plotting(costs_df, costs_df_vec)


# --------------------------------------------------------


def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("day_min", type=int)
    parser.add_argument("day_max", type=int)
    parser.add_argument("data", type=str, help="file with time series")

    parser.add_argument("--folder2", type=str, default=False)
    parser.add_argument("--day_mid", type=int, default=False)

    parser.add_argument("--ylim", type=int, default=1000)
    args = parser.parse_args()

    args.cost_file = args.folder + "/costs.dat"
    args.results_file = args.folder + "/results.csv"

    # Check 2 optional parameters are given together (XOR)
    if bool(args.folder2) ^ bool(args.day_mid):
        parser.error("--folder2 and --day_mid must be given together")

    if args.day_mid is not False:
        args.cost_file2 = args.folder2 + "/costs.dat"
        args.results_file2 = args.folder2 + "/results.csv"
        print(f"Using {args.folder}, {args.folder2}")
    else:
        print(f"Using {args.folder}")
    return args


# --------------------------------------------------------


def val_test_split(df):
    validation_list = []
    test_list = []
    for i in range(len(df.columns) - 2):
        if (i % (args.day_max - args.day_min)) < (args.day_mid - args.day_min):
            validation_list.append(i)
        else:
            test_list.append(i)

    validation_list = np.array(validation_list) + 2
    test_list = np.array(test_list) + 2

    df_val = df.copy()
    df_val.drop(df.columns[test_list], axis=1, inplace=True)
    df_test = df.copy()
    df_test.drop(df.columns[validation_list], axis=1, inplace=True)

    return df_val, df_test


# --------------------------------------------------------


def val_test(results_df_vec):
    # Split data and results
    data_val = get_data("validation")
    data_test = get_data("test")
    df_val, df_test = val_test_split(results_df_vec)

    cost_val_vect = compute_cost(data_val, df_val, which="validation", columns=2)
    df_val["cost_val"] = cost_val_vect

    cost_test_vect = compute_cost(data_test, df_test, which="test", columns=2)
    df_test["cost_test"] = cost_test_vect

    df_val.set_index("seed", inplace=True)

    # best seed for each generation
    best_gen = [
        int(df_val[df_val["gen"] == gen + 1].idxmin()["cost_val"])
        for gen in range(N_GENERATIONS)
    ]

    # create DataFrame array with each generation
    df = [df_test[df_test["gen"] == gen + 1] for gen in range(N_GENERATIONS)]
    # filter best seed
    for gen in range(N_GENERATIONS):
        df[gen] = df[gen][df[gen]["seed"] == best_gen[gen]]

    # merge
    df = reduce(lambda x, y: pd.merge(x, y, how="outer"), df)
    df.set_index("gen", inplace=True)

    return df["cost_test"]


# --------------------------------------------------------


def get_dataframe(file) -> pd.DataFrame:
    """Return DataFrame from input data with gaps filled"""

    df = pd.read_csv(file, delim_whitespace=True)

    # Pad non-present generations with the last changing value
    new_index = pd.Index(np.arange(1, N_GENERATIONS + 1), name="gen")

    # Create a vector of dataframes for each seed
    df_vec = [
        df[df.seed == seed].set_index("gen").reindex(new_index).pad().reset_index()
        for seed in range(42, 42 + N_SEEDS)
    ]

    # Filter seeds where only 5 generations where executed (configurator bug)
    df_filter = [
        df[df.seed == seed]["gen"].max() > 5 for seed in range(42, 42 + N_SEEDS)
    ]
    df_vec = list(compress(df_vec, df_filter))

    # Average over seeds
    df = reduce(lambda x, y: pd.merge(x, y, how="outer"), df_vec)
    df = df.groupby("gen").mean()
    del df["seed"]
    return df, df_vec


# --------------------------------------------------------


def get_data(which="test") -> list:
    """Filter wanted days from original data"""

    args2 = deepcopy(args)
    if which == "validation":
        args2.day_min, args2.day_max = args.day_min, args.day_mid
    elif which == "test":
        args2.day_min = (args.day_mid is not False) * args.day_mid + (
            args.day_mid is False
        ) * args.day_min
        args2.day_max = args.day_max
    elif which == "all":
        args2.day_min, args2.day_max = 0, args.day_max
    else:
        raise ValueError("Choose 'validation' or 'test' in `compute_cost`")

    data = get_time_series(args2)

    return data


# --------------------------------------------------------


def compute_cost(data, df, which="test", columns=0) -> np.array:
    """Return array with cost for each generation"""

    if which == "validation":
        dt = args.day_mid - args.day_min
    elif which == "test":
        dt = (
            args.day_max
            - (args.day_mid is not False) * args.day_mid
            - (args.day_mid is False) * args.day_min
        )
    else:
        raise ValueError("Choose 'validation' or 'test' in `compute_cost`")

    cost = np.zeros(len(df))
    for comp in range(3):
        for day in range(dt):
            cost += (df.iloc[:, columns + comp * dt + day] - data[day][comp]) ** 2

    return cost / 1e6


# --------------------------------------------------------


def title_labels(save=False):
    """Add title, labels and show figure
    Specify save name if wanted"""
    plt.title("SIDARTHE")
    plt.xlabel("generations")
    plt.ylabel("cost")
    plt.ylim([0, args.ylim])
    plt.legend()
    if save is not False:
        save = args.folder + "/" + save
        if args.day_mid is False:
            save += f"_{args.day_min}_{args.day_max}.png"
        else:
            save += f"_{args.day_min}_{args.day_mid}_{args.day_max}.png"

        plt.savefig(save, dpi=300)
    plt.show()


# --------------------------------------------------------


def plotting(costs_df, costs_df_vec):
    """Create plots"""

    dt_str = f"(days {args.day_min+1}-{args.day_max})"
    sf = 2  # scale factor
    costs_df["test_av_cost"] /= sf
    costs_df["test_av_curve"] /= sf

    # main plot
    plt.plot(
        costs_df["train_cost"],
        label=f"train (cost average) ({args.day_min} days)",
        c="tab:blue",
    )
    plt.plot(
        costs_df["test_av_cost"],
        label=f"test (cost average) /{sf} {dt_str}",
        c="tab:orange",
    )
    plt.plot(
        costs_df["test_av_curve"],
        label=f"test (realization average) /{sf} {dt_str}",
        c="tab:green",
    )

    if args.day_mid is not False:
        dt_str2 = f"(days {args.day_mid+1}-{args.day_max})"
        costs_df["test_of_best_val"] /= sf
        plt.plot(
            costs_df["test_of_best_val"],
            label=f"test (best validation seed) /{sf} {dt_str2}",
            c="tab:purple",
        )

    title_labels("sidarthe_train_test")

    # training seeds
    for df_i in costs_df_vec:
        plt.plot(df_i["train_cost"], alpha=0.3, lw=1)
    plt.plot(
        costs_df["train_cost"],
        label=f"train (cost average) ({args.day_min} days)",
        c="tab:blue",
        lw=2,
    )
    title_labels("sidarthe_train")

    # test seeds
    for df_i in costs_df_vec:
        plt.plot(df_i["test_av_cost"] / sf, alpha=0.3, lw=1)
    plt.plot(
        costs_df["test_av_cost"],
        label=f"test (cost average) /{sf} {dt_str}",
        c="tab:orange",
        lw=2,
    )
    plt.plot(
        costs_df["test_av_curve"],
        label=f"test (realization average) /{sf} {dt_str}",
        c="tab:green",
        lw=2,
    )
    if args.day_mid is not False:
        plt.plot(
            costs_df["test_of_best_val"],
            label=f"test (best validation seed) /{sf} {dt_str2}",
            c="tab:purple",
            lw=2,
        )
    title_labels("sidarthe_test")


# --------------------------------------------------------

if __name__ == "__main__":
    global args
    args = parsing()
    main()
