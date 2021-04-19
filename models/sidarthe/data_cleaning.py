"""Create train/test cost comparison plot"""

from functools import reduce
from itertools import compress
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_GENERATIONS = 500
N_SEEDS = 20

# --------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cost_file", type=str, default="costs.dat")
    parser.add_argument("-results_file", type=str, default="results.csv")
    parser.add_argument("--min_cost_day", type=int, default=42)
    parser.add_argument("--max_cost_day", type=int, default=46)
    args = parser.parse_args()

    costs_df, costs_df_vec = get_dataframe(args.cost_file)
    print(f"Average over {len(costs_df_vec)} seeds")

    results_df, results_df_vec = get_dataframe(args.results_file)
    data = get_data(args)
    cost = compute_cost(args, data, results_df)

    costs_df["test_av_curve"] = cost

    plotting(costs_df, costs_df_vec)


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


def get_data(args) -> list:
    """Filter wanted days from original data"""

    # fmt: off
    Guariti = np.array([0, 0, 0, 1, 1, 1, 3, 45, 46, 50, 83, 149, 160, 276, 414, 523, 589, 622, 724, 1004, 1045, 1258, 1439, 1966, 2335, 2749, 2941, 4025, 4440, 5129, 6072, 7024, 7432, 8326, 9362, 10361, 10950, 12384, 13030, 14620, 15729, 16847, 18278, 19758, 20996, 21815])
    Isolamento_domiciliare = np.array([49, 91, 162, 221, 284, 412, 543, 798, 927, 1000, 1065, 1155, 1060, 1843, 2180, 2936, 2599, 3724, 5036, 6201, 7860, 9268, 10197, 11108, 12090, 14935, 19185, 22116, 23783, 26522, 28697, 30920, 33648, 36653, 39533, 42588, 43752, 45420, 48134, 50456, 52579, 55270, 58320])
    Ricoverati_sintomi = np.array([54, 99, 114, 128, 248, 345, 401, 639, 742, 1034, 1346, 1790, 2394, 2651, 3557, 4316, 5038, 5838, 6650, 7426, 8372, 9663, 11025, 12894, 14363, 15757, 16020, 17708, 19846, 20692, 21937, 23112, 24753, 26029, 26676, 27386, 27795, 28192, 28403, 28540, 28741, 29010, 28949])
    Terapia_intensiva = np.array([26, 23, 35, 36, 56, 64, 105, 140, 166, 229, 295, 351, 462, 567, 650, 733, 877, 1028, 1153, 1328, 1518, 1672, 1851, 2060, 2257, 2498, 2655, 2857, 3009, 3204, 3396, 3489, 3612, 3732, 3856, 3906, 3981, 4023, 4035, 4053, 4068, 3994, 3977])
    # fmt: on

    min_cost_day, max_cost_day = args.min_cost_day, args.max_cost_day
    start = (min_cost_day <= 4) * 0 + (min_cost_day > 4) * (min_cost_day - 3)

    data = [
        Guariti[min_cost_day:max_cost_day],
        Isolamento_domiciliare[start:max_cost_day],
        Ricoverati_sintomi[start:max_cost_day],
        Terapia_intensiva[start:max_cost_day],
    ]

    return data


# --------------------------------------------------------


def compute_cost(args, data, results_df) -> float:
    """Return array with cost for each generation"""

    cost = np.zeros(N_GENERATIONS)
    dt = args.max_cost_day - args.min_cost_day
    for comp in range(4):
        for day in range(dt):
            cost += (results_df.iloc[:, comp * dt + day] - data[comp][day]) ** 2

    return cost / 1e6


# --------------------------------------------------------


def title_labels(save=False):
    """Add title, labels and show figure
    Specify save name if wanted"""
    plt.title("SIDARTHE")
    plt.xlabel("generations")
    plt.ylabel("cost")
    plt.ylim([0, 2500])
    plt.legend()
    if save is not False:
        plt.savefig(save, dpi=300, transparent=True)
    plt.show()


# --------------------------------------------------------


def plotting(costs_df, costs_df_vec):
    """Create plots"""

    # main plot
    plt.plot(costs_df["train_cost"], label="train cost (42 days)", c="tab:blue")
    plt.plot(
        costs_df["test_av_cost"] / 2, label="test cost1 /2 (days 43-46)", c="tab:orange"
    )
    plt.plot(
        costs_df["test_av_curve"] / 2,
        label="test cost2 /2 (days 43-46)",
        c="tab:green",
    )
    title_labels("sidarthe_train_test.png")

    # training seeds
    for df_i in costs_df_vec:
        plt.plot(df_i["train_cost"], alpha=0.3, lw=1)
    plt.plot(costs_df["train_cost"], label="train cost (42 days)", c="tab:blue", lw=2)
    title_labels()

    # test seeds
    for df_i in costs_df_vec:
        plt.plot(df_i["test_av_cost"] / 2, alpha=0.3, lw=1)
    plt.plot(
        costs_df["test_av_cost"] / 2,
        label="test cost1 /2 (days 43-46)",
        c="tab:orange",
        lw=2,
    )
    plt.plot(
        costs_df["test_av_curve"] / 2,
        label="test cost2 /2 (days 43-46)",
        c="tab:green",
        lw=2,
    )
    title_labels()


# --------------------------------------------------------

if __name__ == "__main__":
    main()
