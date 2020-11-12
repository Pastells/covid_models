"""
Stochastic mean-fiel SEIR model using the Gillespie algorithm
Pol Pastells, october 2020

equations of the deterministic system
s[t] = s[t-1] - beta1*e[t-1]*s[t-1] - beta2*i[t-1]*s[t-1]
e[t] = e[t-1] + beta1*e[t-1]*s[t-1] + beta2*i[t-1]*s[t-1] - (epsilon+delta1)*e[t-1]
i[t] = i[t-1] + epsilon*e[t-1] - delta2 * i[t-1]
r[t] = r[t-1] + delta1 *e[t-1] + delta2 * i[t-1]
"""

import numpy as np
import utils


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    args = parsing()
    (
        e_0,
        i_0,
        r_0,
        t_steps,
        t_total,
        nseed,
        seed0,
        plot,
        save,
        infected_time_series,
        n,
        beta1,
        beta2,
        delta1,
        epsilon,
        delta2,
    ) = parameters_init(args)

    # results per day and seed
    days_gap = 5
    i_day, i_m = (
        np.zeros([nseed, t_total + days_gap]),
        np.zeros(t_total + days_gap),
    )

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for seed in range(seed0, seed0 + nseed):
        np.random.seed(seed)

        # -------------------------
        # initialization
        s, e, i, r = (
            np.zeros(t_steps),
            np.zeros(t_steps),
            np.zeros(t_steps),
            np.zeros(t_steps),
        )
        # T = np.zeros(t_steps)
        # T[0]=0
        e[0] = e_0
        i[0] = i_0
        r[0] = r_0
        s[0] = n - i_0 - r_0 - e_0
        i_day[mc_step, 0] = i_0
        t, time, day = 0, 0, 1

        # -------------------------
        # Time loop
        # -------------------------
        while i[t] > 0.1 and day < t_total - 1:
            day, day_max = utils.day_data(mc_step, t, time, day, day_max, i, i_day)
            t, time = gillespie(
                t_total, t, time, s, e, i, r, beta1, beta2, delta1, delta2, epsilon
            )

        # -------------------------
        day, day_max = utils.day_data(mc_step, t, time, day, day_max, i, i_day, True)

        # final value for the rest of time, otherwise it contributes with a zero when averaged
        i_day[mc_step, day:] = i_day[mc_step, day - 1]

        mc_step += 1
    # =========================
    i_m, i_std = utils.mean_alive(i_day, t_total, day_max, nseed, days_gap)

    utils.cost_func(infected_time_series, i_m, i_std)

    if save:
        utils.saving(args, i_m, i_std, day_max)

    if plot:
        utils.plotting(infected_time_series, i_day, day_max, i_m, i_std)


# -------------------------


def parsing():
    """
    input parameters
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic mean-fiel SEIR model using the Gillespie algorithm"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="Fixed number of (effecitve) people [1000,1000000]",
    )
    parser.add_argument(
        "--e_0", type=int, default=0, help="initial number of latent individuals [1,n]"
    )
    parser.add_argument(
        "--i_0",
        type=int,
        default=20,
        help="initial number of infected individuals [1,n]",
    )
    parser.add_argument(
        "--r_0", type=int, default=0, help="initial number of inmune individuals [1,n]"
    )
    parser.add_argument(
        "--delta1",
        type=float,
        default=0,
        help="ratio of recovery from latent fase (e->r) [1e-2,1]",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.2,
        help="ratio of recovery from infected fase (i->r) [1e-2,1]",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0,
        help="ratio of infection due to latent [1e-2,1]",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.5,
        help="ratio of infection due to infected [1e-2,1]",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1, help="ratio of latency (e->i) [1e-2,1]"
    )

    parser.add_argument(
        "--llavor", type=int, default=1, help="Llavor from the automatic configuration"
    )
    parser.add_argument(
        "--data", type=str, default="../data/italy_i.csv", help="File with time series"
    )
    parser.add_argument(
        "--day_min", type=int, default=33, help="First day to consider on data series"
    )
    parser.add_argument(
        "--day_max", type=int, default=58, help="Last day to consider on data series"
    )

    parser.add_argument(
        "--nseed",
        type=int,
        default=int(5),
        help="number of realizations, not really a parameter",
    )
    parser.add_argument(
        "--seed0", type=int, default=1, help="initial seed, not really a parameter"
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")
    parser.add_argument("--save", action="store_true", help="specify for outputfile")
    args = parser.parse_args()
    print(args)
    return args


# -------------------------
# Parameters


def parameters_init(args):
    """
    initial parameters from argparse
    """
    from numpy import genfromtxt

    e_0 = args.e_0
    i_0 = args.i_0
    r_0 = args.r_0
    t_steps = int(1e6)  # max simulation steps
    t_total = (args.day_max - args.day_min) * 2  # max simulated days
    nseed = args.nseed  # MC realizations
    seed0 = args.seed0
    plot = args.plot
    save = args.save
    infected_time_series = genfromtxt(args.data, delimiter=",")[
        args.day_min : args.day_max
    ]
    # print(infected_time_series)
    n = args.n
    beta1 = args.beta1 / n
    beta2 = args.beta2 / n
    delta1 = args.delta1
    delta2 = args.delta2
    epsilon = args.epsilon

    return (
        e_0,
        i_0,
        r_0,
        t_steps,
        t_total,
        nseed,
        seed0,
        plot,
        save,
        infected_time_series,
        n,
        beta1,
        beta2,
        delta1,
        epsilon,
        delta2,
    )


# -------------------------


def gillespie(t_total, t, time, s, e, i, r, beta1, beta2, delta1, delta2, epsilon):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (epsilon + delta1) * e[t] + delta2 * i[t] + (beta1 * e[t] + beta2 * i[t]) * s[t]
    )

    prob_heal1 = delta1 * e[t] / lambda_sum
    prob_heal2 = delta2 * i[t] / lambda_sum
    prob_latent = epsilon * e[t] / lambda_sum

    t += 1
    time += utils.time_dist(lambda_sum)
    # T[t] = time

    gillespie_step(t, s, e, i, r, prob_heal1, prob_heal2, prob_latent)
    return t, time


# -------------------------


def gillespie_step(t, s, e, i, r, prob_heal1, prob_heal2, prob_latent):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < prob_heal1:
        # e->r
        s[t] = s[t - 1]
        e[t] = e[t - 1] - 1
        i[t] = i[t - 1]
        r[t] = r[t - 1] + 1
    elif random < (prob_heal1 + prob_heal2):
        # i->r
        s[t] = s[t - 1]
        e[t] = e[t - 1]
        i[t] = i[t - 1] - 1
        r[t] = r[t - 1] + 1
    elif random < (prob_heal1 + prob_heal2 + prob_latent):
        # e->i
        s[t] = s[t - 1]
        e[t] = e[t - 1] - 1
        i[t] = i[t - 1] + 1
        r[t] = r[t - 1]
    else:
        # s->e
        s[t] = s[t - 1] - 1
        e[t] = e[t - 1] + 1
        i[t] = i[t - 1]
        r[t] = r[t - 1]


# -------------------------


if __name__ == "__main__":
    import traceback

    try:
        main()
    # handle error when running with --help
    except SystemExit as error:
        print(f"GGA CRASHED {1e20}")
        print(repr(error))
    except:
        print(f"GGA CRASHED {1e20}")
        traceback.print.exc()
