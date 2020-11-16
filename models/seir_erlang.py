"""
Stochastic mean-field SEIR model using the Gillespie algorithm
Pol Pastells,  october 2020

equations of the deterministic system
s[t] = s[t-1] - beta1*e[t-1]*s[t-1] - beta2*i[t-1]*s[t-1]
e[t] = e[t-1] + beta1*e[t-1]*s[t-1] + beta2*i[t-1]*s[t-1] - (epsilon+delta1)*e[t-1]
i[t] = i[t-1] + epsilon*e[t-1] - delta2 * i[t-1]
r[t] = r[t-1] + delta1 *e[t-1] + delta2 * i[t-1]
"""

import numpy as np
import utils


def main():
    args = parsing()
    (
        e_0,
        i_0,
        r_0,
        t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        k_inf,
        k_rec,
        k_lat,
        beta1,
        beta2,
        delta1,
        epsilon,
        delta2,
    ) = parameters_init(args)

    # results per day and seed
    # s_day, s_m, s_95 = np.zeros([mc_nseed, t_total]), np.zeros(t_total), np.zeros([t_total, 2])
    # e_day, e_m, e_95 = np.zeros([mc_nseed, t_total]), np.zeros(t_total), np.zeros([t_total, 2])
    i_day, i_m = (
        np.zeros([mc_nseed, t_total]),
        np.zeros(t_total),
    )
    # r_day, r_m, r_95 = np.zeros([mc_nseed, t_total]), np.zeros(t_total), np.zeros([t_total, 2])

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for seed in range(mc_seed0, mc_seed0 + mc_nseed):
        np.random.seed(seed)

        # -------------------------
        # initialization
        s, e, i, r = (
            np.zeros([t_steps, k_inf + 1]),
            np.zeros([t_steps, k_lat + 1, 2]),
            np.zeros([t_steps, k_rec + 1]),
            np.zeros(t_steps),
        )
        s[0, :-1] = (n - i_0 - r_0) / k_inf
        s[0, -1], e[0, :-1] = e_0 / k_lat, e_0 / k_lat
        e[0, -1], i[0, :-1] = i_0 / k_rec, i_0 / k_rec
        i[0, -1], r[0] = r_0, r_0

        # s_day[mc_step, 0]=s[0]
        # e_day[mc_step, 0]=e_0
        i_day[mc_step, 0] = i_0
        # r_day[mc_step, 0]=r_0
        # T = np.zeros(t_steps)
        # T[0]=0
        t, time, day = 0, 0, 1

        # Time loop
        while i[t, :-1].sum() > 0 and day < t_total - 1:
            day, day_max = utils.day_data_k(
                mc_step, t, time, t_total, day, day_max, i, i_day
            )
            t, time = gillespie(
                t_total,
                t,
                time,
                s,
                e,
                i,
                r,
                beta1,
                beta2,
                delta1,
                delta2,
                epsilon,
                k_rec,
                k_inf,
                k_lat,
            )
            if time is True:
                break

        # -------------------------
        day, day_max = utils.day_data_k(
            mc_step, t, time, day, t_total, day_max, i, i_day, True
        )

        """
        if plot:
            plt.plot(T[:t], s[:t, :-1].sum(1), c='r')
            plt.plot(T[:t], e[:t, :-1, 0].sum(1), c='g')
            plt.plot(T[:t], e[:t, :-1, 1].sum(1), c='b')
            plt.plot(T[:t], i[:t, :-1].sum(1), c='c')
            plt.plot(T[:t], r[:t], c='m')
        """

        mc_step += 1
    # =========================

    i_m, i_std = utils.mean_alive(i_day, t_total, day_max, mc_nseed)

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
        description="Stochastic mean-field SEIR model using the Gillespie algorithm and Erlang \
            distribution transition times.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n", type=int, default=int(1e4), help="Fixed number of people"
    )
    parser.add_argument(
        "--e_0", type=int, default=0, help="initial number of latent individuals"
    )
    parser.add_argument(
        "--i_0", type=int, default=20, help="initial number of infected individuals"
    )
    parser.add_argument(
        "--r_0", type=int, default=0, help="initial number of inmune individuals"
    )
    parser.add_argument(
        "--delta1",
        type=float,
        default=0,
        help="ratio of recovery from latent fase (e->r)",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.2,
        help="ratio of recovery from infected fase (i->r)",
    )
    parser.add_argument(
        "--k_rec",
        type=int,
        default=1,
        help="k parameter for the recovery time erlang distribution",
    )
    parser.add_argument(
        "--beta1", type=float, default=0, help="ratio of infection due to latent"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.5, help="ratio of infection due to infected"
    )
    parser.add_argument(
        "--k_inf",
        type=int,
        default=1,
        help="k parameter for both infection times erlang distribution \
                             (1/beta1,  1/beta2)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=1, help="ratio of latency (e->i)"
    )
    parser.add_argument(
        "--k_lat",
        type=int,
        default=1,
        help="k parameter for both latent times erlang distribution \
                             (1/delta1,  1/delta2)",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="Seed for the automatic configuration"
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
        "--mc_nseed",
        type=int,
        default=int(5),
        help="Number of MC realizations, not really a parameter",
    )
    parser.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="Initial MC seed, not really a parameter",
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")
    parser.add_argument("--save", action="store_true", help="specify for outputfile")

    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters
def parameters_init(args):
    """
    Initial parameters from argparse
    """
    from numpy import genfromtxt

    e_0 = args.e_0
    i_0 = args.i_0
    r_0 = args.r_0
    t_steps = int(1e7)  # max simulation steps
    t_total = args.day_max - args.day_min  # max simulated days
    mc_nseed = args.mc_nseed  # MC realizations
    mc_seed0 = args.mc_seed0
    plot = args.plot
    save = args.save
    infected_time_series = genfromtxt(args.data, delimiter=",")[
        args.day_min : args.day_max
    ]
    # print(infected_time_series)
    n = args.n
    k_inf = args.k_inf
    k_rec = args.k_rec
    k_lat = args.k_lat
    beta1 = args.beta1 / n * k_inf
    beta2 = args.beta2 / n * k_inf
    delta1 = args.delta1 * k_lat
    epsilon = args.epsilon * k_lat
    delta2 = args.delta2 * k_rec
    return (
        e_0,
        i_0,
        r_0,
        t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        k_inf,
        k_rec,
        k_lat,
        beta1,
        beta2,
        delta1,
        epsilon,
        delta2,
    )


def gillespie(
    t_total,
    t,
    time,
    s,
    e,
    i,
    r,
    beta1,
    beta2,
    delta1,
    delta2,
    epsilon,
    k_rec,
    k_inf,
    k_lat,
):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    stot = s[t, :-1].sum()
    itot = i[t, :-1].sum()
    etot_rec = e[t, :-1, 0].sum()
    etot_inf = e[t, :-1, 1].sum()
    etot = etot_inf + etot_rec - e[t, 0, 0]

    lambda_sum = (
        epsilon * etot_inf
        + delta1 * etot_rec
        + delta2 * itot
        + (beta1 * etot + beta2 * itot) * stot
    )

    prob_heal1 = delta1 * e[t, :-1, 0] / lambda_sum
    prob_heal2 = delta2 * i[t, :-1] / lambda_sum
    prob_latent = epsilon * e[t, :-1, 1] / lambda_sum
    prob_infect = (beta1 * etot + beta2 * itot) * s[t, :-1] / lambda_sum

    t += 1
    time += utils.time_dist(lambda_sum)
    if time > t_total:
        return t, True  # rare,  but sometimes long times may appear
    # T[t] = time

    gillespie_step(
        t,
        s,
        e,
        i,
        r,
        prob_heal1,
        prob_heal2,
        prob_latent,
        prob_infect,
        k_rec,
        k_lat,
        k_inf,
    )
    return t, time


# -------------------------


def gillespie_step(
    t, s, e, i, r, prob_heal1, prob_heal2, prob_latent, prob_infect, k_rec, k_lat, k_inf
):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    s and i have one extra dimension to temporally store the infected and recovered
    after k stages, due to the Erlang distribution
    """
    random = np.random.random()
    prob_heal1_tot = prob_heal1.sum()
    prob_heal2_tot = prob_heal2.sum()
    prob_latent_tot = prob_latent.sum()

    # e(k)-> e(k+1)/r
    if random < prob_heal1_tot:
        for k in range(k_lat):
            if random < prob_heal1[: k + 1].sum():
                s[t, :-1] = s[t - 1, :-1]
                i[t, :-1] = i[t - 1, :-1]
                e[t, k, 0] = -1
                e[t, k + 1, 0] = 1
                r[t] = r[t - 1] + e[t, k_lat, 0]
                e[t, 0, 1] = e[t, 0, 0]
                e[t] += e[t - 1]
                break

    # i(k)-> i(k+1)/r
    elif random < (prob_heal1_tot + prob_heal2_tot):
        random -= prob_heal1_tot
        for k in range(k_rec):
            if random < prob_heal2[: k + 1].sum():
                s[t, :-1] = s[t - 1, :-1]
                e[t, :-1] = e[t - 1, :-1]
                i[t, k] = -1
                i[t, k + 1] = 1
                r[t] = r[t - 1] + i[t, k_rec]
                i[t] += i[t - 1]
                break

    # e(k)-> e(k+1)/i(0)
    elif random < (prob_heal1_tot + prob_heal2_tot + prob_latent_tot):
        random -= prob_heal1_tot + prob_heal2_tot
        for k in range(k_lat):
            if random < prob_latent[: k + 1].sum():
                s[t, :-1] = s[t - 1, :-1]
                i[t, :-1] = i[t - 1, :-1]
                r[t] = r[t - 1]
                e[t, k, 1] = -1
                e[t, k + 1, 1] = 1
                i[t, 0] += e[t, k_lat, 1]
                e[t, 0, 0] = e[t, 0, 1]
                e[t] += e[t - 1]
                break

    # s(k)-> s(k+1)/e(0)
    else:
        random -= prob_heal1_tot + prob_heal2_tot + prob_latent_tot
        for k in range(k_inf):
            if random < prob_infect[: k + 1].sum():
                e[t, :-1] = e[t - 1, :-1]
                i[t, :-1] = i[t - 1, :-1]
                r[t] = r[t - 1]
                s[t, k] = -1
                s[t, k + 1] = 1
                e[t, 0] += s[t, k_inf]
                s[t] += s[t - 1]
                break


# -------------------------


if __name__ == "__main__":
    import traceback
    import sys

    try:
        main()
    # handle error when running with --help
    except SystemExit as error:
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        sys.stdout.write(f"{repr(error)}\n")
    except:
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print.exc()
