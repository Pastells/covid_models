"""
Stochastic mean-field SEIR model using the Gillespie algorithm
Pol Pastells,  october 2020

equations of the deterministic system
s[t] = S[t-1] - beta1*e[t-1]*s[t-1] - beta2*i[t-1]*s[t-1]
e[t] = E[t-1] + beta1*e[t-1]*s[t-1] + beta2*i[t-1]*s[t-1] - (epsilon+delta1)*e[t-1]
i[t] = I[t-1] + epsilon*e[t-1] - delta2 * I[t-1]
r[t] = R[t-1] + delta1 *e[t-1] + delta2 * I[t-1]
"""

import random
import sys
import traceback
import numpy as np
import utils


def main():
    args = parsing()
    (
        E_0,
        I_0,
        R_0,
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
        delta2,
        epsilon,
    ) = parameters_init(args)

    # results per day and seed
    # S_day, S_m, S_95 = np.zeros([mc_nseed, t_total]), np.zeros(t_total), np.zeros([t_total, 2])
    # E_day, E_m, E_95 = np.zeros([mc_nseed, t_total]), np.zeros(t_total), np.zeros([t_total, 2])
    I_day, I_m = (
        np.zeros([mc_nseed, t_total]),
        np.zeros(t_total),
    )
    # R_day, R_m, R_95 = np.zeros([mc_nseed, t_total]), np.zeros(t_total), np.zeros([t_total, 2])

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for seed in range(mc_seed0, mc_seed0 + mc_nseed):
        random.seed(seed)
        np.random.seed(seed)

        # -------------------------
        # initialization
        S, E, I, R = (
            np.zeros([t_steps, k_inf + 1]),
            np.zeros([t_steps, k_lat + 1, 2]),
            np.zeros([t_steps, k_rec + 1]),
            np.zeros(t_steps),
        )
        S[0, :-1] = (n - I_0 - R_0) / k_inf
        S[0, -1], E[0, :-1] = E_0 / k_lat, E_0 / k_lat
        E[0, -1], I[0, :-1] = I_0 / k_rec, I_0 / k_rec
        I[0, -1], R[0] = R_0, R_0

        # S_day[mc_step, 0]=s[0]
        # E_day[mc_step, 0]=e_0
        I_day[mc_step, 0] = I_0
        # R_day[mc_step, 0]=r_0
        # T = np.zeros(t_steps)
        # T[0]=0
        t, time, day = 0, 0, 1

        # Time loop
        while I[t, :-1].sum() > 0 and day < t_total:
            day, day_max = utils.day_data(
                time, t_total, day, day_max, I[t, :-1].sum(), I_day[mc_step]
            )
            t, time = gillespie(
                t_total,
                t,
                time,
                S,
                E,
                I,
                R,
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

        """
        if plot:
            plt.plot(T[:t], S[:t, :-1].sum(1), c='r')
            plt.plot(T[:t], E[:t, :-1, 0].sum(1), c='g')
            plt.plot(T[:t], E[:t, :-1, 1].sum(1), c='b')
            plt.plot(T[:t], I[:t, :-1].sum(1), c='c')
            plt.plot(T[:t], R[:t], c='m')
        """

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save:
        utils.saving(args, I_m, I_std, day_max, "seir_erlang")

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# -------------------------
def parsing():
    """input parameters"""
    import argparse

    parser = utils.ArgumentParser(
        description="Stochastic mean-field SEIR model using the Gillespie algorithm and Erlang \
            distribution transition times.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="parameter: fixed number of (effecitve) people [1000,1000000]",
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
        default=0.01,
        help="parameter: ratio of recovery from latent fase (e->r) [1e-2,1]",
    )
    parser.add_argument(
        "--delta2",
        type=float,
        default=0.2,
        help="parameter: ratio of recovery from infected fase (i->r) [1e-2,1]",
    )
    parser.add_argument(
        "--k_rec",
        type=int,
        default=1,
        help="parameter: k for the recovery time erlang distribution [1,5]",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.01,
        help="parameter: ratio of infection due to latent [1e-2,1]",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.5,
        help="parameter: ratio of infection due to infected [1e-2,1]",
    )
    parser.add_argument(
        "--k_inf",
        type=int,
        default=1,
        help="parameter: k for the infection time erlang distribution [1,5]",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1,
        help="parameter: ratio of latency (e->i) [1e-2,1]",
    )
    parser.add_argument(
        "--k_lat",
        type=int,
        default=1,
        help="parameter: k for the latent time erlang distribution [1,5]",
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="seed for the automatic configuration"
    )
    parser.add_argument(
        "--data", type=str, default="../data/italy_i.csv", help="file with time series"
    )
    parser.add_argument(
        "--day_min", type=int, default=33, help="first day to consider on data series"
    )
    parser.add_argument(
        "--day_max", type=int, default=58, help="last day to consider on data series"
    )

    parser.add_argument(
        "--mc_nseed",
        type=int,
        default=int(5),
        help="number of mc realizations",
    )
    parser.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="initial mc seed",
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")
    parser.add_argument("--save", action="store_true", help="specify for outputfile")

    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters
def parameters_init(args):
    """Initial parameters from argparse"""
    from numpy import genfromtxt

    E_0 = args.e_0
    I_0 = args.i_0
    R_0 = args.r_0
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
    delta2 = args.delta2 * k_rec
    epsilon = args.epsilon * k_lat
    return (
        E_0,
        I_0,
        R_0,
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
        delta2,
        epsilon,
    )


def gillespie(
    t_total,
    t,
    time,
    S,
    E,
    I,
    R,
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
    stot = S[t, :-1].sum()
    itot = I[t, :-1].sum()
    etot_rec = E[t, :-1, 0].sum()
    etot_inf = E[t, :-1, 1].sum()
    etot = etot_inf + etot_rec - E[t, 0, 0]

    lambda_sum = (
        epsilon * etot_inf
        + delta1 * etot_rec
        + delta2 * itot
        + (beta1 * etot + beta2 * itot) * stot
    )

    prob_heal1 = delta1 * E[t, :-1, 0] / lambda_sum
    prob_heal2 = delta2 * I[t, :-1] / lambda_sum
    prob_latent = epsilon * E[t, :-1, 1] / lambda_sum
    prob_infect = (beta1 * etot + beta2 * itot) * S[t, :-1] / lambda_sum

    t += 1
    time += utils.time_dist(lambda_sum)
    if time > t_total:
        return t, True  # rare,  but sometimes long times may appear
    # T[t] = time

    gillespie_step(
        t,
        S,
        E,
        I,
        R,
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
    t, S, E, I, R, prob_heal1, prob_heal2, prob_latent, prob_infect, k_rec, k_lat, k_inf
):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    S and I have one extra dimension to temporally store the infected and recovered
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
                S[t, :-1] = S[t - 1, :-1]
                I[t, :-1] = I[t - 1, :-1]
                E[t, k, 0] = -1
                E[t, k + 1, 0] = 1
                R[t] = R[t - 1] + E[t, k_lat, 0]
                E[t, 0, 1] = E[t, 0, 0]
                E[t] += E[t - 1]
                break

    # i(k)-> i(k+1)/r
    elif random < (prob_heal1_tot + prob_heal2_tot):
        random -= prob_heal1_tot
        for k in range(k_rec):
            if random < prob_heal2[: k + 1].sum():
                S[t, :-1] = S[t - 1, :-1]
                E[t, :-1] = E[t - 1, :-1]
                I[t, k] = -1
                I[t, k + 1] = 1
                R[t] = R[t - 1] + I[t, k_rec]
                I[t] += I[t - 1]
                break

    # e(k)-> e(k+1)/i(0)
    elif random < (prob_heal1_tot + prob_heal2_tot + prob_latent_tot):
        random -= prob_heal1_tot + prob_heal2_tot
        for k in range(k_lat):
            if random < prob_latent[: k + 1].sum():
                S[t, :-1] = S[t - 1, :-1]
                I[t, :-1] = I[t - 1, :-1]
                R[t] = R[t - 1]
                E[t, k, 1] = -1
                E[t, k + 1, 1] = 1
                I[t, 0] += E[t, k_lat, 1]
                E[t, 0, 0] = E[t, 0, 1]
                E[t] += E[t - 1]
                break

    # s(k)-> s(k+1)/e(0)
    else:
        random -= prob_heal1_tot + prob_heal2_tot + prob_latent_tot
        for k in range(k_inf):
            if random < prob_infect[: k + 1].sum():
                E[t, :-1] = E[t - 1, :-1]
                I[t, :-1] = I[t - 1, :-1]
                R[t] = R[t - 1]
                S[t, k] = -1
                S[t, k + 1] = 1
                E[t, 0] += S[t, k_inf]
                S[t] += S[t - 1]
                break


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
