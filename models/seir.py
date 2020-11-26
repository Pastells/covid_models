"""
Stochastic mean-field SEIR model using the Gillespie algorithm
Pol Pastells, october 2020

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


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
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
        beta1,
        beta2,
        delta1,
        delta2,
        epsilon,
    ) = parameters_init(args)

    # results per day and seed
    I_day, I_m = (
        np.zeros([mc_nseed, t_total]),
        np.zeros(t_total),
    )

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
            np.zeros(t_steps),
            np.zeros(t_steps),
            np.zeros(t_steps),
            np.zeros(t_steps),
        )
        # T = np.zeros(t_steps)
        # T[0]=0
        E[0] = E_0
        I[0] = I_0
        R[0] = R_0
        S[0] = n - I_0 - R_0 - E_0
        I_day[mc_step, 0] = I_0
        t, time, day = 0, 0, 1

        # -------------------------
        # Time loop
        # -------------------------
        while I[t] > 0.1 and day < t_total:
            day, day_max = utils.day_data(
                time, t_total, day, day_max, I[t], I_day[mc_step]
            )
            t, time = gillespie(
                t_total, t, time, S, E, I, R, beta1, beta2, delta1, delta2, epsilon
            )

        # -------------------------

        mc_step += 1
    # =========================
    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save:
        utils.saving(args, I_m, I_std, day_max, "seir")

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# -------------------------


def parsing():
    """input parameters"""
    import argparse

    parser = utils.ArgumentParser(
        description="stochastic mean-field SEIR model using the Gillespie algorithm",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="parameter: fixed number of (effecitve) people [1000,1000000]",
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
        "--epsilon",
        type=float,
        default=1,
        help="parameter: ratio of latency (e->i) [1e-2,1]",
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
    """initial parameters from argparse"""
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
    beta1 = args.beta1 / n
    beta2 = args.beta2 / n
    delta1 = args.delta1
    delta2 = args.delta2
    epsilon = args.epsilon

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
        beta1,
        beta2,
        delta1,
        delta2,
        epsilon,
    )


# -------------------------


def gillespie(t_total, t, time, S, E, I, R, beta1, beta2, delta1, delta2, epsilon):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    lambda_sum = (
        (epsilon + delta1) * E[t] + delta2 * I[t] + (beta1 * E[t] + beta2 * I[t]) * S[t]
    )

    prob_heal1 = delta1 * E[t] / lambda_sum
    prob_heal2 = delta2 * I[t] / lambda_sum
    prob_latent = epsilon * E[t] / lambda_sum

    t += 1
    time += utils.time_dist(lambda_sum)
    # T[t] = time

    gillespie_step(t, S, E, I, R, prob_heal1, prob_heal2, prob_latent)
    return t, time


# -------------------------


def gillespie_step(t, S, E, I, R, prob_heal1, prob_heal2, prob_latent):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < prob_heal1:
        # e->r
        S[t] = S[t - 1]
        E[t] = E[t - 1] - 1
        I[t] = I[t - 1]
        R[t] = R[t - 1] + 1
    elif random < (prob_heal1 + prob_heal2):
        # i->r
        S[t] = S[t - 1]
        E[t] = E[t - 1]
        I[t] = I[t - 1] - 1
        R[t] = R[t - 1] + 1
    elif random < (prob_heal1 + prob_heal2 + prob_latent):
        # e->i
        S[t] = S[t - 1]
        E[t] = E[t - 1] - 1
        I[t] = I[t - 1] + 1
        R[t] = R[t - 1]
    else:
        # s->e
        S[t] = S[t - 1] - 1
        E[t] = E[t - 1] + 1
        I[t] = I[t - 1]
        R[t] = R[t - 1]


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
