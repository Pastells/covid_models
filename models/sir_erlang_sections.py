"""
Stochastic mean-field SIR model
using the Gillespie algorithm and Erlang distribution transition times
It allows for different sections with different n, delta and beta
Pol Pastells, october 2020

Equations of the deterministic system
s[t] = s[t-1] - beta*i[t-1]*s[t-1]
i[t] = i[t-1] + beta*i[t-1]*s[t-1] - delta * i[t-1]
r[t] = r[t-1] + delta * i[t-1]
"""

import numpy as np
import utils

# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    args = parsing()
    (
        i_0,
        r_0,
        t_steps,
        t_total,
        nseed,
        seed0,
        plot,
        save,
        infected_time_series,
        n_sections,
    ) = parameters_init(args)

    # results per day and seed
    days_gap = 10
    # s_day,s_m,s_95 = np.zeros([nseed,t_total+days_gap]),np.zeros(t_total+days_gap),np.zeros([t_total+days_gap,2])
    i_day, i_m = (
        np.zeros([nseed, t_total + days_gap]),
        np.zeros(t_total + days_gap),
    )
    # r_day,r_m,r_95 = np.zeros([nseed,t_total+days_gap]),np.zeros(t_total+days_gap),np.zeros([t_total+days_gap,2])

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for seed in range(seed0, seed0 + nseed):
        np.random.seed(seed)

        # initialization
        section = 0
        n, k_inf, k_rec, beta, delta, section_day = parameters_section(args, section)
        s, i, r = (
            np.zeros([t_steps, k_inf + 1]),
            np.zeros([t_steps, k_rec + 1]),
            np.zeros(t_steps),
        )
        t, time, day = 0, 0, 1
        i_day[mc_step, 0] = i_0

        s[t, :-1] = (n - i_0 - r_0) / k_inf
        s[t, -1], i[t, :-1] = i_0 / k_rec, i_0 / k_rec
        i[t, -1], r[t] = r_0, r_0
        # T = np.zeros(t_steps)
        # T[0]=0

        # Sections
        while section < n_sections:
            # Time loop
            while i[t, :-1].sum() > 0 and day < section_day:
                day, day_max = utils.day_data_k(
                    mc_step, t, time, day, day_max, i, i_day
                )
                t, time = gillespie(t, time, s, i, r, beta, delta, k_rec, k_inf)

            section += 1
            if section < n_sections:
                n, k_inf, k_rec, beta, delta, section_day = parameters_section(
                    args, section
                )
                s[t, :-1] = (n - i[t:-1].sum() - r[t]) / k_inf

        # -------------------------
        day, day_max = utils.day_data_k(mc_step, t, time, day, day_max, i, i_day, True)
        # final value for the rest of time, otherwise it contributes with a zero when averaged
        # s_day[mc_step,day:] = s_day[mc_step,day-1]
        i_day[mc_step, day:] = i_day[mc_step, day - 1]
        # r_day[mc_step,day:] = r_day[mc_step,day-1]

        # plot all trajectories
        # if plot:
        # plt.plot(i_day[mc_step,:])
        # plt.plot(T[:t],i[:t,:-1].sum(1),c='c')
        mc_step += 1
    # =========================

    i_m, i_std = utils.mean_alive(i_day, t_total, day_max, nseed, days_gap)

    utils.cost_func(infected_time_series, i_m, i_std)

    if save:
        utils.saving(args, i_m, i_std, day_max)

    if plot:
        utils.plotting(infected_time_series, i_day, day_max, i_m, i_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


# -------------------------
def parsing():
    """
    input parameters
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic mean-field SIR model using the Gillespie algorithm and Erlang \
            distribution transition times. It allows for different sections with different \
            n, delta and beta: same number of arguments must be specified for all three, \
            and one more for section_days.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=[int(1e4)],
        nargs="*",
        help="Fixed number of (effecitve) people [1000,1000000]",
    )
    parser.add_argument(
        "--i_0",
        type=int,
        default=20,
        help="initial number of infected individuals [1,n]",
    )
    parser.add_argument(
        "--r_0", type=int, default=0, help="initial number of inmune individuals [0,n]"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=[0.2],
        nargs="*",
        help="Mean ratio of recovery [1e-2,1]",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=[0.5],
        nargs="*",
        help="ratio of infection [1e-2,1]",
    )
    parser.add_argument(
        "--k_rec",
        type=int,
        default=1,
        help="k parameter for the recovery time Erlang distribution [1,5]",
    )
    parser.add_argument(
        "--k_inf",
        type=int,
        default=1,
        help="k parameter for the infection time Erlang distribution [1,5]",
    )
    parser.add_argument(
        "--section_days",
        type=int,
        default=[0, 100],
        nargs="*",
        help="starting day for each section, firts one must be 0,\
                        and final day for last one",
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
        help="Number of realizations, not really a parameter",
    )
    parser.add_argument(
        "--seed0", type=int, default=1, help="initial seed, not really a parameter"
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

    i_0 = args.i_0
    r_0 = args.r_0
    t_steps = int(1e7)  # max simulation steps
    t_total = args.section_days[-1]  # max simulated days
    nseed = args.nseed  # MC realizations
    seed0 = args.seed0
    plot = args.plot
    save = args.save
    infected_time_series = genfromtxt(args.data, delimiter=",")[
        args.day_min : args.day_max
    ]
    n_sections = len(args.section_days) - 1
    # print(infected_time_series)
    return (
        i_0,
        r_0,
        t_steps,
        t_total,
        nseed,
        seed0,
        plot,
        save,
        infected_time_series,
        n_sections,
    )


def parameters_section(args, section):
    """
    Section dependent parameters from argparse
    """
    n = args.n[section]
    k_inf = args.k_inf
    k_rec = args.k_rec
    beta = args.beta[section] / n * k_inf
    delta = args.delta[section] * k_rec
    section_day = args.section_days[section + 1]
    return n, k_inf, k_rec, beta, delta, section_day


# -------------------------


def gillespie(t, time, s, i, r, beta, delta, k_rec, k_inf):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    stot = s[t, :-1].sum()
    itot = i[t, :-1].sum()

    lambda_sum = (delta + utils.beta_func(beta, t) * stot) * itot
    prob_heal = delta * i[t, :-1] / lambda_sum
    prob_infect = utils.beta_func(beta, t) * s[t, :-1] * itot / lambda_sum

    t += 1
    time += utils.time_dist(lambda_sum)
    # T[t] = time

    gillespie_step(t, s, i, r, prob_heal, prob_infect, k_rec, k_inf)
    return t, time


# -------------------------


def gillespie_step(t, s, i, r, prob_heal, prob_infect, k_rec, k_inf):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    s and i have one extra dimension to temporally store the infected and recovered after k stages,
    due to the Erlang distribution
    """
    random = np.random.random()
    prob_heal_tot = prob_heal.sum()

    # i(k)-> i(k+1)/r
    if random < prob_heal_tot:
        for k in range(k_rec):
            if random < prob_heal[: k + 1].sum():
                s[t, :-1] = s[t - 1, :-1]
                i[t, k] = -1
                i[t, k + 1] = 1
                r[t] = r[t - 1] + i[t, k_rec]
                i[t] += i[t - 1]
                break

    # s(k)-> s(k+1)/i(0)
    else:
        for k in range(k_inf):
            if random < (prob_heal_tot + prob_infect[: k + 1].sum()):
                r[t] = r[t - 1]
                i[t, :-1] = i[t - 1, :-1]
                s[t, k] = -1
                s[t, k + 1] = 1
                i[t, 0] += s[t, k_inf]
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
