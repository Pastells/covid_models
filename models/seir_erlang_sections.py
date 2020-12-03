"""
Stochastic mean-field SIR model
using the Gillespie algorithm and Erlang distribution transition times
It allows for different sections with different n, delta and beta
Pol Pastells, october 2020

Equations of the deterministic system
s[t] = S[t-1] - beta*i[t-1]*s[t-1]
i[t] = I[t-1] + beta*i[t-1]*s[t-1] - delta * I[t-1]
r[t] = R[t-1] + delta * I[t-1]
"""

import random
import sys
import traceback
import numpy as np
import utils
import seir_erlang


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%
def main():
    args = parsing()
    (
        E_0,
        I_0,
        R_0,
        n_t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n_sections,
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

        # initialization
        section = 0
        (
            n,
            ratios,
            shapes,
            section_day,
            ratios_old,
            section_day_old,
        ) = parameters_section(args, section)

        comp = seir_erlang.Compartments(n_t_steps, shapes, args)

        t_step, time, day = 0, 0, 1
        I_day[mc_step, 0] = I_0

        # Sections
        while section < n_sections:
            # Time loop
            while comp.I[t_step, :-1].sum() > 0 and day < section_day:
                day, day_max = utils.day_data(
                    time,
                    t_total,
                    day,
                    day_max,
                    comp.I[t_step, :-1].sum(),
                    I_day[mc_step],
                )
                t_step, time = gillespie(
                    t_total,
                    t_step,
                    time,
                    section_day_old,
                    comp,
                    ratios,
                    ratios_old,
                    shapes,
                )

            section += 1
            if section < n_sections:
                (
                    n,
                    ratios,
                    shapes,
                    section_day,
                    ratios_old,
                    section_day_old,
                ) = parameters_section(
                    args,
                    section,
                    ratios_old,
                    section_day,
                )
                comp.S[t_step, :-1] = (
                    n
                    - comp.I[t_step:-1].sum()
                    - comp.E[t_step:-1].sum()
                    - comp.R[t_step]
                ) / shapes["k_inf"]

        # -------------------------

        # plot all trajectories
        # if plot:
        # plt.plot(I_day[mc_step,:])
        # plt.plot(T[:t_step],i[:t_step,:-1].sum(1),c='c')
        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save:
        utils.saving(args, I_m, I_std, day_max, "seir_erlang_sections")

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


# -------------------------
def parsing():
    """input parameters"""
    import argparse

    parser = utils.ArgumentParser(
        description="Stochastic mean-field SEIR model using the Gillespie algorithm and Erlang \
            distribution transition times. It allows for different sections with different \
            n, delta and beta: same number of arguments must be specified for all three, \
            and one more for section_days.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=[int(1e4)],
        nargs="*",
        help="parameter: fixed number of (effecitve) people, must be monotonically increasing [1000,1000000]",
    )
    parser.add_argument(
        "--E_0", type=int, default=0, help="initial number of latent individuals"
    )
    parser.add_argument(
        "--I_0", type=int, default=20, help="initial number of infected individuals"
    )
    parser.add_argument(
        "--R_0", type=int, default=0, help="initial number of inmune individuals [0,n]"
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
        "--section_days",
        type=int,
        default=[0, 100],
        nargs="*",
        help="parameter: starting day for each section, firts one must be 0,\
                        and final day for last one",
    )

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
        "--day_min", type=int, default=33, help="first day to consider on data series"
    )
    parser.add_argument(
        "--day_max", type=int, default=58, help="last day to consider on data series"
    )

    parser.add_argument(
        "--mc_nseed",
        type=int,
        default=int(5),
        help="number of mc realizations, not really a parameter",
    )
    parser.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="initial mc seed, not really a parameter",
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

    if not utils.monotonically_increasing(args.n):
        raise ValueError("n should be monotonically increasing")

    E_0 = args.E_0
    I_0 = args.I_0
    R_0 = args.R_0
    n_t_steps = int(1e7)  # max simulation steps
    t_total = args.section_days[-1]  # max simulated days
    mc_nseed = args.mc_nseed  # MC realizations
    mc_seed0 = args.mc_seed0
    plot = args.plot
    save = args.save
    infected_time_series = genfromtxt(args.data, delimiter=",")[
        args.day_min : args.day_max
    ]
    n_sections = len(args.section_days) - 1
    # print(infected_time_series)
    return (
        E_0,
        I_0,
        R_0,
        n_t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n_sections,
    )


def parameters_section(
    args,
    section,
    ratios_old=None,
    section_day_old=0,
):
    """
    Section dependent parameters from argparse
    """
    n = args.n[section]
    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec, "k_lat": args.k_lat}
    ratios = {
        "beta1": args.beta1 / n * args.k_inf,
        "beta2": args.beta2 / n * args.k_inf,
        "delta1": args.delta1 * args.k_rec,
        "delta2": args.delta2 * args.k_rec,
        "epsilon": args.epsilon * args.k_lat,
    }
    section_day = args.section_days[section + 1]
    return (
        n,
        ratios,
        shapes,
        section_day,
        ratios_old,
        section_day_old + 1,
    )


# -------------------------


def gillespie(
    t_total,
    t_step,
    time,
    section_day_old,
    comp,
    ratios,
    ratios_old,
    shapes,
):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    stot = comp.S[t_step, :-1].sum()
    itot = comp.I[t_step, :-1].sum()
    etot_rec = comp.E[t_step, :-1, 0].sum()
    etot_inf = comp.E[t_step, :-1, 1].sum()
    etot = etot_inf + etot_rec - comp.E[t_step, 0, 0]

    beta1_eval, beta2_eval, delta1_eval, delta2_eval, epsilon_eval = utils.ratios_seir(
        time,
        ratios,
        ratios_old,
        section_day_old,
    )

    lambda_sum = (
        epsilon_eval * etot_inf
        + delta1_eval * etot_rec
        + delta2_eval * itot
        + (beta1_eval * etot + beta2_eval * itot) * stot
    )

    prob_heal1 = delta1_eval * comp.E[t_step, :-1, 0] / lambda_sum
    prob_heal2 = delta2_eval * comp.I[t_step, :-1] / lambda_sum
    prob_latent = epsilon_eval * comp.E[t_step, :-1, 1] / lambda_sum
    prob_infect = (
        (beta1_eval * etot + beta2_eval * itot) * comp.S[t_step, :-1] / lambda_sum
    )

    t_step += 1
    time += utils.time_dist(lambda_sum)
    if time > t_total:
        return t_step, True  # rare,  but sometimes long times may appear

    seir_erlang.gillespie_step(
        t_step,
        comp,
        prob_heal1,
        prob_heal2,
        prob_latent,
        prob_infect,
        shapes,
    )
    return t_step, time


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
