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
import plots
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
    for mc_seed in range(mc_seed0, mc_seed0 + mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)

        # initialization
        section = 0
        (
            n,
            ratios,
            shapes,
            section_day,
            ratios_old,
            section_day_old,
            n_ind,
        ) = parameters_section(args, section)

        comp = seir_erlang.Compartments(n_t_steps, shapes, args)

        t_step, time, day = 0, 0, 1
        I_day[mc_step, 0] = I_0
        index_n = 1  # just to avoid pylint complaining

        # Sections
        while section < n_sections:
            # Time loop
            while comp.I[t_step, :-1].sum() > 0 and day < section_day:

                # add individuals
                if n_ind is not None:
                    if time // n_ind[index_n, 1] == 1:
                        comp.S[t_step] += n_ind[index_n] / shapes["k_inf"]
                        if index_n < (len(n_ind) - 1):
                            index_n += 1
                        else:
                            n_ind = None

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
                    n_ind,
                ) = parameters_section(args, section, ratios, section_day, n)
                comp.S[t_step, :-1] += n_ind[0, 0] / shapes["k_inf"]
                index_n = 1

        # -------------------------

        # plot all trajectories
        # if plot:
        # plt.plot(I_day[mc_step,:])
        # plt.plot(T[:t_step],i[:t_step,:-1].sum(1),c='c')
        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save is not None:
        utils.saving(args, I_m, I_std, day_max, "net_sir", save)

    if plot:
        plots.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


# -------------------------
def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic mean-field SEIR model using the Gillespie algorithm and Erlang \
            distribution transition times. It allows for different sections with different \
            n, delta and beta: same number of arguments must be specified for all three, \
            and one more for section_days.",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser_init = parser.add_argument_group("initial conditions")
    parser_params = parser.add_argument_group("parameters")

    parser_init.add_argument(
        "--E_0", type=int, default=0, help="initial number of latent individuals"
    )
    parser_init.add_argument(
        "--I_0", type=int, default=20, help="initial number of infected individuals"
    )
    parser_init.add_argument(
        "--R_0", type=int, default=0, help="initial number of inmune individuals"
    )

    parser_params.add_argument(
        "--n",
        type=int,
        default=[int(1e4)],
        nargs="*",
        help="fixed number of (effecitve) people, initial and increments [1000,1000000]",
    )
    parser_params.add_argument(
        "--delta1",
        type=float,
        default=[0.01],
        nargs="*",
        help="ratio of recovery from latent fase (e->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--delta2",
        type=float,
        default=[0.2],
        nargs="*",
        help="ratio of recovery from infected fase (i->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--k_rec",
        type=int,
        default=1,
        help="k for the recovery time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--beta1",
        type=float,
        default=[0.01],
        nargs="*",
        help="ratio of infection due to latent [0.05,1]",
    )
    parser_params.add_argument(
        "--beta2",
        type=float,
        default=[0.5],
        nargs="*",
        help="ratio of infection due to infected [0.05,1]",
    )
    parser_params.add_argument(
        "--k_inf",
        type=int,
        default=1,
        help="k for the infection time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--epsilon",
        type=float,
        default=[1],
        nargs="*",
        help="ratio of latency (e->i) [0.05,1]",
    )
    parser_params.add_argument(
        "--k_lat",
        type=int,
        default=1,
        help="k for the latent time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--section_days",
        type=int,
        default=[0, 100],
        nargs="*",
        help="starting day for each section, first one must be 0,\
                        and final day for last one",
    )
    parser_params.add_argument(
        "--transition_days",
        type=int,
        default=4,
        help="days it takes to transition from one number of individuals to the next [1,10]",
    )

    utils.parser_common(parser)

    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters


def parameters_init(args):
    """Initial parameters from argparse"""
    from numpy import genfromtxt

    E_0 = args.E_0
    I_0 = args.I_0
    R_0 = args.R_0
    n_t_steps = args.n_t_steps  # max simulation steps
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


def parameters_section(args, section, ratios_old=None, section_day_old=0, n_old=None):
    """
    Section dependent parameters from argparse
    """
    n = sum(args.n[: section + 1])
    n_ind = utils.n_individuals(n, n_old, section_day_old, args.transition_days)
    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec, "k_lat": args.k_lat}
    ratios = {
        "beta1": args.beta1[section] / n * args.k_inf,
        "beta2": args.beta2[section] / n * args.k_inf,
        "delta1": args.delta1[section] * args.k_rec,
        "delta2": args.delta2[section] * args.k_rec,
        "epsilon": args.epsilon[section] * args.k_lat,
    }
    section_day = args.section_days[section + 1]

    # should return section_day_old , given that the input will be section_day.
    # but I want to add 2 to it for the tanh in ratios_sir/n_individuals
    return (n, ratios, shapes, section_day, ratios_old, section_day_old + 1, n_ind)


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
