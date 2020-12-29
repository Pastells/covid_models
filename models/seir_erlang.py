"""
Stochastic mean-field SEIR model.
Uses the Gillespie algorithm and Erlang distribution transition times

Pol Pastells, 2020

Equations of the deterministic system:

dS(t)/dt = - beta_e*E(t)*S(t) - beta_i*I(t)*S(t) \n
dE(t)/dt =   beta_e*E(t)*S(t) + beta_i*I(t)*S(t) -(epsilon+delta_e)*E(t)\n
dI(t)/dt = - delta_i * I(t)                      + epsilon*E(t)\n
dR(t)/dt =   delta_i * I(t)                      + delta_e * E(t)
"""

import random
import sys
import traceback
import numpy as np
from utils import utils, config


def main():
    args = parsing()
    # print(args)
    t_total, infected_time_series, ratios, shapes = parameters_init(args)

    # results per day and seed
    I_day, I_m = (
        np.zeros([args.mc_nseed, t_total]),
        np.zeros(t_total),
    )

    mc_step, day_max = 0, 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(args.mc_seed0, args.mc_seed0 + args.mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)

        # -------------------------
        # initialization
        comp = Compartments(shapes, args)

        I_day[mc_step, 0] = args.I_0
        # T = np.zeros(n_t_steps)
        # T[0]=0
        t_step, time, day = 0, 0, 1

        # Time loop
        while comp.I[t_step, :-1].sum() > 0 and day < t_total:
            day, day_max = utils.day_data(
                time, t_total, day, day_max, comp.I[t_step, :-1].sum(), I_day[mc_step]
            )
            t_step, time = gillespie(
                t_total,
                t_step,
                time,
                comp,
                ratios,
                shapes,
            )
            if time is True:
                break

        # -------------------------

        """
        if plot:
            plt.plot(T[:t_step], S[:t_step, :-1].sum(1), c='r')
            plt.plot(T[:t_step], E[:t_step, :-1, 0].sum(1), c='g')
            plt.plot(T[:t_step], E[:t_step, :-1, 1].sum(1), c='b')
            plt.plot(T[:t_step], I[:t_step, :-1].sum(1), c='c')
            plt.plot(T[:t_step], R[:t_step], c='m')
        """

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, args.mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if args.save is not None:
        utils.saving(args, I_m, I_std, day_max, "seir_erlang", args.save)

    if args.plot:
        from utils import plots

        plots.plotting(args, I_day, day_max, I_m, I_std)


# -------------------------
def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic mean-field SEIR model using the Gillespie algorithm and Erlang \
            distribution transition times. \
            Dependencies: config.py, utils.py",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )

    parser_params = parser.add_argument_group("parameters")

    parser_params.add_argument(
        "--n",
        type=int,
        default=config.N,
        help="fixed number of (effecitve) people [1000,1000000]",
    )
    parser_params.add_argument(
        "--delta_e",
        type=float,
        default=config.DELTA_E,
        help="ratio of recovery from latent fase (e->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--delta_i",
        type=float,
        default=config.DELTA,
        help="ratio of recovery from infected fase (i->r) [0.05,1]",
    )
    parser_params.add_argument(
        "--k_rec",
        type=int,
        default=config.K_REC,
        help="k for the recovery time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--beta_e",
        type=float,
        default=config.BETA_E,
        help="ratio of infection due to latent [0.05,1]",
    )
    parser_params.add_argument(
        "--beta_i",
        type=float,
        default=config.BETA,
        help="ratio of infection due to infected [0.05,1]",
    )
    parser_params.add_argument(
        "--k_inf",
        type=int,
        default=config.K_INF,
        help="k for the infection time erlang distribution [1,5]",
    )
    parser_params.add_argument(
        "--epsilon",
        type=float,
        default=config.EPSILON,
        help="ratio of latency (e->i) [0.05,2]",
    )
    parser_params.add_argument(
        "--k_lat",
        type=int,
        default=config.K_LAT,
        help="k for the latent time erlang distribution [1,5]",
    )

    utils.parser_common(parser, True)
    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, infected_time_series = utils.parameters_init_common(args)

    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec, "k_lat": args.k_lat}
    ratios = {
        "beta_e": args.beta_e / args.n * args.k_inf,
        "beta_i": args.beta_i / args.n * args.k_inf,
        "delta_e": args.delta_e * args.k_rec,
        "delta_i": args.delta_i * args.k_rec,
        "epsilon": args.epsilon * args.k_lat,
    }
    return t_total, infected_time_series, ratios, shapes


# -------------------------


class Compartments:
    """Compartments for the SEIR Erlang model"""

    def __init__(self, shapes, args):
        """Initialization"""
        self.S = np.zeros([args.n_t_steps, shapes["k_inf"] + 1])
        self.E = np.zeros([args.n_t_steps, shapes["k_lat"] + 1, 2])
        self.I = np.zeros([args.n_t_steps, shapes["k_rec"] + 1])
        self.R = np.zeros(args.n_t_steps)

        # Used for both seir_erlang and seir_erlang sections, where args.n is a vector
        try:
            self.S[0, :-1] = (args.n - args.I_0 - args.R_0) / shapes["k_inf"]
        except TypeError:
            self.S[0, :-1] = (args.n[0] - args.I_0 - args.R_0) / shapes["k_inf"]

        self.S[0, -1] = self.E[0, :-1] = args.E_0 / shapes["k_lat"]
        self.E[0, -1] = self.I[0, :-1] = args.I_0 / shapes["k_rec"]
        self.I[0, -1] = self.R[0] = args.R_0

    def latent_adv_s(self, t_step, k):
        """Turn latent or advance in S
        S(k)-> S(k+1)/E(0)"""
        self.E[t_step, :-1] = self.E[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.R[t_step] = self.R[t_step - 1]
        self.S[t_step, k] = -1
        self.S[t_step, k + 1] = 1
        self.E[t_step, 0] += self.S[t_step, -1]
        self.S[t_step] += self.S[t_step - 1]

    def infect_adv_e(self, t_step, k):
        """Turn infectious or advance in E
        E(k)-> E(k+1)/I(0)"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.R[t_step] = self.R[t_step - 1]
        self.E[t_step, k, 1] = -1
        self.E[t_step, k + 1, 1] = 1
        self.I[t_step, 0] += self.E[t_step, -1, 1]
        self.E[t_step, 0, 0] = self.E[t_step, 0, 1]
        self.E[t_step] += self.E[t_step - 1]

    def recover_adv_e(self, t_step, k):
        """Recover or advance in E
        E(k)-> E(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.E[t_step, k, 0] = -1
        self.E[t_step, k + 1, 0] = 1
        self.R[t_step] = self.R[t_step - 1] + self.E[t_step, -1, 0]
        self.E[t_step, 0, 1] = self.E[t_step, 0, 0]
        self.E[t_step] += self.E[t_step - 1]

    def recover_adv_i(self, t_step, k):
        """Recover or advance in I
        I(k)-> I(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.E[t_step, :-1] = self.E[t_step - 1, :-1]
        self.I[t_step, k] = -1
        self.I[t_step, k + 1] = 1
        self.R[t_step] = self.R[t_step - 1] + self.I[t_step, -1]
        self.I[t_step] += self.I[t_step - 1]


# -------------------------


def gillespie(
    t_total,
    t_step,
    time,
    comp,
    ratios,
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

    lambda_sum = (
        ratios["epsilon"] * etot_inf
        + ratios["delta_e"] * etot_rec
        + ratios["delta_i"] * itot
        + (ratios["beta_e"] * etot + ratios["beta_i"] * itot) * stot
    )

    probs = {}
    probs["heal1"] = ratios["delta_e"] * comp.E[t_step, :-1, 0] / lambda_sum
    probs["heal2"] = ratios["delta_i"] * comp.I[t_step, :-1] / lambda_sum
    probs["latent"] = ratios["epsilon"] * comp.E[t_step, :-1, 1] / lambda_sum
    probs["infect"] = (
        (ratios["beta_e"] * etot + ratios["beta_i"] * itot)
        * comp.S[t_step, :-1]
        / lambda_sum
    )

    t_step += 1
    time += utils.time_dist(lambda_sum)
    if time > t_total:
        return t_step, True  # rare,  but sometimes long times may appear
    # T[t_step] = time

    gillespie_step(t_step, comp, probs, shapes)
    return t_step, time


# -------------------------


def gillespie_step(t_step, comp, probs, shapes):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    S and I have one extra dimension to temporally store the infected and recovered
    after k stages, due to the Erlang distribution
    """
    random = np.random.random()
    prob_heal1_tot = probs["heal1"].sum()
    prob_heal2_tot = probs["heal2"].sum()
    prob_latent_tot = probs["latent"].sum()

    # E(k)-> E(k+1)/R
    if random < prob_heal1_tot:
        for k in range(shapes["k_lat"]):
            if random < probs["heal1"][: k + 1].sum():
                comp.recover_adv_e(t_step, k)
                return

    # I(k)-> I(k+1)/R
    if random < (prob_heal1_tot + prob_heal2_tot):
        random -= prob_heal1_tot
        for k in range(shapes["k_rec"]):
            if random < probs["heal2"][: k + 1].sum():
                comp.recover_adv_i(t_step, k)
                return

    # E(k)-> E(k+1)/I(0)
    if random < (prob_heal1_tot + prob_heal2_tot + prob_latent_tot):
        random -= prob_heal1_tot + prob_heal2_tot
        for k in range(shapes["k_lat"]):
            if random < probs["latent"][: k + 1].sum():
                comp.infect_adv_e(t_step, k)
                return

    # S(k)-> S(k+1)/E(0)
    random -= prob_heal1_tot + prob_heal2_tot + prob_latent_tot
    for k in range(shapes["k_inf"]):
        if random < probs["infect"][: k + 1].sum():
            comp.latent_adv_s(t_step, k)
            return


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
