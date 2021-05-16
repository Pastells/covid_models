"""
Stochastic mean-field SAIR model.
Uses the Gillespie algorithm and Erlang distribution transition times

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta_a/N*A(t)*S(t) - beta/N*I(t)*S(t) \n
dA(t)/dt =   beta_a/N*A(t)*S(t) + beta/N*I(t)*S(t) -(alpha+delta_a)*A(t)\n
dI(t)/dt = - delta * I(t)                          + alpha*A(t)\n
dR(t)/dt =   delta * I(t)                          + delta_a * A(t)
"""

import random
import sys
import traceback
import numpy as np
from utils import utils, config


def main():
    args = parsing()
    t_total, time_series, rates, shapes = parameters_init(args)
    # print(args)

    # results per day and seed
    I_day = np.zeros([args.mc_nseed, t_total], dtype=int)

    day_max = 0
    # =========================
    # MC loop
    # =========================
    for mc_seed in range(args.seed, args.seed + args.mc_nseed):
        random.seed(mc_seed)
        np.random.seed(mc_seed)
        mc_step = mc_seed - args.seed

        # -------------------------
        # initialization
        comp = Compartments(shapes, args)

        I_day[mc_step, 0] = args.initial_infected
        t_step, time = 0, 0

        # Time loop
        while comp.I[t_step, :-1].sum() > 0 and time < t_total:
            t_step, time = gillespie(
                t_total,
                t_step,
                time,
                comp,
                rates,
                shapes,
            )
            if time is True:
                break

        # -------------------------

        if config.CUMULATIVE is True:
            i_var = comp.I_cum
        else:
            i_var = comp.I[:, :-1].sum(axis=1)

        day_max = utils.day_data(
            comp.T[:t_step], i_var[:t_step], I_day[mc_step], day_max
        )

    # =========================

    utils.cost_save_plot(I_day, t_total, day_max, args, time_series)


# -------------------------
def parsing():
    """input parameters"""

    description = "Stochastic mean-field SAIR model using the Gillespie algorithm and Erlang \
        distribution transition times. Dependencies: config.py, utils.py"

    parser = utils.ParserCommon(description)
    parser.n()
    parser.sir()
    parser.asymptomatic()
    parser.erlang(True)

    return parser.parse_args()


# -------------------------
# Parameters


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    shapes = {"k_inf": args.k_inf, "k_rec": args.k_rec, "k_asym": args.k_asym}
    rates = {
        "beta_a": args.beta_a / args.n * args.k_inf,
        "beta": args.beta / args.n * args.k_inf,
        "delta_a": args.delta_a * args.k_rec,
        "delta": args.delta * args.k_rec,
        "alpha": args.alpha * args.k_asym,
    }
    return t_total, time_series, rates, shapes


# -------------------------


class Compartments:
    """Compartments for the SAIR Erlang model"""

    def __init__(self, shapes, args):
        """Initialization
        S, A and I are vectors, with one dimension more than the Erlang shape.
        This extra dimension is used to facilitate notation.
        E.g.: both infection and advance in S remove an individual from the k
        dimension and add one to the k+1 in S. In case where k+1 is the last entry,
        the individual is added to the first I compartment."""

        self.S = np.zeros([args.n_t_steps, shapes["k_inf"] + 1])
        self.A = np.zeros([args.n_t_steps, shapes["k_asym"] + 1, 2])
        self.I = np.zeros([args.n_t_steps, shapes["k_rec"] + 1])
        self.R = np.zeros(args.n_t_steps)
        self.T = np.zeros(args.n_t_steps)

        # Used for both sair_erlang and sair_erlang sections, where args.n is a vector
        try:
            self.S[0, :-1] = (args.n - args.initial_infected - args.initial_recovered) / shapes["k_inf"]
        except TypeError:
            self.S[0, :-1] = (args.n[0] - args.initial_infected - args.initial_recovered) / shapes["k_inf"]

        self.S[0, -1] = self.A[0, :-1] = args.initial_asymptomatic / shapes["k_asym"]
        self.A[0, -1] = self.I[0, :-1] = args.initial_infected / shapes["k_rec"]
        self.I[0, -1] = self.R[0] = args.initial_recovered
        self.T[0] = 0
        self.I_cum = np.zeros(args.n_t_steps)
        self.I_cum[0] = args.initial_infected

    def asymptomatic_adv_s(self, t_step, k):
        """Turn asymptomatic or advance in S
        S(k)-> S(k+1)/A(0)"""
        self.A[t_step, :-1] = self.A[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.R[t_step] = self.R[t_step - 1]
        self.S[t_step, k] = -1
        self.S[t_step, k + 1] = 1
        self.A[t_step, 0] += self.S[t_step, -1]
        self.S[t_step] += self.S[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def infect_adv_a(self, t_step, k):
        """Turn infectious or advance in A
        A(k)-> A(k+1)/I(0)"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.R[t_step] = self.R[t_step - 1]
        self.A[t_step, k, 1] = -1
        self.A[t_step, k + 1, 1] = 1
        self.I[t_step, 0] += self.A[t_step, -1, 1]
        self.A[t_step, 0, 0] = self.A[t_step, 0, 1]
        self.A[t_step] += self.A[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + self.S[t_step, -1]

    def recover_adv_a(self, t_step, k):
        """Recover or advance in A
        A(k)-> A(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.A[t_step, k, 0] = -1
        self.A[t_step, k + 1, 0] = 1
        self.R[t_step] = self.R[t_step - 1] + self.A[t_step, -1, 0]
        self.A[t_step, 0, 1] = self.A[t_step, 0, 0]
        self.A[t_step] += self.A[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]

    def recover_adv_i(self, t_step, k):
        """Recover or advance in I
        I(k)-> I(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.A[t_step, :-1] = self.A[t_step - 1, :-1]
        self.I[t_step, k] = -1
        self.I[t_step, k + 1] = 1
        self.R[t_step] = self.R[t_step - 1] + self.I[t_step, -1]
        self.I[t_step] += self.I[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]


# -------------------------


def gillespie(
    t_total,
    t_step,
    time,
    comp,
    rates,
    shapes,
):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    stot = comp.S[t_step, :-1].sum()
    itot = comp.I[t_step, :-1].sum()
    etot_rec = comp.A[t_step, :-1, 0].sum()
    etot_inf = comp.A[t_step, :-1, 1].sum()
    etot = etot_inf + etot_rec - comp.A[t_step, 0, 0]

    lambda_sum = (
        rates["alpha"] * etot_inf
        + rates["delta_a"] * etot_rec
        + rates["delta"] * itot
        + (rates["beta_a"] * etot + rates["beta"] * itot) * stot
    )

    probs = {}
    probs["heal_a"] = rates["delta_a"] * comp.A[t_step, :-1, 0] / lambda_sum
    probs["heal_i"] = rates["delta"] * comp.I[t_step, :-1] / lambda_sum
    probs["asymptomatic"] = rates["alpha"] * comp.A[t_step, :-1, 1] / lambda_sum
    probs["infect"] = (
        (rates["beta_a"] * etot + rates["beta"] * itot)
        * comp.S[t_step, :-1]
        / lambda_sum
    )

    t_step += 1
    time += utils.time_dist(lambda_sum)
    if time > t_total:
        return t_step, True  # rare,  but sometimes long times may appear
    comp.T[t_step] = time

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
    prob_heal_a_tot = probs["heal_a"].sum()
    prob_heal_i_tot = probs["heal_i"].sum()
    prob_asymptomatic_tot = probs["asymptomatic"].sum()

    # A(k)-> A(k+1)/R
    if random < prob_heal_a_tot:
        for k in range(shapes["k_asym"]):
            if random < probs["heal_a"][: k + 1].sum():
                comp.recover_adv_a(t_step, k)
                return

    # I(k)-> I(k+1)/R
    if random < (prob_heal_a_tot + prob_heal_i_tot):
        random -= prob_heal_a_tot
        for k in range(shapes["k_rec"]):
            if random < probs["heal_i"][: k + 1].sum():
                comp.recover_adv_i(t_step, k)
                return

    # A(k)-> A(k+1)/I(0)
    if random < (prob_heal_a_tot + prob_heal_i_tot + prob_asymptomatic_tot):
        random -= prob_heal_a_tot + prob_heal_i_tot
        for k in range(shapes["k_asym"]):
            if random < probs["asymptomatic"][: k + 1].sum():
                comp.infect_adv_a(t_step, k)
                return

    # S(k)-> S(k+1)/A(0)
    random -= prob_heal_a_tot + prob_heal_i_tot + prob_asymptomatic_tot
    for k in range(shapes["k_inf"]):
        if random < probs["infect"][: k + 1].sum():
            comp.asymptomatic_adv_s(t_step, k)
            return


# -------------------------


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc(ex)
