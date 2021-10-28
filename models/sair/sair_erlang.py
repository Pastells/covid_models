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
import functools
import random

import numpy as np
from optilog.autocfg import ac, Int, Real

from .sair import simulate_evolution, Result, get_cost
from ..utils import utils


@ac
def sair_erlang(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    n_t_steps: int,
    metric: str,
    n: Int(70000, 90000) = 70000,
    initial_infected: Int(410, 440) = 410,
    initial_recovered: Int(4, 6) = 4,
    initial_asymptomatic: Int(0, 100) = 0,
    alpha: Real(0.05, 2.0) = 0.05,
    delta_a: Real(0.05, 1.0) = 0.05,
    delta: Real(0.03, 0.06) = 0.03,
    beta_a: Real(0.05, 1.0) = 0.05,
    beta: Real(0.3, 0.4) = 0.3,
    k_rec: Int(1, 5) = 1,
    k_inf: Int(1, 5) = 1,
    k_asym: Int(1, 5) = 1,
):
    rates = {
        "beta_a": beta_a / n * k_inf,
        "beta": beta / n * k_inf,
        "delta_a": delta_a * k_rec,
        "delta": delta * k_rec,
        "alpha": alpha * k_asym,
    }
    shapes = {"k_inf": k_inf, "k_rec": k_rec, "k_asym": k_asym}

    func = functools.partial(
        gillespie_simulation,
        n=n,
        n_t_steps=n_t_steps,
        initial_asymptomatic=initial_asymptomatic,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
        t_total=t_total,
        rates=rates,
        shapes=shapes,
    )
    evolution_df, day_max = simulate_evolution(func, n_seeds, seed, t_total)

    cost = get_cost(time_series, evolution_df.infected, t_total, day_max, metric)
    print(f"GGA SUCCESS {cost}")

    return cost, evolution_df


def gillespie_simulation(
    seed: int,
    n: int,
    n_t_steps: int,
    initial_asymptomatic: int,
    initial_infected: int,
    initial_recovered: int,
    t_total: int,
    rates: dict,
    shapes: dict,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)
    comp = Compartments(
        n,
        n_t_steps,
        initial_asymptomatic,
        initial_infected,
        initial_recovered,
        shapes,
    )

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

    event_timestamps = comp.T[:t_step]

    # Note that S, A, and I compartments have an extra
    # dimension not relevant for the results, used only
    # for an easier notation
    _, susceptible = utils.day_data(
        event_timestamps, comp.S[:t_step, :-1].sum(axis=1), t_total
    )
    # TODO asymptomatic has an extra dimension
    _, asymptomatic = (
        None,
        None,
    )  # utils.day_data(event_timestamps, comp.A[:t_step, :-1].sum(axis=1), t_total)
    day_max, infected = utils.day_data(
        event_timestamps, comp.I[:t_step, :-1].sum(axis=1), t_total
    )
    _, recovered = utils.day_data(event_timestamps, comp.R[:t_step], t_total)

    return Result(susceptible, asymptomatic, infected, recovered, day_max)


class Compartments:
    """Compartments for the SAIR Erlang model"""

    def __init__(
        self,
        n,
        n_t_steps,
        initial_asymptomatic,
        initial_infected,
        initial_recovered,
        shapes,
    ):
        """Initialization
        S, A and I are vectors, with one dimension more than the Erlang shape.
        This extra dimension is used to facilitate notation.
        E.g.: both infection and advance in S remove an individual from the k
        dimension and add one to the k+1 in S. In case where k+1 is the last entry,
        the individual is added to the first I compartment."""
        self.S = np.zeros([n_t_steps, shapes["k_inf"] + 1])
        self.A = np.zeros([n_t_steps, shapes["k_asym"] + 1, 2])
        self.I = np.zeros([n_t_steps, shapes["k_rec"] + 1])
        self.R = np.zeros(n_t_steps)
        self.T = np.zeros(n_t_steps)

        # Used for both sair_erlang and sair_erlang sections, where n is a vector
        try:
            self.S[0, :-1] = (n - initial_infected - initial_recovered) / shapes[
                "k_inf"
            ]
        except TypeError:
            self.S[0, :-1] = (n[0] - initial_infected - initial_recovered) / shapes[
                "k_inf"
            ]

        self.S[0, -1] = self.A[0, :-1] = initial_asymptomatic / shapes["k_asym"]
        self.A[0, -1] = self.I[0, :-1] = initial_infected / shapes["k_rec"]
        self.I[0, -1] = self.R[0] = initial_recovered
        self.T[0] = 0
        self.I_cum = np.zeros(n_t_steps)
        self.I_cum[0] = initial_infected

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


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)
    return t_total, time_series


def main(args):
    t_total, time_series = parameters_init(args)
    return sair_erlang(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.n_t_steps,
        args.metric,
        n=args.n,
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        initial_asymptomatic=args.initial_asymptomatic,
        alpha=args.alpha,
        delta_a=args.delta_a,
        delta=args.delta,
        beta_a=args.beta_a,
        beta=args.beta,
        k_rec=args.k_rec,
        k_inf=args.k_inf,
        k_asym=args.k_asym,
    )
