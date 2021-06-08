"""
Stochastic mean-field SIR model.
Uses the Gillespie algorithm and Erlang distribution transition times

Pol Pastells, 2020-2021

Equations of the deterministic system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                      delta * I(t)
"""

import random
from collections import namedtuple

import numpy as np
from optilog.autocfg import ac, Int, Real

from ..utils import utils, config

Result = namedtuple("Result", "infected day_max")


def check_successful_simulation(result: Result, time_total: int):
    return not result.infected[time_total - 1] == 0


def get_cost(time_series: np.ndarray, infected, t_total, day_max, n_seeds, metric):
    var_m = utils.mean_alive(infected, t_total, day_max, n_seeds)
    return utils.cost_func(time_series[:, 0], var_m, metric)


@ac
def sir_erlang(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    n_t_steps: int,
    metric: str,
    n: Int(70000, 90000) = 70000,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
    delta: Real(0.1, 1.0) = 0.2,
    beta: Real(0.1, 1.0) = 0.5,
    k_rec: Int(1, 5) = 1,
    k_inf: Int(1, 5) = 1,
):
    # Create shapes and rates dictionaries
    # Normalize beta for the number of individuals
    # Scale beta and delta with the Erlang shapes
    shapes = {"k_inf": k_inf, "k_rec": k_rec}
    rates = {"beta": beta / n * k_inf, "delta": delta * k_rec}

    mc_step = 0
    day_max = 0
    current_seed = seed - 1  # we increase the seed at the start of the loop

    results = list()

    while mc_step < n_seeds:
        current_seed += 1
        result = gillespie_simulation(
            current_seed,
            n,
            n_t_steps,
            initial_infected,
            initial_recovered,
            t_total,
            rates,
            shapes,
            day_max,
        )
        day_max = result.day_max

        if check_successful_simulation(result, t_total):
            mc_step += 1
            results.append(result)

    # results per day and seed
    infected = np.zeros([n_seeds, t_total], dtype=int)

    for mc_step, result in enumerate(results):
        infected[mc_step] = result.infected

    cost = get_cost(time_series, infected, t_total, day_max, n_seeds, metric)
    print(f"GGA SUCCESS {cost}")
    return cost


def gillespie_simulation(
    seed: int,
    n: int,
    n_t_steps: int,
    initial_infected: int,
    initial_recovered: int,
    t_total: int,
    rates: dict,
    shapes: dict,
    day_max: int,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    # initialization
    comp = Compartments(n, n_t_steps, initial_infected, initial_recovered, shapes)

    infected = np.zeros(t_total, dtype=int)
    infected[0] = initial_infected

    t_step, time = 0, 0

    while comp.I[t_step, :-1].sum() > 0 and time < t_total:
        t_step, time = gillespie(t_step, time, comp, rates=rates, shapes=shapes)

    day_max = utils.day_data(
        comp.T[:t_step], comp.I[:t_step, :-1].sum(axis=1), infected, day_max
    )

    return Result(infected, day_max)


class Compartments:
    """Compartments for the SIR Erlang model"""

    def __init__(self, n, n_t_steps, initial_infected, initial_recovered, shapes):
        """Initialization
        S and I are vectors, with one dimension more than the
        This extra dimension is used to facilitate notation.
        E.g.: both infection and advance in S remove an indiv
        dimension and add one to the k+1 in S. In case where
        the individual is added to the first I compartment."""

        self.S = np.zeros([n_t_steps, shapes["k_inf"] + 1])
        self.I = np.zeros([n_t_steps, shapes["k_rec"] + 1])
        self.R = np.zeros(n_t_steps)
        self.T = np.zeros(n_t_steps)

        # Used for both sir_erlang and sir_erlang sections, where n is a vector
        try:
            self.S[0, :-1] = (n - initial_infected - initial_recovered) / shapes[
                "k_inf"
            ]
        except TypeError:
            self.S[0, :-1] = (n[0] - initial_infected - initial_recovered) / shapes[
                "k_inf"
            ]

        if self.S[0, 0] < 0:
            raise ValueError("S cannot be negative, check initial conditions")

        self.S[0, -1] = self.I[0, :-1] = initial_infected / shapes["k_rec"]
        self.I[0, -1] = self.R[0] = initial_recovered
        self.T[0] = 0
        self.I_cum = np.zeros(n_t_steps, dtype=int)
        self.I_cum[0] = initial_infected

    def infect_adv_s(self, t_step, k):
        """Infect or advance in S
        S(k)-> S(k+1)/I(0)"""
        self.R[t_step] = self.R[t_step - 1]
        self.I[t_step, :-1] = self.I[t_step - 1, :-1]
        self.S[t_step, k] = -1
        self.S[t_step, k + 1] = 1
        self.I[t_step, 0] += self.S[t_step, -1]
        self.S[t_step] += self.S[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1] + self.S[t_step, -1]

    def recover_adv_i(self, t_step, k):
        """Recover or advance in I
        I(k)-> I(k+1)/R"""
        self.S[t_step, :-1] = self.S[t_step - 1, :-1]
        self.I[t_step, k] = -1
        self.I[t_step, k + 1] = 1
        self.R[t_step] = self.R[t_step - 1] + self.I[t_step, -1]
        self.I[t_step] += self.I[t_step - 1]
        self.I_cum[t_step] = self.I_cum[t_step - 1]


# -------------------------


def gillespie(t_step, time, comp, rates, shapes):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    stot = comp.S[t_step, :-1].sum()
    itot = comp.I[t_step, :-1].sum()

    lambda_sum = (rates["delta"] + rates["beta"] * stot) * itot
    probs = {}
    probs["heal"] = rates["delta"] * comp.I[t_step, :-1] / lambda_sum
    probs["infect"] = rates["beta"] * comp.S[t_step, :-1] * itot / lambda_sum

    t_step += 1
    time += utils.time_dist(lambda_sum)
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
    prob_heal_tot = probs["heal"].sum()

    # I(k)-> I(k+1)/R
    if random < prob_heal_tot:
        for k in range(shapes["k_rec"]):
            if random < probs["heal"][: k + 1].sum():
                comp.recover_adv_i(t_step, k)
                return

    # S(k)-> S(k+1)/I(0)
    for k in range(shapes["k_inf"]):
        if random < (prob_heal_tot + probs["infect"][: k + 1].sum()):
            comp.infect_adv_s(t_step, k)
            return


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)
    return t_total, time_series


def main(args):
    t_total, time_series = parameters_init(args)
    sir_erlang(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.n_t_steps,
        args.metric,
        n=args.n,  # due to a bug, naming the configurable parameters is mandatory
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        delta=args.delta,
        beta=args.beta,
        k_rec=args.k_rec,
        k_inf=args.k_inf,
    )
