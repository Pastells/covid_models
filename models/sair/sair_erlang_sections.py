"""
Stochastic mean-field SAIR model.
Uses the Gillespie algorithm and Erlang distribution transition times.
It allows for different sections with different n, delta and beta

Pol Pastells, 2020

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

from .sair import simulate_evolution, get_cost, Result
from .sair_erlang import Compartments, gillespie_step
from ..utils import utils


@ac
def sair_erlang_sections(
    time_series: np.ndarray,
    seed: int,
    n_seeds: int,
    t_total: int,
    n_t_steps: int,
    metric: str,
    n_sections: int,
    initial_infected: Int(410, 440) = 410,
    initial_recovered: Int(4, 6) = 4,
    initial_asymptomatic: Int(0, 100) = 0,
    k_rec: Int(1, 5) = 1,
    k_inf: Int(1, 5) = 1,
    k_asym: Int(1, 5) = 1,
    section_day1: Int(0, 1000) = 10,
    section_day2: Int(0, 1000) = 0,
    section_day3: Int(0, 1000) = 0,
    section_day4: Int(0, 1000) = 0,
    section_day5: Int(0, 1000) = 0,
    n1: Int(70000, 90000) = 70000,
    n2: Int(0, 90000) = 7000,
    n3: Int(0, 90000) = 0,
    n4: Int(0, 90000) = 0,
    n5: Int(0, 90000) = 0,
    alpha1: Real(0.1, 1.0) = 0.5,
    alpha2: Real(0.1, 1.0) = 0.7,
    alpha3: Real(0.1, 1.0) = 0.5,
    alpha4: Real(0.1, 1.0) = 0.3,
    alpha5: Real(0.1, 1.0) = 0.5,
    delta_a1: Real(0.1, 1.0) = 0.2,
    delta_a2: Real(0.1, 1.0) = 0.2,
    delta_a3: Real(0.1, 1.0) = 0.2,
    delta_a4: Real(0.1, 1.0) = 0.2,
    delta_a5: Real(0.1, 1.0) = 0.2,
    delta1: Real(0.1, 1.0) = 0.2,
    delta2: Real(0.1, 1.0) = 0.2,
    delta3: Real(0.1, 1.0) = 0.2,
    delta4: Real(0.1, 1.0) = 0.2,
    delta5: Real(0.1, 1.0) = 0.2,
    beta_a1: Real(0.1, 1.0) = 0.5,
    beta_a2: Real(0.1, 1.0) = 0.7,
    beta_a3: Real(0.1, 1.0) = 0.5,
    beta_a4: Real(0.1, 1.0) = 0.3,
    beta_a5: Real(0.1, 1.0) = 0.5,
    beta1: Real(0.1, 1.0) = 0.5,
    beta2: Real(0.1, 1.0) = 0.7,
    beta3: Real(0.1, 1.0) = 0.5,
    beta4: Real(0.1, 1.0) = 0.3,
    beta5: Real(0.1, 1.0) = 0.5,
):
    # "reconstruct" arrays and shapes dictionary
    section_days = [
        0,
        section_day1,
        section_day2,
        section_day3,
        section_day4,
        section_day5,
    ]

    func = functools.partial(
        gillespie_simulation,
        n_vect=[n1, n2, n3, n4, n5],
        alpha_vect=np.array([alpha1, alpha2, alpha3, alpha4, alpha5]) * k_asym,
        delta_a_vect=np.array([delta_a1, delta_a2, delta_a3, delta_a4, delta_a5])
        * k_rec,
        delta_vect=np.array([delta1, delta2, delta3, delta4, delta5]) * k_rec,
        beta_a_vect=np.array([beta_a1, beta_a2, beta_a3, beta_a4, beta_a5]) * k_inf,
        beta_vect=np.array([beta1, beta2, beta3, beta4, beta5]) * k_inf,
        section_days=section_days,
        n_sections=n_sections,
        n_t_steps=n_t_steps,
        initial_infected=initial_infected,
        initial_recovered=initial_recovered,
        initial_asymptomatic=initial_asymptomatic,
        t_total=t_total,
        shapes={"k_inf": k_inf, "k_rec": k_rec, "k_asym": k_asym},
    )

    evolution_df, day_max = simulate_evolution(func, n_seeds, seed, t_total)

    cost = get_cost(time_series, evolution_df.infected, t_total, day_max, metric)
    print(f"GGA SUCCESS {cost}")
    return cost, evolution_df


def gillespie_simulation(
    seed: int,
    n_vect: list,
    alpha_vect: list,
    delta_a_vect: list,
    delta_vect: list,
    beta_a_vect: list,
    beta_vect: list,
    section_days: list,
    n_sections: int,
    n_t_steps: int,
    initial_infected: int,
    initial_recovered: int,
    initial_asymptomatic: int,
    t_total: int,
    shapes: dict,
) -> Result:
    random.seed(seed)
    np.random.seed(seed)

    section = 0
    (n, rates, section_day, rates_old, section_day_old, n_ind,) = parameters_section(
        n_vect,
        alpha_vect,
        delta_a_vect,
        delta_vect,
        beta_a_vect,
        beta_vect,
        section_days,
        section,
    )

    comp = Compartments(
        n, n_t_steps, initial_asymptomatic, initial_infected, initial_recovered, shapes
    )

    t_step, time = 0, 0
    index_n = 1  # just to avoid pylint complaining

    # Sections
    while section < n_sections:
        # Time loop
        while comp.I[t_step, :-1].sum() > 0 and time < section_day:
            # add individuals
            if (n_ind is not None) and (time // n_ind[index_n, 1] == 1):
                comp.S[t_step] += n_ind[index_n, 0] / shapes["k_inf"]
                if index_n < (len(n_ind) - 1):
                    index_n += 1
                else:
                    n_ind = None

            t_step, time = gillespie(
                t_total,
                t_step,
                time,
                section_day_old,
                comp,
                rates,
                rates_old,
                shapes,
            )

        section += 1
        if section < n_sections:
            (
                n,
                rates,
                section_day,
                rates_old,
                section_day_old,
                n_ind,
            ) = parameters_section(
                n_vect,
                alpha_vect,
                delta_a_vect,
                delta_vect,
                beta_a_vect,
                beta_vect,
                section_days,
                section,
                rates_old=rates,
                section_day_old=section_day,
                n_old=n,
            )
            if n_ind is not None:
                comp.S[t_step, :-1] += n_ind[0, 0] / shapes["k_inf"]
            index_n = 1

    day_max, infected = utils.day_data(
        comp.T[:t_step], comp.I[:t_step, :-1].sum(axis=1), t_total
    )
    # TODO fill missing compartments
    return Result(None, None, infected, None, day_max)


def parameters_section(
    n_vect,
    alpha_vect,
    delta_a_vect,
    delta_vect,
    beta_a_vect,
    beta_vect,
    section_days,
    section,
    rates_old=None,
    section_day_old=0,
    n_old=None,
):
    """
    Section dependent parameters from argparse
    """
    n = sum(n_vect[: section + 1])
    n_ind = utils.n_individuals(n, n_old, section_day_old)
    rates = {
        "beta_a": beta_a_vect[section] / n,
        "beta": beta_vect[section] / n,
        "delta_a": delta_a_vect[section],
        "delta": delta_vect[section],
        "alpha": alpha_vect[section],
    }
    section_day = section_days[section + 1]

    # should return section_day_old , given that the input will be section_day.
    # but I want to add 2 to it for the tanh in rates_sir/n_individuals
    return (n, rates, section_day, rates_old, section_day_old + 1, n_ind)


def gillespie(
    t_total,
    t_step,
    time,
    section_day_old,
    comp,
    rates,
    rates_old,
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

    rates_eval = utils.section_rates(time, rates, rates_old, section_day_old)
    lambda_sum = (
        rates_eval["alpha"] * etot_inf
        + rates_eval["delta_a"] * etot_rec
        + rates_eval["delta"] * itot
        + (rates_eval["beta_a"] * etot + rates_eval["beta"] * itot) * stot
    )

    probs = {}
    probs["heal_a"] = rates_eval["delta_a"] * comp.A[t_step, :-1, 0] / lambda_sum
    probs["heal_i"] = rates_eval["delta"] * comp.I[t_step, :-1] / lambda_sum
    probs["asymptomatic"] = rates_eval["alpha"] * comp.A[t_step, :-1, 1] / lambda_sum
    probs["infect"] = (
        (rates_eval["beta_a"] * etot + rates_eval["beta"] * itot)
        * comp.S[t_step, :-1]
        / lambda_sum
    )

    t_step += 1
    time += utils.time_dist(lambda_sum)
    comp.T[t_step] = time
    if time > t_total:
        return t_step, True  # rare,  but sometimes long times may appear

    gillespie_step(t_step, comp, probs, shapes)
    return t_step, time


def pad(var, length=5):
    return np.pad(var, (0, length - len(var)))


def parameters_init(args):
    """Initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days)

    if (
        not len(args.beta)
        == len(args.delta)
        == len(args.beta_a)
        == len(args.delta_a)
        == len(args.alpha)
        == n_sections
        >= len(args.n)
    ):
        raise ValueError("All rates and n must have same dimension")

    args.section_days = pad(args.section_days)
    args.alpha = pad(args.alpha)
    args.beta_a = pad(args.beta_a)
    args.beta = pad(args.beta)
    args.delta_a = pad(args.delta_a)
    args.delta = pad(args.delta)
    args.n = pad(args.n)
    return t_total, time_series, n_sections


def main(args):
    t_total, time_series, n_sections = parameters_init(args)
    return sair_erlang_sections(
        time_series,
        args.seed,
        args.mc_nseed,
        t_total,
        args.n_t_steps,
        args.metric,
        n_sections,
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        initial_asymptomatic=args.initial_asymptomatic,
        k_rec=args.k_rec,
        k_inf=args.k_inf,
        k_asym=args.k_asym,
        section_day1=args.section_days[0],
        section_day2=args.section_days[1],
        section_day3=args.section_days[2],
        section_day4=args.section_days[3],
        section_day5=args.section_days[4],
        n1=args.n[0],
        n2=args.n[1],
        n3=args.n[2],
        n4=args.n[3],
        n5=args.n[4],
        alpha1=args.alpha[0],
        alpha2=args.alpha[1],
        alpha3=args.alpha[2],
        alpha4=args.alpha[3],
        alpha5=args.alpha[4],
        delta_a1=args.delta[0],
        delta_a2=args.delta[1],
        delta_a3=args.delta[2],
        delta_a4=args.delta[3],
        delta_a5=args.delta[4],
        delta1=args.delta[0],
        delta2=args.delta[1],
        delta3=args.delta[2],
        delta4=args.delta[3],
        delta5=args.delta[4],
        beta_a1=args.beta[0],
        beta_a2=args.beta[1],
        beta_a3=args.beta[2],
        beta_a4=args.beta[3],
        beta_a5=args.beta[4],
        beta1=args.beta[0],
        beta2=args.beta[1],
        beta3=args.beta[2],
        beta4=args.beta[3],
        beta5=args.beta[4],
    )
