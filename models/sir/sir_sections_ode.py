"""
Deterministic mean-field SIR model
It allows for different sections with different n, delta and beta

Pol Pastells, 2020-2021

Equations of the system:

dS(t)/dt = - beta/N*I(t)*S(t) \n
dI(t)/dt =   beta/N*I(t)*S(t) - delta * I(t) \n
dR(t)/dt =                      delta * I(t)
"""

from collections import namedtuple
import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
from optilog.autocfg import ac, Int, Real
from ..utils import utils, config


def get_cost(time_series: np.ndarray, infected, t_total, day_max, metric):
    inf_m = np.zeros([t_total, 2], dtype=int)
    inf_m[:, 0] = infected
    return utils.cost_func(time_series[:, 0], inf_m, metric)


@ac
def sir(
    time_series: np.ndarray,
    t_total: int,
    metric: str,
    n_sections: int,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
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
    delta1: Real(0.1, 1.0) = 0.2,
    delta2: Real(0.1, 1.0) = 0.2,
    delta3: Real(0.1, 1.0) = 0.2,
    delta4: Real(0.1, 1.0) = 0.2,
    delta5: Real(0.1, 1.0) = 0.2,
    beta1: Real(0.1, 1.0) = 0.5,
    beta2: Real(0.1, 1.0) = 0.7,
    beta3: Real(0.1, 1.0) = 0.5,
    beta4: Real(0.1, 1.0) = 0.3,
    beta5: Real(0.1, 1.0) = 0.5,
):
    # "reconstruct" arrays dictionary
    section_days = [
        0,
        section_day1,
        section_day2,
        section_day3,
        section_day4,
        section_day5,
    ]

    n_vect = [n1, n2, n3, n4, n5]
    beta_vect = np.array([beta1, beta2, beta3, beta4, beta5])
    delta_vect = np.array([delta1, delta2, delta3, delta4, delta5])

    section = 0
    n, rates, section_day, rates_old, section_day_old, n_old = parameters_section(
        n_vect,
        beta_vect,
        delta_vect,
        section_days,
        section,
    )

    # Sections
    initial_cond = (
        n - initial_infected - initial_recovered,
        initial_infected,
        initial_recovered,
    )

    solution = []
    time = []
    while section < n_sections:
        # solve ODE
        _time = np.linspace(section_day_old - 1, section_day, num=t_total * 1000)
        params = (rates, rates_old, section_day_old, n, n_old)
        _solution = odeint(SIR_ODE, initial_cond, _time, args=tuple(params))
        solution.extend(_solution)
        time.extend(_time)
        section += 1
        if section < n_sections:
            (
                n,
                rates,
                section_day,
                rates_old,
                section_day_old,
                n_old,
            ) = parameters_section(
                n_vect,
                beta_vect,
                delta_vect,
                section_days,
                section,
                rates,
                section_day,
                n,
            )
            initial_cond = np.array(solution[-1])

    # results per day
    solution = np.array(solution)
    day_max, infected = utils.day_data(time, solution[:, 1], t_total)
    cost = get_cost(time_series, infected, t_total, day_max, metric)
    print(f"GGA SUCCESS {cost}")
    plt.plot(time, solution)
    plt.show()
    return cost


def SIR_ODE(x, time, *params, transition_days=config.TRANSITION_DAYS):
    rates, rates_old, section_day_old, n, n_old = params
    rates_eval = utils.section_rates(time, rates, rates_old, section_day_old)
    S, I, R = x
    beta, delta = rates_eval.values()

    dS_dt = -beta * S * I
    dI_dt = (beta * S - delta) * I
    dR_dt = delta * I

    if n_old is not None:
        transition_weight = 0.5 * (
            5.33
            / transition_days
            / np.cosh(
                (time - section_day_old - transition_days / 2) * 5.33 / transition_days
            )
            ** 2
        )
        dS_dt += (n - n_old) * transition_weight

    return dS_dt, dI_dt, dR_dt


def parameters_section(
    n_vect,
    beta_vect,
    delta_vect,
    section_days,
    section,
    rates_old=None,
    section_day_old=1,
    n_old=None,
):
    """
    Section dependent parameters
    """
    n = sum(n_vect[: section + 1])
    rates = {"beta": beta_vect[section] / n, "delta": delta_vect[section]}
    section_day = section_days[section + 1]

    return (n, rates, section_day, rates_old, section_day_old + 1, n_old)


def pad(var, length=5):
    return np.pad(var, (0, length - len(var)))


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)

    n_sections = len(args.section_days)

    if not len(args.beta) == len(args.delta) == n_sections >= len(args.n):
        raise ValueError("All rates and n must have same dimension")

    args.section_days[-1] -= 1e-3  # do not finish at day_max
    args.section_days = pad(args.section_days)
    args.beta = pad(args.beta)
    args.delta = pad(args.delta)
    args.n = pad(args.n)

    return t_total, time_series, n_sections


def main(args):
    t_total, time_series, n_sections = parameters_init(args)
    sir(
        time_series,
        t_total,
        args.metric,
        n_sections,
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
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
        delta1=args.delta[0],
        delta2=args.delta[1],
        delta3=args.delta[2],
        delta4=args.delta[3],
        delta5=args.delta[4],
        beta1=args.beta[0],
        beta2=args.beta[1],
        beta3=args.beta[2],
        beta4=args.beta[3],
        beta5=args.beta[4],
    )
