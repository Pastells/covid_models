"""
Deterministic mean-field SIR model

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
from ..utils import utils


def get_cost(time_series: np.ndarray, infected, t_total, day_max, metric):
    inf_m = np.zeros([t_total, 2], dtype=int)
    inf_m[:, 0] = infected
    return utils.cost_func(time_series[:, 0], inf_m, metric)


@ac
def sir_ode(
    time_series: np.ndarray,
    t_total: int,
    metric: str,
    n: Int(70000, 90000) = 70000,
    initial_infected: Int(1, 1000) = 10,
    initial_recovered: Int(0, 1000) = 4,
    delta: Real(0.1, 1.0) = 0.2,
    beta: Real(0.1, 1.0) = 0.5,
):
    # Normalize beta for the number of individuals
    params = (beta / n, delta)
    day_max = 0

    # solve ODE
    time = np.linspace(0, t_total - 0.01, num=t_total * 100)
    S0 = (n - initial_infected - initial_recovered) / 2
    initial_cond = (
        np.array([S0, S0]),
        initial_infected,
        initial_recovered,
    )
    solution = odeint(SIR_ODE, initial_cond, time, args=tuple(params))
    plt.plot(time, solution)
    plt.show()

    # results per day
    infected = np.zeros(t_total, dtype=int)
    day_max = utils.day_data(time, solution[:, 1], infected, day_max)
    cost = get_cost(time_series, infected, t_total, day_max, metric)
    print(f"GGA SUCCESS {cost}")
    return cost


def SIR_ODE(x, t, *params):
    S, I, R = x
    beta, delta, k_inf, k_rec = params
    dS_dt = -beta * S * I
    dI_dt = (beta * S - delta) * I
    dR_dt = delta * I
    return dS_dt, dI_dt, dR_dt


def parameters_init(args):
    """initial parameters from argparse"""
    t_total, time_series = utils.parameters_init_common(args)
    return t_total, time_series


def main(args):
    t_total, time_series = parameters_init(args)
    sir_ode(
        time_series,
        t_total,
        args.metric,
        n=args.n,
        initial_infected=args.initial_infected,
        initial_recovered=args.initial_recovered,
        delta=args.delta,
        beta=args.beta,
    )
