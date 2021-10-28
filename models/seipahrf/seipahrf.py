import numpy
import numpy as np
from scipy.integrate import solve_ivp
from optilog.autocfg import ac, Int, Real

from models.utils import utils

"""
Reported best:
python -m models seipahrf --data data/wuhan.csv --beta 2.55 --l 1.56 \
    --beta_p 7.65 --k 0.25 --rho1 0.580 --rho2 0.001 --gamma_a 0.94 \
    --gamma_i 0.27 --gamma_r 0.5 --delta_i 3.5 --delta_p 1 --delta_h 0.3 \
    --n 44000 --day_max 66 --day_min 0
"""


def ode(t, x, *rates):
    (
        beta,
        beta_p,
        l,
        k,
        rho1,
        rho2,
        gamma_a,
        gamma_i,
        gamma_r,
        delta_i,
        delta_p,
        delta_h,
    ) = rates

    S, E, I, P, A, H, R, F = x
    N = sum(x)

    # Scale rates for infection and dead
    beta = beta / N
    beta_p = beta_p / N
    delta_i = delta_i / N
    delta_p = delta_p / N
    delta_h = delta_h / N

    dS_dt = -beta * I * S - l * beta * H * S - beta_p * P * S
    dE_dt = beta * I * S + l * beta * H * S + beta_p * P * S - k * E
    dI_dt = k * rho1 * E - (gamma_a + gamma_i) * I - delta_i * I
    dP_dt = k * rho2 * E - (gamma_a + gamma_i) * P - delta_p * P
    dA_dt = k * (1 - rho1 - rho2) * E
    dH_dt = gamma_a * (I + P) - gamma_r * H - delta_h * H
    dR_dt = gamma_i * (I + P) + gamma_r * H
    dF_dt = delta_i * I + delta_p * P + delta_h * H

    return dS_dt, dE_dt, dI_dt, dP_dt, dA_dt, dH_dt, dR_dt, dF_dt


def get_cost(time_series: np.ndarray, infected, t_total, day_max, metric):
    inf_m = np.zeros([t_total, 2], dtype=int)
    inf_m[:, 0] = infected
    return utils.cost_func(time_series[:, 0], inf_m, metric)


@ac
def seipahrf(
    time_series: numpy.ndarray,
    t_total: int,
    metric: str,
    n,
    initial_exposed: Int(0, 20) = 0,
    initial_infected: Int(0, 20) = 1,
    initial_superspreader: Int(0, 20) = 5,
    initial_asymptomatic=0,
    initial_hospitalized=0,
    initial_recovered=0,
    initial_dead=0,
    beta: Real(0.01, 5.0) = 2.5,
    beta_p: Real(0.01, 10.0) = 7.65,
    l: Real(0.01, 3.0) = 1.56,
    k: Real(0.01, 1.0) = 0.25,
    rho1: Real(0.01, 1.0) = 0.580,
    rho2: Real(0.001, 0.5) = 0.001,
    gamma_a: Real(0.01, 1.5) = 0.94,
    gamma_i: Real(0.01, 1.5) = 0.27,
    gamma_r: Real(0.01, 5.0) = 0.5,
    delta_i: Real(1.0, 5.0) = 3.5,
    delta_p: Real(0.5, 1.5) = 1.0,
    delta_h: Real(0.01, 1.0) = 0.3,
):
    # Solve the ODE
    time = numpy.linspace(0, t_total - 0.01, num=t_total * 100)

    initial_s = (
        n
        - initial_exposed
        - initial_infected
        - initial_superspreader
        - initial_asymptomatic
        - initial_hospitalized
        - initial_recovered
        - initial_dead
    )
    initial_conditions = (
        initial_s,
        initial_exposed,
        initial_infected,
        initial_superspreader,
        initial_asymptomatic,
        initial_hospitalized,
        initial_recovered,
        initial_dead,
    )

    args = (
        beta,
        beta_p,
        l,
        k,
        rho1,
        rho2,
        gamma_a,
        gamma_i,
        gamma_r,
        delta_i,
        delta_p,
        delta_h,
    )

    t_span = (time[0], time[-1])
    solution = solve_ivp(
        fun=ode, t_span=t_span, t_eval=time, y0=initial_conditions, args=args
    )

    prediction = solution.y
    confirmed_seq = prediction[2] + prediction[3] + prediction[5]

    # confirmed is used for the cost calculation
    day_max, confirmed = utils.day_data(time, confirmed_seq, t_total)

    _, susceptible = utils.day_data(time, prediction[0], t_total)
    _, exposed = utils.day_data(time, prediction[1], t_total)
    _, infected = utils.day_data(time, prediction[2], t_total)
    _, superspreader = utils.day_data(time, prediction[3], t_total)
    _, asymptomatic = utils.day_data(time, prediction[4], t_total)
    _, hospitalized = utils.day_data(time, prediction[5], t_total)
    _, recovered = utils.day_data(time, prediction[6], t_total)
    _, fatal = utils.day_data(time, prediction[7], t_total)

    # Results per day
    evolution_df = utils.evolution_to_dataframe(
        [susceptible, exposed, infected, superspreader, asymptomatic,
         hospitalized, recovered, fatal],
        ["susceptible", "exposed", "infected", "superspreader", "asymptomatic",
         "hospitalized", "recovered", "fatal"],
    )

    cost = get_cost(time_series, confirmed, t_total, day_max, metric)

    print(f"GGA SUCCESS {cost}")
    return cost, evolution_df


def parameters_init(args):
    t_total, time_series = utils.parameters_init_common(args)
    return t_total, time_series


def main(args):
    t_total, time_series = parameters_init(args)
    return seipahrf(
        time_series,
        t_total,
        args.metric,
        n=args.n,
        initial_exposed=args.initial_exposed,
        initial_infected=args.initial_infected,
        initial_superspreader=args.initial_superspreader,
        initial_asymptomatic=args.initial_asymptomatic,
        initial_hospitalized=args.initial_hospitalized,
        initial_recovered=args.initial_recovered,
        initial_dead=args.initial_dead,
        beta=args.beta,
        beta_p=args.beta_p,
        l=args.l,
        k=args.k,
        rho1=args.rho1,
        rho2=args.rho2,
        gamma_a=args.gamma_a,
        gamma_i=args.gamma_i,
        gamma_r=args.gamma_r,
        delta_i=args.delta_i,
        delta_p=args.delta_p,
        delta_h=args.delta_h,
    )
