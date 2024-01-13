"""
Adaptation of the SIDARTHE model
If n_old = 0:
    no previous section is assumed and the _old parameters are ignored
otherwise:
    n is the **increase** in population
"""
import sys
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from optilog.tuning import ac, Int, Real
from ..utils import utils, config


@ac
def sidarthe(
    data,
    day_min,
    day_max,
    initial_infected: Int(1, 1000) = 10,
    initial_diagnosed: Int(1, 1000) = 10,
    initial_ailing: Int(1, 1000) = 10,
    initial_recovered: Int(1, 1000) = 10,
    n: Int(1000, 100000000) = int(60e6),
    n_old: Int(0, 100000000) = int(60e6),
    alfa: Real(0.1, 1.0) = 0.6,
    beta: Real(0.001, 1.0) = 0.01,
    gamma: Real(0.05, 1.0) = 0.5,
    delta: Real(0.001, 1.0) = 0.01,
    epsilon: Real(0.1, 0.5) = 0.15,
    theta: Real(0.1, 0.5) = 0.3,
    zeta: Real(0.01, 0.5) = 0.1,
    eta: Real(0.01, 0.5) = 0.1,
    mu: Real(0.001, 0.1) = 0.04,
    nu: Real(0.001, 0.1) = 0.03,
    tau: Real(0.001, 0.1) = 0.03,
    lambd: Real(0.01, 0.5) = 0.05,
    rho: Real(0.01, 0.5) = 0.05,
    kappa: Real(0.001, 0.5) = 0.005,
    xi: Real(0.001, 0.5) = 0.005,
    sigma: Real(0.001, 0.5) = 0.005,
    alfa_old: Real(0.1, 1.0) = 0.6,
    beta_old: Real(0.001, 0.05) = 0.01,
    gamma_old: Real(0.05, 1.0) = 0.5,
    delta_old: Real(0.001, 0.05) = 0.01,
    epsilon_old: Real(0.1, 0.5) = 0.15,
    theta_old: Real(0.1, 0.5) = 0.3,
    zeta_old: Real(0.01, 0.5) = 0.1,
    eta_old: Real(0.01, 0.5) = 0.1,
    mu_old: Real(0.001, 0.1) = 0.04,
    nu_old: Real(0.001, 0.1) = 0.03,
    tau_old: Real(0.001, 0.1) = 0.03,
    lambda_old: Real(0.01, 0.5) = 0.05,
    rho_old: Real(0.01, 0.5) = 0.05,
    kappa_old: Real(0.001, 0.5) = 0.005,
    xi_old: Real(0.001, 0.5) = 0.005,
    sigma_old: Real(0.001, 0.5) = 0.005,
):
    # fmt: off
    if n_old == 0:
        n_init = n
    else:
        n_init = n_old

    initial_cond = (
        n_init - initial_infected - initial_diagnosed - initial_ailing - initial_recovered,
        initial_infected,
        initial_diagnosed,
        initial_ailing,
        initial_recovered,
        0, 0, 0, 0,
        initial_infected + initial_diagnosed + initial_ailing + initial_recovered,
    )
    assert initial_cond[0] > 0, f"Insuficient individuals ({n_init}) for this initial settings"

    rates = [ alfa, beta, gamma, delta, epsilon, theta, zeta,
              eta, mu, nu, tau, lambd, rho, kappa, xi, sigma]

    if n_old == 0:
        rates_old = rates
    else:
        rates_old = [
            alfa_old, beta_old, gamma_old, delta_old, epsilon_old,
            theta_old, zeta_old, eta_old, mu_old, nu_old, tau_old,
            lambda_old, rho_old, kappa_old, xi_old, sigma_old,
        ]
    # fmt: on

    params = (rates, rates_old, n, n_old, day_min)
    time = np.linspace(day_min, day_max, num=(day_max - day_min + 1) * 1000)

    solution = odeint(SIDARTHE_ODE, initial_cond, time, args=tuple(params))

    import matplotlib.pyplot as plt

    plt.plot(time, solution)
    plt.show()

    time_series = get_data(data, day_min, day_max)
    # TODO review if it is needed
    # infected = np.zeros(day_max - day_min + 1, dtype=int)
    # day_max = utils.day_data(time, solution[:, 1], infected, day_max)
    cost = get_cost(time_series, solution)

    # last values as input for next section
    print(np.array(solution[-1, 1:5], dtype=int))

    sys.stdout.write(f"GGA SUCCESS {cost}\n")


def get_data(data, day_min, day_max):
    df = pd.read_csv(data)
    df = df.loc[day_min:day_max]
    guariti = df.dimessi_guariti.values
    isol = df.isolamento_domiciliare.values
    ricov = df.ricoverati_con_sintomi.values
    terapia = df.terapia_intensiva.values
    data = np.array([guariti, isol, ricov, terapia])
    return data


def get_cost(time_series, solution):
    cost = 0
    cost += np.linalg.norm(time_series[0] - solution[:, 8::1000])
    cost += np.linalg.norm(time_series[1] - solution[:, 2::1000])
    cost += np.linalg.norm(time_series[2] - solution[:, 4::1000])
    cost += np.linalg.norm(time_series[3] - solution[:, 5::1000])
    return cost / 1e6


def SIDARTHE_ODE(x, time, *params, transition_days=config.TRANSITION_DAYS):
    rates, rates_old, n, n_old, day_min = params

    # If this is not the first section
    if n_old != 0:
        rates_eval = []
        weight = utils.transition_weight(time - day_min, transition_days)
        for i in range(len(rates)):
            rates_eval.append(rates_old[i] + (rates[i] - rates_old[i]) * weight)
    else:
        rates_eval = rates
        n_eval = n

    dS = 0
    if n_old != 0:
        # If we increment the population
        if n != 0:
            transition_weight = 0.5 * (
                5.33
                / transition_days
                / np.cosh(
                    (time - day_min - transition_days / 2) * 5.33 / transition_days
                )
                ** 2
            )
            n_increment = n * transition_weight
            n_eval = n_old + n_increment
            dS += n_increment
        else:
            n_eval = n_old

    # fmt: off
    (
        alfa, beta, gamma, delta, epsilon,
        theta, zeta, eta, mu, nu, tau, lambd, rho, kappa, xi, sigma,
    ) = rates_eval
    # fmt: on
    alfa = alfa / n_eval
    beta = beta / n_eval
    gamma = gamma / n_eval
    delta = delta / n_eval

    S, I, D, A, R, T, H, E, H_diag, I_real = x
    dS -= (alfa * I + beta * D + gamma * A + delta * R) * S
    dI = (alfa * I + beta * D + gamma * A + delta * R) * S - (
        epsilon + zeta + lambd
    ) * I
    dD = epsilon * I - (eta + rho) * D
    dA = zeta * I - (theta + mu + kappa) * A
    dR = eta * D + theta * A - (nu + xi) * R
    dT = mu * A + nu * R - (sigma + tau) * T
    dH = lambd * I + rho * D + kappa * A + xi * R + sigma * T
    dE = tau * T
    dH_diag = rho * D + xi * R + sigma * T
    dI_real = (alfa * I + beta * D + gamma * A + delta * R) * S
    return dS, dI, dD, dA, dR, dT, dH, dE, dH_diag, dI_real


def main(args):
    sidarthe(
        args.data,
        args.day_min,
        args.day_max,
        initial_infected=args.initial_infected,
        initial_diagnosed=args.initial_diagnosed,
        initial_ailing=args.initial_ailing,
        initial_recovered=args.initial_recovered,
        n=args.n,
        n_old=args.n_old,
        alfa=args.alfa,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        epsilon=args.epsilon,
        theta=args.theta,
        zeta=args.zeta,
        eta=args.eta,
        mu=args.mu,
        nu=args.nu,
        tau=args.tau,
        lambd=args.lambd,
        rho=args.rho,
        kappa=args.kappa,
        xi=args.xi,
        sigma=args.sigma,
        alfa_old=args.alfa_old,
        beta_old=args.beta_old,
        gamma_old=args.gamma_old,
        delta_old=args.delta_old,
        epsilon_old=args.epsilon_old,
        theta_old=args.theta_old,
        zeta_old=args.zeta_old,
        eta_old=args.eta_old,
        mu_old=args.mu_old,
        nu_old=args.nu_old,
        tau_old=args.tau_old,
        lambda_old=args.lambda_old,
        rho_old=args.rho_old,
        kappa_old=args.kappa_old,
        xi_old=args.xi_old,
        sigma_old=args.sigma_old,
    )
