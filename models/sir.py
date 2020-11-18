"""
Stochastic mean-field sir model using the Gillespie algorithm
Pol Pastells, october 2020

equations of the deterministic system
s[t] = S[t-1] - beta*i[t-1]*s[t-1]
i[t] = I[t-1] + beta*i[t-1]*s[t-1] - delta * I[t-1]
r[t] = R[t-1] + delta * I[t-1]
"""


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def main():
    args = parsing()
    (
        I_0,
        R_0,
        t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        beta,
        delta,
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
    for seed in range(mc_seed0, mc_seed0 + mc_nseed):
        np.random.seed(seed)

        # -------------------------
        # initialization
        S, I, R = np.zeros(t_steps), np.zeros(t_steps), np.zeros(t_steps)
        I[0] = I_0
        R[0] = R_0
        S[0] = n - I_0 - R_0
        I_day[mc_step, 0] = I_0
        # T = np.zeros(t_steps)
        # T[0]=0
        # S_day[mc_step,0]=s[0]
        t, time, day = 0, 0, 1

        # Time loop
        while I[t] > 0.1 and day < t_total - 1:
            day, day_max = utils.day_data(
                time, t_total, day, day_max, I[t], I_day[mc_step]
            )
            t, time = gillespie(t, time, S, I, R, beta, delta)
        # -------------------------
        day, day_max = utils.day_data(
            time, t_total, day, day_max, I[t], I_day[mc_step], True
        )

        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save:
        utils.saving(args, I_m, I_std, day_max, "sir")

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="stochastic mean-fiel sir model using the Gillespie algorithm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n",
        type=int,
        default=int(1e4),
        help="Fixed number of (effecitve) people [1000,1000000]",
    )
    parser.add_argument(
        "--i_0",
        type=int,
        default=20,
        help="initial number of infected individuals [1,n]",
    )
    parser.add_argument(
        "--r_0", type=int, default=0, help="initial number of inmune individuals [0,n]"
    )
    parser.add_argument(
        "--delta", type=float, default=0.2, help="ratio of recovery [1e-2,1]"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="ratio of infection [1e-2,1]"
    )

    parser.add_argument(
        "--seed", type=int, default=1, help="Seed for the automatic configuration"
    )
    parser.add_argument(
        "--data", type=str, default="../data/italy_i.csv", help="File with time series"
    )
    parser.add_argument(
        "--day_min", type=int, default=33, help="First day to consider on data series"
    )
    parser.add_argument(
        "--day_max", type=int, default=58, help="Last day to consider on data series"
    )

    parser.add_argument(
        "--mc_nseed",
        type=int,
        default=int(1e3),
        help="Number of MC realizations, not really a parameter",
    )
    parser.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="Initial MC seed, not really a parameter",
    )
    parser.add_argument("--plot", action="store_true", help="specify for plots")
    parser.add_argument("--save", action="store_true", help="specify for outputfile")

    args = parser.parse_args()
    # print(args)
    return args


# -------------------------
# Parameters


def parameters_init(args):
    """initial parameters from argparse"""
    from numpy import genfromtxt

    I_0 = args.i_0
    R_0 = args.r_0
    t_steps = int(1e7)  # max simulation steps
    t_total = args.day_max - args.day_min  # max simulated days
    mc_nseed = args.mc_nseed  # MC realizations
    mc_seed0 = args.mc_seed0
    plot = args.plot
    save = args.save
    infected_time_series = genfromtxt(args.data, delimiter=",")[
        args.day_min : args.day_max
    ]
    # print(infected_time_series)
    n = args.n
    beta = args.beta / n
    delta = args.delta
    return (
        I_0,
        R_0,
        t_steps,
        t_total,
        mc_nseed,
        mc_seed0,
        plot,
        save,
        infected_time_series,
        n,
        beta,
        delta,
    )


# -------------------------


def gillespie(t, time, S, I, R, beta, delta):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """

    lambda_sum = (delta + utils.beta_func(beta, t) * S[t]) * I[t]
    prob_heal = delta * I[t] / lambda_sum

    t += 1
    time += utils.time_dist(lambda_sum)
    # T[t] = time

    gillespie_step(t, S, I, R, prob_heal)
    return t, time


# -------------------------


def gillespie_step(t, S, I, R, prob_heal):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    """
    random = np.random.random()

    if random < prob_heal:
        # heal
        S[t] = S[t - 1]
        I[t] = I[t - 1] - 1
        R[t] = R[t - 1] + 1
    else:
        # infect
        S[t] = S[t - 1] - 1
        I[t] = I[t - 1] + 1
        R[t] = R[t - 1]


# ~~~~~~~~~~~~~~~~~~~


if __name__ == "__main__":
    import numpy as np
    import utils
    import sys
    import traceback

    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        sys.stdout.write(f"{repr(ex)}\n")
        traceback.print_exc(ex)
