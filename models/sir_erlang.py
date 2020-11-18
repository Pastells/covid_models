"""
Stochastic mean-field SIR model
Uses the Gillespie algorithm and erlang distribution transition times
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
        k_inf,
        k_rec,
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
        S, I, R = (
            np.zeros([t_steps, k_inf + 1]),
            np.zeros([t_steps, k_rec + 1]),
            np.zeros(t_steps),
        )
        S[0, :-1] = (n - I_0 - R_0) / k_inf
        S[0, -1], I[0, :-1] = I_0 / k_rec, I_0 / k_rec
        I[0, -1], R[0] = R_0, R_0

        # S_day[mc_step,0]=s[0]
        I_day[mc_step, 0] = I_0
        # R_day[mc_step,0]=r_0
        # T = np.zeros(t_steps)
        # T[0]=0
        t, time, day = 0, 0, 1

        # Time loop
        while I[t, :-1].sum() > 0.1 and day < t_total - 1:
            # Add individuals periodically
            # if time//add_n==1:
            # add_n += 30
            # S[t] += float(n)/2
            day, day_max = utils.day_data(
                time, t_total, day, day_max, I[t, :-1].sum(), I_day[mc_step]
            )
            t, time = gillespie(t, time, S, I, R, beta, delta, k_rec, k_inf)
        # -------------------------
        day, day_max = utils.day_data(
            time, day, t_total, day_max, I[t, :-1].sum(), I_day[mc_step], True
        )

        # final value for the rest of time, otherwise it contributes with a zero when averaged
        # S_day[mc_step,day:] = S_day[mc_step,day-1]
        I_day[mc_step, day:] = I_day[mc_step, day - 1]
        # R_day[mc_step,day:] = R_day[mc_step,day-1]

        # plot all trajectories
        # if plot:
        # plt.plot(I_day[mc_step,:])
        # plt.plot(T[:t],i[:t,:-1].sum(1),c='c')
        mc_step += 1
    # =========================

    I_m, I_std = utils.mean_alive(I_day, t_total, day_max, mc_nseed)

    utils.cost_func(infected_time_series, I_m, I_std)

    if save:
        utils.saving(args, I_m, I_std, day_max, "sir_erlang")

    if plot:
        utils.plotting(infected_time_series, I_day, day_max, I_m, I_std)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%


def parsing():
    """input parameters"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stochastic mean-field SIR model using the Gillespie algorithm and Erlang \
            distribution transition times.",
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
        "--delta", type=float, default=0.2, help="Mean ratio of recovery [1e-2,1]"
    )
    parser.add_argument(
        "--beta", type=float, default=0.5, help="ratio of infection [1e-2,1]"
    )
    parser.add_argument(
        "--k_rec",
        type=int,
        default=1,
        help="k parameter for the recovery time erlang distribution [1,5]",
    )
    parser.add_argument(
        "--k_inf",
        type=int,
        default=1,
        help="k parameter for the infection time erlang distribution [1,5]",
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
        default=int(5),
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
    """Initial parameters from argparse"""
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
    k_inf = args.k_inf
    k_rec = args.k_rec
    beta = args.beta / n * k_inf
    delta = args.delta * k_rec
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
        k_inf,
        k_rec,
        beta,
        delta,
    )


# -------------------------


def gillespie(t, time, S, I, R, beta, delta, k_rec, k_inf):
    """
    Time elapsed for the next event
    Calls gillespie_step
    """
    stot = S[t, :-1].sum()
    itot = I[t, :-1].sum()

    lambda_sum = (delta + utils.beta_func(beta, t) * stot) * itot
    prob_heal = delta * I[t, :-1] / lambda_sum
    prob_infect = utils.beta_func(beta, t) * S[t, :-1] * itot / lambda_sum

    t += 1
    time += utils.time_dist(lambda_sum)
    # T[t] = time

    gillespie_step(t, S, I, R, prob_heal, prob_infect, k_rec, k_inf)
    return t, time


# -------------------------


def gillespie_step(t, S, I, R, prob_heal, prob_infect, k_rec, k_inf):
    """
    Perform an event of the algorithm, either infect or recover a single individual
    S and I have one extra dimension to temporally store the infected and recovered
    after k stages, due to the Erlang distribution
    """
    random = np.random.random()
    prob_heal_tot = prob_heal.sum()

    # i(k)-> i(k+1)/r
    if random < prob_heal_tot:
        for k in range(k_rec):
            if random < prob_heal[: k + 1].sum():
                S[t, :-1] = S[t - 1, :-1]
                I[t, k] = -1
                I[t, k + 1] = 1
                R[t] = R[t - 1] + I[t, k_rec]
                I[t] += I[t - 1]
                break

    # s(k)-> s(k+1)/i(0)
    else:
        for k in range(k_inf):
            if random < (prob_heal_tot + prob_infect[: k + 1].sum()):
                R[t] = R[t - 1]
                I[t, :-1] = I[t - 1, :-1]
                S[t, k] = -1
                S[t, k + 1] = 1
                I[t, 0] += S[t, k_inf]
                S[t] += S[t - 1]
                break


# -------------------------


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
