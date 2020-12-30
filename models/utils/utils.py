""" Common functions for all models """

import sys
import random
import numpy as np
from . import config

# -------------------------


def n_individuals(
    n,
    n_old,
    t_0=0,
    transition_days=config.TRANSITION_DAYS,
    points_per_day=config.POINTS_PER_DAY,
):
    """returns a vector n_ind with n increment as a function of time:
    interpolates between the two given values using a tanh"""

    if (n_old is None) or (n == n_old):
        return None

    def weight(time):
        return 0.5 * (
            1 + np.tanh((time - transition_days / 2) * 5.33 / transition_days)
        )

    points = points_per_day * transition_days
    n_ind = np.zeros([points + 1, 2])
    for time in range(points):
        n_ind[time, 0] = int((n - n_old) * weight(time / 4))
        n_ind[time, 1] = t_0 + time / 4
    n_ind[-1, 0] = n - n_old
    n_ind[-1, 1] = t_0 + 4
    return n_ind


# -------------------------


def rates_sir(
    time, rates, rates_old=None, t_0=0, transition_days=config.TRANSITION_DAYS
):
    """returns beta and delta as a function of time:
    interpolates between the two given values using a tanh"""

    if rates_old is None:
        return rates

    if time > t_0 + transition_days:
        return rates

    rates_eval = {}
    weight = 0.5 * (1 + np.tanh((time - transition_days / 2) * 5.33 / transition_days))
    rates_eval["delta"] = (
        rates_old["delta"] + (rates["delta"] - rates_old["delta"]) * weight
    )
    rates_eval["beta"] = (
        rates_old["beta"] + (rates["beta"] - rates_old["beta"]) * weight
    )
    return rates_eval


# -------------------------


def rates_sair(
    time, rates, rates_old=None, t_0=0, transition_days=config.TRANSITION_DAYS
):
    """returns beta_a/2, delta_a/2 and alpha as a function of time:
    interpolates between the two given values using a tanh"""

    if rates_old is None:
        return rates

    if time > t_0 + 4:
        return rates

    rates_eval = {}
    weight = 0.5 * (1 + np.tanh((time - transition_days / 2) * 5.33 / transition_days))
    rates_eval["delta_a"] = (
        rates_old["delta_a"] + (rates["delta_a"] - rates_old["delta_a"]) * weight
    )
    rates_eval["delta_i"] = (
        rates_old["delta_i"] + (rates["delta_i"] - rates_old["delta_i"]) * weight
    )
    rates_eval["beta_a"] = (
        rates_old["beta_a"] + (rates["beta_a"] - rates_old["beta_a"]) * weight
    )
    rates_eval["beta_i"] = (
        rates_old["beta_i"] + (rates["beta_i"] - rates_old["beta_i"]) * weight
    )
    rates_eval["alpha"] = (
        rates_old["alpha"] + (rates["alpha"] - rates_old["alpha"]) * weight
    )
    return rates_eval


# -------------------------


def monotonically_increasing(array):
    """Check if array is monotonically increasing
    Use:
    if not utils.monotonically_increasing(args.n):
        raise ValueError("n should be monotonically increasing")
    """

    return all(x <= y for x, y in zip(array, array[1:]))


# -------------------------


def time_dist(lambd):
    """ Time intervals of a Poisson process follow an exponential distribution"""
    return random.expovariate(lambd)


# -------------------------


def day_data(time, t_total, day, day_max, I, I_day, last_event=False):
    """
    Write number of infected per day instead of event
    Also tracks day_max

    I :  number of infected for the last event, i.e. I[t]
    I_day : number of infected when time is a day multiple for the current seed,
            i.e. I_day[mc_seed]

    :Returns:

    day : adds one, or one + days_jumped if an event was slow
    day_max : updates if day is bigger than the resulting from previous seeds

    :Modifies:

    I_day : adds the infected number for day (before updating), if several days have
            elapsed they all get the same value as the previous

    if last_event:
        # final value for rest of time, otherwise contributes with zero when averaged
        _days_jumped = int(time - day)
        max_days = min(day + _days_jumped + 1, t_total)
        I_day[day : max_days - 1] = I_day[day - 1]
        I_day[max_days - 1] = I
        day = max_days
        day_max = max(day_max, day)
        return day, day_max
    """
    if (time // day == 1) and (day < t_total):
        _days_jumped = int(time - day)
        max_days = min(day + _days_jumped + 1, t_total)
        I_day[day : max_days - 1] = I_day[day - 1]
        I_day[max_days - 1] = I
        # print(time, day, _days_jumped, max_days, t_total)
        # print(I_day, I)
        day = max_days
        day_max = max(day_max, day)
        return day, day_max
    return day, day_max


# -------------------------


def mean_alive(I_day, t_total, day_max, nseed):
    """
    Given that we already have a pandemic to study,
    we average only the alive realizations after an
    (arbitrary) time equal to half the time of the longest
    realization
    The running variance is computed according to:
        https://www.johndcook.com/blog/standard_deviation/

    :Returns:

    I_m :mean
    I_std : standard deviation

    :Modifies:

    I_day : adds the infected number for day (before updating), if several days have
            elapsed they all get the same value as the previous
    """
    check_realization_alive = day_max // 2

    # first seed alive
    for day in range(nseed):
        if I_day[day, check_realization_alive] != 0:
            _x_var = I_day[day]
            alive_realizations = 1
            _s_var = np.zeros(t_total)
            I_m = _x_var
            break

    for j in range(day + 1, nseed):
        if I_day[j, check_realization_alive] != 0:
            alive_realizations += 1
            _x_var = I_day[j]
            _I_m_1 = I_m
            I_m = _I_m_1 + (_x_var - _I_m_1) / alive_realizations
            _s_var = _s_var + (_x_var - _I_m_1) * (_x_var - I_m)

    if alive_realizations < 1:
        raise ValueError("Not enough alive realizations")
    if alive_realizations == 1:
        # raise ValueError("Not enough alive realizations")
        I_std = I_m * 0.0
    else:
        I_std = np.sqrt(_s_var / (alive_realizations - 1))

    if nseed - alive_realizations > nseed * 0.1:
        sys.stderr.write("The initial number of infected may be too low\n")
        sys.stderr.write(
            f"Alive realizations after {check_realization_alive} days = {alive_realizations},\
              out of {nseed}\n"
        )
    return I_m, I_std


# -------------------------


def saving(args, I_m, I_std, day_max, program_name, custom_name=None):
    """If --save is added creates an output file with date and time
    Modifies args.save to filename for the plots module to use (if specified)"""
    import time

    if custom_name is None:
        filename = (
            config.SAVE_FOLDER
            + f"{program_name}_"
            + time.strftime("%d%m_%H%M%S")
            + ".dat"
        )
    else:
        filename = config.SAVE_FOLDER + f"{program_name}_" + custom_name
        args.save = filename
        filename += ".dat"

    I_cum = 0
    I_cum_std = 0
    with open(filename, "w") as out_file:
        out_file.write(f"#{args}\n")
        out_file.write("# day, I_m, I_std, I_cum, I_cum_std\n")
        for day in range(day_max):
            I_cum += I_m[day]
            I_cum_std += I_std[day]
            out_file.write(f"{day} {I_m[day]} {I_std[day]} {I_cum} {I_cum_std}\n")


# ~~~~~~~~~~~~~~~~~~~
# Output
# ~~~~~~~~~~~~~~~~~~~
def cost_func(infected_time_series, I_m, I_std):
    """compute cost function with a weighted mean squared error
    comparing with data from input
    if config.CUMULATIVE is True us cumulative data

    :Input:

    I_m :  number of _new_ infected per day

    :Generates:

    I_cum :  number of _cumulative_ infected

    """

    import sys

    pad = len(infected_time_series) - len(I_m)

    if pad > 0:
        I_m = np.pad(I_m, (0, pad), "constant")
        I_std = np.pad(I_std, (0, pad), "constant")

    cost = 0
    if config.CUMULATIVE is True:
        I_cum = 0
        I_cum_std = 0
        for day, _ in enumerate(infected_time_series):
            I_cum += I_m[day]
            I_cum_std += I_std[day]
            cost += (I_cum - infected_time_series[day]) ** 2 / infected_time_series[day]
            """ print(
                day,
                "{:12.2f}".format((I_cum - infected_time_series[day]) ** 2),
                "{:10.2f}".format(infected_time_series[day]),
                "{:5.2f}".format(
                    (I_cum - infected_time_series[day]) ** 2 / infected_time_series[day]
                ),
            ) """
        sys.stdout.write(f"GGA SUCCESS {cost}\n")
        return

    for day, _ in enumerate(infected_time_series):
        cost += (I_m[day] - infected_time_series[day]) ** 2 / infected_time_series[day]

    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# -------------------------


def parser_common(parser, A_0=False, E_0=False):
    """ create init, configuraten, data and actions groups for the parser"""

    parser_init = parser.add_argument_group("initial conditions")
    parser_config = parser.add_argument_group("configuraten")
    parser_data = parser.add_argument_group("data")
    parser_act = parser.add_argument_group("actions")

    if A_0 is True:
        parser_init.add_argument(
            "--A_0",
            type=int,
            default=config.A_0,
            help="initial number of asymptomatic individuals",
        )

    if E_0 is True:
        parser_init.add_argument(
            "--E_0",
            type=int,
            default=config.E_0,
            help="initial number of latent individuals",
        )

    parser_init.add_argument(
        "--I_0",
        type=int,
        default=config.I_0,
        help="initial number of infected individuals,\
                if None is specified is set to first day of input data",
    )
    parser_init.add_argument(
        "--R_0",
        type=int,
        default=config.R_0,
        help="initial number of inmune individuals",
    )

    parser_config.add_argument(
        "--seed",
        type=int,
        default=config.SEED,
        help="seed for the automatic configuraten",
    )
    parser_config.add_argument(
        "--timeout",
        type=int,
        default=config.TIMEOUT,
        help="timeout for the automatic configuraten",
    )
    parser_config.add_argument(
        "--mc_nseed",
        type=int,
        default=config.MC_NSEED,
        help="number of mc realizations to average over",
    )
    parser_config.add_argument(
        "--mc_seed0",
        type=int,
        default=config.MC_SEED0,
        help="initial mc seed",
    )
    parser_config.add_argument(
        "--n_t_steps",
        type=int,
        default=config.N_T_STEPS,
        help="maximum number of simulation steps, dimension for the arrays",
    )

    parser_data.add_argument(
        "--data",
        type=str,
        default=config.DATA,
        help="file with time series",
    )
    parser_data.add_argument(
        "--day_min",
        type=int,
        default=config.DAY_MIN,
        help="first day to consider of the data series",
    )
    parser_data.add_argument(
        "--day_max",
        type=int,
        default=config.DAY_MAX,
        help="last day to consider of the data series",
    )
    parser_data.add_argument(
        "--undiagnosed",
        type=float,
        default=config.UNDIAGNOSED,
        help="percentage of undiagnosed cases, used to rescale the data, default taken from \
                Alex Arenas 2020 paper (Physical Review X, 10(4), 041055.)",
    )

    parser_act.add_argument("--plot", action="store_true", help="specify for plots")
    parser_act.add_argument(
        "--save", type=str, default=None, help="specify a name for outputfile"
    )


# -------------------------


def parameters_init_common(args):
    """initial parameters from argparse"""
    from numpy import genfromtxt

    t_total = args.day_max - args.day_min  # max simulated days
    infected_time_series = (
        genfromtxt(args.data, delimiter=",")[args.day_min : args.day_max]
        * 100
        / (100 - args.undiagnosed)
    )

    if config.CUMULATIVE is False:
        for day in range(len(infected_time_series) - 1, 0, -1):
            infected_time_series[day] = (
                infected_time_series[day] - infected_time_series[day - 1]
            )

    if args.I_0 is None:
        args.I_0 = int(infected_time_series[0])

    if (hasattr(args, "E_0")) and (args.A_0 is None):
        args.E_0 = int(infected_time_series[0])

    if (hasattr(args, "A_0")) and (args.A_0 is None):
        args.A_0 = int(infected_time_series[0])

    return t_total, infected_time_series
