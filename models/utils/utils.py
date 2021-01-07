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


def day_data(times, values, values_day):
    """ Values per day instead of event"""

    day_max = int(times.max()) + 1

    for day in range(day_max):
        values_day[day] = values[np.searchsorted(times, day)]

    for day in range(1, day_max - 1):
        if values_day[day] == values_day[day + 1]:
            values_day[day] = values_day[day - 1]

    return day_max


# -------------------------


def mean_alive(I_day, t_total, day_max, nseed):
    """
    Given that we already have a pandemic to study,
    we average only the alive realizations after:
     a) an (arbitrary) time equal to half the time of the longest realization
     b) the whole time (useful at the beginning of the pandemic)
    The running variance is computed according to:
        https://www.johndcook.com/blog/standard_deviation/

    :Returns:

    I_m :mean
    I_std : standard deviation

    :Modifies:

    I_day : adds the infected number for day (before updating), if several days have
            elapsed they all get the same value as the previous
    """
    # check_realization_alive = day_max // 2
    check_realization_alive = day_max - 1
    alive_realizations = 0

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


def saving(args, I_m, I_std, day_max):
    """If --save is added creates a file with the daily results to config.SAVE_FOLDER
    Uses the name of the program that generated it and the given name to args.save
        e.g. sir.py --save test will generate sir_test.dat
    If that name already exists uses date and time after program name
    Modifies args.save to filename for the plots function to use (if specified)"""

    import time
    import inspect
    import os.path

    # get calling program name
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    program_name = module.__file__.split("/")[-1][:-3]

    # create filename
    filename = config.SAVE_FOLDER + f"{program_name}_" + args.save
    args.save = filename
    filename += ".dat"

    # if already exist use date and time
    if os.path.isfile(filename):
        print(f"{filename} already exists, changed to:")
        filename = (
            config.SAVE_FOLDER + f"{program_name}_" + time.strftime("%d%m_%H%M%S")
        )
        args.save = filename
        filename += ".dat"
        print(filename)

    with open(filename, "w") as out_file:
        out_file.write("#")
        for key, value in vars(args).items():
            out_file.write(f"--{key} {value} ")
        out_file.write("\n")
        out_file.write("# day        I_m        I_std\n")

        form = "{:3.0f} {:12.2f} {:12.2f}\n"
        for day in range(day_max):
            out_file.write(form.format(day, I_m[day], I_std[day]))


# ~~~~~~~~~~~~~~~~~~~
# Output
# ~~~~~~~~~~~~~~~~~~~


def cost_func(time_series, I_m, I_std):
    """compute cost function with a weighted mean squared error
    comparing with data from input
    if config.CUMULATIVE is True uses cumulative data

    Can also be used with R_m or D_m instead of I_m

    :Input:

    I_m   : mean number infected per day
    I_std : standard deviation for I_m

    """

    import sys
    import warnings

    warnings.filterwarnings("error")

    pad = len(time_series) - len(I_m)

    if pad > 0:
        I_m = np.pad(I_m, (0, pad), "constant")
        I_std = np.pad(I_std, (0, pad), "constant")

    cost = 0
    for day, _ in enumerate(time_series):
        # catch division by zero
        try:
            cost += abs(I_m[day] - time_series[day]) / time_series[day]
        except RuntimeWarning:
            cost += abs(I_m[day])

        """print(
            day,
            "{:10.2f}".format(time_series[day]),
            "{:12.2f}".format(abs(I_m[day] - time_series[day])),
            "{:5.2f}".format(
                abs(I_m[day] - time_series[day]) / time_series[day]
            ),
            "{:12.2f}".format(cost),
        )"""

    # Normalize with number of days
    cost = cost / len(time_series) * 100

    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# -------------------------


def parser_common(parser, A_0=False, E_0=False, D_0=False):
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
            help="initial number of asymptomatic individuals \
                if None is specified is set to first day of input data",
        )
    if E_0 is True:
        parser_init.add_argument(
            "--E_0",
            type=int,
            default=config.E_0,
            help="initial number of latent individuals \
                if None is specified is set to first day of input data",
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
    if D_0 is True:
        parser_init.add_argument(
            "--D_0",
            type=int,
            default=config.D_0,
            help="initial number of dead individuals",
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
        help="percentage of undiagnosed cases, used to rescale the data",
    )

    parser_act.add_argument("--plot", action="store_true", help="specify for plots")
    parser_act.add_argument(
        "--save", type=str, default=None, help="specify a name for outputfile"
    )


# -------------------------


def parameters_init_common(args):
    """initial parameters from argparse"""

    t_total = args.day_max - args.day_min  # max simulated days
    time_series = np.loadtxt(args.data, delimiter=",").astype(int)[
        args.day_min : args.day_max
    ]

    # Scale for undiagnosed cases
    if args.undiagnosed != 0:
        time_series[:, 0] = (time_series[:, 0] * 100 / (100 - args.undiagnosed)).astype(
            int
        )
        time_series[:, 3] = time_series[:, 0:3].sum(axis=1)

    if args.I_0 is None:
        args.I_0 = int(time_series[0, 0])

    if (hasattr(args, "E_0")) and (args.E_0 is None):
        args.E_0 = int(time_series[0, 0])

    if (hasattr(args, "A_0")) and (args.A_0 is None):
        args.A_0 = int(time_series[0, 0])

    return t_total, time_series
