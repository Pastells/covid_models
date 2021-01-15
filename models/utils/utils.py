"""Common functions for all models"""

import sys
import random
import numpy as np
from . import config

# -------------------------


def n_individuals(
    number,
    n_old,
    t_0=0,
    transition_days=config.TRANSITION_DAYS,
    points_per_day=config.POINTS_PER_DAY,
):
    """returns a vector n_ind with number increment as a function of time:
    interpolates between the two given values using a tanh"""

    if (n_old is None) or (number == n_old):
        return None

    def weight(time):
        return 0.5 * (
            1 + np.tanh((time - transition_days / 2) * 5.33 / transition_days)
        )

    points = points_per_day * transition_days
    n_ind = np.zeros([points + 1, 2])
    for time in range(points):
        n_ind[time, 0] = int((number - n_old) * weight(time / 4))
        n_ind[time, 1] = t_0 + time / 4
    n_ind[-1, 0] = number - n_old
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
    """Time intervals of a Poisson process follow an exponential distribution"""
    return random.expovariate(lambd)


# -------------------------


def day_data(times, var, var_day, day_max):
    """Values per day instead of event"""

    _day_max = int(times[-1]) + 1

    for day in range(_day_max):
        var_day[day] = var[np.searchsorted(times, day)]

    for day in range(1, _day_max - 1):
        if var_day[day] == var_day[day + 1]:
            var_day[day] = var_day[day - 1]

    return max(day_max, _day_max)


# -------------------------


def mean_alive(var_day, t_total, day_max, mc_nseed):
    """
    Given that we already have a pandemic to study,
    we average only the alive realizations after:
     a) the whole time (useful at the beginning of the pandemic)
     b) an (arbitrary) time equal to half the time of the longest realization
    The running variance is computed according to:
        https://www.johndcook.com/blog/standard_deviation/

    :Returns:

    var_m : array with mean and standard deviation

    :Modifies:

    var_day : adds the infected number for day (before updating), if several days have
            elapsed they all get the same value as the previous
    """
    # check_realization_alive = day_max // 2
    check_realization_alive = t_total - 1
    alive_realizations = 0

    # first mc_nseed alive
    for mc_seed in range(mc_nseed):
        if var_day[mc_seed, check_realization_alive] != 0:
            _x_var = var_day[mc_seed]
            alive_realizations = 1
            _s_var = np.zeros(t_total)
            var_m = _x_var
            break

    for j in range(mc_seed + 1, mc_nseed):
        if var_day[j, check_realization_alive] != 0:
            alive_realizations += 1
            _x_var = var_day[j]
            var_m_1 = var_m
            var_m = var_m_1 + (_x_var - var_m_1) / alive_realizations
            _s_var = _s_var + (_x_var - var_m_1) * (_x_var - var_m)

    if alive_realizations < 1:
        raise ValueError("Not enough alive realizations")
    if alive_realizations == 1:
        # raise ValueError("Not enough alive realizations")
        var_std = var_m * 0.0
    else:
        var_std = np.sqrt(_s_var / (alive_realizations - 1))

    if mc_nseed - alive_realizations > mc_nseed * 0.1:
        sys.stderr.write("The initial number of infected may be too low\n")
        sys.stderr.write(
            f"Alive realizations after {check_realization_alive} days = {alive_realizations},\
              out of {mc_nseed}\n"
        )
    # return var_m, var_std
    return np.column_stack([var_m, var_std])


# -------------------------


def mean_alive_rd(var_day, t_total, day_max, mc_nseed, var2_day=False, var3_day=False):
    """Same as above, except it computes means for two other variables,
    usually R and D"""

    # check_realization_alive = day_max // 2
    check_realization_alive = day_max - 1
    alive_realizations = 0

    # first mc_nseed alive
    for mc_seed in range(mc_nseed):
        if var_day[mc_seed, check_realization_alive] != 0:
            _x_var = np.array([var_day[mc_seed], var2_day[mc_seed], var3_day[mc_seed]])
            alive_realizations = 1
            _s_var = np.zeros([3, t_total])
            var_m = _x_var
            break

    for j in range(mc_seed + 1, mc_nseed):
        if var_day[j, check_realization_alive] != 0:
            alive_realizations += 1
            _x_var = np.array([var_day[j], var2_day[j], var3_day[j]])
            var_m_1 = var_m
            var_m = var_m_1 + (_x_var - var_m_1) / alive_realizations
            _s_var = _s_var + (_x_var - var_m_1) * (_x_var - var_m)

    if alive_realizations < 1:
        raise ValueError("Not enough alive realizations")
    if alive_realizations == 1:
        # raise ValueError("Not enough alive realizations")
        var_std = var_m * 0.0
    else:
        var_std = np.sqrt(_s_var / (alive_realizations - 1))

    if mc_nseed - alive_realizations > mc_nseed * 0.1:
        sys.stderr.write("The initial number of infected may be too low\n")
        sys.stderr.write(
            f"Alive realizations after {check_realization_alive} days: "
            f"{alive_realizations} out of {mc_nseed}\n"
        )

    return (
        np.column_stack([var_m[0], var_std[0]]),
        np.column_stack([var_m[1], var_std[1]]),
        np.column_stack([var_m[2], var_std[2]]),
    )


# -------------------------


def saving(args, var_m, day_max):
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
            out_file.write(form.format(day, var_m[day, 0], var_m[day, 1]))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cost function based on different metrics to choose from
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class StrCallable(object):
    """Call a function given its name as module.function"""

    def __init__(self, name):
        self.n, self.f = name, None
        if self.f is None:
            modn, funcn = self.n.rsplit(".", 1)
            self.f = getattr(sys.modules[modn], funcn)

    def call(self, *a, **k):
        output = self.f(*a, **k)
        return output


# -------------------------


def sq_diff(var_m, time_series, *day):
    """Sum of absolute differences"""
    return (var_m - time_series) ** 2 / 1e6


def sq_diff_weight(var_m, time_series, day):
    """Sum of absolute differences"""
    return (day + 1) * (var_m - time_series) ** 2 / 1e8


def sq_diff_scaled(var_m, time_series, *day):
    """Sum of squared differences, scaled"""

    # catch division by zero
    try:
        cost = (var_m - time_series) ** 2 / time_series ** 2
    except RuntimeWarning:
        cost = var_m ** 2
    return cost


def abs_diff(var_m, time_series, *day):
    """Sum of squared differences"""
    return abs(var_m - time_series) / 1e2


def abs_diff_scaled(var_m, time_series, *day):
    """Sum of absolute differences, scaled"""

    # catch division by zero
    try:
        cost = abs(var_m - time_series) / time_series
    except RuntimeWarning:
        cost = abs(var_m)
    return cost * 10


# -------------------------


def cost_func(time_series, var_m, metric=abs_diff):
    """compute cost function with a selected metric
    comparing with data from input file
    if config.CUMULATIVE is True uses cumulative data

    :Input:
    var_m : array with mean and standard deviation
    time_series : ndarray with daily data
    """

    import sys
    import warnings

    warnings.filterwarnings("error")

    """
    pad = len(time_series) - len(var_m)
    if pad > 0:
        var_m = np.pad(var_m, (0, pad), "constant")
        var_std = np.pad(var_std, (0, pad), "constant")
    """

    metric_func = StrCallable(metric)
    cost = 0
    for day, _ in enumerate(time_series):
        cost += metric_func.call(var_m[day, 0], time_series[day], day)

    # Normalize with number of days
    cost = cost / len(time_series) * 100

    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# -------------------------


def cost_return(time_series, var_m, metric=sq_diff_weight):
    """Same as above, but returns cost instead of
    printing GGA success, useful for adding multiple costs"""
    import warnings

    warnings.filterwarnings("error")
    metric_func = StrCallable(metric)
    cost = 0
    for day, _ in enumerate(time_series):
        cost += metric_func.call(var_m[day, 0], time_series[day], day)

    return cost / len(time_series) * 100


# -------------------------


def parser_common(parser, A_0=False, E_0=False, D_0=False):
    """create init, configuraten, data and actions groups for the parser"""

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
    parser_data.add_argument(
        "--metric",
        type=str,
        default=config.METRIC,
        choices=config.METRICS,
        help=f"Metric to use to compute the cost function, choose from \
              {config.METRICS}",
        metavar="",
    )

    parser_act.add_argument("--plot", action="store_true", help="specify for plots")
    parser_act.add_argument(
        "--save", type=str, default=None, help="specify a name for outputfile"
    )


# -------------------------


def get_time_series(args):
    """Time_series from input file"""
    time_series = np.loadtxt(args.data, delimiter=",", dtype=int, usecols=(0, 1, 2, 3))[
        args.day_min : args.day_max
    ]

    # Scale for undiagnosed cases
    if args.undiagnosed != 0:
        time_series[:, 0] = (time_series[:, 0] * 100 / (100 - args.undiagnosed)).astype(
            int
        )
        time_series[:, 3] = time_series[:, 0:3].sum(axis=1)
    return time_series


# -------------------------


def parameters_init_common(args):
    """initial parameters from argparse"""

    # max simulated days
    if hasattr(args, "section_days"):
        t_total = args.section_days[-1]
    else:
        t_total = args.day_max - args.day_min

    time_series = get_time_series(args)
    args.metric = __name__ + "." + args.metric

    if args.I_0 is None:
        args.I_0 = int(time_series[0, 0])

    if (hasattr(args, "E_0")) and (args.E_0 is None):
        args.E_0 = int(time_series[0, 0])

    if (hasattr(args, "A_0")) and (args.A_0 is None):
        args.A_0 = int(time_series[0, 0])

    return t_total, time_series
