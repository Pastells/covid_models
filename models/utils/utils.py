"""Common functions for all models"""

import sys
import random
from typing import Tuple, List, Optional

import numpy
import numpy as np
import pandas

from . import config


def evolution_to_dataframe(evolution: numpy.ndarray,
                           compartments: List[str],
                           seeds: Optional[List[int]]=None) -> pandas.DataFrame:
    if seeds:
        evolution_df = pandas.DataFrame(
            [],
            columns=pandas.MultiIndex.from_product(
                [compartments, seeds]
            )
        )
    else:
        evolution_df = pandas.DataFrame(
            [],
            columns=compartments
        )

    for i, compartment in enumerate(compartments):
        if seeds:
            for step, seed in enumerate(seeds):
                evolution_df[(compartment, seed)] = evolution[i, step]
        else:
            evolution_df[compartment] = evolution[i]

    return evolution_df

# -------------------------


def transition_weight(time, transition_days=config.TRANSITION_DAYS):
    """tanh to transition between two values, used in section models"""
    return 0.5 * (1 + np.tanh((time - transition_days / 2) * 5.33 / transition_days))


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

    points = points_per_day * transition_days
    n_ind = np.zeros([points + 1, 2])
    for time in range(points):
        n_ind[time, 0] = int(
            (number - n_old) * transition_weight(time / 4, transition_days)
        )
        n_ind[time, 1] = t_0 + time / 4
    n_ind[-1, 0] = number - n_old
    n_ind[-1, 1] = t_0 + 4
    n_ind[1:, 0] = np.diff(n_ind[:, 0])
    return n_ind


def section_rates(
    time, rates, rates_old=None, t_0=0, transition_days=config.TRANSITION_DAYS
):
    """returns rates as a function of time:
    interpolates between the two given values using a tanh
    works for both sir and sair models"""

    if rates_old is None:
        return rates

    if time > t_0 + transition_days:
        return rates

    rates_eval = {}
    weight = transition_weight(time - t_0, transition_days)
    for rate in rates:
        rates_eval[rate] = rates_old[rate] + (rates[rate] - rates_old[rate]) * weight

    return rates_eval


# -------------------------


def time_dist(lambd):
    """Time intervals of a Poisson process follow an exponential distribution"""
    return random.expovariate(lambd)


# -------------------------


def day_data(event_timestamps, events_sequence, t_total) -> Tuple[int, numpy.ndarray]:
    """
    Converts values per event to values per day.
    Parameters:
        event_timestamps: Sequence of the timestamps for the events
        events_sequence: Sequence of values per event
        time_total: Amount of days to contemplate

    Return:
        A tuple containing the max day used, and the daily sequence
    """
    daily_sequence = numpy.zeros(t_total)

    available_day_max = int(event_timestamps[-1]) + 1
    for day in range(available_day_max):
        daily_sequence[day] = events_sequence[np.searchsorted(event_timestamps, day)]

    for day in range(1, available_day_max - 1):
        if daily_sequence[day] == daily_sequence[day + 1]:
            daily_sequence[day] = daily_sequence[day - 1]

    return available_day_max, daily_sequence


# -------------------------


def mean_std(var_day):
    """Returns mean and std in a 2-dim array
    Used in SIRD model"""
    return np.column_stack([np.mean(var_day, axis=0), np.std(var_day, axis=0)])


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
    # check_realization_alive = day_max // 2  # a)
    check_realization_alive = t_total - 1  # b)
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
            f"Alive realizations after {check_realization_alive} days = {alive_realizations}, out of {mc_nseed}\n"
        )
    return np.column_stack([var_m, var_std])


# -------------------------


def mean_alive_rd(var_day, t_total, day_max, mc_nseed, var2_day=False, var3_day=False):
    """Same as above, except it computes means for two other variables,
    usually R and D"""

    # check_realization_alive = day_max // 2  # a)
    check_realization_alive = day_max - 1  # b)
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


def get_program_name(level):
    """Get program calling the function
    level : how deep is the function call
    e.g.: sir.py calls cost_save_plot, which in turn calls saving
          => level = 2"""
    import inspect

    frame = inspect.stack()[level]
    module = inspect.getmodule(frame[0])
    return module.__file__.split("/")[-1][:-3]


# -------------------------


def saving(args, var_m, day_max, var="I", level=2):
    """If --save is added creates a file with the daily results to config.SAVE_FOLDER
    Uses the name of the program that generated it and the given name to args.save
        e.g. sir.py --save test will generate sir_test.dat
    If that name already exists uses date and time after program name
    Modifies args.save to filename for the plots function to use (if specified)"""

    import os.path

    # get calling program name
    program_name = get_program_name(level)

    # create filename
    filename = config.SAVE_FOLDER + f"{program_name}_" + args.save
    args.save = filename
    filename += ".dat"

    # if already exist use date and time
    if os.path.isfile(filename):
        import time

        print(f"{filename} already exists, changed to:")
        filename = (
            config.SAVE_FOLDER + f"{program_name}_" + time.strftime("%d%m_%H%M%S")
        )
        args.save = filename
        filename += ".dat"
        print(filename)

    # get time series
    if var == "I":
        time_series = get_time_series(args)[:, 0]
    elif var == "R":
        time_series = get_time_series(args)[:, 1]
    elif var == "D":
        time_series = get_time_series(args)[:, 2]
    else:
        raise ValueError("Bad variable name in saving")

    # save to file
    with open(filename, "w") as out_file:
        out_file.write("#")
        for key, value in vars(args).items():
            out_file.write(f"--{key} {value} ")
        out_file.write("\n")
        out_file.write(f"# day       {var}_m       {var}_std      {var}_data\n")

        form = "{:3.0f}{:12.2f}{:12.2f}{:12.0f}\n"
        for day in range(day_max):
            out_file.write(
                form.format(day, var_m[day, 0], var_m[day, 1], time_series[day])
            )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Cost function based on different metrics to choose from
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class StrCallable:
    """Call a function given its name as module.function"""

    def __init__(self, name):
        self.name, self.function = name, None
        if self.function is None:
            modn, funcn = self.name.rsplit(".", 1)
            self.function = getattr(sys.modules[modn], funcn)

    def call(self, *a, **k):
        """Call function"""
        output = self.function(*a, **k)
        return output


def sum_sq(var_m, time_series):
    """Sum of squared differences"""
    cost = 0
    for day, _ in enumerate(time_series):
        cost += (var_m[day, 0] - time_series[day]) ** 2 / 1e6
    return cost


def max_sq(var_m, time_series):
    """Max of squared differences"""
    cost = 0
    for day, _ in enumerate(time_series):
        _cost = (var_m[day, 0] - time_series[day]) ** 2 / 1e6
        cost = max(cost, _cost)
    return cost


def sum_sq_weight(var_m, time_series, day):
    """Sum of weighted quared differences"""
    cost = 0
    for day, _ in enumerate(time_series):
        cost += (day + 1) * (var_m[day, 0] - time_series[day]) ** 2 / 1e8
    return cost


def sum_sq_scaled(var_m, time_series, *day):
    """Sum of scaled squared differences"""

    cost = 0
    for day, _ in enumerate(time_series):
        # catch division by zero
        try:
            cost += (var_m[day, 0] - time_series[day]) ** 2 / time_series[day] ** 2
        except RuntimeWarning:
            cost += var_m ** 2
    return cost


def cost_func(time_series, var_m, metric=sum_sq):
    """compute cost function with a selected metric
    comparing with data from input file
    if config.CUMULATIVE is True uses cumulative data

    :Input:
    var_m : array with mean and standard deviation
    time_series : ndarray with daily data
    """
    cost = cost_return(time_series, var_m, metric)
    # Normalize with number of days
    # cost = cost / len(time_series) * 100
    print(f"GGA SUCCESS {cost}")
    return cost


def cost_return(time_series, var_m, metric=sum_sq):
    """Same as above, but returns cost instead of
    printing GGA success, useful for adding multiple costs"""
    import warnings

    warnings.filterwarnings("error")
    metric_func = StrCallable(metric)
    return metric_func.call(var_m, time_series)


def cost_save_plot(var_day, t_total, day_max, args, time_series):
    """Compute mean and std, cost function and save/plot if needed
    I had the same code in all files so I put it here"""

    var_m = mean_alive(var_day, t_total, day_max, args.mc_nseed)

    if config.CUMULATIVE is True:
        cost_func(time_series[:, 3], var_m, args.metric)
    else:
        cost_func(time_series[:, 0], var_m, args.metric)

    if args.save is not None:
        saving(args, var_m, day_max, level=3)

    if args.plot:
        from . import plots

        plots.plotting(args, day_max, var_m)  # , comp=comp, t_step=t_step)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def get_time_series(args):
    """Time_series from input file"""
    if hasattr(args, "section_days"):
        time_series = np.loadtxt(
            args.data, delimiter=",", dtype=int, usecols=(0, 1, 2, 3)
        )[args.day_min : args.day_min + args.section_days[-1]]
    else:
        time_series = np.loadtxt(
            args.data, delimiter=",", dtype=int, usecols=(0, 1, 2, 3)
        )[args.day_min : args.day_max]

    # Scale for undiagnosed cases
    if hasattr(args, "undiagnosed") and args.undiagnosed != 0:
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

    if args.initial_infected is None:
        args.initial_infected = int(time_series[0, 0])

    if args.initial_infected <= 0:
        raise ValueError(
            "initial_infected must be a positive integer. \
            Change given initial condition or check day_min data"
        )

    initial_ind = args.initial_infected + args.initial_recovered

    if (hasattr(args, "initial_exposed")) and (args.initial_exposed is None):
        args.initial_exposed = int(time_series[0, 0])
        initial_ind += args.initial_exposed

    if (hasattr(args, "initial_asymptomatic")) and (args.initial_asymptomatic is None):
        args.initial_asymptomatic = int(time_series[0, 0])
        initial_ind += args.initial_asymptomatic

    # If the model has sections, the initial population is the one from the
    # first section. Otherwise it is the one given by the parameter
    # Section models have n as a list, try to access the first element or
    # assume it is a single integer
    try:
        n0 = args.n[0]
    except TypeError:
        n0 = args.n

    assert (
        n0 - initial_ind > 0
    ), f"Insuficient individuals ({args.n}) for this initial settings"

    return t_total, time_series
