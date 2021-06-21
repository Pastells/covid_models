"""Common functions for all models"""

import sys
import random
import numpy as np
from . import config

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
    weight = transition_weight(time, transition_days)
    for rate in rates:
        rates_eval[rate] = rates_old[rate] + (rates[rate] - rates_old[rate]) * weight

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


def mean_std(var_day):
    """Returns mean and std in a 2-dim array"""
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
            f"Alive realizations after {check_realization_alive} days = {alive_realizations}, out of {mc_nseed}\n"
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


# -------------------------


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


# -------------------------


def cost_func(time_series, var_m, metric=sum_sq):
    """compute cost function with a selected metric
    comparing with data from input file
    if config.CUMULATIVE is True uses cumulative data

    :Input:
    var_m : array with mean and standard deviation
    time_series : ndarray with daily data
    """

    import warnings

    warnings.filterwarnings("error")

    metric_func = StrCallable(metric)

    cost = metric_func.call(var_m, time_series)

    # Normalize with number of days
    # cost = cost / len(time_series) * 100

    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# -------------------------


def cost_return(time_series, var_m, metric=sum_sq):
    """Same as above, but returns cost instead of
    printing GGA success, useful for adding multiple costs"""
    import warnings

    warnings.filterwarnings("error")
    metric_func = StrCallable(metric)
    cost = metric_func.call(var_m, time_series)
    return cost / len(time_series) * 100


# -------------------------


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


class ParserCommon:
    """Handle all different parsers in an incremental fashion, in order
    to avoid repetitions"""

    def __init__(self, description):
        """create parser with init, configuration, data and actions groups
        with a given description"""
        import argparse

        self.parser = argparse.ArgumentParser(
            description=description,
            # formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            formatter_class=argparse.MetavarTypeHelpFormatter,
        )

        self.parser_params = self.parser.add_argument_group("parameters")
        self.parser_init = self.parser.add_argument_group("initial conditions")
        self.parser_config = self.parser.add_argument_group("configuration")
        self.parser_data = self.parser.add_argument_group("data")
        self.parser_act = self.parser.add_argument_group("actions")

        self.parser_init.add_argument(
            "--initial_infected",
            type=int,
            default=config.initial_infected,
            help="initial number of infected individuals,\
                    if None is specified is set to first day of input data",
        )
        self.parser_init.add_argument(
            "--initial_recovered",
            type=int,
            default=config.initial_recovered,
            help="initial number of inmune individuals",
        )

        self.parser_config.add_argument(
            "--seed",
            type=int,
            default=config.SEED,
            help="seed metaparameter for the automatic configuration, also used as MC seed",
        )
        self.parser_config.add_argument(
            "--timeout",
            type=int,
            default=config.TIMEOUT,
            help="timeout metaparameter for the automatic configuration",
        )
        self.parser_config.add_argument(
            "--mc_nseed",
            type=int,
            default=config.MC_NSEED,
            help="number of mc realizations to average over",
        )
        self.parser_config.add_argument(
            "--n_t_steps",
            type=int,
            default=config.N_T_STEPS,
            help="maximum number of simulation steps, dimension for the arrays",
        )

        self.parser_data.add_argument(
            "--data",
            type=str,
            default=config.DATA,
            help="file with time series",
        )
        self.parser_data.add_argument(
            "--day_min",
            type=int,
            default=config.DAY_MIN,
            help="first day to consider of the data series",
        )
        self.parser_data.add_argument(
            "--day_max",
            type=int,
            default=config.DAY_MAX,
            help="last day to consider of the data series",
        )
        self.parser_data.add_argument(
            "--undiagnosed",
            type=float,
            default=config.UNDIAGNOSED,
            help="percentage of undiagnosed cases, \
                    used to rescale the data to account for underreporting",
        )
        self.parser_data.add_argument(
            "--metric",
            type=str,
            default=config.METRIC,
            choices=config.METRICS,
            help=f"metric to use to compute the cost function {config.METRICS_STR}",
            metavar="str",
        )
        self.parser_data.add_argument(
            "--cost_day",
            type=int,
            default=None,
            help="At which day start computing the cost, only implemented in sird_parallel for now",
        )

        self.parser_act.add_argument(
            "--plot", action="store_true", help="specify for plots"
        )
        self.parser_act.add_argument(
            "--save", type=str, default=None, help="specify a name for outputfile"
        )

    # -------------------------

    def parse_args(self):
        """Return parsed args object"""
        return self.parser.parse_args()

    # -------------------------

    def add_argument(self, *args, **kwargs):
        """Add argument"""
        self.parser.add_argument(*args, **kwargs)

    # -------------------------

    def n(self):
        """Number of individuals"""
        self.parser_params.add_argument(
            "--n",
            type=int,
            default=config.N,
            help="fixed number of (effecitve) individuals [1000,1000000]",
        )

    # -------------------------

    def n_sections(self):
        """Number of individuals and days for different sections"""
        self.parser_params.add_argument(
            "--n",
            type=int,
            default=[config.N],
            nargs="*",
            help="fixed number of (effecitve) individuals, \
                    initial and increments [1000,1000000]",
        )
        self.parser_params.add_argument(
            "--section_days",
            type=int,
            default=config.SECTIONS_DAYS,
            nargs="*",
            help="ending day for each section",
        )
        self.parser_params.add_argument(
            "--transition_days",
            type=int,
            default=config.TRANSITION_DAYS,
            help="days it takes to transition from one number of individuals \
                    to the next [1,10]",
        )

    # -------------------------

    def sir(self):
        """Arguments for SIR model"""
        self.parser_params.add_argument(
            "--delta",
            type=float,
            default=config.DELTA,
            help="rate of recovery from infected phase (i->r) [0.05,1.0]",
        )
        self.parser_params.add_argument(
            "--beta",
            type=float,
            default=config.BETA,
            help="infectivity due to infected [0.05,1.0]",
        )

    # -------------------------

    def sir_sections(self):
        """Arguments for SIR model with sections"""
        self.parser_params.add_argument(
            "--delta",
            type=float,
            default=[config.DELTA],
            nargs="*",
            help="rate of recovery from infected phase (i->r) [0.05,1.0]",
        )
        self.parser_params.add_argument(
            "--beta",
            type=float,
            default=[config.BETA],
            nargs="*",
            help="infectivity due to infected [0.05,1.0]",
        )

    # -------------------------

    def asymptomatic(self):
        """Add asymptomatic compartment: initial and all transition rates"""
        self.parser_init.add_argument(
            "--initial_asymptomatic",
            type=int,
            default=config.initial_asymptomatic,
            help="initial number of asymptomatic individuals, \
                if None is specified is set to first day of input data",
        )
        self.parser_params.add_argument(
            "--delta_a",
            type=float,
            default=config.DELTA_A,
            help="rate of recovery from asymptomatic phase (a->r) [0.05,1.0]",
        )
        self.parser_params.add_argument(
            "--beta_a",
            type=float,
            default=config.BETA_A,
            help="infectivity due to asymptomatic [0.05,1.0]",
        )
        self.parser_params.add_argument(
            "--alpha",
            type=float,
            default=config.ALPHA,
            help="asymptomatic rate (a->i) [0.05,2.0]",
        )

    # -------------------------

    def asymptomatic_sections(self):
        """Add asymptomatic compartment for model with sections: \
                initial and all transition rates"""
        self.parser_init.add_argument(
            "--initial_asymptomatic",
            type=int,
            default=config.initial_asymptomatic,
            help="initial number of asymptomatic individuals \
                if None is specified is set to first day of input data",
        )
        self.parser_params.add_argument(
            "--delta_a",
            type=float,
            default=[config.DELTA_A],
            nargs="*",
            help="rate of recovery from asymptomatic phase (a->r) [0.05,1.0]",
        )
        self.parser_params.add_argument(
            "--beta_a",
            type=float,
            default=[config.BETA_A],
            nargs="*",
            help="infectivity due to asymptomatic [0.05,1.0]",
        )
        self.parser_params.add_argument(
            "--alpha",
            type=float,
            default=[config.ALPHA],
            nargs="*",
            help="asymptomatic rate (a->i) [0.05,1]",
        )

    # -------------------------

    def erlang(self, k_asym=False):
        """Arguments for Erlang model"""

        self.parser_params.add_argument(
            "--k_rec",
            type=int,
            default=config.K_REC,
            help="k for the recovery time erlang distribution [1,5]",
        )
        self.parser_params.add_argument(
            "--k_inf",
            type=int,
            default=config.K_INF,
            help="k for the infection time erlang distribution [1,5]",
        )
        if k_asym is True:
            self.parser_params.add_argument(
                "--k_asym",
                type=int,
                default=config.K_ASYM,
                help="k for the asymptomatic time erlang distribution [1,5]",
            )

    # -------------------------

    def exposed(self):
        """Add exposed compartment: initial and transition rate"""
        self.parser_init.add_argument(
            "--initial_exposed",
            type=int,
            default=config.initial_exposed,
            help="initial number of latent individuals \
                if None is specified is set to first day of input data",
        )
        self.parser_params.add_argument(
            "--epsilon",
            type=float,
            default=config.EPSILON,
            help="latency rate (e->a) [0.2,1.0]",
        )

    # -------------------------

    def dead(self):
        """Add dead compartment: initial and transition rate"""
        self.parser_init.add_argument(
            "--initial_dead",
            type=int,
            default=config.initial_dead,
            help="initial number of dead individuals",
        )
        self.parser_params.add_argument(
            "--theta",
            type=float,
            default=config.THETA,
            help="death probability [0.001,0.1]",
        )

    # -------------------------

    def network(self):
        """Network type and parameter"""
        self.parser_params.add_argument(
            "--network",
            type=str,
            choices=["er", "ba"],
            default=config.NETWORK,
            help="Erdos-Renyi or Barabasi-Albert {er,ba}",
        )
        self.parser_params.add_argument(
            "--network_param",
            type=int,
            default=config.NETWORK_PARAM,
            help="mean number of edges [1,50]",
        )

        # -------------------------

    def parallel(self):
        self.parser_act.add_argument(
            "--sequential",
            action="store_true",
            help="specify for sequential execution, by default is parallel",
        )


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

    if np.array(args.n).size > 1:
        n0 = args.n[0]
    else:
        n0 = args.n
    assert (
        n0 - initial_ind > 0
    ), f"Insuficient individuals ({args.n}) for this initial settings"

    return t_total, time_series
