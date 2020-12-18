"""
Common functions for all models
"""

import sys
import random
import numpy as np

# -------------------------


def n_individuals(n, n_old, t_0=0, transition_days=4, points_per_day=4):
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


def ratios_sir(time, ratios, ratios_old=None, t_0=0, transition_days=4):
    """returns beta and delta as a function of time:
    interpolates between the two given values using a tanh"""

    if ratios_old is None:
        return ratios["beta"], ratios["delta"]

    if time > t_0 + transition_days:
        return ratios["beta"], ratios["delta"]

    weight = 0.5 * (1 + np.tanh((time - transition_days / 2) * 5.33 / transition_days))
    delta = ratios_old["delta"] + (ratios["delta"] - ratios_old["delta"]) * weight
    beta = ratios_old["beta"] + (ratios["beta"] - ratios_old["beta"]) * weight
    return beta, delta


# -------------------------


def ratios_seir(time, ratios, ratios_old=None, t_0=0, transition_days=4):
    """returns beta1/2, delta1/2 and epsilon as a function of time:
    interpolates between the two given values using a tanh"""

    if ratios_old is None:
        return (
            ratios["beta1"],
            ratios["beta2"],
            ratios["delta1"],
            ratios["delta2"],
            ratios["epsilon"],
        )

    if time > t_0 + 4:
        return (
            ratios["beta1"],
            ratios["beta2"],
            ratios["delta1"],
            ratios["delta2"],
            ratios["epsilon"],
        )

    weight = 0.5 * (1 + np.tanh((time - transition_days / 2) * 5.33 / transition_days))
    delta1 = ratios_old["delta1"] + (ratios["delta1"] - ratios_old["delta1"]) * weight
    delta2 = ratios_old["delta2"] + (ratios["delta2"] - ratios_old["delta2"]) * weight
    beta1 = ratios_old["beta1"] + (ratios["beta1"] - ratios_old["beta1"]) * weight
    beta2 = ratios_old["beta2"] + (ratios["beta2"] - ratios_old["beta2"]) * weight
    epsilon = (
        ratios_old["epsilon"] + (ratios["epsilon"] - ratios_old["epsilon"]) * weight
    )
    return beta1, beta2, delta1, delta2, epsilon


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

    I :  is the the number of infected for the last event, i.e. I[t]
    I_day : is the number of infected when time is a day multiple for the current seed,
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
    for u in range(nseed):
        if I_day[u, check_realization_alive] != 0:
            _x_var = I_day[u]
            alive_realizations = 1
            _s_var = np.zeros(t_total)
            I_m = _x_var
            break

    for j in range(u + 1, nseed):
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
    """ If --save is added creates an output file with date and time"""
    import time

    if custom_name is None:
        filename = f"../results/{program_name}_" + time.strftime("%d%m_%H%M%S") + ".dat"
    else:
        filename = f"../results/{program_name}_" + custom_name + ".dat"
    with open(filename, "w") as out_file:
        out_file.write(f"#{args}\n")
        for day in range(day_max):
            out_file.write(f"{day} {I_m[day]} {I_std[day]}\n")


# ~~~~~~~~~~~~~~~~~~~
# Output
# ~~~~~~~~~~~~~~~~~~~
def cost_func(infected_time_series, I_m, I_std):
    """ compute cost function with a weighted mean squared error"""
    import sys

    pad = len(infected_time_series) - len(I_m)

    if pad > 0:
        I_m = np.pad(I_m, (0, pad), "constant")
        I_std = np.pad(I_std, (0, pad), "constant")

    cost = 0
    for u, _ in enumerate(infected_time_series):
        cost += (I_m[u] - infected_time_series[u]) ** 2 / (1 + I_std[u])
    cost = np.sqrt(cost)
    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# -------------------------


def parser_common(parser):
    """ create configuration, data and actions groups for the parser"""

    parser_config = parser.add_argument_group("configuration")
    parser_data = parser.add_argument_group("data")
    parser_act = parser.add_argument_group("actions")

    parser_config.add_argument(
        "--seed", type=int, default=1, help="seed for the automatic configuration"
    )
    parser_config.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="timeout for the automatic configuration",
    )
    parser_config.add_argument(
        "--mc_nseed",
        type=int,
        default=int(5),
        help="number of mc realizations, not really a parameter",
    )
    parser_config.add_argument(
        "--mc_seed0",
        type=int,
        default=1,
        help="initial mc seed, not really a parameter",
    )
    parser_config.add_argument(
        "--n_t_steps",
        type=int,
        default=int(1e7),
        help="maximum number of simulation steps, dimension for the arrays",
    )

    parser_data.add_argument(
        "--data",
        type=str,
        default="../../data/italy_i.csv",
        help="file with time series",
    )
    parser_data.add_argument(
        "--day_min",
        type=int,
        default=33,
        help="first day to consider on data series",
    )
    parser_data.add_argument(
        "--day_max",
        type=int,
        default=58,
        help="last day to consider on data series",
    )

    parser_act.add_argument("--plot", action="store_true", help="specify for plots")
    parser_act.add_argument(
        "--save", type=str, default=None, help="specify a name for outputfile"
    )
