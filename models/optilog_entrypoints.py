import numpy

from .sair import sair
from .utils import config
from .sird import sird


RESULT_REGEX = r"Result: ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"


def _report_result(cost):
    print(f"Result: {cost}")


def load_data(dataset, day_min, day_max):
    time_series = numpy.loadtxt(dataset, delimiter=",", dtype=int, usecols=(0, 1, 2, 3))
    time_series = time_series[day_min:day_max]
    return time_series


def _auto_sird(dataset, seed):
    # Constants:
    day_min = 0
    day_max = 54
    mc_nseed = 100

    t_total = day_max - day_min
    time_series = load_data(dataset, day_max, day_min)

    cost = sird.sird(
        time_series=time_series,
        seed=seed,
        n_seeds=mc_nseed,
        t_total=t_total,
        n_t_steps=config.N_T_STEPS,
        metric="models.utils.utils.sum_sq"
    )
    _report_result(cost)


def _auto_sair(dataset, seed):
    # Constants:
    day_min = 0
    day_max = 54
    mc_nseed = 100

    t_total = day_max - day_min
    time_series = load_data(dataset, day_max, day_min)

    cost = sair.sair(
        time_series=time_series,
        seed=seed,
        n_seeds=mc_nseed,
        t_total=t_total,
        n_t_steps=config.N_T_STEPS,
        metric="models.utils.utils.sum_sq"
    )
    _report_result(cost)


_entrypoints = {
    "sird": (_auto_sird, [sird.sird]),
    "sair": (_auto_sair, [sair.sair])
}


def get_entrypoint_for_model(model):
    return _entrypoints[model]