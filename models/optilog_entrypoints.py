from typing import Tuple

import numpy

from .seair import seair
from .sir import sir, sir_erlang, net_sir
from .sair import sair, net_sair, sair_erlang
from .utils import config
from .sird import sird
from .seipahrf import seipahrf


class Entrypoint:
    NAME = None
    CFG_CALLS = None
    RESULT_REGEX = r"Result: ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
    MAX_COST = 100000000000000000000

    @staticmethod
    def report_result(cost):
        print(f"Result: {cost}")

    @staticmethod
    def load_dataset(dataset, day_min, day_max):
        time_series = numpy.loadtxt(dataset, delimiter=",", dtype=int, usecols=(0, 1, 2, 3))
        time_series = time_series[day_min:day_max]
        return time_series

    @classmethod
    def entrypoint(cls, data, seed):
        dataset, day_min, day_max, mc_nseed = cls.load_data(data)

        t_total = day_max - day_min
        time_series = cls.load_dataset(dataset, day_min, day_max)

        cls._run_model(time_series, seed, t_total, mc_nseed)

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        raise NotImplementedError

    @classmethod
    def create_data(cls, file, dataset, day_min, day_max, mc_nseed):
        with open(file, "w+") as file_:
            file_.writelines([
                f"dataset={dataset}\n",
                f"day_min={day_min}\n",
                f"day_max={day_max}\n",
                f"mc_nseed={mc_nseed}\n"
            ])

    @classmethod
    def load_data(cls, file) -> Tuple[str, int, int, int]:
        dataset = day_min = day_max = mc_nseed = None
        with open(file, "r") as file_:
            for line in file_:
                name, value = line.strip().split("=")
                if name == "dataset":
                    dataset = value
                elif name == "day_min":
                    day_min = int(value)
                elif name == "day_max":
                    day_max = int(value)
                elif name == "mc_nseed":
                    mc_nseed = int(value)
        return dataset, day_min, day_max, mc_nseed


class SairEntrypoint(Entrypoint):
    NAME = "sair"
    CFG_CALLS = [sair.sair]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        cost, _ = sair.sair(
            time_series=time_series,
            seed=seed,
            n_seeds=mc_nseed,
            t_total=t_total,
            n_t_steps=config.N_T_STEPS,
            metric="models.utils.utils.sum_sq"
        )
        cls.report_result(cost)


class SairErlangEntrypoint(Entrypoint):
    NAME = "sair-erlang"
    CFG_CALLS = [sair_erlang.sair_erlang]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        cost, _ = sair_erlang.sair_erlang(
            time_series=time_series,
            seed=seed,
            n_seeds=mc_nseed,
            t_total=t_total,
            n_t_steps=config.N_T_STEPS,
            metric="models.utils.utils.sum_sq"
        )
        cls.report_result(cost)


class SairNetworkEntrypoint(Entrypoint):
    NAME = "sair-network"
    CFG_CALLS = [net_sair.net_sair]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        cost, _ = net_sair.net_sair(
            time_series=time_series,
            seed=seed,
            n_seeds=mc_nseed,
            t_total=t_total,
            metric="models.utils.utils.sum_sq",
        )
        cls.report_result(cost)


class SeairEntrypoint(Entrypoint):
    NAME = "seair"
    CFG_CALLS = [seair.seair]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        cost, _ = seair.seair(
            time_series=time_series,
            seed=seed,
            n_seeds=mc_nseed,
            t_total=t_total,
            n_t_steps=config.N_T_STEPS,
            metric="models.utils.utils.sum_sq"
        )
        cls.report_result(cost)


class SeipahrfEntrypoint(Entrypoint):
    NAME = "seipahrf"
    CFG_CALLS = [seipahrf.seipahrf]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        del seed  # Not used

        cost, _ = seipahrf.seipahrf(
            time_series=time_series,
            t_total=t_total,
            metric="models.utils.utils.sum_sq",
        )
        cls.report_result(cost)


class SirEntrypoint(Entrypoint):
    NAME = "sir"
    CFG_CALLS = [sir.sir]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        cost, _ = sir.sir(
            time_series=time_series,
            seed=seed,
            n_seeds=mc_nseed,
            t_total=t_total,
            n_t_steps=config.N_T_STEPS,
            metric="models.utils.utils.sum_sq"
        )
        cls.report_result(cost)


class SirErlangEntrypoint(Entrypoint):
    NAME = "sir-erlang"
    CFG_CALLS = [sir_erlang.sir_erlang]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        cost, _ = sir_erlang.sir_erlang(
            time_series=time_series,
            seed=seed,
            n_seeds=mc_nseed,
            t_total=t_total,
            n_t_steps=config.N_T_STEPS,
            metric="models.utils.utils.sum_sq"
        )
        cls.report_result(cost)


class SirNetworkEntrypoint(Entrypoint):
    NAME = "sir-network"
    CFG_CALLS = [net_sir.net_sir]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        cost, _ = net_sir.net_sir(
            time_series=time_series,
            seed=seed,
            n_seeds=mc_nseed,
            t_total=t_total,
            n_t_steps=config.N_T_STEPS,
            metric="models.utils.utils.sum_sq"
        )
        cls.report_result(cost)


class SirdEntrypoint(Entrypoint):
    NAME = "sird"
    CFG_CALLS = [sird.sird]

    @classmethod
    def _run_model(cls, time_series, seed, t_total, mc_nseed):
        cost, _ = sird.sird(
            time_series=time_series,
            seed=seed,
            n_seeds=mc_nseed,
            t_total=t_total,
            n_t_steps=config.N_T_STEPS,
            metric="models.utils.utils.sum_sq"
        )
        cls.report_result(cost)


_entrypoints = {
    e.NAME: e for e in [
        SairEntrypoint,
        SairErlangEntrypoint,
        SairNetworkEntrypoint,
        SeairEntrypoint,
        SeipahrfEntrypoint,
        SirEntrypoint,
        SirErlangEntrypoint,
        SirNetworkEntrypoint,
        SirdEntrypoint,
    ]
}


def get_entrypoint_for_model(model) -> Entrypoint:
    return _entrypoints[model]


def get_available_models():
    return _entrypoints.keys()
