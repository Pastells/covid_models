import os.path
import re
import shutil
import sys

import numpy as np

from optilog.autocfg.configurators import SMACConfigurator

# this is required as running > if __name__ == "__main__"
# from inside the module itself is an antipattern and we
# must force the path to the project top-level module
PACKAGE_PARENT = '../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from models.sird.configurable import sird
from models.utils import config

RESULT_REGEX = r"Result: ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"

DATA_BASE_PATH = "/home/saas/test_optilog/sird_china/data/"
RUNSOLVER_PATH = "/home/saas/opt/runsolver/runsolver-3.4"


# Entrypoint
def auto_sird(dataset, seed):
    # Constants:
    day_min = 0
    day_max = 54
    mc_nseed = 100

    t_total = day_max - day_min
    time_series = np.loadtxt(dataset, delimiter=",", dtype=int, usecols=(0, 1, 2, 3))
    time_series = time_series[day_min:day_max]

    cost = sird(
        time_series=time_series,
        seed=seed,
        n_seeds=mc_nseed,
        t_total=t_total,
        n_t_steps=config.N_T_STEPS,
        metric="models.utils.utils.sum_sq"
    )
    print(f"Result: {cost}")
    print(re.match(RESULT_REGEX, f"Result: {cost}"))


def create_scenario(scenario_path):
    smac_kwargs = {
        "cutoff": 300,
        "wallclock_limit": 24*60*60
    }
    data = [
        os.path.join(DATA_BASE_PATH, "china.dat"),
    ]

    configurator = SMACConfigurator(
        entrypoint=auto_sird,
        global_cfgcalls=[sird],
        input_data=data,
        runsolver_path=RUNSOLVER_PATH,
        memory_limit=6*1024,
        run_obj="quality",
        data_kwarg="dataset",
        seed_kwarg="seed",
        quality_regex=RESULT_REGEX,
        **smac_kwargs
    )
    configurator.generate_scenario(scenario_path)


def remove_old_scenario(scenario_path):
    try:
        shutil.rmtree(scenario_path)
    except FileNotFoundError:
        pass


def main():
    scenario_path = sys.argv[1]
    scenario_path = os.path.abspath(scenario_path)
    print(f"Creating scenario at {scenario_path}")

    # OPTIONAL remove old scenario
    # remove_old_scenario(scenario_path)

    create_scenario(scenario_path)


if __name__ == "__main__":
    main()
