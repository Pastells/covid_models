import os.path
import shutil
import sys

from optilog.autocfg.configurators import SMACConfigurator
from models import optilog_entrypoints


DATA_BASE_PATH = "/home/saas/test_optilog/sird_china/data/"
RUNSOLVER_PATH = "/home/saas/opt/runsolver/runsolver-3.4"


def create_smac_scenario(scenario_path, model):
    smac_kwargs = {
        "cutoff": 300,
        "wallclock_limit": 24 * 60 * 60
    }
    data = [
        os.path.join(DATA_BASE_PATH, "china.dat")
    ]

    entrypoint, cfg_calls = optilog_entrypoints.get_entrypoint_for_model(model)

    configurator = SMACConfigurator(
        entrypoint=entrypoint,
        global_cfgcalls=cfg_calls,
        input_data=data,
        runsolver_path=RUNSOLVER_PATH,
        memory_limit=6*1024,
        run_obj="quality",
        data_kwarg="dataset",
        seed_kwarg="seed",
        quality_regex=optilog_entrypoints.RESULT_REGEX,
        **smac_kwargs
    )
    configurator.generate_scenario(scenario_path)


def remove_old_scenario(scenario_path):
    try:
        shutil.rmtree(scenario_path)
    except FileNotFoundError:
        pass


def main():
    model = sys.argv[1]
    scenario_path = sys.argv[2]

    scenario_path = os.path.abspath(scenario_path)
    print(f"Creating scenario at {scenario_path}")

    # OPTIONAL remove old scenario
    # remove_old_scenario(scenario_path)
    create_smac_scenario(scenario_path, model)


if __name__ == "__main__":
    main()
