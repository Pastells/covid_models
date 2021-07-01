from argparse import ArgumentParser
import os.path
import shlex
import shutil
import subprocess
import sys

from optilog.autocfg.configurators import SMACConfigurator
from models import optilog_entrypoints


RUNSOLVER_PATH = shutil.which(
    "runsolver"
)  # Note: change if the binary is not in the path.


def create_smac_scenario(scenario_path, model, data, smac_kwargs):
    entrypoint, cfg_calls = optilog_entrypoints.get_entrypoint_for_model(model)

    configurator = SMACConfigurator(
        entrypoint=entrypoint,
        global_cfgcalls=cfg_calls,
        input_data=list(data),
        runsolver_path=RUNSOLVER_PATH,
        memory_limit=6 * 1024,
        run_obj="quality",
        data_kwarg="dataset",
        seed_kwarg="seed",
        quality_regex=optilog_entrypoints.RESULT_REGEX,
        **smac_kwargs,
    )
    configurator.generate_scenario(scenario_path)


def remove_old_scenario(scenario_path):
    try:
        shutil.rmtree(scenario_path)
    except FileNotFoundError:
        pass


def run(scenario_path):
    command = shlex.split(f"smac --scenario {scenario_path}/scenario.txt")
    subprocess.run(command)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "model",
        help="The model to configure",
        choices=optilog_entrypoints.get_available_models(),
    )
    parser.add_argument(
        "scenario_path", help="The path where the SMAC scenario will be created"
    )
    parser.add_argument(
        "--data",
        help="The data files that will be used to configure the"
        " model. At least one is required",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--remove-old",
        help="If set, remove the scenario_path before creating" " the new one",
        action="store_true",
    )
    parser.add_argument(
        "--run",
        help="Run SMAC to configure the model after the" " scenario is created",
        action="store_true",
    )
    parser.add_argument(
        "--cutoff",
        help="The time limit (in seconds) set to each execution"
        " for a new configuration tested. By default it"
        " is set to 300 (5 minutes)",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--time-limit",
        help="The global time limit for the configuration"
        " process, in seconds. By default set to 86400"
        " (i.e. one full day).",
        type=int,
        default=86400,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Creating scenario at {args.scenario_path}", file=sys.stderr)

    if args.remove_old:
        print("Deleting old scenario...", file=sys.stderr)
        remove_old_scenario(args.scenario_path)

    data = map(os.path.realpath, args.data)

    smac_kwargs = {"cutoff": args.cutoff, "wallclock_limit": args.time_limit}

    create_smac_scenario(args.scenario_path, args.model, data, smac_kwargs)

    if args.run:
        print("Run the scenario...", file=sys.stderr)
        run(args.scenario_path)


if __name__ == "__main__":
    main()
