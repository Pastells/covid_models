from argparse import ArgumentParser
import os.path
import shlex
import shutil
import subprocess
import sys
from typing import Dict
from inspect import signature

from optilog.autocfg.configurators import SMACConfigurator, GGAConfigurator
from optilog.autocfg.basis import Parameter as ConfigurableParameter
from models import optilog_entrypoints

RUNSOLVER_PATH = shutil.which(
    "runsolver"
)  # Note: change if the binary is not in the path.


def get_number_of_configurable_parameters(entrypoint):
    count = 0
    for configurable in entrypoint.CFG_CALLS:
        sig = signature(configurable)
        for par in sig.parameters.values():
            if isinstance(par.annotation, ConfigurableParameter):
                count += 1
    return count
    

def create_scenario(
    configurator_choice,
    scenario_path,
    model,
    dataset,
    day_min,
    day_max,
    mc_nseed,
    configurator_kwargs
):
    entrypoint = optilog_entrypoints.get_entrypoint_for_model(model)

    if configurator_choice == "gga":
        # Mutate at least one parameter
        n_params = get_number_of_configurable_parameters(entrypoint)
        min_mutation_prob = 1 / n_params
        current_mut = configurator_kwargs["mutation_probability"]
        new_mut = max(current_mut, min_mutation_prob)
        configurator_kwargs["mutation_probability"] = new_mut

    data_file = os.path.realpath(os.path.join(scenario_path, "data.dat"))

    if configurator_choice == "gga":
        configurator_class = GGAConfigurator
    elif configurator_choice == "smac":
        configurator_class = SMACConfigurator
    else:
        raise ValueError(f"Unknown configurator {configurator_choice}")

    configurator = configurator_class(
        entrypoint=entrypoint.entrypoint,
        global_cfgcalls=entrypoint.CFG_CALLS,
        input_data=[data_file],
        runsolver_path=RUNSOLVER_PATH,
        memory_limit=20 * 1024,
        run_obj="quality",
        data_kwarg="data",
        seed_kwarg="seed",
        quality_regex=optilog_entrypoints.Entrypoint.RESULT_REGEX,
        **configurator_kwargs,
    )
    configurator.generate_scenario(scenario_path)

    entrypoint.create_data(
        data_file,
        os.path.realpath(dataset),
        day_min,
        day_max,
        mc_nseed
    )


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
        "configurator",
        choices=["gga", "smac"],
        help="The configurator that will be used during the solving process",
    )
    parser.add_argument(
        "--data",
        help="The data file that will be used to configure the model.",
        required=True
    )
    parser.add_argument(
        "--day-min",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--day-max",
        type=int,
        default=54
    )
    parser.add_argument(
        "--mc-nseed",
        type=int,
        default=100
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


def get_configurator_kwargs(configurator, args) -> Dict:
    if configurator == "gga":
        return {
            "population": 50,  # default 25
            "generations": 500,  # default 30
            "min_generations": 490,  # default 0
            "max_age": 3,
            # "rand_replace_prob": None,
            "eval_group_size": 8,
            "eval_time_limit": args.cutoff,
            "tuner_rt_limit": args.time_limit,
            # "tuner_evals_limit": None,
            "winners_percentage": 0.1,
            "mutation_probability": 0.1,
            "sigma_percentage": 1.0,
            "crossover_operator": "gga",
            "use_elite_group": True,
            "objective": "sum",
            "cost_min": 0,
            "cost_max": optilog_entrypoints.Entrypoint.MAX_COST,
            "cost_tolerance": 0.0,
            # "cancel": None,
            # "cancel_min_evals": None,
            "instances_selector": "rlinear",
            "instances_min": 1,
            "instances_max": 0,
            "instances_gen_max": -2,
            "seed": 42,
        }
    elif configurator == "smac":
        return {"cutoff": args.cutoff, "wallclock_limit": args.time_limit}
    else:
        raise ValueError(f"Unknown configurator {configurator}")


def main():
    args = parse_args()
    print(f"Creating scenario at {args.scenario_path}", file=sys.stderr)

    if args.remove_old:
        print("Deleting old scenario...", file=sys.stderr)
        remove_old_scenario(args.scenario_path)

    configurator_kwargs = get_configurator_kwargs(args.configurator, args)
    create_scenario(
        args.configurator, args.scenario_path,
        args.model, args.data, args.day_min, args.day_max, args.mc_nseed,
        configurator_kwargs
    )

    if args.run:
        print("Run the scenario...", file=sys.stderr)
        run(args.scenario_path)


if __name__ == "__main__":
    main()
