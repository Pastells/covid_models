from argparse import ArgumentParser

from .utils import config
from .sird.sird import main as sird_main
from .seair.seair import main as seair_main
from .sair import sair, net_sair, net_sair_sections, sair_erlang, sair_erlang_sections


def run_sir(args):
    print("Using model sir")
    # Available:
    # - Network
    # - Network with sections
    # - Normal
    # - Erlang distribution
    # - Erlang distribution with sections
    raise NotImplementedError


class CommonParser:
    """
    Handle all different parsers in an incremental fashion, in order
    to avoid repetitions
    """

    @classmethod
    def add_arguments(cls, parser, function_to_run):
        parser.set_defaults(run_fn=function_to_run)

        parameters_group = parser.add_argument_group("parameters")
        initial_conditions_group = parser.add_argument_group("initial_conditions")
        configuration_group = parser.add_argument_group("configuration")
        data_group = parser.add_argument_group("data")
        actions_group = parser.add_argument_group("actions")

        cls.initialize_parameters_group(parameters_group)
        cls.initialize_initial_conditions_group(initial_conditions_group)
        cls.initialize_configuration_group(configuration_group)
        cls.initialize_data_group(data_group)
        cls.initialize_actions_group(actions_group)

    @classmethod
    def initialize_parameters_group(cls, group):
        pass

    @classmethod
    def initialize_initial_conditions_group(cls, group):
        group.add_argument(
            "--I_0", type=int, default=config.I_0,
            help="initial number of infected individuals, if None is specified"
                 "is set to first day of input data"
        )
        group.add_argument(
            "--R_0", type=int, default=config.R_0,
            help="initial number of inmune individuals",
        )

    @classmethod
    def initialize_configuration_group(cls, group):
        group.add_argument(
            "--seed", type=int, default=config.SEED,
            help="seed metaparameter for the automatic configuration, also used"
                 "as MC seed",
        )
        group.add_argument(
            "--timeout", type=int, default=config.TIMEOUT,
            help="timeout metaparameter for the automatic configuration",
        )
        group.add_argument(
            "--mc_nseed", type=int, default=config.MC_NSEED,
            help="number of mc realizations to average over",
        )
        group.add_argument(
            "--n_t_steps", type=int, default=config.N_T_STEPS,
            help="maximum number of simulation steps, dimension for the arrays",
        )

    @classmethod
    def initialize_data_group(cls, group):
        group.add_argument(
            "--data", type=str, required=True,
            help="file with time series",
        )
        group.add_argument(
            "--day_min", type=int, default=config.DAY_MIN,
            help="first day to consider of the data series",
        )
        group.add_argument(
            "--day_max", type=int, default=config.DAY_MAX,
            help="last day to consider of the data series",
        )
        group.add_argument(
            "--undiagnosed", type=float, default=config.UNDIAGNOSED,
            help="percentage of undiagnosed cases, used to rescale the data to"
                 "account for underreporting",
        )
        group.add_argument(
            "--metric", type=str, default=config.METRIC, choices=config.METRICS,
            help="metric to use to compute the cost function"
                 f"{config.METRICS_STR}",
            metavar="str",
        )

    @classmethod
    def initialize_actions_group(cls, group):
        group.add_argument(
            "--plot", action="store_true", help="specify for plots"
        )
        group.add_argument(
            "--save", type=str, default=None,
            help="specify a name for outputfile"
        )


class SairParser(CommonParser):
    # .n(), .sir(), .asymptomatic()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
        # N:
        group.add_argument(
            "--n", type=int, default=config.N,
            help="fixed number of (effective) individuals [1000,1000000]"
        )
        # Sir:
        group.add_argument(
            "--delta", type=float, default=config.DELTA,
            help="rate of recovery from infected phase (i->r) [0.05,1.0]",
        )
        group.add_argument(
            "--beta", type=float, default=config.BETA,
            help="infectivity due to infected [0.05,1.0]",
        )
        # Asymptomatic
        group.add_argument(
            "--A_0", type=int, default=config.A_0,
            help="initial number of asymptomatic individuals, if None is"
                 "specified is set to first day of input data",
        )
        group.add_argument(
            "--delta_a", type=float, default=config.DELTA_A,
            help="rate of recovery from asymptomatic phase (a->r) [0.05,1.0]",
        )
        group.add_argument(
            "--beta_a", type=float, default=config.BETA_A,
            help="infectivity due to asymptomatic [0.05,1.0]",
        )
        group.add_argument(
            "--alpha", type=float, default=config.ALPHA,
            help="asymptomatic rate (a->i) [0.05,2.0]",
        )


class NetworkSairParser(SairParser):
    # .network()
    @classmethod
    def initialize_parameters_group(cls, group):
        group.add_argument(
            "--network", type=str, choices=["er", "ba"], default=config.NETWORK,
            help="Erdos-Renyi or Barabasi-Albert {er,ba}",
        )
        group.add_argument(
            "--network_param", type=int, default=config.NETWORK_PARAM,
            help="mean number of edges [1,50]",
        )


class NetworkSairSections(CommonParser):
    # .n_sections(), sir_sections(), asymptomatic_sections(), .network()
    @classmethod
    def initialize_parameters_group(cls, group):
        # N sections:
        group.add_argument(
            "--n", type=int, default=[config.N], nargs="*",
            help="fixed number of (effecitve) individuals, initial and"
                 "increments [1000,1000000]",
        )
        group.add_argument(
            "--section_days", type=int, default=config.SECTIONS_DAYS,
            nargs="*",
            help="starting day for each section, first one must be 0, and"
                 "final day for last one",
        )
        group.add_argument(
            "--transition_days",
            type=int,
            default=config.TRANSITION_DAYS,
            help="days it takes to transition from one number of individuals \
                            to the next [1,10]",
        )
        # Sir sections:
        group.add_argument(
            "--delta", type=float, default=[config.DELTA], nargs="*",
            help="rate of recovery from infected phase (i->r) [0.05,1.0]",
        )
        group.add_argument(
            "--beta", type=float, default=[config.BETA], nargs="*",
            help="infectivity due to infected [0.05,1.0]",
        )
        # Asymptomatic sections
        group.add_argument(
            "--A_0", type=int, default=config.A_0,
            help="initial number of asymptomatic individuals if None is"
                 "specified is set to first day of input data",
        )
        group.add_argument(
            "--delta_a", type=float, default=[config.DELTA_A], nargs="*",
            help="rate of recovery from asymptomatic phase (a->r) [0.05,1.0]",
        )
        group.add_argument(
            "--beta_a", type=float, default=[config.BETA_A], nargs="*",
            help="infectivity due to asymptomatic [0.05,1.0]",
        )
        group.add_argument(
            "--alpha", type=float, default=[config.ALPHA], nargs="*",
            help="asymptomatic rate (a->i) [0.05,1]",
        )
        # Network:
        group.add_argument(
            "--network", type=str, choices=["er", "ba"], default=config.NETWORK,
            help="Erdos-Renyi or Barabasi-Albert {er,ba}",
        )
        group.add_argument(
            "--network_param", type=int, default=config.NETWORK_PARAM,
            help="mean number of edges [1,50]",
        )


class SeairParser(CommonParser):
    # .n(), .sir(), .exposed(), .asymptomatic()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
        # N:
        group.add_argument(
            "--n", type=int, default=config.N,
            help="fixed number of (effective) individuals [1000,1000000]"
        )
        # Sir:
        group.add_argument(
            "--delta", type=float, default=config.DELTA,
            help="rate of recovery from infected phase (i->r) [0.05,1.0]",
        )
        group.add_argument(
            "--beta", type=float, default=config.BETA,
            help="infectivity due to infected [0.05,1.0]",
        )
        # Exposed
        group.add_argument(
            "--E_0", type=int, default=config.E_0,
            help="initial number of latent individuals if None is specified"
                 "is set to first day of input data",
        )
        group.add_argument(
            "--epsilon", type=float, default=config.EPSILON,
            help="latency rate (e->a) [0.2,1.0]",
        )
        # Asymptomatic
        group.add_argument(
            "--A_0", type=int, default=config.A_0,
            help="initial number of asymptomatic individuals, if None is"
                 "specified is set to first day of input data",
        )
        group.add_argument(
            "--delta_a", type=float, default=config.DELTA_A,
            help="rate of recovery from asymptomatic phase (a->r) [0.05,1.0]",
        )
        group.add_argument(
            "--beta_a", type=float, default=config.BETA_A,
            help="infectivity due to asymptomatic [0.05,1.0]",
        )
        group.add_argument(
            "--alpha", type=float, default=config.ALPHA,
            help="asymptomatic rate (a->i) [0.05,2.0]",
        )


class SirParser(CommonParser):
    pass


class SirdParser(CommonParser):
    # .n(), .sir(), .dead()
    @classmethod
    def initialize_parameters_group(cls, group):
        # N:
        super().initialize_parameters_group(group)
        group.add_argument(
            "--n", type=int, default=config.N,
            help="fixed number of (effective) individuals [1000,1000000]"
        )
        # Sir:
        group.add_argument(
            "--delta", type=float, default=config.DELTA,
            help="rate of recovery from infected phase (i->r) [0.05,1.0]",
        )
        group.add_argument(
            "--beta", type=float, default=config.BETA,
            help="infectivity due to infected [0.05,1.0]",
        )
        # Dead
        group.add_argument(
            "--D_0", type=int, default=config.D_0,
            help="initial number of dead individuals",
        )
        group.add_argument(
            "--theta", type=float, default=config.THETA,
            help="death probability [0.001,0.1]",
        )


def parse_args():
    parser = ArgumentParser(allow_abbrev=False)

    model_parser = parser.add_subparsers(help="model to use")

    models = [
        ("sair", SairParser, sair.main),
        ("sair-network", NetworkSairParser, net_sair.main),
        ("sair-network-sections", NetworkSairSections, net_sair_sections.main),
        ("sair-erlang", None, sair_erlang.main),
        ("sair-erlang-sections", None, sair_erlang_sections.main),

        ("seair", SeairParser, seair_main),

        ("sir", SirParser, run_sir),
        ("sir-network", None, None),
        ("sir-network-sections", None, None),
        ("sir-erlang", None, None),
        ("sir-erlang-sections", None, None),

        ("sird", SirdParser, sird_main),
    ]

    for model_name, parser_class, run_fn in models:
        if parser_class is None:
            continue
        sub_parser = model_parser.add_parser(model_name)
        parser_class.add_arguments(sub_parser, run_fn)

    return parser.parse_args()


def main():
    args = parse_args()
    args.run_fn(args)


if __name__ == "__main__":
    main()