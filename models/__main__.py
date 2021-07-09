import sys
import traceback
from argparse import ArgumentParser

from .utils import config

from .sir import (
    sir,
    sir_erlang,
    sir_erlang_sections,
    net_sir,
    net_sir_sections,
    sir_ode,
    sir_sections_ode,
)
from .sird import sird, sird_discrete, sird_parallel
from .sair import (
    sair,
    net_sair,
    net_sair_sections,
    sair_erlang,
    sair_erlang_sections,
    sair_sections_ode,
)
from .seair import seair
from .sidarthe import sidarthe, sidarthe_sections

# fmt: off


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
            "--initial_infected", type=int, default=config.initial_infected,
            help="initial number of infected individuals, if None is specified"
                 "is set to first day of input data"
        )
        group.add_argument(
            "--initial_recovered", type=int, default=config.initial_recovered,
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
            help="specify a name for outputfile",
        )
        group.add_argument(
            "--limit_memory", action="store_true",
            help="specify to limit process memory in order to avoid memory errors.",
        )


# Basic models (SIR)
class SirParser(CommonParser):
    # n(), sir()
    @classmethod
    def initialize_parameters_group(cls, group):
        # N
        group.add_argument(
            "--n", type=int, default=config.N,
            help="fixed number of (effective) individuals [1000, 1000000]",
        )
        # Sir
        group.add_argument(
            "--delta",
            type=float,
            default=config.DELTA,
            help="rate of recovery from infected phase (i->r) [0.05,1.0]",
        )
        group.add_argument(
            "--beta",
            type=float,
            default=config.BETA,
            help="infectivity due to infected [0.05,1.0]",
        )


class NetworkSirParser(SirParser):
    # + network()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
        # Newtork
        group.add_argument(
            "--network", type=str, choices=["er", "ba"], default=config.NETWORK,
            help="Erdos-Renyi or Barabasi-Albert {er,ba}",
        )
        group.add_argument(
            "--network_param", type=int, default=config.NETWORK_PARAM,
            help="mean number of edges [1, 50]",
        )


class NetworkSirSectionsParser(CommonParser):
    # n_sections(), sir_sections(), network()
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
        # Network:
        group.add_argument(
            "--network", type=str, choices=["er", "ba"], default=config.NETWORK,
            help="Erdos-Renyi or Barabasi-Albert {er,ba}",
        )
        group.add_argument(
            "--network_param", type=int, default=config.NETWORK_PARAM,
            help="mean number of edges [1,50]",
        )


class SirErlangParser(SirParser):
    # + erlang()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
        # Erlang
        group.add_argument(
            "--k_rec", type=int, default=config.K_REC,
            help="k for the recovery time erlang distribution [1, 5]"
        )
        group.add_argument(
            "--k_inf", type=int, default=config.K_INF,
            help="k for the infection time erlang distribution [1, 5]",
        )


class SirErlangSectionsParser(CommonParser):
    # n_sections(), sir_sections(), erlang()
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
        # Erlang
        group.add_argument(
            "--k_rec", type=int, default=config.K_REC,
            help="k for the recovery time erlang distribution [1, 5]"
        )
        group.add_argument(
            "--k_inf", type=int, default=config.K_INF,
            help="k for the infection time erlang distribution [1, 5]",
        )


# SAIR versions:
class SairParser(SirParser):
    # + .asymptomatic()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
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

    @classmethod
    def initialize_initial_conditions_group(cls, group):
        super().initialize_initial_conditions_group(group)
        group.add_argument(
            "--initial_asymptomatic", type=int, default=config.initial_asymptomatic,
            help="initial number of asymptomatic individuals, if None is"
                 "specified is set to first day of input data",
        )


class NetworkSairParser(NetworkSirParser):
    # + .asymptomatic()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
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

    @classmethod
    def initialize_initial_conditions_group(cls, group):
        super().initialize_initial_conditions_group(group)
        group.add_argument(
            "--initial_asymptomatic", type=int, default=config.initial_asymptomatic,
            help="initial number of asymptomatic individuals, if None is"
                 "specified is set to first day of input data",
        )


class NetworkSairSectionsParser(NetworkSirSectionsParser):
    # + asymptomatic_sections()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
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

    @classmethod
    def initialize_initial_conditions_group(cls, group):
        super().initialize_initial_conditions_group(group)
        group.add_argument(
            "--initial_asymptomatic", type=int, default=config.initial_asymptomatic,
            help="initial number of asymptomatic individuals if None is"
                 "specified is set to first day of input data",
        )


class SairErlangParser(SairParser):
    # + .erlang(True)
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
        group.add_argument(
            "--k_rec", type=int, default=config.K_REC,
            help="k for the recovery time erlang distribution [1, 5]"
        )
        group.add_argument(
            "--k_inf", type=int, default=config.K_INF,
            help="k for the infection time erlang distribution [1, 5]",
        )
        group.add_argument(
            "--k_asym", type=int, default=config.K_ASYM,
            help="k for the asymptomatic time erlang distribution [1, 5]",
        )


class SairErlangSectionsParser(SirErlangSectionsParser):
    # + asymptomatic_sections()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
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
        group.add_argument(
            "--k_asym", type=int, default=config.K_ASYM,
            help="k for the asymptomatic time erlang distribution [1, 5]",
        )

    @classmethod
    def initialize_initial_conditions_group(cls, group):
        super().initialize_initial_conditions_group(group)
        group.add_argument(
            "--initial_asymptomatic", type=int, default=config.initial_asymptomatic,
            help="initial number of asymptomatic individuals if None is"
                 "specified is set to first day of input data",
        )


# SEAIR versions:
class SeairParser(SairParser):
    # + .exposed()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
        group.add_argument(
            "--epsilon", type=float, default=config.EPSILON,
            help="latency rate (e->a) [0.2,1.0]",
        )

    @classmethod
    def initialize_initial_conditions_group(cls, group):
        super().initialize_initial_conditions_group(group)
        group.add_argument(
            "--initial_exposed", type=int, default=config.initial_exposed,
            help="initial number of latent individuals if None is specified"
                 "is set to first day of input data",
        )


# SIRD versions:
class SirdParser(SirParser):
    # + .dead()
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
        group.add_argument(
            "--theta", type=float, default=config.THETA,
            help="death probability [0.001,0.1]",
        )

    @classmethod
    def initialize_initial_conditions_group(cls, group):
        super().initialize_initial_conditions_group(group)
        group.add_argument(
            "--initial_dead", type=int, default=config.initial_dead,
            help="initial number of dead individuals",
        )


class SirdDiscreteParser(SirdParser):
    @classmethod
    def initialize_parameters_group(cls, group):
        super().initialize_parameters_group(group)
        group.add_argument(
            "--q",
            type=float,
            default=0.1,
            help="Ratio of the real population that is effective [0.002,1]",
        )
        group.add_argument(
            "--w",
            type=float,
            default=0.9,
            help="Forgetting factor, weight for cost sum (0,1]",
        )


class SirdParallelParser(SirdParser):
    @classmethod
    def initialize_configuration_group(cls, group):
        super().initialize_configuration_group(group)
        group.add_argument(
            "--sequential",
            action="store_true",
            help="specify for sequential execution, by default is parallel",
        )

    @classmethod
    def initialize_data_group(cls, group):
        super().initialize_data_group(group)
        group.add_argument(
            "--cost_day",
            type=int,
            default=None,
            help="At which day start computing the cost",
        )


# SIDARTHE
class SidartheParser(CommonParser):
    @classmethod
    def add_arguments(cls, parser, function_to_run):
        # noqa: E241
        parser.set_defaults(run_fn=function_to_run)
        parser.add_argument("--alfa1",    type=float, default=0.57)
        parser.add_argument("--beta1",    type=float, default=0.0114)
        parser.add_argument("--gamma1",   type=float, default=0.456)
        parser.add_argument("--epsilon1", type=float, default=0.171)
        parser.add_argument("--theta1",   type=float, default=0.3705)
        parser.add_argument("--zeta1",    type=float, default=0.1254)
        parser.add_argument("--mu1",      type=float, default=0.0171)
        parser.add_argument("--nu1",      type=float, default=0.0274)
        parser.add_argument("--tau1",     type=float, default=0.01)
        parser.add_argument("--lambda1",  type=float, default=0.0342)
        parser.add_argument("--kappa1",   type=float, default=0.0171)
        parser.add_argument("--alfa2",    type=float, default=0.4218)
        parser.add_argument("--beta2",    type=float, default=0.0057)
        parser.add_argument("--gamma2",   type=float, default=0.285)
        parser.add_argument("--epsilon3", type=float, default=0.1425)
        parser.add_argument("--alfa4",    type=float, default=0.36)
        parser.add_argument("--beta4",    type=float, default=0.005)
        parser.add_argument("--gamma4",   type=float, default=0.2)
        parser.add_argument("--zeta4",    type=float, default=0.034)
        parser.add_argument("--mu4",      type=float, default=0.008)
        parser.add_argument("--nu4",      type=float, default=0.015)
        parser.add_argument("--lambda4",  type=float, default=0.08)
        parser.add_argument("--rho4",     type=float, default=0.0171)
        parser.add_argument("--alfa5",    type=float, default=0.21)
        parser.add_argument("--gamma5",   type=float, default=0.11)
        parser.add_argument("--epsilon6", type=float, default=0.2)
        parser.add_argument("--rho6",     type=float, default=0.02)
        parser.add_argument("--sigma6",   type=float, default=0.01)
        parser.add_argument("--zeta6",    type=float, default=0.025)
        parser.add_argument("--data",     type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--timeout", type=int)
        parser.add_argument("--limit_memory", action="store_true")
        parser.add_argument("--day_min", type=int, default=config.DAY_MIN,
                            help="first day to consider of the data series")
        parser.add_argument("--day_max", type=int, default=config.DAY_MAX,
                            help="last day to consider of the data series")


# SIDARTHE sections
class SidartheSectionsParser(CommonParser):
    @classmethod
    def add_arguments(cls, parser, function_to_run):
        # noqa: E241
        parser.set_defaults(run_fn=function_to_run)
        parser.add_argument("--n", type=int, default=int(60e6))
        parser.add_argument("--n_old", type=int, default=0)
        parser.add_argument("--initial_infected", type=int, default=1000)
        parser.add_argument("--initial_diagnosed", type=int, default=1000)
        parser.add_argument("--initial_ailing", type=int, default=1000)
        parser.add_argument("--initial_recovered", type=int, default=1000)
        parser.add_argument("--alfa",    type=float, default=0.5)
        parser.add_argument("--beta",    type=float, default=0.5)
        parser.add_argument("--gamma",   type=float, default=0.5)
        parser.add_argument("--delta",   type=float, default=0.5)
        parser.add_argument("--epsilon", type=float, default=0.2)
        parser.add_argument("--theta",   type=float, default=0.2)
        parser.add_argument("--zeta",    type=float, default=0.2)
        parser.add_argument("--eta",     type=float, default=0.2)
        parser.add_argument("--mu",      type=float, default=0.02)
        parser.add_argument("--nu",      type=float, default=0.02)
        parser.add_argument("--tau",     type=float, default=0.02)
        parser.add_argument("--lambd",   type=float, default=0.2)
        parser.add_argument("--rho",     type=float, default=0.02)
        parser.add_argument("--kappa",   type=float, default=0.02)
        parser.add_argument("--xi",      type=float, default=0.02)
        parser.add_argument("--sigma",   type=float, default=0.02)
        parser.add_argument("--alfa_old",    type=float, default=0.5)
        parser.add_argument("--beta_old",    type=float, default=0.5)
        parser.add_argument("--gamma_old",   type=float, default=0.5)
        parser.add_argument("--delta_old",   type=float, default=0.5)
        parser.add_argument("--epsilon_old", type=float, default=0.2)
        parser.add_argument("--theta_old",   type=float, default=0.2)
        parser.add_argument("--zeta_old",    type=float, default=0.2)
        parser.add_argument("--eta_old",     type=float, default=0.2)
        parser.add_argument("--mu_old",      type=float, default=0.02)
        parser.add_argument("--nu_old",      type=float, default=0.02)
        parser.add_argument("--tau_old",     type=float, default=0.02)
        parser.add_argument("--lambda_old",  type=float, default=0.2)
        parser.add_argument("--rho_old",     type=float, default=0.02)
        parser.add_argument("--kappa_old",   type=float, default=0.02)
        parser.add_argument("--xi_old",      type=float, default=0.02)
        parser.add_argument("--sigma_old",   type=float, default=0.02)
        parser.add_argument("--data",     type=str)
        parser.add_argument("--seed", type=int)
        parser.add_argument("--timeout", type=int)
        parser.add_argument("--limit_memory", action="store_true")
        parser.add_argument("--day_min", type=int, default=config.DAY_MIN,
                            help="first day to consider of the data series")
        parser.add_argument("--day_max", type=int, default=config.DAY_MAX,
                            help="last day to consider of the data series")


def parse_args():
    parser = ArgumentParser(allow_abbrev=False)

    model_parser = parser.add_subparsers(help="model to use")

    models = [
        ("sair", SairParser, sair.main),
        ("sair-network", NetworkSairParser, net_sair.main),
        ("sair-network-sections", NetworkSairSectionsParser, net_sair_sections.main),
        ("sair-erlang", SairErlangParser, sair_erlang.main),
        ("sair-erlang-sections", SairErlangSectionsParser, sair_erlang_sections.main),
        ("sair-sections-ode", SairErlangSectionsParser, sair_sections_ode.main),

        ("seair", SeairParser, seair.main),

        ("sir", SirParser, sir.main),
        ("sir-network", NetworkSirParser, net_sir.main),
        ("sir-network-sections", NetworkSirSectionsParser, net_sir_sections.main),
        ("sir-erlang", SirErlangParser, sir_erlang.main),
        ("sir-erlang-sections", SirErlangSectionsParser, sir_erlang_sections.main),
        ("sir-ode", SirParser, sir_ode.main),
        ("sir-sections-ode", SirErlangSectionsParser, sir_sections_ode.main),

        ("sird", SirdParser, sird.main),
        ("sird-discrete", SirdDiscreteParser, sird_discrete.main),
        ("sird-parallel", SirdParallelParser, sird_parallel.main),

        ("sidarthe", SidartheParser, sidarthe.main),
        ("sidarthe-sections", SidartheSectionsParser, sidarthe_sections.main),
    ]

    for model_name, parser_class, run_fn in models:
        if parser_class is None:
            continue
        sub_parser = model_parser.add_parser(model_name)
        parser_class.add_arguments(sub_parser, run_fn)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit_memory:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (int(1024 ** 3 * 5.5), hard))

    try:
        args.run_fn(args)
    except MemoryError as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        sys.stdout.write("MemoryError in python\n")
        sys.stdout.write(f"GGA MEMOUT {1e20}\n")
    except Exception as ex:
        sys.stderr.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc()


if __name__ == "__main__":
    main()

# fmt: on
