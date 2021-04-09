import sys
import traceback
import argparse
from oct2py import Oct2Py

# --------------------


def parsing():
    """
    default:
        --alfa1 0.57 --beta1 0.0114 --gamma1 0.456 --epsilon1 0.171 \
        --theta1 0.3705 --zeta1 0.1254 --mu1 0.0171 --nu1 0.0274 \
        --tau1 0.01 --lambda1 0.0342 --kappa1 0.0171 --alfa2 0.4218 \
        --beta2 0.0057 --gamma2 0.285 --epsilon3 0.1425 --alfa4 0.36 \
        --beta4 0.005 --gamma4 0.2 --mu4 0.008 --nu4 0.015 --zeta4 0.034 \
        --lambda4 0.08 --rho4 0.0171 --alfa5 0.21 --gamma5 0.11 \
        --epsilon6 0.2 --rho6 0.02 --sigma6 0.01 --zeta6 0.025 \
        --delta1 0.0114 --eta1 0.1254 --rho1 0.0342  --xi1 0.0171 \
        --sigma1 0.0171 --delta2 0.0057 --delta4 0.005 --eta4 0.034 \
        --kappa4 0.0171 --xi4 0.0171 --sigma4 0.0171 --kappa6 0.02 \
        --xi6 0.02 --eta6 0.025
    """
    parser = argparse.ArgumentParser(
        description="python parser for sidarthe code",
        formatter_class=argparse.MetavarTypeHelpFormatter,
    )
    parser.add_argument("--alfa1", type=float)
    parser.add_argument("--beta1", type=float)
    parser.add_argument("--gamma1", type=float)
    parser.add_argument("--epsilon1", type=float)
    parser.add_argument("--theta1", type=float)
    parser.add_argument("--zeta1", type=float)
    parser.add_argument("--mu1", type=float)
    parser.add_argument("--nu1", type=float)
    parser.add_argument("--tau1", type=float)
    parser.add_argument("--lambda1", type=float)
    parser.add_argument("--kappa1", type=float)
    parser.add_argument("--alfa2", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--gamma2", type=float)
    parser.add_argument("--epsilon3", type=float)
    parser.add_argument("--alfa4", type=float)
    parser.add_argument("--beta4", type=float)
    parser.add_argument("--gamma4", type=float)
    parser.add_argument("--zeta4", type=float)
    parser.add_argument("--mu4", type=float)
    parser.add_argument("--nu4", type=float)
    parser.add_argument("--lambda4", type=float)
    parser.add_argument("--rho4", type=float)
    parser.add_argument("--alfa5", type=float)
    parser.add_argument("--gamma5", type=float)
    parser.add_argument("--epsilon6", type=float)
    parser.add_argument("--rho6", type=float)
    parser.add_argument("--sigma6", type=float)
    parser.add_argument("--zeta6", type=float)

    parser.add_argument("--data", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int)

    parser.add_argument("--delta1", type=float)
    parser.add_argument("--eta1", type=float)
    parser.add_argument("--rho1", type=float)
    parser.add_argument("--xi1", type=float)
    parser.add_argument("--sigma1", type=float)
    parser.add_argument("--delta2", type=float)
    parser.add_argument("--delta4", type=float)
    parser.add_argument("--eta4", type=float)
    parser.add_argument("--kappa4", type=float)
    parser.add_argument("--xi4", type=float)
    parser.add_argument("--sigma4", type=float)
    parser.add_argument("--kappa6", type=float)
    parser.add_argument("--xi6", type=float)
    parser.add_argument("--eta6", type=float)
    return parser.parse_args()


# --------------------


def get_params(args):

    params1 = [
        args.alfa1,
        args.beta1,
        args.gamma1,
        args.delta1,
        args.epsilon1,
        args.theta1,
        args.zeta1,
        args.eta1,
        args.mu1,
        args.nu1,
        args.tau1,
        args.lambda1,
        args.rho1,
        args.kappa1,
        args.xi1,
        args.sigma1,
    ]

    # --------------------

    params2 = [args.alfa2, args.beta2, args.gamma2, args.delta2]

    # --------------------

    params3 = [args.epsilon3]

    # --------------------

    params4 = [
        args.alfa4,
        args.beta4,
        args.gamma4,
        args.delta4,
        args.mu4,
        args.nu4,
        args.zeta4,
        args.eta4,
        args.lambda4,
        args.rho4,
        args.kappa4,
        args.xi4,
        args.sigma4,
    ]

    # --------------------

    params5 = [args.alfa5, args.gamma5]

    # --------------------

    params6 = [
        args.epsilon6,
        args.rho6,
        args.kappa6,
        args.xi6,
        args.sigma6,
        args.zeta6,
        args.eta6,
    ]
    # --------------------

    return params1, params2, params3, params4, params5, params6


# --------------------


def main():
    args = parsing()
    params1, params2, params3, params4, params5, params6 = get_params(args)

    oc = Oct2Py()
    cost = oc.sidarthe(params1, params2, params3, params4, params5, params6)
    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# --------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc()
