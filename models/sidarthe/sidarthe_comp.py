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
        --epsilon6 0.2 --rho6 0.02 --sigma6 0.01 --zeta6 0.025
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

    parser.add_argument("--alfa1c", type=float)
    parser.add_argument("--beta1c", type=float)
    parser.add_argument("--gamma1c", type=float)
    parser.add_argument("--epsilon1c", type=float)
    parser.add_argument("--theta1c", type=float)
    parser.add_argument("--zeta1c", type=float)
    parser.add_argument("--mu1c", type=float)
    parser.add_argument("--nu1c", type=float)
    parser.add_argument("--tau1c", type=float)
    parser.add_argument("--lambda1c", type=float)
    parser.add_argument("--kappa1c", type=float)
    parser.add_argument("--alfa2c", type=float)
    parser.add_argument("--beta2c", type=float)
    parser.add_argument("--gamma2c", type=float)
    parser.add_argument("--epsilon3c", type=float)
    parser.add_argument("--alfa4c", type=float)
    parser.add_argument("--beta4c", type=float)
    parser.add_argument("--gamma4c", type=float)
    parser.add_argument("--zeta4c", type=float)
    parser.add_argument("--mu4c", type=float)
    parser.add_argument("--nu4c", type=float)
    parser.add_argument("--lambda4c", type=float)
    parser.add_argument("--rho4c", type=float)
    parser.add_argument("--alfa5c", type=float)
    parser.add_argument("--gamma5c", type=float)
    parser.add_argument("--epsilon6c", type=float)
    parser.add_argument("--rho6c", type=float)
    parser.add_argument("--sigma6c", type=float)
    parser.add_argument("--zeta6c", type=float)

    parser.add_argument("--data", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--timeout", type=int)

    return parser.parse_args()


# --------------------


def get_params(args):

    """
    delta = beta
    eta = zeta
    rho = lambd
    xi = kappa
    sigma = kappa
    """

    params1 = [
        args.alfa1,
        args.beta1,
        args.gamma1,
        args.beta1,
        args.epsilon1,
        args.theta1,
        args.zeta1,
        args.zeta1,
        args.mu1,
        args.nu1,
        args.tau1,
        args.lambda1,
        args.lambda1,
        args.kappa1,
        args.kappa1,
        args.kappa1,
    ]

    params1c = [
        args.alfa1c,
        args.beta1c,
        args.gamma1c,
        args.beta1c,
        args.epsilon1c,
        args.theta1c,
        args.zeta1c,
        args.zeta1c,
        args.mu1c,
        args.nu1c,
        args.tau1c,
        args.lambda1c,
        args.lambda1c,
        args.kappa1c,
        args.kappa1c,
        args.kappa1c,
    ]

    # --------------------

    """
    delta = beta
    """

    params2 = [args.alfa2, args.beta2, args.gamma2, args.beta2]
    params2c = [args.alfa2c, args.beta2c, args.gamma2c, args.beta2c]

    # --------------------

    params3 = [args.epsilon3]
    params3c = [args.epsilon3c]

    # --------------------

    """
    delta = beta
    eta = zeta
    kappa = rho
    xi = rho
    sigma = rho
    """

    params4 = [
        args.alfa4,
        args.beta4,
        args.gamma4,
        args.beta4,
        args.mu4,
        args.nu4,
        args.zeta4,
        args.zeta4,
        args.lambda4,
        args.rho4,
        args.rho4,
        args.rho4,
        args.rho4,
    ]

    params4c = [
        args.alfa4c,
        args.beta4c,
        args.gamma4c,
        args.beta4c,
        args.mu4c,
        args.nu4c,
        args.zeta4c,
        args.zeta4c,
        args.lambda4c,
        args.rho4c,
        args.rho4c,
        args.rho4c,
        args.rho4c,
    ]

    # --------------------

    params5 = [args.alfa5, args.gamma5]
    params5c = [args.alfa5c, args.gamma5c]

    # --------------------

    """
    kappa = rho
    xi = rho
    eta = zeta
    """

    params6 = [
        args.epsilon6,
        args.rho6,
        args.rho6,
        args.rho6,
        args.sigma6,
        args.zeta6,
        args.zeta6,
    ]

    params6c = [
        args.epsilon6c,
        args.rho6c,
        args.rho6c,
        args.rho6c,
        args.sigma6c,
        args.zeta6c,
        args.zeta6c,
    ]
    # --------------------

    return (
        params1,
        params2,
        params3,
        params4,
        params5,
        params6,
        params1c,
        params2c,
        params3c,
        params4c,
        params5c,
        params6c,
    )


# --------------------


def main():
    args = parsing()
    (
        params1,
        params2,
        params3,
        params4,
        params5,
        params6,
        params1c,
        params2c,
        params3c,
        params4c,
        params5c,
        params6c,
    ) = get_params(args)

    oc = Oct2Py()
    cost = oc.sidarthe_comp(
        params1,
        params2,
        params3,
        params4,
        params5,
        params6,
        params1c,
        params2c,
        params3c,
        params4c,
        params5c,
        params6c,
    )

    sys.stdout.write(f"GGA SUCCESS {cost}\n")


# --------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        sys.stdout.write(f"{repr(ex)}\n")
        sys.stdout.write(f"GGA CRASHED {1e20}\n")
        traceback.print_exc()
