import sys
from oct2py import Oct2Py


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


def main(args):
    params1, params2, params3, params4, params5, params6 = get_params(args)

    oc = Oct2Py()
    cost = oc.sidarthe(params1, params2, params3, params4, params5, params6)
    sys.stdout.write(f"GGA SUCCESS {cost}\n")
