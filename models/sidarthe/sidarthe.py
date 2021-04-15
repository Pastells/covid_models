import sys
from oct2py import Oct2Py


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

    # --------------------

    """
    delta = beta
    """

    params2 = [args.alfa2, args.beta2, args.gamma2, args.beta2]

    # --------------------

    params3 = [args.epsilon3]

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

    # --------------------

    params5 = [args.alfa5, args.gamma5]

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
    # --------------------

    return params1, params2, params3, params4, params5, params6


# --------------------


def main(args):
    params1, params2, params3, params4, params5, params6 = get_params(args)

    oc = Oct2Py()
    cost = oc.sidarthe(params1, params2, params3, params4, params5, params6)
    sys.stdout.write(f"GGA SUCCESS {cost}\n")
