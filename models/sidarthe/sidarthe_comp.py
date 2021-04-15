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


def main(args):
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
