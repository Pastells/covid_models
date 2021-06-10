import sys
from oct2py import Oct2Py
from optilog.autocfg import ac, Real

# fmt: off


@ac
def sidarthe(
    alfa1, beta1, gamma1, epsilon1, theta1,
    zeta1, mu1, nu1, tau1, lambda1,
    kappa1, alfa2, beta2, gamma2,
    epsilon3, alfa4, beta4, gamma4,
    mu4, nu4, zeta4, lambda4, rho4,
    alfa5, gamma5, epsilon6, rho6,
    sigma6, zeta6,
):
    """
    delta = beta
    eta = zeta
    rho = lambd
    xi = kappa
    sigma = kappa
    """

    params1 = [
        alfa1, beta1, gamma1, beta1, epsilon1,
        theta1, zeta1, zeta1, mu1, nu1, tau1,
        lambda1, lambda1, kappa1, kappa1, kappa1,
    ]

    """
    delta = beta
    """

    params2 = [alfa2, beta2, gamma2, beta2]

    params3 = [epsilon3]

    """
    delta = beta
    eta = zeta
    kappa = rho
    xi = rho
    sigma = rho
    """

    params4 = [
        alfa4, beta4, gamma4, beta4, mu4,
        nu4, zeta4, zeta4, lambda4, rho4,
        rho4, rho4, rho4,
    ]

    params5 = [alfa5, gamma5]

    """
    kappa = rho
    xi = rho
    eta = zeta
    """

    params6 = [
        epsilon6, rho6, rho6, rho6,
        sigma6, zeta6, zeta6,
    ]

    oc = Oct2Py()
    oc.addpath("models/sidarthe")
    cost = oc.sidarthe(params1, params2, params3, params4, params5, params6)
    sys.stdout.write(f"GGA SUCCESS {cost}\n")


def main(args):

    sidarthe(
        args.alfa1, args.beta1, args.gamma1, args.epsilon1, args.theta1,
        args.zeta1, args.mu1, args.nu1, args.tau1, args.lambda1,
        args.kappa1, args.alfa2, args.beta2, args.gamma2,
        args.epsilon3, args.alfa4, args.beta4, args.gamma4,
        args.mu4, args.nu4, args.zeta4, args.lambda4, args.rho4,
        args.alfa5, args.gamma5, args.epsilon6, args.rho6,
        args.sigma6, args.zeta6,
        )
# fmt: on
