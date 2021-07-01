import argparse

# --------------------


def parsing():
    parser = argparse.ArgumentParser(
        description="""Parse params to tex for paper""",
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

    return parser.parse_args()


# --------------------


# fmt: off
def main():
    args = parsing()

    text = f"""At day 1 the parameters are set as $\\alpha={args.alfa1:.3f}$, $\\beta=\\delta={args.beta1:.3f}$, $\\gamma={args.gamma1:.3f}$, $\\epsilon={args.epsilon1:.3f}$, $\\theta={args.theta1:.3f}$, $\\zeta=\\eta={args.zeta1:.3f}$, $\\mu={args.mu1:.3f}$, $\\nu={args.nu1:.3f}$, $\\tau={args.tau1:.3f}$, $\\lambda=\\rho={args.lambda1:.3f}$ and $\\kappa=\\xi=\\sigma={args.kappa1:.3f}$.
After day 4, $\\alpha={args.alfa2:.3f}$, $\\beta=\\delta={args.beta2:.3f}$ and $\\gamma={args.gamma2:.3f}$.
After day 12, $\\epsilon={args.epsilon3:.3f}$.
After day 22, $\\alpha={args.alfa4:.3f}$, $\\beta=\\delta={args.beta4:.3f}$ and $\\gamma={args.gamma4:.3f}$; also, $\\zeta=\\eta={args.zeta4:.3f}$, $\\mu={args.mu4:.3f}$, $\\nu={args.nu4:.3f}$, $\\lambda={args.lambda4:.3f}$ and $\\rho=\\kappa=\\xi=\\sigma={args.rho4:.3f}$.
After day 28, $\\alpha={args.alfa5:.3f}$ and $\\gamma={args.gamma5:.3f}$.
Finally, after day 38, $\\epsilon={args.epsilon6:.3f}$, $\\rho=\\kappa=\\xi={args.rho6:.3f}$, $\\sigma={args.sigma6:.3f}$ and $\\zeta=\\eta={args.zeta6:.3f}$."""

    with open("params.tex", "w") as file:
        file.write(text)

# fmt: on

if __name__ == "__main__":
    main()
