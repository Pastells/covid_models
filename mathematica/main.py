from argparse import ArgumentParser
import os.path

import pandas
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr


def sird(data, n, initial_infected, initial_recovered, initial_dead, p1, p2, p3):
    data["Susceptible"] = n - data["Infected"] - data["Dead"] - data["Recovered"]

    # Reorder
    data = data[["Day", "Susceptible", "Infected", "Dead", "Recovered"]]
    # print(data.head())

    # TODO: WolframClient should be able to use Numpy and Pandas,
    # but I don't know yet how to do it.
    data = data.values.tolist()

    rates = [p1, p2, p3]

    with WolframLanguageSession("/usr/local/bin/WolframKernel") as session:
        session.evaluate(wl.Needs("Sird`"))
        cost = session.evaluate(
            wl.Sird.Score(*rates,
                data, n,
                initial_infected, initial_recovered, initial_dead
            )
        )

    print(f"GGA SUCCESS {cost}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=45*10**6)
    parser.add_argument("--initial-infected", type=int, default=70)
    parser.add_argument("--initial-recovered", type=int, default=0)
    parser.add_argument("--initial-dead", type=int, default=0)

    parser.add_argument("--p1", type=float, default=0.05)
    parser.add_argument("--p2", type=float, default=0.05)
    parser.add_argument("--p3", type=float, default=0.05)

    return parser.parse_args()


def main():
    args = parse_args()
    data = pandas.read_csv(args.data)
    n = 45 * 10**6

    sird(data, n, args.initial_infected, args.initial_recovered, args.initial_dead, args.p1, args.p2, args.p3)


if __name__ == "__main__":
    main()

