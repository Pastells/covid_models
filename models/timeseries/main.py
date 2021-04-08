import sys
from argparse import ArgumentParser

from matplotlib import pyplot as plt
import numpy

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from optilog.autocfg import ac, Bool, Int, Real, Categorical, Choice

from .errors import *
from ..utils import utils, config
from ..utils import utils, config


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()

    for i in range(len(sequence)):
        end_idx = i + n_steps_in
        out_end_idx = end_idx + n_steps_out

        if out_end_idx > len(sequence):
            break

        X.append(sequence[i:end_idx])
        y.append(sequence[end_idx:out_end_idx])

    return numpy.array(X), numpy.array(y)


@ac
def timeserie(data: numpy.ndarray,
              n_steps_in: int,
              n_steps_out: int,
              kernel: Categorical("linear", "poly", "rbf", "sigmoid", "precomputed") = "rbf",
              degree: Int(0, 10) = 3,  # only for kernel=poly
              gamma: Choice(Categorical("scale", "auto", default="scale"), Real(0.0, 1.0, default=0.5)) = Categorical("scale", "auto", default="scale"),
              coef0: Real(0.0, 1.0) = 0.0,
              C: Real(0.0, 30.0) = 1.0,
              epsilon: Real(0.0, 1.0) = 1.0,
              shrinking: Bool = True,
              cache_size: Int(100, 300) = 200,
              max_iter: Choice(Int(-1, -1, default=-1), Int(1, 30, default=10)) = Int(-1, -1, default=-1)):
    X, y = split_sequence(data, n_steps_in, n_steps_out)

    regressor = SVR(
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        C=C,
        epsilon=epsilon,
        shrinking=shrinking,
        cache_size=cache_size,
        max_iter=max_iter
    )

    model = MultiOutputRegressor(regressor)

    model.fit(X, y)
    pred = model.predict(X)
    return pred


def entrypoint(data):
    n_steps_in = 10
    n_steps_out = 1
    pred = timeserie(data, n_steps_in=n_steps_in, n_steps_out=n_steps_out,
                     kernel="rbf", C=20, gamma=0.28, epsilon=0.0125)

    pred = pred.flatten()  # We predict one day only
    print(f"Result: {rmse(data[n_steps_in:], pred)}")

    days = numpy.arange(len(data))
    plt.plot(days, data, label="real")
    plt.plot(days[n_steps_in:], pred, label="pred")
    plt.show()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--day-min", default=config.DAY_MIN)
    parser.add_argument("--day-max", default=config.DAY_MAX)

    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Using dataset {args.data}")

    data = utils.get_time_series(args)
    entrypoint(data)


if __name__ == "__main__":
    main()
