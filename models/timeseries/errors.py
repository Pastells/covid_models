import math


def mae(real, predicted):
    """
    L’error absolut mitjà (mean absolute error, MAE)
    és la mètrica més simple i directa per a avaluar
    el grau de divergència entre dos conjunts de valors.

    En aquest cas, tots els residus tenen la mateixa
    contribució a l’error absolutfi-nal
    """
    return sum(
        abs(r - p) for r,p in zip(real, predicted)
    ) / len(real)


def mse(real, predicted):
    """
    L’error quadràtic mitjà (mean square error, MSE)
    és probablement la mètrica més emprada per a
    avaluar models de regressió.
    Utilitzant aquesta mètrica, es penalitzen els residus grans.
    """
    return sum(
        (r - p)**2 for r,p in zip(real, predicted)
    ) / len(real)


def rmse(real, predicted):
    """
    Mateix que MSE, però expressa l'error amb la mateixa
    escala que les dades
    """
    return math.sqrt(mse(real, predicted))


def mape(real, predicted):
    """
    L’error percentual absolut mitjà (mean absolute
    percentage error, MAPE) és una mesura de la
    precisió que s’utilitza com una funció de pèrdua
    per a problemes de regressió en l’aprenentatge automàtic.
    """
    return sum(
        abs(r - p) / r for r,p in zip(real, predicted)
    ) * 100 / len(predicted)

