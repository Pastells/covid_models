import pandas as pd

get_list = [1361, 1366, 1374, 1367, 1384, 1377]
names = ["SIR", "SIR-Erlang", "SEIR", "SAIR", "SAIR-Erlang", "SEAIR"]
RESULTS = "results/ac/"
params = {"delimiter": "|", "header": 0, "names": ["params", "values"]}
df_vec = []

for get in get_list:
    df = pd.read_csv(RESULTS + str(get) + ".get", **params)
    df.dropna(inplace=True)
    df["params"] = df["params"].str.strip()
    df_vec.append(df)


models = pd.concat(df_vec, keys=names, axis=0).reset_index(level=0)
models.rename(columns={"level_0": "models"}, inplace=True)
models = models.pivot(columns="params", index="models").round(2)
models = models.reindex(names)
models.columns = [col[1] for col in models.columns.values]


models.fillna(0, inplace=True)
models["n"] = models["n"] / 1000
models = models.astype({"k_asym": int, "k_inf": int, "k_rec": int, "n": int})


models = models[
    [
        "n",
        "beta",
        "beta_a",
        "delta",
        "delta_a",
        "alpha",
        "epsilon",
        "k_inf",
        "k_rec",
        "k_asym",
    ]
]
models


print(models.to_latex())
