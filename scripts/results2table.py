"""Convert AC 'get' output of several problems into LaTex table"""
import pandas as pd

# Results folder
RESULTS = "results/ac/"

# List of configurator problems and their name
get_list = [1361, 1366, 1374, 1367, 1384, 1377]
names = ["SIR", "SIR-Erlang", "SEIR", "SAIR", "SAIR-Erlang", "SEAIR"]

args = {"delimiter": "|", "header": 0, "names": ["params", "values"]}
df_vec = []

# convert 'get' output to a DataFrame
for get in get_list:
    df = pd.read_csv(RESULTS + str(get) + ".get", **args)
    df.dropna(inplace=True)
    df["params"] = df["params"].str.strip()
    df_vec.append(df)

# Join DFs and format
models = pd.concat(df_vec, keys=names, axis=0).reset_index(level=0)
models.rename(columns={"level_0": "models"}, inplace=True)
models = models.pivot(columns="params", index="models").round(2)
models = models.reindex(names)
models.columns = [col[1] for col in models.columns.values]

# Integer columns, missing values as '-'
models.fillna(0, inplace=True)
models["n"] = models["n"] / 1000
models = models.astype({"k_asym": int, "k_inf": int, "k_rec": int, "n": int})
models.replace(to_replace=0, value="-", inplace=True)


# Order columns
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

print(models.to_latex())
