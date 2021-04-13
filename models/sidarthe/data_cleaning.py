from functools import reduce
from itertools import compress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_GENERATIONS = 200

df = pd.read_csv("costs.dat", delim_whitespace=True)

# Pad non-present generations with the last changing value
new_index = pd.Index(np.arange(1, N_GENERATIONS + 1), name="gen")

# Create a vector of dataframes for each seed
df_vec = [
    df[df.seed == i].set_index("gen").reindex(new_index).pad().reset_index()
    for i in range(42, 52)
]

# Filter seeds where only 5 generations where executed (configurator bug)
df_filter = [df[df.seed == i]["gen"].max() > 10 for i in range(42, 52)]
df_vec = list(compress(df_vec, df_filter))

# Average over seeds
df = reduce(lambda x, y: pd.merge(x, y, how="outer"), df_vec)
df = df.groupby("gen").mean()
del df["seed"]
plt.plot(df["train_cost"], label="train")
plt.plot(df["test_cost"] / 2, label="test/2")
plt.legend()
plt.savefig("train_test.png", dpi=300)
plt.show()
