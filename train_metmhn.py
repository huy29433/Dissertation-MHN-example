import logging
import matplotlib.pyplot as plt
import pandas as pd
import jax.numpy as jnp
import jax.random as jrp
import numpy as np
import jax as jax

import sys
sys.path.append(metMHN)

import metmhn.regularized_optimization as reg_opt
import metmhn.Utilityfunctions as utils
jax.config.update("jax_enable_x64", True)

df = pd.read_csv("data/paired.csv", index_col=0)
df.loc[df["Observation Order"]==0, [c for c in df.columns if c.endswith("MT")]] = 0
df.loc[df["Observation Order"]==0, "Observation Order"] = -99
df["type"] = df["Seeding"] * 3

dat = jnp.array(df.to_numpy(dtype=np.int8))
