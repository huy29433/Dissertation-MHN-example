from jax import random as jrp
from jax import numpy as jnp
import jax as jax
import pandas as pd

import sys
sys.path.append("../metMHN")

import metmhn.simulations as simul

jax.config.update("jax_enable_x64", True)

seed = 0
key = jrp.key(seed)

n_sim = 1000

log_theta = pd.read_csv(
    "../metMHN/results/luad/luad_g14_cv_20muts_8cnvs.csv", index_col=0)
log_theta.index = ["Obs PT", "Obs MT"] + log_theta.columns.to_list()

events = [
    "TP53 (M)",
    "TERT/5p (Amp)",
    "MCL1/1q (Amp)",
    "KRAS (M)",
    "EGFR (M)",
    "Seeding",
]
log_theta = log_theta.loc[["Obs PT", "Obs MT"] + events, events]

key, sub_key = jrp.split(key)
dat = simul.simulate_dat(
    log_theta=jnp.array(log_theta.loc[events]),
    pt_d_ef=jnp.array(log_theta.loc[["Obs PT"]]),
    mt_d_ef=jnp.array(log_theta.loc[["Obs MT"]]),
    n_sim=n_sim,
    original_key=sub_key)

pd.DataFrame(dat, columns=[
    event + entity for event in events[:-1] for entity in [" PT", " MT"]
] + ["Seeding", "Observation Order"]).to_csv("paired.csv")
pd.DataFrame(dat[:, :-2:2], columns=events[:-1]).to_csv("primary.csv")
