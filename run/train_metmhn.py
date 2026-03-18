import pandas as pd
import jax.numpy as jnp
import jax.random as jrp
import numpy as np
import jax as jax

import sys
sys.path.append("../metMHN")

import metmhn.Utilityfunctions as utils
import metmhn.regularized_optimization as reg_opt

jax.config.update("jax_enable_x64", True)

df = pd.read_csv("../data/paired.csv", index_col=0)
df.loc[df["Observation Order"] == 0, [
    c for c in df.columns if c.endswith("MT")]] = 0
df.loc[df["Observation Order"] == 0, "Observation Order"] = -99
df["type"] = df["Seeding"] * 3

dat = jnp.array(df.to_numpy(dtype=np.int8))

w_corr = df["type"].value_counts()[3] / len(df)

log_lams = np.linspace(-3.5, -2.5, 5)
lams = 10**log_lams
key = jrp.key(42)
penal_weights = utils.cross_val(dat=dat,
                                penal_fun=reg_opt.symmetric_penal,
                                splits=lams,
                                n_folds=5,
                                m_p_corr=w_corr,
                                key=key)

# The cross_val function returns a n_folds x log_lams.size shaped
# dataframe
best_lam = lams[np.argmax(np.mean(penal_weights, axis=0))]

log_theta_init, dp_init, dm_init = utils.indep(dat)
log_theta, d_p, d_m = reg_opt.learn_mhn(th_init=log_theta_init,
                                    dp_init=dp_init,
                                    dm_init=dm_init,
                                    dat=dat,
                                    perc_met=w_corr,
                                    penal=reg_opt.symmetric_penal,
                                    w_penal=best_lam,
                                    opt_ftol=1e-05
                                    )

log_theta_final = np.row_stack((d_p.reshape((1, -1)),
                        d_m.reshape((1, -1)),
                        log_theta))

pd.DataFrame(log_theta_final).to_csv("results/metmhn.csv")
pd.DataFrame(log_theta_final, columns=[c.rstrip(" PT") for c in df.columns.to_list()[
             :-1:2]]).to_csv("results/metmhn.csv")
