import numpy as np
import mhn.model
from mhn.mcmc.mcmc import MCMC
from mhn.optimizers import Penalty

mhn_model = mhn.model.oMHN.load(
    f"results/omhn.csv")
data = np.loadtxt(f"../data/primary.csv", delimiter=",", skiprows=1,
                  usecols=range(1, mhn_model.log_theta.shape[1] + 1), dtype=np.int32)

mcmc_sampler = MCMC(
    mhn_model=mhn_model,
    data=data,
    penalty=Penalty.SYM_SPARSE,
    seed=0,
)

mcmc_sampler.run()

np.save(
    "results/mcmc.npy",
    mcmc_sampler.log_thetas)
