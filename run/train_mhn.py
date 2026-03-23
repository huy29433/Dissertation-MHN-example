import mhn
import numpy as np

np.random.seed(0)

cmhn_opt = mhn.optimizers.cMHNOptimizer()
cmhn_opt.set_penalty(mhn.optimizers.Penalty.SYM_SPARSE)
cmhn_opt.load_data_from_csv("../data/primary.csv", index_col=0)

lam = cmhn_opt.lambda_from_cv()
cmhn_opt.train(lam=lam)

cmhn_opt.result.save("results/cmhn.csv")

np.random.seed(1)

omhn_opt = mhn.optimizers.oMHNOptimizer()
omhn_opt.set_penalty(mhn.optimizers.Penalty.SYM_SPARSE)
omhn_opt.load_data_from_csv("../data/primary.csv", index_col=0)

lam = omhn_opt.lambda_from_cv()
omhn_opt.train(lam=lam)

omhn_opt.result.save("results/omhn.csv")