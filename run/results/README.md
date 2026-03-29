# Run results

This directory contains the results of the model training and posterior sampling.

## Contents

- [c](cmhn.csv)/[o](omhn.csv)/[metmhn.csv](metmhn.csv) is the parameter matrix of the c/o/metMHN model.
Note that the first two rows of metmhn.csv contain the observation effects.
- [c](cmhn_meta.json)/[omhn_meta.json](omhn_meta.json) is the meta data of the c/oMHN model.
- [mcmc.npy](mcmc.npy) are the results of the posterior sampling from the omhn model and can be opened in Python with:
  ```python
  import numpy as np

  chains = np.load(f"mcmc.npy")  
  
  print(type(chains))
  >>> <class 'numpy.ndarray'>