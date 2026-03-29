# Run

This directory contains the code to train the toy MHNs, metMHN and MHN posterior sampling, and the corresponding results.

## Contents

- [train_mhn.py](train_mhn.py) trains a cMHN and an oMHN on the [primary tumor data](../data/primary.csv).
- [train_metmhn.py](train_metmhn.py) trains a metMHN on the [paired data](../data/paired.csv).
- [mcmc.py](mcmc.py) runs posterior sampling for the [oMHN model](../run/results/omhn.csv).

## Note

To run the scripts, simply call
```bash
python <script_name>.py
```
from this directory.