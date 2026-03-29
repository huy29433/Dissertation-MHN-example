# Dissertation MHN Example

This repository contains the Python code to reproduce the figures from Y. Linda Hu's dissertation "Modeling Cancer Progression with Mutual Hazard Networks".


## Setup

We use code from the [`metMHN`](https://github.com/cbg-ethz/metMHN/) repository and from the [MCMC-sampling-for-MHN](https://github.com/huy29433/MCMC-sampling-for-MHN) repository and included them as submodules.
Therefore, make sure to clone this repository including its submodules with
```bash
git clone --recurse-submodules https://github.com/huy29433/Dissertation-MHN-example.git
```

After cloning, install the required Python packages with
```bash
pip install -r requirements.txt
```

## Contents

This repository contains:

- in [data](data) the paired and unpaired data used in the toy example as well as the script to prepare them.
- in [run](run) the scripts to train an o/c/metMHN, as well as the MCMC posterior analysis of an MHN and the corresponding results.
- in [analysis](analysis) the scripts and utitlities to perform the analyses and produce the figures.
