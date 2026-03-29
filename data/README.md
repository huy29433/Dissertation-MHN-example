# Data

This directory contains the data on which the models from the dissertation are based.
They are both subsets of the data used in [Rupp et al, 2024](https://doi.org/10.1093/bioinformatics/btae250) and taken from the [`metMHN`](https://github.com/cbg-ethz/metMHN/) repository.


## Contents

- [prepare_data.py](prepare_data.py) is a Python script that takes the paired lung cancer data from the [`metMHN`](https://github.com/cbg-ethz/metMHN/) repository, annotates it regarding the metastasis status and the observation order of the primary tumor and the metastasis.
  It saves the event data for _TP53 (M)_, _TERT/5p (Amp)_, _MCL1/1q (Amp)_, _KRAS (M)_, and _EGFR (M)_ of the primary tumor and the metastasis, the seeding event, the observation order and the metastasis status.
  The latter is encoded as follows:

    | paired | metastasis status | metastasis status encoding
    | -------|-------------------|-----
    | no     | absent            | 0
    | no     | present           | 1
    | no     | is Metastasis     | 2
    | yes    | -                 | 3

  This closely follows the [examples\data_analysis.ipynb](..\metMHN\examples\data_analysis.ipynb) from the [`metMHN`](https://github.com/cbg-ethz/metMHN/) repository.
- [paired.csv](paired.csv) contains the paired data.
- [primary.csv](primary.csv) contains only the primary tumor data, i. e., the primary tumor information from all samples in [paired.csv](paired.csv) whose metastasis status is encoded with 0, 1, or 3.

