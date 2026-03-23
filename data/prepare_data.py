import sys
import pandas as pd
import jax as jax
jax.config.update("jax_enable_x64", True)

sys.path.append("../metMHN")
import metmhn.Utilityfunctions as utils
import metmhn.regularized_optimization as reg_opt

mut_handle = "../metMHN/data/luad/G14_LUAD_Events.csv"
annot_handle = "../metMHN/data/luad/G14_LUAD_sampleSelection.csv"

events = [
    "TP53 (M)",
    "TERT/5p (Amp)",
    "MCL1/1q (Amp)",
    "KRAS (M)",
    "EGFR (M)",
    "Seeding",
]

annot_data = pd.read_csv(annot_handle, index_col=0)
mut_data = pd.read_csv(mut_handle, index_col=0)
mut_data.drop(columns=[c for c in mut_data
                       if c.replace("P." , "").replace("M.", "")
                       not in events + ["paired", "AgeAtSeqRep", "metaStatus"]],
              inplace=True)
mut_data["metaStatus"] = annot_data["metaStatus"]
muts = [entity + event for event in events[:-1]
        for entity in ("P.", "M.")]


# Paired | metaStatus   | type
# ----------------------------
# 0      | absent       | 0
# 0      | present      | 1
# 0      | isMetastasis | 2
# 1      | -            | 3
# Else -> pd.NA

mut_data["type"] = mut_data.apply(utils.categorize, axis=1)

# Add the seeding event
mut_data["Seeding"] = mut_data["type"].apply(
    lambda x: pd.NA if pd.isna(x) else 0 if x == 0 else 1)

mut_data["M.AgeAtSeqRep"] = pd.to_numeric(
    mut_data["M.AgeAtSeqRep"], errors='coerce')
mut_data["P.AgeAtSeqRep"] = pd.to_numeric(
    mut_data["P.AgeAtSeqRep"], errors='coerce')

# Define the order of diagnosis for paired datapoints
mut_data["diag_order"] = mut_data["M.AgeAtSeqRep"] - mut_data["P.AgeAtSeqRep"]
mut_data["diag_order"] = mut_data["diag_order"].apply(
    lambda x: pd.NA if pd.isna(x) else 2 if x < 0 else 1 if x > 0 else 0)
mut_data["diag_order"] = mut_data["diag_order"].astype(pd.Int64Dtype())

events_data = muts + ["Seeding"]

# Only use datapoints where the state of the seeding is known
cleaned = mut_data.loc[~pd.isna(
    mut_data["type"]), muts+["Seeding", "diag_order", "type"]]

cleaned.iloc[:1000].to_csv("paired.csv")
primary = cleaned[cleaned["type"] != 2].iloc[:1000, :-3:2]
primary.columns = events[:-1]
primary.to_csv("primary.csv")
