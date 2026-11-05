# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py//py:percent
#     notebook_metadata_filter: kernelspec,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.0
# ---

# %% [markdown]
# # Coherence Analysis
#
# Aim is to compute number of coherent people (i.e., with transitive preferences), to estimate the upper bound on how well our multilevel models can do.
#

# %% [markdown]
# ## 0. Setup

# %%
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime
# from sklearn.preprocessing import StandardScaler
from joptim.utils.export_path import get_export_paths



EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
STUDY = "confirmatory"  # "exploratory" or "confirmatory"
ANALYSIS_NAME = "coherence"

with open(Path(EXPORT_PATH_DATA, STUDY, "trials.pkl"), "rb") as pickle_file:
    trials = pickle.load(pickle_file)

trial_file_id = trials["file_id"]
del trials

data = pd.read_csv(Path(EXPORT_PATH_DATA, STUDY, f"data_preprocessed_{str(trial_file_id).replace('.','-')}.csv"))

print(len(data.participant.unique()), "participants loaded.")
print(len(data), "trials loaded.")

# %% [markdown]
# ## 1. Compute Coherence Ceiling

# %% [markdown]
# Filter to coherence trials and to participants for who all 3 coherence trials are available

# %%
coherence_trial_idx = {6: "AB", 12: "AC", 19: "BC"}

# Filter to coherence trials

data_filtered = data.loc[data["trial_id"].isin(coherence_trial_idx)].reset_index(
    drop=True
)[
    [
        "participant",
        "trial_file_id",
        "trial_file_user_id",
        "trial_id",
        "choice_RvL",
        "choice_cat_RvL",
    ]
]

# Remove participants for who less than 3 coherence trials are available

group_cols = ["participant", "trial_file_id", "trial_file_user_id"]
complete_participants = data_filtered.groupby(group_cols).count()

complete_participants = complete_participants.loc[
    complete_participants["trial_id"] == 3
].reset_index()[group_cols]

data_filtered = data_filtered.merge(complete_participants, on=group_cols).reset_index(
    drop=True
)

print(len(data_filtered.participant.unique()), "participants remain.")

# Compute the choices

data_filtered["coherence_stimuli"] = data_filtered.trial_id.map(coherence_trial_idx)
data_filtered["coherence_choice"] = data_filtered.apply(
    lambda row: (
        row.coherence_stimuli[0] if row.choice_RvL < 0 else row.coherence_stimuli[1]
    ),
    axis=1,
)
data_filtered = data_filtered[group_cols + ["coherence_choice"]]

data_filtered_grp = data_filtered.groupby(group_cols).agg({"coherence_choice": "".join})

# %% [markdown]
# Coherent trials can only have two winning stimuli: the overall best stimulus wins twice, the second best wins once

# %%
data_filtered_grp["coherent"] = data_filtered_grp["coherence_choice"].apply(
    lambda x: len(set(x)) == 2
)

print(
    f"""{data_filtered_grp["coherent"].mean():.2%} of the participants passed the coherence check."""
)

# %% [markdown]
# ## 2. Publication metrics export (JSON)

# %%
n_checked = int(len(data_filtered_grp))
n_transitive = int(data_filtered_grp["coherent"].sum())
n_intransitive = int(n_checked - n_transitive)
p_coherent = float(data_filtered_grp["coherent"].mean())



out = {
    "meta": {
        "analysis_name": ANALYSIS_NAME,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    },
    "n_participants_checked": n_checked,
    "n_transitive":n_transitive,
    "n_intransitive":n_intransitive,
    "p_coherent": p_coherent
}

out_path = Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{ANALYSIS_NAME}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {out_path}")

# %%
