# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
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
# Compare completion times and per-trial times betwenn the exploratory and the confirmatory study

# %%
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# Optional convenience: seaborn for heatmaps (style comes from figures.setup_style)
import seaborn as sns

from joptim.utils.export_path import get_export_paths



EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()


data = {}
data_demo = {}

for STUDY in ["exploratory","confirmatory"]:

    with open(Path(EXPORT_PATH_DATA, STUDY, "trials.pkl"), "rb") as pickle_file:
        trials = pickle.load(pickle_file)

    TRIAL_FILE_ID = trials["file_id"]
    
    data[STUDY] = pd.read_csv(
        Path(
            EXPORT_PATH_DATA,
            STUDY,
            f"""data_preprocessed_{str(TRIAL_FILE_ID).replace(".", "-")}.csv""",
        ),
        sep=",",
        dtype={"trial_file_id": float},
    )
    
    
    
    # filter to TRIAL_FILE_ID (loading from above file should be enough,
    # but we want to be on the safe side)
    data[STUDY] = data[STUDY].loc[data[STUDY]["trial_file_id"] == TRIAL_FILE_ID].reset_index(drop=True)
    data[STUDY] = data[STUDY].drop(["trial_file_id", "trial_file_user_id"], axis=1)
    
    data[STUDY]["log_rt"] = np.log(data[STUDY]["rt"])
    
    with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
        results = pickle.load(pickle_file)
    
    
    data_demo[STUDY] = pd.read_csv(
        Path(EXPORT_PATH_DATA, STUDY, "demographics", "prolific_demographics.csv")
    )


# %% [markdown]
# Overall completion times

# %%
data_demo["exploratory"]["Time taken"].mean()

# %%
data_demo["confirmatory"]["Time taken"].mean()

# %%
data_demo["confirmatory"]["Time taken"].mean() - data_demo["exploratory"]["Time taken"].mean()

# %%
from scipy.stats import ttest_ind

ttest_ind(data_demo["exploratory"]["Time taken"], data_demo["confirmatory"]["Time taken"])

# %% [markdown]
# Age

# %%
data_demo["exploratory"]["Age"].mean()

# %%
data_demo["confirmatory"]["Age"].mean()

# %%
ttest_ind(data_demo["exploratory"]["Age"], data_demo["confirmatory"]["Age"])

# %% [markdown]
# Sex

# %%
(data_demo["exploratory"]["Sex"] == "Female").mean()

# %%
(data_demo["confirmatory"]["Sex"] == "Female").mean()

# %% [markdown]
# Location

# %%
(data_demo["exploratory"]["Country of residence"] == "United Kingdom").mean()

# %%
(data_demo["confirmatory"]["Country of residence"] == "United Kingdom").mean()

# %% [markdown]
# Approvals

# %%
data_demo["exploratory"]["Total approvals"].mean()

# %%
data_demo["confirmatory"]["Total approvals"].mean()

# %%
ttest_ind(data_demo["exploratory"]["Total approvals"], data_demo["confirmatory"]["Total approvals"])

# %% [markdown]
# Employment

# %%
(data_demo["exploratory"]['Employment status'] == "Full-Time").mean()

# %%
(data_demo["confirmatory"]['Employment status'] == "Full-Time").mean()

# %% [markdown]
# Education

# %%
education_mapping = {
    'No formal qualifications': 0,
    'Secondary education (e.g. GED/GCSE)': 1,
    'High school diploma/A-levels': 2,
    'Technical/community college': 3,
    'Undergraduate degree (BA/BSc/other)': 4,
    'Graduate degree (MA/MSc/MPhil/other)': 5,
    'Doctorate degree (PhD/other)': 6
}

# %%
data_demo["exploratory"]['Highest education level completed'].map(education_mapping).mean()

# %%
data_demo["confirmatory"]['Highest education level completed'].map(education_mapping).mean()

# %% [markdown]
# Trials

# %%
ttest_ind(data["confirmatory"]["rt"], data["exploratory"]["rt"])

# %%
data["confirmatory"]["rt"].mean() - data["exploratory"]["rt"].mean()

# %% [markdown]
# Each datapoint is a participant and its mean rt

# %%
ttest_ind(data["confirmatory"].groupby("participant")["rt"].mean(), data["exploratory"].groupby("participant")["rt"].mean())

# %%
ttest_ind(data["confirmatory"].groupby("participant")["rt"].mean(), data["exploratory"].groupby("participant")["rt"].mean())

# %%
ttest_ind(data["confirmatory"].groupby("participant")["rt_solving"].mean(), data["exploratory"].groupby("participant")["rt_solving"].mean())

# %%
data["confirmatory"]["rt_solving"].mean() - data["exploratory"]["rt_solving"].mean()

# %% [markdown]
# Use location as predictors

# %%
rt_c = data["confirmatory"].groupby("participant")["rt"].mean().reset_index()
rt_e = data["exploratory"].groupby("participant")["rt"].mean().reset_index()

# %%
data_demo["confirmatory"]

# %%
rt_e

# %%
rt_c

# %%
rt_c.merge(data_demo["confirmatory"], left_on="participant", right_on="Participant id")

# %%
from scipy.stats import pearsonr

# %%
pearsonr(data_demo["confirmatory"]["Time taken"], data_demo["confirmatory"]["Age"])

# %%
pearsonr(data_demo["exploratory"]["Time taken"], data_demo["exploratory"]["Age"])

# %% [markdown]
# Final message
#
# The difference in the evaluation trials is -4 seconds per trial and it is significant (even if I use the mean per participant as observations -> 0.0005), the difference in solving trials is -50 seconds per trial and its also significant (0.03).There is a highly significant difference in number of prolific approvals (3,000 in confirmatory vs. 900 in exploratory). However, within each sample rt does not correlate with number of approvals. Another difference is age with confirmatory people being 9 years older on average (45 vs 36 y). There is a sign. pos. correlation between age and rt in the exploratory sample, not in the confirmatory sample. Gender ratio is almost exactly the same with 43% female. The UK proportion is also almost exactly the same with 81% in both. Overall it points to more experience, but evidence is mixed. Since the experiment and the sampling procedure did not change, it is likely that the factors are some "changing world" phenomenon outside of our experiment.

# %%
