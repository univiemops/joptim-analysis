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
# Preprocess data to be used for Multilevel and Machine Learning Analysis
# ===

# %% [markdown]
# TODO:
#
# Make sure the different studies (`STUDY`) are cleanly separated (no aggregating of values across the two studies)!

# %% [markdown]
# ## Imports and Data

# %% [markdown]
# **Important Prerequisites:**
# - Latest `results.pkl` (run `scripts/data_processing/download_experiment_data.py`)
# - Latest `EXPORT_PATH_DATA/<STUDY>/prolific_demographics.csv` exists. If not, run script `analysis/demographics.ipynb`
# - Check `download_eye_data.py/ipynb` for any participants on mobile devices for potential exclusion

# %%
# # !python download_experimental_data.py 39 # includes filip
# !python ../data_processing/download_experiment_data.py 40

# %%
import json
import pickle
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joptim.utils.export_path import get_export_paths
from joptim.utils.elements import find_element_by_id
from joptim.disorder import disorder_index
import statsmodels.formula.api as smf
from marginaleffects import datagrid, predictions
from scipy.stats import zscore, pearsonr
from joptim.utility import compute_utility
from joptim.standardizer import zscore_LR
from joptim.graph_edit_distance import graph_edit_distance_diagonal_approximated, graph_edit_distance_diagonal


STUDY = "confirmatory"  # "exploratory" or "confirmatory"
EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()

if STUDY not in ["exploratory", "confirmatory"]:
    raise ValueError(f"STUDY can only be 'exploratory' or 'confirmatory', got: {STUDY}")

with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
    results = pickle.load(pickle_file)

with open(Path(EXPORT_PATH_DATA, "results_no_exclusion.pkl"), "rb") as pickle_file:
    results_no_exclusion = pickle.load(pickle_file)

with open(Path(EXPORT_PATH_DATA, STUDY, "trials.pkl"), "rb") as pickle_file:
    trials = pickle.load(pickle_file)

TRIAL_FILE_ID = trials["file_id"]
trial_data = {}
trial_data[str(TRIAL_FILE_ID)] = trials["trials"]

print("\nLoaded results from", len(results["user"]), "participants.")

df = results["preference_trials"][
    [
        "xid",
        "trial_file_id",
        "trial_file_user_id",
        "trial_id",
        "choice",
        "use_case",
        "task_loads",
        "worker_capacities",
        "task_order_1",
        "task_order_2",
        "worker_order_1",
        "worker_order_2",
        "rt",
    ]
].copy()

df_ne = results_no_exclusion["preference_trials"][
    [
        "xid",
        "trial_file_id",
        "trial_file_user_id",
        "trial_id",
        "choice",
        "use_case",
        "task_loads",
        "worker_capacities",
        "task_order_1",
        "task_order_2",
        "worker_order_1",
        "worker_order_2",
        "rt",
    ]
].copy()


df_prolific_demographics = pd.read_csv(
    Path(EXPORT_PATH_DATA, STUDY, "demographics", "prolific_demographics.csv")
)

participant_ids = list(df_prolific_demographics["Participant id"].values.flatten())

participant_xids = list(
    results_no_exclusion["user"].loc[results_no_exclusion["user"].pid.isin(participant_ids), "xid"].values
)

del results_no_exclusion


# create xid field
pid2xid_dict = {x: y for x, y in results["user"][["pid", "xid"]].values}
df_prolific_demographics["xid"] = df_prolific_demographics["Participant id"].map(
    pid2xid_dict
)

# filter to TRIAL_FILE_ID
df = df.loc[df["trial_file_id"] == str(TRIAL_FILE_ID)].reset_index(drop=True)
df_ne = df_ne.loc[df_ne["trial_file_id"] == str(TRIAL_FILE_ID)].reset_index(drop=True)

print(
    f"""Filtered results to {len(df_ne['xid'].unique())} ({len(df_ne)} trials) from {STUDY} study (trial file: {TRIAL_FILE_ID})."""
)


# Filter to prolific participants only
df = df.loc[df["xid"].isin(participant_xids)].reset_index(drop=True)
df_ne = df_ne.loc[df_ne["xid"].isin(participant_xids)].reset_index(drop=True)

print(f"Filtered results to {len(df_ne['xid'].unique())} participants ({len(df_ne)} trials) with prolific ids participating in the experiment and who are not blacklisted (see demographics.ipynb/py).")

n_total = len(df_ne["xid"].unique())
n_included = len(df["xid"].unique())
n_excluded = n_total-n_included


# "Filter" to included participants only (= we will simply use df from now on, not df_ne) 

print(
    f"Filtered results to {n_included} participants ({len(df)} trials) that met the experimental inclusion criteria (catch trials). {n_excluded} of a total of {n_total} ({n_excluded/n_total:.1%}) participants were excluded."
)

del df_ne

fout = f"""data_preprocessed_{str(TRIAL_FILE_ID).replace(".","-")}.csv"""
print(f"\nPreprocessed data will be written to {fout}")

# %% [markdown]
# In case of confirmatory study, cap sample size to be equal with exploratory study

# %% [markdown]
# ## 1. Get problem, solution and presentation metrics

# %% [markdown]
# ### 1.1. Complexity metrics

# %%
# load problem instances

with open(
    Path(EXPORT_PATH_DATA, "problems_filtered_incl_solution_metrics.pkl"), "rb"
) as pickle_file:
    problems = pickle.load(pickle_file)
    print(len(problems), "problem instances loaded.")


def get_problem_solution_properties(
    trial_file_id, trial_file_user_id, trial_id, problems, trial_data
):
    """Helper function to add missing problem and solution metrics to dataframe.
    Makes use of trials and problems object which are in the outer scope
    of this function."""

    _, sol1_id, sol2_id = trial_data[trial_file_id][trial_file_user_id - 1][
        "preference_trials"
    ][trial_id]["id"].split("_")

    sol1_id = int(sol1_id)
    sol2_id = int(sol2_id)

    w = trial_data[trial_file_id][trial_file_user_id - 1]["preference_trials"][
        trial_id
    ]["worker_capacities"]
    z = trial_data[trial_file_id][trial_file_user_id - 1]["preference_trials"][
        trial_id
    ]["task_loads"]

    problem_id = (
        "-".join(list([str(i) for i in w])) + "|" + "-".join(list([str(j) for j in z]))
    )

    problem = find_element_by_id(problems, str(problem_id))

    ac_L = problem["assignment_complexity"][sol1_id]
    ac_R = problem["assignment_complexity"][sol2_id]

    cc_v2_L = problem["composition_complexity_v2"][sol1_id]
    cc_v2_R = problem["composition_complexity_v2"][sol2_id]
    cc_v1_L = problem["composition_complexity_v1"][sol1_id]
    cc_v1_R = problem["composition_complexity_v1"][sol2_id]

    if STUDY == "confirmatory":
        cc_v2_L_confirmatory = problem["composition_complexity_v2_confirmatory"][
            sol1_id
        ]
        cc_v2_R_confirmatory = problem["composition_complexity_v2_confirmatory"][
            sol2_id
        ]
        cc_v1_L_confirmatory = problem["composition_complexity_v1_confirmatory"][
            sol1_id
        ]
        cc_v1_R_confirmatory = problem["composition_complexity_v1_confirmatory"][
            sol2_id
        ]

    # load-capacity ratio (aka problem difficulty)
    load_capacity_ratio = problem["capacity_load_ratio"]

    # utility difference
    utility_optimal = problem["utility"]
    utility_heuristic = compute_utility(
        problem["task_loads"], problem["greedy_solution"]
    )
    heuristic_optimality = utility_heuristic / utility_optimal

    # iterations of mixed-integer programming solver
    mip_iterations = problem["iterations"]
    if STUDY == "confirmatory":
        confirmatory_cc_values = [
            cc_v2_L_confirmatory,
            cc_v2_R_confirmatory,
            cc_v1_L_confirmatory,
            cc_v1_R_confirmatory,
        ]
    else:
        confirmatory_cc_values = []

    return (
        [ac_L, ac_R, cc_v2_L, cc_v2_R, cc_v1_L, cc_v1_R]
        + confirmatory_cc_values
        + [
            load_capacity_ratio,
            utility_optimal,
            utility_heuristic,
            heuristic_optimality,
            mip_iterations,
        ]
    )


if STUDY == "confirmatory":
    confirmatory_cc_cols = ["cc_v2_L_opt", "cc_v2_R_opt", "cc_v1_L_opt", "cc_v1_R_opt"]

else:
    confirmatory_cc_cols = []
cols = (
    ["ac_L", "ac_R", "cc_v2_L", "cc_v2_R", "cc_v1_L", "cc_v1_R"]
    + confirmatory_cc_cols
    + [
        "load_capacity_ratio",
        "utility_optimal",
        "utility_heuristic",
        "heuristic_optimality",
        "mip_iterations",
    ]
)


df[cols] = df.apply(
    lambda row: pd.Series(
        get_problem_solution_properties(
            row.trial_file_id,
            row.trial_file_user_id,
            row.trial_id,
            problems,
            trial_data,
        )
    ),
    axis=1,
)

# differences
df["diff_ac_RvL"] = df["ac_R"] - df["ac_L"]
df["diff_cc_v2_RvL"] = df["cc_v2_R"] - df["cc_v2_L"]
df["diff_cc_v1_RvL"] = df["cc_v1_R"] - df["cc_v1_L"]

# abs. differences
df["abs_diff_ac_RvL"] = df["diff_ac_RvL"].abs()
df["abs_diff_cc_v2_RvL"] = df["diff_cc_v2_RvL"].abs()
df["abs_diff_cc_v1_RvL"] = df["diff_cc_v1_RvL"].abs()

if STUDY == "confirmatory":
    df["diff_cc_v2_RvL_opt"] = df["cc_v2_R_opt"] - df["cc_v2_L_opt"]
    df["diff_cc_v1_RvL_opt"] = df["cc_v1_R"] - df["cc_v1_L_opt"]
    df["abs_diff_cc_v2_RvL_opt"] = df["diff_cc_v2_RvL_opt"].abs()
    df["abs_diff_cc_v1_RvL_opt"] = df["diff_cc_v1_RvL_opt"].abs()

# %% [markdown]
# ### 1.2. Disorder indexes
#

# %%
# Disorder is defined in a way that it does not take worker identity into account,
# but only respects the size of the workers.

df["task_loads_L"] = df.apply(
    lambda row: row["task_loads"][row["task_order_1"]], axis=1
)
df["task_loads_R"] = df.apply(
    lambda row: row["task_loads"][row["task_order_2"]], axis=1
)
df["worker_capacities_L"] = df.apply(
    lambda row: row["worker_capacities"][row["worker_order_1"]], axis=1
)
df["worker_capacities_R"] = df.apply(
    lambda row: row["worker_capacities"][row["worker_order_2"]], axis=1
)

df["disorder_L"] = df.apply(
    lambda row: disorder_index(row["worker_capacities_L"], row["task_loads_L"]),
    axis=1,
)
df["disorder_R"] = df.apply(
    lambda row: disorder_index(row["worker_capacities_R"], row["task_loads_R"]),
    axis=1,
)

df["diff_disorder_RvL"] = df["disorder_R"] - df["disorder_L"]
df["abs_diff_disorder_RvL"] = df["diff_disorder_RvL"].abs()

df["max_disorder"] = df[["disorder_L", "disorder_R"]].max(axis=1)
df["sum_disorder"] = df[["disorder_L", "disorder_R"]].sum(axis=1)


# %% [markdown]
# ### 1.3. Compute diagonal dissimilarity scores
#
# - whether a solution looks diagonal in an experiment depends on how the problem is sorted
# - the diagonal dissimilarity scores are all computed for the ordered cases, so we need to compute them taking order into account

# %%
def fetch_diag_for_trial(trial_file_id, trial_file_user_id, trial_id, trials=trials):    
    if not trial_file_id == str(trials["file_id"]):
        raise ValueError("Wrong trial file.")
    
    solution_L = trials["trials"][trial_file_user_id-1]["preference_trials"][trial_id]["solution_1"]
    solution_R = trials["trials"][trial_file_user_id-1]["preference_trials"][trial_id]["solution_2"]
    
    worker_order_L = trials["trials"][trial_file_user_id-1]["preference_trials"][trial_id]["worker_order_1"]
    worker_order_R = trials["trials"][trial_file_user_id-1]["preference_trials"][trial_id]["worker_order_2"]
    task_order_L = trials["trials"][trial_file_user_id-1]["preference_trials"][trial_id]["task_order_1"]
    task_order_R = trials["trials"][trial_file_user_id-1]["preference_trials"][trial_id]["task_order_2"]
    
    solution_L_display = solution_L[task_order_L][:, worker_order_L]
    solution_R_display = solution_R[task_order_R][:, worker_order_R]
    
    du_L, dl_L = graph_edit_distance_diagonal(solution_L_display)
    du_R, dl_R = graph_edit_distance_diagonal(solution_R_display)

    da_L = graph_edit_distance_diagonal_approximated(solution_L_display)
    da_R = graph_edit_distance_diagonal_approximated(solution_R_display)
    
    return du_L, du_R, dl_L, dl_R, da_L, da_R

df[["diagu_L","diagu_R","diagl_L","diagl_R", "diaga_L", "diaga_R"]] = df.apply(lambda row: pd.Series(fetch_diag_for_trial(row.trial_file_id, row.trial_file_user_id, row.trial_id)), axis=1)

# differences
df["diff_diagu_RvL"] = df["diagu_R"] - df["diagu_L"]
df["diff_diagl_RvL"] = df["diagl_R"] - df["diagl_L"]
df["diff_diaga_RvL"] = df["diaga_R"] - df["diaga_L"]

# abs. differences
df["abs_diff_diagu_RvL"] = df["diff_diagu_RvL"].abs()
df["abs_diff_diagl_RvL"] = df["diff_diagl_RvL"].abs()
df["abs_diff_diaga_RvL"] = df["diff_diaga_RvL"].abs()


# %% [markdown]
# ### 1.4. PSI scores

# %%
def compute_psi_score(x):

    values = np.array(
        list(x.values()),
        dtype=int,
    )

    # problem-solving inventory reversal mask
    reverse = np.array(
        [
            7,
            7,
            7,
            7,
            0,
            0,
            0,
            0,
            0,
            7,
            0,
            7,
            7,
            0,
            7,
            0,
            0,
            0,
            7,
            0,
            0,
            7,
            7,
            0,
            0,
            7,
            0,
            7,
            0,
            7,
            0,
        ]
    )

    score = np.sum(np.abs(reverse - values), axis=0)
    return score


results["survey"]["psi"] = results["survey"]["result_psi"].apply(
    lambda x: compute_psi_score(x)
)
results["survey"] = results["survey"].merge(
    results["user"][["xid", "trial_file_id", "trial_file_user_id"]],
    on="xid",
    how="left",
)

df = df.merge(
    results["survey"][["trial_file_id", "trial_file_user_id", "psi"]],
    on=["trial_file_id", "trial_file_user_id"],
    how="left",
)

# %% [markdown]
# ### 1.5. Solving Time

# %%
rt_solving = (
    results["solving_trials"]
    .groupby(["trial_file_id", "trial_file_user_id"])[["rt"]]
    .sum()
).rename(columns={"rt": "rt_solving"})


df = df.merge(rt_solving, on=["trial_file_id", "trial_file_user_id"], how="left")

# %% [markdown]
# ### 1.6. Problem-Solving Efficiency

# %%
results["solving_trials"]

# %%
from joptim.utils.representations import assignments2solution

results["solving_trials"]["utility_achieved"] = results["solving_trials"].apply(
    lambda row: compute_utility(
        row.task_loads,
        assignments2solution(row.assignments, row.task_loads, row.worker_capacities),
    ),
    axis=1,
)

results["solving_trials"]["rt_seconds"] = results["solving_trials"]["rt"] / 1000

results["solving_trials"]["efficiency_raw"] = (
    results["solving_trials"]["utility_achieved"]
    / results["solving_trials"]["utility"]
    / results["solving_trials"]["rt_seconds"]
)

trial_difficulty_weights = (
    results["solving_trials"]
    .groupby(["trial_file_id", "trial_id"])[["efficiency_raw"]]
    .agg(difficulty_weight=("efficiency_raw", "mean"))
)

trial_difficulty_weights = trial_difficulty_weights.merge(
    trial_difficulty_weights.groupby("trial_file_id").agg(
        difficulty_weight_sum=("difficulty_weight", "sum")
    ),
    how="right",
    left_index=True,
    right_index=True,
)

# first normalize: easiness weights
trial_difficulty_weights["difficulty_weight"] = (
    trial_difficulty_weights["difficulty_weight"]
    / trial_difficulty_weights["difficulty_weight_sum"]
)

# convert easiness → hardness: harder trials get larger weights
trial_difficulty_weights["difficulty_weight"] = (
    1.0 - trial_difficulty_weights["difficulty_weight"]
)

# renormalize hardness weights to sum to 1 per trial_file_id
trial_difficulty_weights["difficulty_weight"] = (
    trial_difficulty_weights["difficulty_weight"]
    / trial_difficulty_weights.groupby("trial_file_id")["difficulty_weight"].transform("sum")
)

trial_difficulty_weights.drop("difficulty_weight_sum", axis=1, inplace=True)


# add column difficulty_weight to main dataframe

results["solving_trials"] = results["solving_trials"].merge(
    trial_difficulty_weights, on=["trial_file_id", "trial_id"]
)

# create weighted efficiency

results["solving_trials"]["efficiency_weighted"] = (
    results["solving_trials"]["efficiency_raw"]
    * results["solving_trials"]["difficulty_weight"]
)

# %% [markdown]
# overall efficiency scores weighted by difficulty

# %%
efficiency_data = (
    results["solving_trials"]
    .groupby(["trial_file_id", "trial_file_user_id"])
    .agg(efficiency=("efficiency_weighted", "sum"))
)

efficiency_data

# %% [markdown]
# z-scored overall efficiency scores

# %%
from scipy.stats import zscore

efficiency_data["efficiency_z"] = efficiency_data.groupby(
    ["trial_file_id"]
).efficiency.transform(lambda x: zscore(x, ddof=1))

# %% [markdown]
# Merge with main data

# %%
df = df.merge(efficiency_data, on=["trial_file_id", "trial_file_user_id"], how="left")

# %% [markdown]
# ## 2. Recode variables

# %% [markdown]
# ### 2.1. Choice
#
# For use with correlation

# %%
choice_recode_dict = {1: -2, 2: -1, 3: 1, 4: 2}

df["choice_RvL"] = df["choice"].apply(lambda x: choice_recode_dict[x])

# %% [markdown]
# For use in ordinal regression

# %%
df["choice_cat_RvL"] = df["choice_RvL"].map(
    {
        -2: "definitely left",
        -1: "slightly left",
        1: "slightly right",
        2: "definitely right",
    }
)

df["choice_cat_RvL"] = pd.Categorical(
    df["choice_cat_RvL"],
    # categories=['-2', '-1', '1', '2'],
    categories=[
        "definitely left",
        "slightly left",
        "slightly right",
        "definitely right",
    ],
    ordered=True,
)

# %% [markdown]
# ### 2.2. Create Standardized Predictors

# %%
df["psi_z"] = zscore(df["psi"])

df["load_capacity_ratio_z"] = zscore(df["load_capacity_ratio"])
df["heuristic_optimality_z"] = zscore(df["heuristic_optimality"])
df["mip_iterations_z"] = zscore(df["mip_iterations"])


# disorder RvL

df[["disorder_L_z", "disorder_R_z"]] = zscore_LR(df, "disorder_L", "disorder_R")
df["diff_disorder_RvL_z"] = df["disorder_R_z"] - df["disorder_L_z"]
df["abs_diff_disorder_RvL_z"] = df["diff_disorder_RvL_z"].abs()


# max disorder
df["max_disorder_z"] = df[["disorder_R_z", "disorder_L_z"]].max(axis=1)
df["sum_disorder_z"] = df[["disorder_R_z", "disorder_L_z"]].sum(axis=1)

# ac RvL
df[["ac_L_z", "ac_R_z"]] = zscore_LR(df, "ac_L", "ac_R")
df["diff_ac_RvL_z"] = df["ac_R_z"] - df["ac_L_z"]
df["abs_diff_ac_RvL_z"] = df["diff_ac_RvL_z"].abs()

# cc RvL
df[["cc_v1_L_z", "cc_v1_R_z"]] = zscore_LR(df, "cc_v1_L", "cc_v1_R")
df["diff_cc_v1_RvL_z"] = df["cc_v1_R_z"] - df["cc_v1_L_z"]
df["abs_diff_cc_v1_RvL_z"] = df["diff_cc_v1_RvL_z"].abs()

df[["cc_v2_L_z", "cc_v2_R_z"]] = zscore_LR(df, "cc_v2_L", "cc_v2_R")
df["diff_cc_v2_RvL_z"] = df["cc_v2_R_z"] - df["cc_v2_L_z"]
df["abs_diff_cc_v2_RvL_z"] = df["diff_cc_v2_RvL_z"].abs()

if STUDY == "confirmatory":
    df[["cc_v1_L_opt_z", "cc_v1_R_opt_z"]] = zscore_LR(df, "cc_v1_L_opt", "cc_v1_R_opt")
    df["diff_cc_v1_RvL_opt_z"] = df["cc_v1_R_opt_z"] - df["cc_v1_L_opt_z"]
    df["abs_diff_cc_v1_RvL_opt_z"] = df["diff_cc_v1_RvL_opt_z"].abs()

    df[["cc_v2_L_opt_z", "cc_v2_R_opt_z"]] = zscore_LR(df, "cc_v2_L_opt", "cc_v2_R_opt")
    df["diff_cc_v2_RvL_opt_z"] = df["cc_v2_R_opt_z"] - df["cc_v2_L_opt_z"]
    df["abs_diff_cc_v2_RvL_opt_z"] = df["diff_cc_v2_RvL_opt_z"].abs()

# diagonals
df[["diagu_L_z", "diagu_R_z"]] = zscore_LR(df, "diagu_L", "diagu_R")
df["diff_diagu_RvL_z"] = df["diagu_R_z"] - df["diagu_L_z"]
df["abs_diff_diagu_RvL_z"] = df["diff_diagu_RvL_z"].abs()

df[["diagl_L_z", "diagl_R_z"]] = zscore_LR(df, "diagl_L", "diagl_R")
df["diff_diagl_RvL_z"] = df["diagl_R_z"] - df["diagl_L_z"]
df["abs_diff_diagl_RvL_z"] = df["diff_diagl_RvL_z"].abs()

df[["diaga_L_z", "diaga_R_z"]] = zscore_LR(df, "diaga_L", "diaga_R")
df["diff_diaga_RvL_z"] = df["diaga_R_z"] - df["diaga_L_z"]
df["abs_diff_diaga_RvL_z"] = df["diff_diaga_RvL_z"].abs()

# %%
df["problem_is_sorted"] = df[["task_order_1", "task_order_2"]].apply(
    lambda row: (sorted(list(row.task_order_1)) == list(row.task_order_1))
    & (sorted(list(row.task_order_2)) == list(row.task_order_2)),
    axis=1,
)

# %% [markdown]
# ### 2.3. Create  variable `handedness`

# %%
df_handedness = results["survey"][["xid", "result_demographics"]].apply(
    lambda x: pd.Series(
        {
            "xid": x.xid,
            "handedness": ["Right", "Left", "Ambidextrous"][
                int(x.result_demographics["1_handedness"])
            ],
        }
    ),
    axis=1,
)

df = df.merge(df_handedness, on="xid", how="left")

# %% [markdown]
# Delete, if we don't need it anymore

# %%
# we don't need it anylonger
del results

# %% [markdown]
# ### 2.4. Create variable `location`

# %%
df = df.merge(
    df_prolific_demographics.rename(columns={"Country of residence": "location"})[
        ["xid", "location"]
    ],
    on="xid",
    how="left",
)

# %% [markdown]
# ## 3. Save as csv for usage with R

# %% [markdown]
# ### 3.1. Save

# %%
if STUDY == "confirmatory":
    cc_opt_cols = [
        # values for L and R
        "cc_v2_R_opt",
        "cc_v2_L_opt",
        "cc_v1_R_opt",
        "cc_v1_L_opt",
        # differences
        "diff_cc_v2_RvL_opt",
        "diff_cc_v1_RvL_opt",
        # absolute differences
        "abs_diff_cc_v2_RvL_opt",
        "abs_diff_cc_v1_RvL_opt",
        # z-scored values for L and R
        "cc_v2_R_opt_z",
        "cc_v2_L_opt_z",
        "cc_v1_R_opt_z",
        "cc_v1_L_opt_z",
        # z-scored differences
        "diff_cc_v2_RvL_opt_z",
        "diff_cc_v1_RvL_opt_z",
        # z-scored abs. differences
        "abs_diff_cc_v2_RvL_opt_z",
        "abs_diff_cc_v1_RvL_opt_z",
    ]
else:
    cc_opt_cols = []

data = df[
    [
        "xid",
        "trial_file_id",
        "trial_file_user_id",
        "trial_id",
        # choice
        "choice_RvL",
        "choice_cat_RvL",
        # values for L and R
        "ac_R",
        "ac_L",
        "diagu_R",
        "diagu_L",
        "diagl_R",
        "diagl_L",
        "diaga_R",
        "diaga_L",
        "cc_v2_R",
        "cc_v2_L",
        "cc_v1_R",
        "cc_v1_L",
        "disorder_R",
        "disorder_L",
        # differences
        "diff_ac_RvL",
        "diff_diagu_RvL",
        "diff_diagl_RvL",
        "diff_diaga_RvL",
        "diff_cc_v2_RvL",
        "diff_cc_v1_RvL",
        "diff_disorder_RvL",
        # absolute differences
        "abs_diff_ac_RvL",
        "abs_diff_diagu_RvL",
        "abs_diff_diagl_RvL",
        "abs_diff_diaga_RvL",
        "abs_diff_cc_v2_RvL",
        "abs_diff_cc_v1_RvL",
        "abs_diff_disorder_RvL",
        # other
        "psi",
        "load_capacity_ratio",
        "handedness",
        "max_disorder",
        "sum_disorder",
        "utility_optimal",
        "utility_heuristic",
        "heuristic_optimality",
        "mip_iterations",
        "location",
        # z-scored values for L and R
        "ac_R_z",
        "ac_L_z",
        "diagu_R_z",
        "diagu_L_z",
        "diagl_R_z",
        "diagl_L_z",
        "diaga_R_z",
        "diaga_L_z",
        "cc_v2_R_z",
        "cc_v2_L_z",
        "cc_v1_R_z",
        "cc_v1_L_z",
        "disorder_R_z",
        "disorder_L_z",
        # z-scored differences
        "diff_ac_RvL_z",
        "diff_diagu_RvL_z",
        "diff_diagl_RvL_z",
        "diff_diaga_RvL_z",
        "diff_cc_v2_RvL_z",
        "diff_cc_v1_RvL_z",
        "diff_disorder_RvL_z",
        # z-scored abs. differences
        "abs_diff_ac_RvL_z",
        "abs_diff_diagu_RvL_z",
        "abs_diff_diagl_RvL_z",
        "abs_diff_diaga_RvL_z",
        "abs_diff_cc_v2_RvL_z",
        "abs_diff_cc_v1_RvL_z",
        "abs_diff_disorder_RvL_z",
        # z-scored other
        "psi_z",
        "efficiency",
        "efficiency_z",
        "load_capacity_ratio_z",
        "rt",
        "rt_solving",
        "max_disorder_z",
        "sum_disorder_z",
        "heuristic_optimality_z",
        "mip_iterations_z",
    ]
    + cc_opt_cols
].rename(columns=dict(xid="participant"))

data.to_csv(Path(EXPORT_PATH_DATA, STUDY, fout), index=False)

# %%
Path(EXPORT_PATH_DATA, STUDY, fout)

# %% [markdown]
# +++ only for exploratory data to be used for fitting composition complexity parameters and subsequent confirmatory trial sampling +++
#
# ## 3.2. Update Data
#
# - Now run `parameter_fitting/mixture_model_fit_pilot.ipynb` (adds column `diff_cc_v2_RvL_opt` for parameter-optimized composition complexity) without re-running the optimization, only run parts 1,2, 4 and 6, 7
# - load updated csv file here to check

# %%
# data = pd.read_csv(
#     Path(
#         EXPORT_PATH_DATA,
#         STUDY,
#         f"""data_preprocessed_updated_{str(TRIAL_FILE_ID).replace(".","-")}.csv""",
#     )
# )

# %% [markdown]
# ### 3.3. Update Data II

# %%
# data = pd.read_csv(Path(EXPORT_PATH_DATA, f"""data_preprocessed_updated_{str(TRIAL_FILE_ID).replace(".","-")}.csv"""))

# data["diff_diagu_RvL"] = data["diff_diagu_RvL"] *-1
# data["diff_diagl_RvL"] = data["diff_diagl_RvL"] *-1

# data["diff_diagu_RvL_z"] = data["diff_diagu_RvL_z"] *-1
# data["diff_diagl_RvL_z"] = data["diff_diagl_RvL_z"] *-1

# data.to_csv(Path(EXPORT_PATH_DATA, STUDY, "data_preprocessed_updated.csv"), index=False)
