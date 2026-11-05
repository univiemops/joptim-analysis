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
# Fit Complexity Metric Parameters to Pilot Data
# ===
#
# Note: solution metrics were already computed in `filter_extend_solutions.ipynb`.

# %%
import jupyter_black

jupyter_black.load()

# %% [markdown]
# ## 0. Imports & Setup

# %%
import pickle
import json
from glob import glob
from datetime import datetime
from pathlib import Path

# import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore, spearmanr, pearsonr
from scipy.optimize import minimize

from joptim.utils.export_path import get_export_paths
from joptim.mixture_models import composition_complexity_v2, composition_complexity_v1
from joptim.r_ordinal_logloss import compute_logloss, compute_logloss_hierarchical
from joptim.standardizer import zscore_LR


EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
TRIAL_FILE_EXPLORATORY = "1724688094.1341743"


with open(
    Path(EXPORT_PATH_DATA, "exploratory", "optimal_parameters_cc_v2.json"),
    "r",
) as fp:
    optimal_parameters_cc_v2_exploratory = json.load(
        fp,
    )

best_params_exploratory = optimal_parameters_cc_v2_exploratory["best_params"]
best_combination_exploratory = optimal_parameters_cc_v2_exploratory["best_combination"]


trial_data = {}

with open(Path(EXPORT_PATH_DATA, "exploratory", "trials.pkl"), "rb") as pickle_file:
    trial_data_tmp = pickle.load(pickle_file)
if "file_id" in trial_data_tmp:
    trial_data[str(trial_data_tmp["file_id"])] = trial_data_tmp["trials"]
del trial_data_tmp

for file_id in trial_data:
    print(f"File {file_id}: trials for {len(trial_data[file_id])} participants loaded.")

# %% [markdown]
# ## 1. Load pilot data and combine it with solution information
#
# - solution information: `task_loads`, `worker_capacities`, `solution_matrix`
#
# **NOTE: column `participant` is 1-indexed while `trial_id` is 0-indexed.**

# %%
fin = f"""data_preprocessed_{str(TRIAL_FILE_EXPLORATORY).replace(".","-")}.csv"""
fout = (
    f"""data_preprocessed_updated_{str(TRIAL_FILE_EXPLORATORY).replace(".","-")}.csv"""
)
print(f"Preprocessed data will be written to {fout}")

df_results = pd.read_csv(
    Path(EXPORT_PATH_DATA, "exploratory", fin),
    dtype={"trial_file_id": str},
)

# filter to trial_file_id
df_results = df_results.loc[
    df_results["trial_file_id"] == TRIAL_FILE_EXPLORATORY
].reset_index(drop=True)

print(
    f"""Filtered results to {len(df_results.participant.unique())} ({len(df_results)} trials) from exploratory study (trial file: {TRIAL_FILE_EXPLORATORY})."""
)

samples_preference = []

for trial_file_id in trial_data:
    for i, _ in enumerate(trial_data[trial_file_id]):
        trial_file_user_id = i + 1  # change pid to 1-indexed to match experimental data
        for trial_id, trial_info in enumerate(
            trial_data[trial_file_id][i]["preference_trials"]
        ):
            samples_preference.append(
                [
                    str(trial_file_id),
                    trial_file_user_id,
                    trial_id,
                    trial_info["task_loads"],
                    trial_info["worker_capacities"],
                    trial_info["solution_1"],
                    trial_info["solution_2"],
                    trial_info["use_case"],
                ]
            )

cols = [
    "trial_file_id",
    "trial_file_user_id",
    "trial_id",
    "task_loads",
    "worker_capacities",
    "solution_1",
    "solution_2",
    "use_case",
]

df_samples_preference = pd.DataFrame(samples_preference, columns=cols)

df_results["trial_file_id"] = df_results["trial_file_id"].astype(str)

df = df_results.merge(
    df_samples_preference,
    on=["trial_file_id", "trial_file_user_id", "trial_id"],
    how="left",
    validate="1:1",
)

# define as categorical
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
# ## 2. Recompute composition complexity values for each trial as sanity check

# %% [markdown]
# ### 2.1. Recompute
#
# - `diff_cc_v1_RvL_new` means recomputed

# %%
# Version 1 ("dirichlet")

# initial params
best_params_v1_initial = [0.5, 2.0, 0.8]
best_combination_v1_initial = ["absolute"]  # absolute, relative

df["cc_v1_L"] = df.apply(
    lambda row: composition_complexity_v1(
        row["solution_1"],
        row["worker_capacities"],
        row["task_loads"],
        alpha1=best_params_v1_initial[0],
        alpha2=best_params_v1_initial[1],
        theta=best_params_v1_initial[2],
        weighted=best_combination_v1_initial[0],
    ),
    axis=1,
)

df["cc_v1_R"] = df.apply(
    lambda row: composition_complexity_v1(
        row["solution_2"],
        row["worker_capacities"],
        row["task_loads"],
        alpha1=best_params_v1_initial[0],
        alpha2=best_params_v1_initial[1],
        theta=best_params_v1_initial[2],
        weighted=best_combination_v1_initial[0],
    ),
    axis=1,
)

df["diff_cc_v1_RvL_new"] = df["cc_v1_R"] - df["cc_v1_L"]

# Version 2 (more complex mixture model)

df["cc_v2_L"] = df.apply(
    lambda row: composition_complexity_v2(
        row["solution_1"],
        row["worker_capacities"],
        row["task_loads"],
        geom_p=best_params_exploratory[0],
        empty_param=best_params_exploratory[1],
        comp_alpha=best_params_exploratory[2],
        empty_distribution=best_combination_exploratory[0],
        dirichlet_correction=best_combination_exploratory[1],
    ),
    axis=1,
)

df["cc_v2_R"] = df.apply(
    lambda row: composition_complexity_v2(
        row["solution_2"],
        row["worker_capacities"],
        row["task_loads"],
        geom_p=best_params_exploratory[0],
        empty_param=best_params_exploratory[1],
        comp_alpha=best_params_exploratory[2],
        empty_distribution=best_combination_exploratory[0],
        dirichlet_correction=best_combination_exploratory[1],
    ),
    axis=1,
)

df["diff_cc_v2_RvL_new"] = df["cc_v2_R"] - df["cc_v2_L"]

# %% [markdown]
# ### 2.2. Sanity check

# %%
# Version 1 ("dirichlet")
indices_notequal = np.round(df["diff_cc_v1_RvL"], 5) != np.round(
    df["diff_cc_v1_RvL_new"], 5
)

if len(df[indices_notequal]) == 0:
    print("Check V1 passed!")
else:
    print("Check not passed. Composition metric V1 differs in the following trials:\n")
    print(
        df[indices_notequal][
            [
                "participant",
                "trial_id",
                "use_case",
                "diff_cc_v1_RvL",
                "diff_cc_v1_RvL_new",
            ]
        ]
    )

# Version 2 (more complex variant)
indices_notequal = np.round(df["diff_cc_v2_RvL"], 5) != np.round(
    df["diff_cc_v2_RvL_new"], 5
)

if len(df[indices_notequal]) == 0:
    print("Check V2 passed!")
else:
    print("Check not passed. Composition metric V2 differs in the following trials:\n")
    print(
        df[indices_notequal][
            [
                "participant",
                "trial_id",
                "use_case",
                "diff_cc_v2_RvL",
                "diff_cc_v2_RvL_new",
            ]
        ]
    )

del df["diff_cc_v2_RvL_new"]

# %% [markdown]
# ## 3. Optimize parameters of complexity metric so that `diff_cc_v2_RvL` best matches the `choice_RvL` outcome variable in the pilot experiment

# %% [markdown]
# ### 3.1. Use the initial complexity scores with default parameters and compare ...

# %% [markdown]
# Initial values and Spearman Correlation

# %%
sns.regplot(
    data=df,
    x="diff_cc_v2_RvL",
    y="choice_RvL",
    scatter_kws=dict(marker=".", color="steelblue", s=60, alpha=0.02),
    label=f"""r = {spearmanr(df["choice_RvL"], df["diff_cc_v2_RvL"])[0]:.3f}""",
)
plt.legend()
plt.show()

# %% [markdown]
# Initial Values and Ordinal Regression

# %%
outcome = df["choice_cat_RvL"]
predictor = df["diff_cc_v2_RvL"]
group = df["participant"]

compute_logloss(outcome, predictor)

# %%
# just to compare. We won't use the hierarchical version for fitting
compute_logloss_hierarchical(outcome, predictor, group)

# %% [markdown]
# ### 3.2. Find optimal parameters that make `diff_cc_v2_RvL` maximally correlated with `choice_RvL`

# %% [markdown]
# **Functions**

# %% [markdown]
# Changes here : `trials` instead of solutions and the objective function needs to compute 2 complexities and compute the difference. this difference needs then to be correlated with the target metric `choice_RvL`

# %%
from optimparallel import minimize_parallel


def objective_fun(
    params,
    trials,
    outcome,  # choice_RvL
    empty_distribution,
    dirichlet_correction,
):
    """
    Objective function for optimization.

    This function fits an ordinal model to the observed data data
    using the given parameters and trials. It is designed
    to be used as the objective function in an optimization routine.

    Parameters
    ----------
    params : array_like
        An array of parameters [geom_p, empty_param, comp_alpha] to be optimized.

    trials : array_like
        A list of trial dictionaries, where each dictionary contains solutions,
        worker capacities, and task loads.

    outcome : array_like
        The observed data to be matched against the predicted data.

    empty_distribution : str
        The type of distribution assumed for the empty spaces. One of ["normal", "laplace", "cont_bern"].

    dirichlet_correction : bool
        Whether to apply Dirichlet correction.

    Returns
    -------
    float
        The log-loss between the one-hot encoded outcomes and the predicted probabilities for each outcome.

    Notes
    -----
    This function uses `composition_complexity` to compute the complexity of each trial's solutions
    and calculates the difference in composition complexity. The predicted data is then compared with
    the observed data, and the negative correlation coefficient is returned for optimization purposes.
    """

    geom_p, empty_param, comp_alpha = params
    predictor = []

    for trial in trials:
        cc_v2_L = composition_complexity_v2(
            trial["solution_1"],
            trial["worker_capacities"],
            trial["task_loads"],
            geom_p=geom_p,
            empty_param=empty_param,
            comp_alpha=comp_alpha,
            empty_distribution=empty_distribution,
            dirichlet_correction=dirichlet_correction,
        )
        cc_v2_R = composition_complexity_v2(
            trial["solution_2"],
            trial["worker_capacities"],
            trial["task_loads"],
            geom_p=geom_p,
            empty_param=empty_param,
            comp_alpha=comp_alpha,
            empty_distribution=empty_distribution,
            dirichlet_correction=dirichlet_correction,
        )
        diff_cc_v2_RvL = cc_v2_R - cc_v2_L
        predictor.append(diff_cc_v2_RvL)
    predictor = np.array(predictor)
    logloss = compute_logloss(outcome, predictor)

    with open("optimization_v2.log", "a") as fp:
        fp.write(f"log-loss, {geom_p},{empty_param},{comp_alpha},{logloss}\n")

    return logloss


def optimize_for_all_combinations(trials, outcome, parallel=False):
    """
    Optimize parameters for all combinations of given settings.

    This function iterates over specified distributions and correction settings, performing
    optimization to find the best parameters that maximize negative correlation with observed data.

    Parameters
    ----------
    trials : array_like
        A list of trial dictionaries, where each dictionary contains solutions,
        worker capacities, and task loads.

    outcome : array_like
        The observed data to be matched against during the optimization.

    parallel : bool, optional
        If True, the optimization will be run in parallel. By default, this is False.

    Returns
    -------
    best_params : ndarray
        The best parameters found during the optimization process.

    best_combination : tuple
        A tuple containing the best 'empty_distribution' and 'dirichlet_correction' found.

    best_result : float
        The best correlation value obtained.

    best_result : OptimizeResult
        The optimization result corresponding to the best correlation.

    all_results : list
        A list of results for all combinations of parameters tested. Each element in the
        list is a list containing the result, `empty_distribution`, and `dirichlet_correction`.

    Notes
    -----
    This function tests multiple combinations of `empty_distribution` and `dirichlet_correction`,
    and optimizes three continuous parameters: `geom_p`, `empty_param`, and `comp_alpha`, using
    bounded optimizations.

    If `parallel` is set to True, a parallel optimization method will be employed; otherwise,
    a sequential optimization method is used.
    """

    best_result = np.float64("inf")
    best_params = None
    best_combination = None
    all_results = []

    # Define discrete parameters to test
    empty_distributions = ["normal", "laplace", "cont_bern"]
    dirichlet_corrections = [True, False]

    for empty_distribution in empty_distributions:
        for dirichlet_correction in dirichlet_corrections:
            try:
                # Initial guess for the continuous parameters
                initial_params = [0.5, 0.05, 2.0]  # [geom_p, empty_param, comp_alpha]
                bounds = [
                    (0.001, 0.999),
                    (0.001, 0.999),
                    (1e-5, 10),
                ]  # Bounds for the parameters

                if not parallel:
                    # Optimization
                    result = minimize(
                        objective_fun,
                        initial_params,
                        args=(
                            trials,
                            outcome,
                            empty_distribution,
                            dirichlet_correction,
                        ),
                        bounds=bounds,
                        method="L-BFGS-B",
                        # method="Powell",
                        # method="Nelder-Mead",
                    )
                else:
                    result = minimize_parallel(
                        objective_fun,
                        initial_params,
                        args=(
                            trials,
                            outcome,
                            empty_distribution,
                            dirichlet_correction,
                        ),
                        bounds=bounds,
                        # method="L-BFGS-B",
                    )
                print(
                    "empty_distribution:",
                    empty_distribution,
                    "- dirichlet_correction:",
                    dirichlet_correction,
                    "- parallel:",
                    parallel,
                    "- best_result:",
                    best_result,
                    "- result.fun:",
                    result.fun,
                )
                all_results.append([result, empty_distribution, dirichlet_correction])

                if result.fun < best_result:
                    best_result = result.fun
                    best_params = result.x
                    best_combination = (
                        empty_distribution,
                        dirichlet_correction,
                    )
            except Exception as e:
                print(e)

    return best_params, best_combination, best_result, all_results


# %% [markdown]
# **Find Optimal Parameters**

# %%
start_time = datetime.now()

trials = df[["task_loads", "worker_capacities", "solution_1", "solution_2"]].to_dict(
    "records"
)
choice_RvL = df["choice_cat_RvL"].values

(
    best_params,
    best_combination,
    best_result,
    _,
) = optimize_for_all_combinations(trials, choice_RvL, parallel=False)

computing_time = (datetime.now() - start_time).total_seconds()

print("computing_time", computing_time)

# %%
optimal_parameters_cc_v2_confirmatory = {
    "description": "optimal parameters for composition complexity V2 based on exploratory data (see trial_file_id).",
    "trial_file_id": trial_file_id,
    "best_params": best_params,
    "best_combination": best_combination,
    "best_result": best_result,
}

with open(
    Path(EXPORT_PATH_DATA, "confirmatory", "optimal_parameters_cc_v2.json"),
    "w",
) as fp:
    json.dump(
        optimal_parameters_cc_v2_confirmatory,
        fp,
    )

# %% [markdown]
# ## 4. Update dataframe with optimized composition complexity V2

# %% [markdown]
# Load optimal parameters

# %%
with open(
    Path(EXPORT_PATH_DATA, "confirmatory", "optimal_parameters_cc_v2.json"),
    "r",
) as fp:
    optimal_parameters_cc_v2_confirmatory = json.load(
        fp,
    )

best_params = optimal_parameters_cc_v2_confirmatory["best_params"]
best_combination = optimal_parameters_cc_v2_confirmatory["best_combination"]

print("Parameters:\n")
print("Exploratory:\n")
print(json.dumps(optimal_parameters_cc_v2_exploratory, indent=4))
print("\nConfirmatory:\n")
print(json.dumps(optimal_parameters_cc_v2_confirmatory, indent=4))

# %% [markdown]
# ### 4.1. Compare correlations initial vs. optimized

# %% [markdown]
# Inital (optimized via PCA)

# %%
print("Initial parameters:", best_params_exploratory, best_combination_exploratory)

print(f"""pearson corr:   {pearsonr(df["choice_RvL"], df["diff_cc_v2_RvL"])[0]:.4f}""")
print(f"""spearman corr:  {spearmanr(df["choice_RvL"], df["diff_cc_v2_RvL"])[0]:.4f}""")
print(
    f"""log-loss:        {compute_logloss(df["choice_cat_RvL"], df["diff_cc_v2_RvL"]):.4f}"""
)

# %% [markdown]
# ### 4.2. update dataframe with optimized variable

# %%
df["cc_v2_L_opt"] = df.apply(
    lambda row: composition_complexity_v2(
        row["solution_1"],
        row["worker_capacities"],
        row["task_loads"],
        geom_p=best_params[0],
        empty_param=best_params[1],
        comp_alpha=best_params[2],
        empty_distribution=best_combination[0],
        dirichlet_correction=best_combination[1],
    ),
    axis=1,
)

df["cc_v2_R_opt"] = df.apply(
    lambda row: composition_complexity_v2(
        row["solution_2"],
        row["worker_capacities"],
        row["task_loads"],
        geom_p=best_params[0],
        empty_param=best_params[1],
        comp_alpha=best_params[2],
        empty_distribution=best_combination[0],
        dirichlet_correction=best_combination[1],
    ),
    axis=1,
)

df["diff_cc_v2_RvL_opt"] = df["cc_v2_R_opt"] - df["cc_v2_L_opt"]
df["abs_diff_cc_v2_RvL_opt"] = df["diff_cc_v2_RvL_opt"].abs()


corr_r = pearsonr(df["choice_RvL"], df["diff_cc_v2_RvL_opt"])[0]
corr_rho = spearmanr(df["choice_RvL"], df["diff_cc_v2_RvL_opt"])[0]
logloss_h = compute_logloss_hierarchical(
    df["choice_cat_RvL"], df["diff_cc_v2_RvL_opt"], df["participant"]
)
logloss = compute_logloss(df["choice_cat_RvL"], df["diff_cc_v2_RvL_opt"])

print(f"""pearson corr:      {corr_r:.4f}""")
print(f"""spearman corr:     {corr_rho:.4f}""")
print(f"""log-loss:           {logloss:.4f}""")

# %% [markdown]
# Manipulate CC to be on a different scale and see if this changes correlation and logloss

# %%
df["cc_v2_R_opt_tmp"] = (df["cc_v2_R_opt"] - 3.2) * 1000
df["cc_v2_L_opt_tmp"] = (df["cc_v2_L_opt"] - 3.2) * 1000

df["diff_cc_v2_RvL_opt_tmp"] = df["cc_v2_R_opt_tmp"] - df["cc_v2_L_opt_tmp"]

# %%
sns.regplot(
    data=df,
    x="diff_cc_v2_RvL_opt_tmp",
    y="choice_RvL",
    scatter_kws=dict(marker=".", color="steelblue", s=60, alpha=0.02),
    label=f"""r = {spearmanr(df["choice_RvL"], df["diff_cc_v2_RvL_opt"])[0]:.3f}""",
)
plt.legend()
plt.show()

# %%
logloss = compute_logloss(df["choice_cat_RvL"], df["diff_cc_v2_RvL_opt_tmp"])

print(f"""pearson corr:      {corr_r:.4f}""")
print(f"""spearman corr:     {corr_rho:.4f}""")
print(f"""log-loss:           {logloss:.4f}""")

# %% [markdown]
# ---
# TODO:
# - Visual sanity check. Show what makes a very complex, a medium complex and a simple solution. Despite the 3.2.
#

# %% [markdown]
# * implement jupytext and maybe even restructure / flatten?
# * why the 3.2 outcome? is it problematic?
# * push trials.pkl form June to experiment and collect participants.

# %% [markdown]
# ---

# %% [markdown]
# ### 4.3. Save updated dataframe

# %% [markdown]
# Sanity check

# %%
print(
    sum(df["cc_v2_R_opt"] - df["cc_v2_L_opt"] == df["diff_cc_v2_RvL_opt"])
    / len(df)
    * 100,
    end=" ",
)
print("% equal")

# %% [markdown]
# Standardizing

# %%
df[["cc_v2_L_opt_z", "cc_v2_R_opt_z"]] = zscore_LR(df, "cc_v2_L_opt", "cc_v2_R_opt")
df["diff_cc_v2_RvL_opt_z"] = df["cc_v2_R_opt_z"] - df["cc_v2_L_opt_z"]
df["abs_diff_cc_v2_RvL_opt_z"] = df["diff_cc_v2_RvL_opt_z"].abs()

cols = list(df_results.columns) + [
    "cc_v2_R_opt",
    "cc_v2_L_opt",
    "cc_v2_R_opt_z",
    "cc_v2_L_opt_z",
    "diff_cc_v2_RvL_opt",
    "diff_cc_v2_RvL_opt_z",
    "abs_diff_cc_v2_RvL_opt",
    "abs_diff_cc_v2_RvL_opt_z",
]

df[cols].to_csv(Path(EXPORT_PATH_DATA, "data_preprocessed_updated.csv"), index=False)

# %% [markdown]
# ## 5. Optimize parameters of dirichlet-only complexity metric so that `diff_cc_v1_RvL` best matches the `choice_RvL` outcome variable in the pilot experiment

# %% [markdown]
# ### 5.1. Use the initial complexity scores with default parameters and compare ...

# %% [markdown]
# ### 5.2. Find optimal parameters that make `diff_cc_v1_RvL` maximally correlated with `choice_RvL`

# %% [markdown]
# Initial values

# %%
print(spearmanr(df["choice_RvL"], df["diff_cc_v1_RvL"]))

# %% [markdown]
# **Functions**

# %%
from optimparallel import minimize_parallel


def objective_fun(
    params,
    trials,
    outcome,  # choice_RvL
    weighted,
):
    """
    Objective function for optimization.

    This function computes the correlation between the observed data and
    the predicted data generated using the given parameters and trials. It is designed
    to be used as the objective function in an optimization routine.

    Parameters
    ----------
    params : array_like
        An array of parameters [alpha1, alpha2, theta] to be optimized.

    trials : array_like
        A list of trial dictionaries, where each dictionary contains solutions,
        worker capacities, and task loads.

    X_observed : array_like
        The observed data to be matched against the predicted data.

    weighted : str
        The type of weighting the individual bins/workers by size. One of [None, "absolute", "relative"].

    Returns
    -------
    float
        The negative correlation coefficient between the observed data and the predicted data.
    """

    alpha1, alpha2, theta = params
    predictor = []
    group = []

    # iterate over all trials
    for trial in trials:
        cc_v2_L = composition_complexity_v1(
            x=trial["solution_1"],
            w=trial["worker_capacities"],
            z=trial["task_loads"],
            alpha1=alpha1,
            alpha2=alpha2,
            theta=theta,
            weighted=weighted,
        )
        cc_v2_R = composition_complexity_v1(
            x=trial["solution_2"],
            w=trial["worker_capacities"],
            z=trial["task_loads"],
            alpha1=alpha1,
            alpha2=alpha2,
            theta=theta,
            weighted=weighted,
        )
        diff_cc_v2_RvL = cc_v2_R - cc_v2_L  # diff. composition complexity RvL
        predictor.append(diff_cc_v2_RvL)
        group.append(trial["participant"])  # remove?

    predictor = np.array(predictor)
    group = np.array(group)  # remove?

    logloss = compute_logloss(outcome, predictor)

    # log optimization process
    with open("optimization_v1.log", "a") as fp:
        fp.write(f"{weighted}, {alpha1},{alpha2},{theta},{logloss}\n")

    return logloss


def optimize_for_all_combinations(trials, outcome, parallel=False):
    """
    Optimize parameters for all combinations of given settings.

    This function iterates over specified distributions and correction settings, performing
    optimization to find the best parameters that maximize the neg. correlation with observed data.

    Parameters
    ----------
    trials : array_like
        A list of trial dictionaries, where each dictionary contains solutions,
        worker capacities, and task loads.

    X_observed : array_like
        The observed data to be matched against during the optimization.

    parallel : bool, optional
        If True, the optimization will be run in parallel. By default, this is False.

    Returns
    -------
    best_params : ndarray
        The best parameters found during the optimization process.

    best_combination : tuple
        A tuple containing the best 'empty_distribution' and 'dirichlet_correction' found.

    best_result : float
        The best correlation value obtained.

    best_result : OptimizeResult
        The optimization result corresponding to the best correlation.

    all_results : list
        A list of results for all combinations of parameters tested. Each element in the
        list is a list containing the result, `empty_distribution`, and `dirichlet_correction`.

    Notes
    -----
    This function tests multiple combinations of `empty_distribution` and `dirichlet_correction`,
    and optimizes three continuous parameters: `alpha1`, `alpha2`, and `theta`, using
    bounded optimizations.

    If `parallel` is set to True, a parallel optimization method will be employed; otherwise,
    a sequential optimization method is used.
    """

    best_result = float("inf")
    best_params = None
    best_combination = [None]
    all_results = []

    # Define discrete parameters to test
    weighted_list = [None, "absolute", "relative"]

    for weighted in weighted_list:
        try:
            # Initial guess for the continuous parameters
            initial_params = [0.5, 2, 0.8]  # [alpha1, alpha2, theta]
            bounds = [
                (0.001, 100),  # alpha of dirichlet 1
                (0.001, 100),  # alpha of dirichlet 2
                (1e-6, 1 - 1e-6),  # theta of geometric
            ]  # Bounds for the parameters

            if not parallel:
                # Optimization
                result = minimize(
                    objective_fun,
                    initial_params,
                    args=(
                        trials,
                        outcome,
                        weighted,
                    ),
                    bounds=bounds,
                    method="L-BFGS-B",
                    # method="Powell",
                    # method="Nelder-Mead",
                )
            else:
                result = minimize_parallel(
                    objective_fun,
                    initial_params,
                    args=(
                        trials,
                        outcome,
                        weighted,
                    ),
                    bounds=bounds,
                    # method="L-BFGS-B",
                )
            all_results.append([result, weighted])

            if result.fun < best_result:
                best_result = result.fun
                best_params = result.x
                best_combination = [weighted]
        except Exception as e:
            print(e)

    return best_params, best_combination, best_result, all_results


# %% [markdown]
# **Find Optimal Parameters**

# %%
start_time = datetime.now()

trials = df[
    ["participant", "task_loads", "worker_capacities", "solution_1", "solution_2"]
].to_dict("records")
choice_RvL = df["choice_cat_RvL"].values

(
    best_params,
    best_combination,
    best_result,
    _,
) = optimize_for_all_combinations(trials, choice_RvL, parallel=False)

computing_time = (datetime.now() - start_time).total_seconds()

print("computing_time", computing_time)

# %%
optimal_parameters_cc_v1_confirmatory = {
    "description": "optimal parameters for composition complexity V2 based on exploratory data (see trial_file_id).",
    "trial_file_id": trial_file_id,
    "best_params": best_params,
    "best_combination": best_combination,
    "best_result": best_result,
}

with open(
    Path(EXPORT_PATH_DATA, "confirmatory", "optimal_parameters_cc_v1.json"),
    "w",
) as fp:
    json.dump(
        optimal_parameters_cc_v1_confirmatory,
        fp,
    )

# %% [markdown]
# ## 6. Update dataframe with optimized composition complexity V1

# %% [markdown]
# Load

# %%
with open(
    Path(EXPORT_PATH_DATA, "confirmatory", "optimal_parameters_cc_v1.json"),
    "r",
) as fp:
    optimal_parameters_cc_v1_confirmatory = json.load(
        fp,
    )

best_params = optimal_parameters_cc_v1_confirmatory["best_params"]
best_combination = optimal_parameters_cc_v1_confirmatory["best_combination"]

print("Parameters:\n")
print("\nConfirmatory:\n")
print(json.dumps(optimal_parameters_cc_v1_confirmatory, indent=4))

# %% [markdown]
# ### 6.1. Update

# %% [markdown]
# Inital

# %%
best_params_inital = [0.5, 2.0, 0.8]
best_combination_inital = ["absolute"]

corr_r = pearsonr(df["choice_RvL"], df["diff_cc_v1_RvL"])[0]
corr_rho = spearmanr(df["choice_RvL"], df["diff_cc_v1_RvL"])[0]
logloss_h = compute_logloss_hierarchical(
    df["choice_cat_RvL"], df["diff_cc_v1_RvL"], df["participant"]
)
logloss = compute_logloss(df["choice_cat_RvL"], df["diff_cc_v1_RvL"])

print(f"""pearson corr:      {corr_r:.4f}""")
print(f"""spearman corr:     {corr_rho:.4f}""")
print(f"""log-loss (hier.):   {logloss_h:.4f}""")
print(f"""log-loss:           {logloss:.4f}""")

# %%
df["cc_v1_L_opt"] = df.apply(
    lambda row: composition_complexity_v1(
        x=row["solution_1"],
        w=row["worker_capacities"],
        z=row["task_loads"],
        alpha1=best_params[0],
        alpha2=best_params[1],
        theta=best_params[2],
        weighted=best_combination[0],
    ),
    axis=1,
)

df["cc_v1_R_opt"] = df.apply(
    lambda row: composition_complexity_v1(
        x=row["solution_2"],
        w=row["worker_capacities"],
        z=row["task_loads"],
        alpha1=best_params[0],
        alpha2=best_params[1],
        theta=best_params[2],
        weighted=best_combination[0],
    ),
    axis=1,
)

df["diff_cc_v1_RvL_opt"] = df["cc_v1_R_opt"] - df["cc_v1_L_opt"]
df["abs_diff_cc_v1_RvL_opt"] = df["diff_cc_v1_RvL_opt"].abs()

corr_r = pearsonr(df["choice_RvL"], df["diff_cc_v1_RvL_opt"])[0]
corr_rho = spearmanr(df["choice_RvL"], df["diff_cc_v1_RvL_opt"])[0]
logloss_h = compute_logloss_hierarchical(
    df["choice_cat_RvL"], df["diff_cc_v1_RvL_opt"], df["participant"]
)
logloss = compute_logloss(df["choice_cat_RvL"], df["diff_cc_v1_RvL_opt"])

print(f"""pearson corr:      {corr_r:.4f}""")
print(f"""spearman corr:     {corr_rho:.4f}""")
print(f"""log-loss (hier.):   {logloss_h:.4f}""")
print(f"""log-loss:           {logloss:.4f}""")

# %% [markdown]
# Note: Above value is for `df` (complete results data), fitting was done on `df` (pilot data only)

# %% [markdown]
# ### 6.2. Save updated dataframe

# %% [markdown]
# Sanity check

# %%
print(
    sum(df["cc_v1_R_opt"] - df["cc_v1_L_opt"] == df["diff_cc_v1_RvL_opt"])
    / len(df)
    * 100,
    end=" ",
)
print("% equal")

# %% [markdown]
# Standardizing
#
# Part with for version 2 is already done in Section 4, is repeated here (may be redundant)

# %%
df[["cc_v1_L_opt_z", "cc_v1_R_opt_z"]] = zscore_LR(df, "cc_v1_L_opt", "cc_v1_R_opt")
df["diff_cc_v1_RvL_opt_z"] = df["cc_v1_R_opt_z"] - df["cc_v1_L_opt_z"]
df["abs_diff_cc_v1_RvL_opt_z"] = df["diff_cc_v1_RvL_opt_z"].abs()

df[["cc_v2_L_opt_z", "cc_v2_R_opt_z"]] = zscore_LR(df, "cc_v2_L_opt", "cc_v2_R_opt")
df["diff_cc_v2_RvL_opt_z"] = df["cc_v2_R_opt_z"] - df["cc_v2_L_opt_z"]
df["abs_diff_cc_v2_RvL_opt_z"] = df["diff_cc_v2_RvL_opt_z"].abs()


cols = list(df_results.columns) + [
    "cc_v2_R_opt",
    "cc_v2_L_opt",
    "cc_v2_R_opt_z",
    "cc_v2_L_opt_z",
    "cc_v1_R_opt",
    "cc_v1_L_opt",
    "cc_v1_R_opt_z",
    "cc_v1_L_opt_z",
    "diff_cc_v2_RvL_opt",
    "diff_cc_v1_RvL_opt",
    "diff_cc_v2_RvL_opt_z",
    "diff_cc_v1_RvL_opt_z",
    "abs_diff_cc_v2_RvL_opt",
    "abs_diff_cc_v1_RvL_opt",
    "abs_diff_cc_v2_RvL_opt_z",
    "abs_diff_cc_v1_RvL_opt_z",
]

df[cols].to_csv(Path(EXPORT_PATH_DATA, fout), index=False)

# %% [markdown]
# ## 7. Update Problems/Solutions ...
#
# ... in `scripts/experiment/extend_solutions_for_confirmatory_analysis.ipynb`

# %% [markdown]
# ## 8. Load dataframe for final comparison between complexity metric V1 and V2 and for testing
#
# NOTE: This is for all data (not only thep pilot data. Pilot data is relevant for decision about parameters and between v1 and v2)

# %%
df = pd.read_csv(
    Path(EXPORT_PATH_DATA, fout),
    dtype={"trial_file_id": str},
)

# %% [markdown]
# Maybe include this by default

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
    categories=[
        "definitely left",
        "slightly left",
        "slightly right",
        "definitely right",
    ],
    ordered=True,
)

# %% [markdown]
# Composition Prior Version 1

# %%
spearmanr(df["choice_RvL"], df["diff_cc_v1_RvL_z"])

# %%
spearmanr(df["choice_RvL"], df["diff_cc_v1_RvL_opt_z"])

# %%
compute_logloss(df["choice_cat_RvL"], df["diff_cc_v1_RvL_opt_z"])

# %% [markdown]
# Composition Prior Version 2

# %%
spearmanr(df["choice_RvL"], df["diff_cc_v2_RvL_z"])

# %%
spearmanr(df["choice_RvL"], df["diff_cc_v2_RvL_opt_z"])

# %%
compute_logloss(df["choice_cat_RvL"], df["diff_cc_v2_RvL_opt_z"])
