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
# Using PCA to Extract One Variable from all the Solution Metrics
# ===
#
# Note: solution metrics were already computed in `extend_filter_solutions.ipynb`.
#
# What this does:
# - ...

# %%
import jupyter_black

jupyter_black.load()

# %% [markdown]
# ## 0. Imports & Setup

# %%
import json
import pickle
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import seaborn as sns
from joptim.utils.export_path import get_export_paths
from joptim.initial_solution_metrics import (
    avg_discrepancy_index,
    avg_ratio_index,
    assignment_variance_index,
)
from joptim.mixture_models import composition_complexity_v2
from scipy.stats import pearsonr, zscore
from sklearn.decomposition import PCA
from scipy.optimize import minimize

plt.rcParams["figure.facecolor"] = "white"

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()


with open(
    Path(EXPORT_PATH_DATA, "problems_filtered_incl_solution_metrics.pkl"), "rb"
) as pickle_file:
    problems = pickle.load(pickle_file)
    print(len(problems), "problems loaded.")

# %% [markdown]
# Add missing solution metrics (if they become more important later, we may want to include them already in `GAP Solutions.ipynb`)

# %%
for problem in problems:
    assignment_variance_list = []
    avg_discrepancy_list = []
    avg_ratio_list = []

    for s in problem["optimal_solutions"]:
        assignment_variance_list.append(assignment_variance_index(problem, s))
        avg_discrepancy_list.append(avg_discrepancy_index(problem, s))
        avg_ratio_list.append(avg_ratio_index(problem, s))

    problem["assignment_variance"] = assignment_variance_list
    problem["avg_discrepancy"] = avg_discrepancy_list
    problem["avg_ratio"] = avg_ratio_list

# %% [markdown]
# Extract solutions and metrics for all problem instances.

# %%
solutions = []

for problem in problems:
    for i, s in enumerate(problem["optimal_solutions"]):
        solution = {
            "id": problem["id"] + "_" + str(i),
            "worker_capacities": problem["worker_capacities"],
            "task_loads": problem["task_loads"],
            "solution_matrix": s,
            "assignment_variance": problem["assignment_variance"][i],
            "avg_discrepancy": problem["avg_discrepancy"][i],
            "avg_ratio": problem["avg_ratio"][i],
        }
        solutions.append(solution)

df = pd.DataFrame(solutions)

print(len(solutions), "solutions")

# %% [markdown]
# ## 1. Visualize Solution Metrics

# %%
df_metrics = df[["assignment_variance", "avg_discrepancy", "avg_ratio"]]

mask = np.zeros_like(df_metrics.corr())
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(4.5, 4.5))
# sns.set(style="ticks", rc={'figure.figsize':(5.5,5.5)})
# sns.set(style="ticks", font_scale = 0.9)
sns.heatmap(
    df_metrics.corr(),
    square=True,
    annot=True,
    cmap="cividis",
    fmt=".2f",
    cbar=False,
    # center=0,
    # vmin=-1,
    # vmax=1,
    mask=mask,
)
plt.xticks(
    0.5 + np.array(range(len(df_metrics.columns) - 1)), (list(df_metrics.columns[:-1]))
)
plt.yticks(
    1.5 + np.array(range(len(df_metrics.columns) - 1)), (list(df_metrics.columns[1:]))
)
plt.tight_layout()
plt.title("Solution metrics")
plt.savefig(Path(EXPORT_PATH_PUB, "metrics_correlations.png"), dpi=200)

# %% [markdown]
# Takes too long ...

# %% [raw]
# # pltsklearn.decomposition
# plt.rc("font", size=12)
# # sns.set_theme("paper")
# # sns.set(style="ticks")
# # sns.set(style="ticks", font_scale = 1.2)
#
# g = sns.PairGrid(df_metrics, height=2, aspect=1)
# g.map_diag(sns.histplot, bins=20)
# g.map_offdiag(
#     sns.regplot,
#     scatter_kws=dict(color="steelblue", alpha=10_000 / len(df_metrics), marker="o", s=10),
#     line_kws=dict(color="salmon"),
# )
#
# plt.suptitle("Problem metrics")
# plt.tight_layout()
# plt.savefig(Path(EXPORT_PATH_PUB, "solution_metrics_gridplot.png"), dpi=200)

# %% [markdown]
# ## 2. PCA
#
# Extract 1 component for assignment_variance avg_discrepancy avg_ratio.

# %%
pca = PCA(n_components=1)

df["compound"] = pca.fit_transform(zscore(df_metrics, axis=0)).reshape(-1)

# %% [markdown]
# Check correlation of new metric with old old metrics

# %%
df[["assignment_variance", "avg_discrepancy", "avg_ratio", "compound"]].corr()

# %% [markdown]
# <div class='alert alert-warning'>Try same with standardized data.</div>

# %% [markdown]
# ## 3. Optimize parameters of complexity prior to best match the compound metric
#
# ### 3.1. Compute initial complexity scores with default parameters and compare with compound metric
#
# Initial values

# %%
complexities = []

# iterate over all solutions
for s in solutions:
    # iterate over all workers of a solution
    complexity = composition_complexity_v2(
        s["solution_matrix"],
        s["worker_capacities"],
        s["task_loads"],
        0.5,
        0.05,
        2.0,
        "cont_bern",
        True,
    )
    complexities.append(complexity)

# %% [markdown]
# Add complexity to DataFrame `df`

# %%
df["complexity_initial"] = complexities

# %% [markdown]
# Compute initial correlation between compound metric and complexity metric

# %%
pearsonr(df["compound"], df["complexity_initial"])

# %%
plt.figure(figsize=(12, 9))
sns.regplot(
    data=df,
    x="compound",
    y="complexity_initial",
    scatter_kws=dict(marker=".", color="steelblue", s=60, alpha=0.02),
)

# %% [markdown]
# ### 3.2. Find optimal parameters that make `complexity` maximally similar to `compound`
#
# **Functions**

# %%
from optimparallel import minimize_parallel
from concurrent.futures import ProcessPoolExecutor


def objective_fun(
    params,
    solutions,
    X_observed,
    empty_distribution,
    dirichlet_correction,
):
    geom_p, empty_param, comp_alpha = params
    X_predicted = []

    # iterate over all solutions
    for s in solutions:
        complexity = composition_complexity_v2(
            s["solution_matrix"],
            s["worker_capacities"],
            s["task_loads"],
            geom_p=geom_p,
            empty_param=empty_param,
            comp_alpha=comp_alpha,
            empty_distribution=empty_distribution,
            dirichlet_correction=dirichlet_correction,
        )
        X_predicted.append(complexity)
    X_predicted = np.array(X_predicted)
    return -np.corrcoef(X_observed, X_predicted)[0][1]


def optimize_for_all_combinations(solutions, X_observed, parallel=False):
    best_corr = float("-inf")
    best_params = None
    best_combination = None
    best_result = None
    all_results = []

    # Define discrete parameters to test
    empty_distributions = ["normal", "laplace", "cont_bern"]
    dirichlet_corrections = [True, False]

    for empty_distribution in empty_distributions:
        for dirichlet_correction in dirichlet_corrections:
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
                        solutions,
                        X_observed,
                        empty_distribution,
                        dirichlet_correction,
                    ),
                    bounds=bounds,
                    method="L-BFGS-B",
                )
            else:
                result = minimize_parallel(
                    objective_fun,
                    initial_params,
                    args=(
                        solutions,
                        X_observed,
                        empty_distribution,
                        dirichlet_correction,
                    ),
                    bounds=bounds,
                    # method="L-BFGS-B",
                )
            all_results.append([result, empty_distribution, dirichlet_correction])
            if result.fun > best_corr:
                best_corr = -result.fun
                best_params = result.x
                best_combination = (
                    empty_distribution,
                    dirichlet_correction,
                )
                best_result = result

    return best_params, best_combination, best_corr, best_result, all_results


# for parallelized loops


def optimize_single_combination(args):
    solutions, X_observed, parallel, empty_distribution, dirichlet_correction = args
    initial_params = [0.5, 0.05, 2.0]
    bounds = [(0.001, 0.999), (0.001, 0.999), (1e-5, 10)]

    if not parallel:
        result = minimize(
            objective_fun,
            initial_params,
            args=(
                solutions,
                X_observed,
                empty_distribution,
                dirichlet_correction,
            ),
            bounds=bounds,
            method="L-BFGS-B",
        )
    else:
        result = minimize_parallel(
            objective_fun,
            initial_params,
            args=(
                solutions,
                X_observed,
                empty_distribution,
                dirichlet_correction,
            ),
            bounds=bounds,
        )

    return result, empty_distribution, dirichlet_correction


def optimize_for_all_combinations_parallel_loop(solutions, X_observed, parallel=False):
    best_corr = float("-inf")
    best_params = None
    best_combination = None
    best_result = None
    all_results = []

    empty_distributions = ["normal", "laplace", "cont_bern"]
    dirichlet_corrections = [True, False]

    args = [
        (solutions, X_observed, parallel, ed, dc)
        for ed in empty_distributions
        for dc in dirichlet_corrections
    ]

    with ProcessPoolExecutor() as executor:
        results = executor.map(optimize_single_combination, args)

    for result, empty_distribution, dirichlet_correction in results:
        all_results.append([result, empty_distribution, dirichlet_correction])
        if result.fun > best_corr:
            best_corr = -result.fun
            best_params = result.x
            best_combination = (empty_distribution, dirichlet_correction)
            best_result = result

    return best_params, best_combination, best_corr, best_result, all_results


# %% [markdown]
# **Find Optimal Parameters**

# %%
from datetime import datetime

# %%
# %%time


result_dict = {
    "no parallelization": {},
    "parallel loops": {},
    "minimize_parallel": {},
    "parallel loops and minimize_parallel": {},
}

n_sols_list = [2, 3, 4, 6, 9, 12, 15, 18, 24, 32, 45, 55, 70, 90, 120, 145]

for parallel_type in list(result_dict):

    result_dict[parallel_type]["times"] = []
    result_dict[parallel_type]["best_corr"] = []
    result_dict[parallel_type]["best_params"] = []
    result_dict[parallel_type]["best_combination"] = []
    result_dict[parallel_type]["best_result"] = []

    for n_sols in n_sols_list:

        start_time = datetime.now()

        sol_subset = solutions[:n_sols]
        compound_subset = df["compound"].tolist()[:n_sols]

        # sol_subset = solutions
        # compound_subset = df["compound"].tolist()

        (
            best_params,
            best_combination,
            best_corr,
            best_result,
            _,
        ) = optimize_for_all_combinations(sol_subset, compound_subset, parallel=False)

        # print("Best Parameters:", best_params)
        # print("Best Combination:", best_combination)
        # print("Best Corr.:", best_corr)

        computing_time = (datetime.now() - start_time).total_seconds()

        result_dict[parallel_type]["times"].append(computing_time)
        result_dict[parallel_type]["best_corr"].append(best_corr)
        result_dict[parallel_type]["best_params"].append(best_params)
        result_dict[parallel_type]["best_combination"].append(best_combination)
        result_dict[parallel_type]["best_result"].append(best_result)


plt.figure(figsize=(8, 6))
for parallel_type in list(result_dict):
    plt.plot(
        n_sols_list,
        result_dict[parallel_type]["times"],
        marker="o",
        label=f"with {parallel_type}",
    )
plt.xlabel("number of solutions")
plt.ylabel("time (s)")
plt.title("optimization")
plt.legend()
plt.show()

# %% [markdown]
# Relative

# %%
plt.figure(figsize=(12, 6))
for parallel_type in list(result_dict):
    y_values = np.array(result_dict[parallel_type]["times"]) / np.array(n_sols_list)
    plt.plot(
        n_sols_list,
        y_values,
        marker="o",
        label=f"with {parallel_type}",
    )
plt.xlabel("number of solutions")
plt.ylabel("time per solution (s)")
plt.title("optimization")
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# Store optimization results

# %%
optimal_parameters_cc_v2_exploratory = {
    "description": "optimal parameters for composition complexity V2 based on PCA for use in exploratory study.",
    "best_params": best_params,
    "best_combination": best_combination,
    "best_result": best_result,
}

with open(
    Path(EXPORT_PATH_DATA, "exploratory", "optimal_parameters_cc_v2.json"),
    "w",
) as fp:
    json.dump(
        optimal_parameters_cc_v2_exploratory,
        fp,
    )

# %% [markdown]
# **Verification**

# %% [markdown]
# Load optimization results

# %%
with open(
    Path(EXPORT_PATH_DATA, "exploratory", "optimal_parameters_cc_v2.json"),
    "r",
) as fp:
    optimal_parameters_cc_v2_exploratory = json.load(
        fp,
    )

best_params = optimal_parameters_cc_v2_exploratory["best_params"]
best_combination = optimal_parameters_cc_v2_exploratory["best_combination"]

print(json.dumps(optimal_parameters_cc_v2_exploratory, indent=4))

# %% [markdown]
# Initial Values

# %%
complexities = []

sol_subset = solutions
compound_subset = df["compound"].tolist()

# iterate over all solutions
for s in sol_subset:
    complexity = composition_complexity_v2(
        s["solution_matrix"],
        s["worker_capacities"],
        s["task_loads"],
        geom_p=0.5,
        empty_param=0.05,
        comp_alpha=2.0,
        empty_distribution="cont_bern",
        dirichlet_correction=True,
    )
    complexities.append(complexity)

print("MSE", np.mean((np.array(compound_subset) - np.array(complexities)) ** 2))
pearsonr(complexities, compound_subset)

# %% [markdown]
# Optimized Values

# %%
complexities = []

# iterate over all solutions
for s in sol_subset:
    complexity = composition_complexity_v2(
        s["solution_matrix"],
        s["worker_capacities"],
        s["task_loads"],
        geom_p=best_params[0],
        empty_param=best_params[1],
        comp_alpha=best_params[2],
        empty_distribution=best_combination[0],
        dirichlet_correction=best_combination[1],
    )
    complexities.append(complexity)

print("MSE", np.mean((np.array(compound_subset) - np.array(complexities)) ** 2))
pearsonr(complexities, compound_subset)

# %%
