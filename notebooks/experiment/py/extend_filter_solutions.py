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
# Solutions to problem instances
# ===
#
# Note: solutions were already created in `generate_problems_solutions.ipynb`.
#
# What this does:
# 1. Removes solutions with identical bin composition
# 2. Removes problems with only 1 solution
# 3. Adds solution metrics for each solution
# 4. Finds the most extreme solution pair for each metric  

# %% [markdown]
# ## 0. Imports & Setup

# %%
import pickle
import numpy as np
from pathlib import Path
from joptim.solvers import find_all_optimal_solutions_cp_max, solve_gap_heuristic
from joptim.mixture_models import composition_complexity, composition_complexity_v1
from joptim.graph_edit_distance import (
    graph_edit_distance,
    graph_edit_distance_diagonal,
    graph_edit_distance_diagonal_approximated
)
from joptim.utils.export_path import get_export_paths

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()

# Load problems.pkl
try:
    with open(Path(EXPORT_PATH_DATA, "problems.pkl"), "rb") as pickle_file:
        problems = pickle.load(pickle_file)
        print(len(problems), "problems loaded.")
except Exception as e:
    print("Failed to load problems from problems.pkl:", e)

# %% [markdown]
# ## 1. Remove solutions in which bins have identical composition

# %%
solutions = [problem["optimal_solutions"] for problem in problems]

solutions_dupl_removed = []


def create_composition_str(sm, task_loads):
    composition = sm.T * task_loads
    composition.sort(axis=1)
    composition = np.flip(composition, axis=1)
    composition = "".join(
        sorted(["".join([str(w) for w in worker]) for worker in composition])
    )
    return composition


def composition_is_new(previous_compositions, new_composition):
    for p in previous_compositions:
        if p == new_composition:
            return False
    return True


for i, gap in enumerate(solutions):
    compositions = []
    gap_solutions = []
    for solution in gap:
        composition = create_composition_str(solution, problems[i]["task_loads"])
        if composition_is_new(compositions, composition):
            compositions.append(composition)
            gap_solutions.append(solution)
    solutions_dupl_removed.append(np.array(gap_solutions))

for i, problem in enumerate(problems):
    problem["optimal_solutions"] = solutions_dupl_removed[i]
    problem["n_optimal_solutions"] = len(solutions_dupl_removed[i])

# %% [markdown]
# ## 2. Remove instances with only 1 optimal solution
#
# (to allow for comparing solution metrics)

# %%
problems = [problem for problem in problems if problem["n_optimal_solutions"] > 1]
len(problems)

# %% [markdown]
# ### 2.1. Save to file

# %%
with open(Path(EXPORT_PATH_DATA, "problems_filtered.pkl"), "wb") as pickle_file:
    pickle.dump(problems, pickle_file)

# %% [markdown]
# ### 2.2. Load from file

# %%
try:
    with open(
        Path(EXPORT_PATH_DATA, "problems_filtered.pkl"), "rb"
    ) as pickle_file:
        problems = pickle.load(pickle_file)
        print(len(problems), "problems loaded.")
except Exception as e:
    print("Failed to load problems from all_simulations.pkl:", e)

# %% [markdown]
# ## 3. Solution metrics
#
# **for all optimal solutions**
#
# - composition_complexity of bins
#   - based on 2 dirichlets
#   - based on complex distribution (geometric, gamma, normal/laplace/continous bernoulli)
#   - pick parameters found in `analysis/parameter_fitting/mixture_model_fit_pca.ipynb`
# - assignment complexities
#   - graph edit distance to greedy heuristic
#   - graph edit distance to diagonals

# %%
# %%time

# for composition_complexity_v2()
with open(
    Path(EXPORT_PATH_DATA, "exploratory", "optimal_parameters_cc_v2.json"),
    "r",
) as fp:
    optimal_parameters_cc_v2_exploratory = json.load(
        fp,
    )
best_params = optimal_parameters_cc_v2_exploratory["best_params"]
best_combination = optimal_parameters_cc_v2_exploratory["best_combination"]

# for composition_complexity_v1 (the simpler version)
# defaults are taken.

for problem in problems:
    problem["greedy_solution"] = solve_gap_heuristic(
        problem["worker_capacities"], problem["task_loads"]
    )
    assignment_complexities = []
    assignment_complexities_diag1 = []
    assignment_complexities_diag2 = []
    assignment_complexities_diag3 = []
    composition_complexities = []
    composition_complexities_dirichlet = []

    for s in problem["optimal_solutions"]:
        ged_greedy = graph_edit_distance(s, problem["greedy_solution"])
        ged_diag1, ged_diag2 = graph_edit_distance_diagonal(s)
        ged_diag3 = graph_edit_distance_diagonal_approximated(s)
        assignment_complexities.append(ged_greedy)
        assignment_complexities_diag1.append(ged_diag1)
        assignment_complexities_diag2.append(ged_diag2)
        assignment_complexities_diag3.append(ged_diag3)
        composition_complexities.append(
            composition_complexity_v2(
                s,
                problem["worker_capacities"],
                problem["task_loads"],
                geom_p=best_params[0],
                empty_param=best_params[1],
                comp_alpha=best_params[2],
                empty_distribution=best_combination[0],
                dirichlet_correction=best_combination[1],
            )
        )
        composition_complexities_dirichlet.append(
            composition_complexity_v1(
                s,
                problem["worker_capacities"],
                problem["task_loads"],
                weighted="absolute",
            )
        )
·
    problem["assignment_complexity"] = assignment_complexities
    problem["assignment_complexity_diag1"] = assignment_complexities_diag1
    problem["assignment_complexity_diag2"] = assignment_complexities_diag2
    problem["assignment_complexity_diag3"] = assignment_complexities_diag3
    problem["composition_complexity"] = composition_complexities
    problem["composition_complexity_v1"] = composition_complexities_dirichlet

# %% [markdown]
# ## 4. Save file incl. solutions and solution metrics

# %%
with open(
    Path(EXPORT_PATH_DATA, "problems_filtered_incl_solution_metrics.pkl"),
    "wb",
) as pickle_file:
    pickle.dump(problems, pickle_file)
