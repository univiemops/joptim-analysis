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
# Extend solution metrics to problem instances for confirmatory experiment
# ===
#
# Note: solutions were already created in `generate_problems_solutions.ipynb` and extended in `extend_filter_solutions.ipynb`.
#
# What this does:
# - Adds updated composition complexity based on new parameters found in `mixture_model_fit_pilot.ipynb` for each solution to be used in the confirmatory experiment

# %% [markdown]
# ## 0. Imports & Setup

# %%
import pickle
import numpy as np
from pathlib import Path
from joptim.solvers import find_all_optimal_solutions_cp_max, solve_gap_heuristic
from joptim.mixture_models import composition_complexity_v1, composition_complexity_v2
from joptim.graph_edit_distance import (
    graph_edit_distance,
    graph_edit_distance_diagonal,
)
from joptim.utils.export_path import get_export_paths

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()

# Load problems.pkl
try:
    with open(
        Path(EXPORT_PATH_DATA, "problems_filtered_incl_solution_metrics.pkl"),
        "rb",
    ) as pickle_file:
        problems = pickle.load(pickle_file)
        print(len(problems), "problems loaded.")
except Exception as e:
    print("Failed to load problems from problems.pkl:", e)

# %% [markdown]
# ## 1. Add composition complexity with parameters found in exploratory experiment
#
#
#
# ### 1.1. Load parameters ...
# ... found in `analysis/parameter_fitting/mixture_model_fit_pilot.ipynb`

# %%
with open(
    Path(EXPORT_PATH_DATA, "confirmatory", "optimal_parameters_cc_v1.json"),
    "r",
) as fp:
    optimal_parameters_cc_v1_confirmatory = json.load(
        fp,
    )

print("v1:", optimal_parameters_cc_v1_confirmatory)


with open(
    Path(EXPORT_PATH_DATA, "confirmatory", "optimal_parameters_cc_v2.json"),
    "r",
) as fp:
    optimal_parameters_cc_v2_confirmatory = json.load(
        fp,
    )

print("v2:", optimal_parameters_cc_v2_confirmatory)

# %% [markdown]
# ### 1.2. Add new composition complexity ...
# ... based on optimized parameters with suffix `_confirmatory` to all optimal solutions of all problems

# %%
# %%time

best_params_v1 = optimal_parameters_cc_v1_confirmatory["best_params"]
best_combination_v1 = optimal_parameters_cc_v1_confirmatory["best_combination"]

best_params_v2 = optimal_parameters_cc_v2_confirmatory["best_params"]
best_combination_v2 = optimal_parameters_cc_v2_confirmatory["best_combination"]


for problem in problems:
    cc_v1_all = []
    cc_v2_all = []

    for s in problem["optimal_solutions"]:

        cc_v1_all.append(
            composition_complexity_v1(
                s,
                problem["worker_capacities"],
                problem["task_loads"],
                alpha1=best_params_v1[0],
                alpha2=best_params_v1[1],
                theta=best_params_v1[2],
                weighted=best_combination_v1[0],
            )
        )

        cc_v2_all.append(
            composition_complexity_v2(
                s,
                problem["worker_capacities"],
                problem["task_loads"],
                geom_p=best_params_v2[0],
                empty_param=best_params_v2[1],
                comp_alpha=best_params_v2[2],
                empty_distribution=best_combination_v2[0],
                dirichlet_correction=best_combination_v2[1],
            )
        )

    problem["composition_complexity_v1_confirmatory"] = cc_v1_all
    problem["composition_complexity_v2_confirmatory"] = cc_v2_all

# %% [markdown]
# Show some examples

# %%
i = 2

for stage in ["", "_confirmatory"]:
    for v in ["1", "2"]:
        print(
            f"""Solutions of problem {i} {stage} have v{v} complexities of {', '.join([str(x.round(3)) for x in problems[i][f"composition_complexity_v{v}{stage}"]])}"""
        )

# %% [markdown]
# ## 4. Save file incl. solutions and solution metrics

# %%
with open(
    Path(EXPORT_PATH_DATA, "problems_filtered_incl_solution_metrics.pkl"),
    "wb",
) as pickle_file:
    pickle.dump(problems, pickle_file)
