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
# Create problem instances
# ===
#
# Note: This is the starting point for generating the GAP stimuli.

# %%
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from joptim.solvers import find_all_optimal_solutions_cp_max, mip_solve_max
from joptim.utils.export_path import get_export_paths


EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()

N_SIMS = 20_000
RANDOMIZE_SIZE = True  # True means +1/-1/0 for rows and columns starting from SIZE
BASE_SIZE = (8, 5)  # (n_tasks, n_workers)
RATIO_INTERVAL = (0.8, 1.0)  # ratio total capacity/total load
TASK_STEPS = 5  # 10
WORKER_STEPS = 10
REPEAT_TASK_SIZES = False
REPEAT_WORKER_SIZES = True
MAX_WORKER_SIZE = 100
MAX_TASK_SIZE = 100
MAX_OPT_SOLUTIONS = 99  # when to stop CP-SAT loop

# %% [markdown]
# <div class="alert alert-warning">Increase N_SIMS and MAX_OPT_SOLUTIONS for more instances to sample from.</div>

# %% [markdown]
# ## Generate

# %% [markdown]
# This generation approach doesn't take care of tasks that are larger than each worker and workers that are smaller than each task. We filter them out in the steps that follow.

# %%
# %%time


problems = []

for _ in range(N_SIMS):

    # define the size of the problem

    n_tasks, n_workers = BASE_SIZE

    if RANDOMIZE_SIZE:  # randomly +/- 1 row, col
        n_tasks += np.random.randint(-1, 2)
        n_workers += np.random.randint(-1, 2)

    # fix the ratio_interval

    target_capacity_load_ratio = np.random.uniform(RATIO_INTERVAL[0], RATIO_INTERVAL[1])

    max_capacity_overflow = True
    while max_capacity_overflow:
        task_loads = np.random.choice(
            range(TASK_STEPS, MAX_TASK_SIZE + 1, TASK_STEPS),
            replace=REPEAT_TASK_SIZES,
            size=n_tasks,
        )
        target_max_capacity = task_loads.sum() * target_capacity_load_ratio
        max_capacity_overflow = target_max_capacity >= MAX_WORKER_SIZE * n_workers

    factor = target_max_capacity // WORKER_STEPS
    rounded_lower = WORKER_STEPS * factor
    rounded_upper = WORKER_STEPS * (factor + 1)

    # check again for ratio

    capacity_load_ratio_lower = rounded_lower / task_loads.sum()
    capacity_load_ratio_upper = rounded_upper / task_loads.sum()

    if capacity_load_ratio_lower > RATIO_INTERVAL[0]:
        if capacity_load_ratio_upper < RATIO_INTERVAL[1]:
            # both within bounds -> choose closer one
            if abs(capacity_load_ratio_lower - target_capacity_load_ratio) < abs(
                capacity_load_ratio_upper - target_capacity_load_ratio
            ):
                max_capacity = rounded_lower
            else:
                max_capacity = rounded_upper
        else:  # upper is out of bounds -> choose lower
            max_capacity = rounded_lower
    else:  # lower is out of bounds --> choose upper
        max_capacity = rounded_upper

    capacity_load_ratio = max_capacity / task_loads.sum()

    def split_capacity(max_capacity, max_worker_size, worker_steps):

        available_capacity = max_capacity
        worker_capacities = []

        for i in range(n_workers):
            free = available_capacity - worker_steps * (n_workers - i - 1)
            can = max_worker_size * (n_workers - i - 1)
            if can < available_capacity:
                diff = abs(can - available_capacity)
            else:
                diff = 0
            min_take = int(np.max([worker_steps, diff]))
            max_take = int(np.min([max_worker_size, free]))
            taken = np.random.choice(range(min_take, max_take + 1, worker_steps))
            available_capacity -= taken
            worker_capacities.append(taken)

        return np.array(worker_capacities)

    worker_capacities = split_capacity(max_capacity, MAX_WORKER_SIZE, WORKER_STEPS)

    # sort
    task_loads.sort()
    worker_capacities.sort()

    problem = {}
    problem["task_loads"] = task_loads[::-1]
    problem["worker_capacities"] = worker_capacities[::-1]
    problem["capacity_load_ratio"] = capacity_load_ratio

    problems.append(problem)

print(len(problems), "instances generated.")


# %% [markdown]
# ### Remove unassignable tasks and unusable resources

# %%
def is_unusable(task_loads, worker_capacities):
    """
    Check if there is a task larger than EVERY worker capacity or if there is a capacity smaller than EVERY task load.
    """
    condition_A = any(
        all(task > capacity for capacity in worker_capacities) for task in task_loads
    )
    condition_B = any(
        all(capacity < task for task in task_loads) for capacity in worker_capacities
    )
    return condition_A or condition_B


problems_filtered = []

for problem in problems:
    to_remove = is_unusable(problem["task_loads"], problem["worker_capacities"])
    if not to_remove:
        problems_filtered.append(problem)

problems = problems_filtered.copy()

len(problems)


# %% [markdown]
# ### Remove duplicates

# %%
def array_new_in_list(arr_list, target_array):
    for arr in arr_list:
        if np.array_equal(arr, target_array):
            return False
    return True


def stack_arrays(problem):
    return np.hstack([problem["worker_capacities"], problem["task_loads"]])


stacked_arrays = []  # need for check for duplicate GAPs
problems_filtered = []

for problem in problems:
    gap_is_new = array_new_in_list(stacked_arrays, stack_arrays(problem))
    if gap_is_new:
        stacked_arrays.append(stack_arrays(problem))
        problems_filtered.append(problem)

problems = problems_filtered.copy()

len(problems)

# %% [markdown]
# ## Problem metrics
#
# - capacity-load ratio (already computed)
# - optimal utility
# - iterations of MIP solver
# - number of optimal solutions

# %%
# %%time

for problem in problems:
    # add metrics
    status, solution_matrix, utility, iterations = mip_solve_max(
        problem["worker_capacities"], problem["task_loads"]
    )
    problem["utility"] = utility
    problem["iterations"] = iterations
    problem["optimal_solutions"] = find_all_optimal_solutions_cp_max(
        problem["worker_capacities"], problem["task_loads"], limit=MAX_OPT_SOLUTIONS + 1
    )
    problem["n_optimal_solutions"] = len(problem["optimal_solutions"])

# %%
from joptim.identifiers import encode_problem_id

for problem in problems:
    problem["id"] = encode_problem_id(
        problem["worker_capacities"], problem["task_loads"]
    )

# %% [markdown]
# ## Save to file

# %%
with open(Path(EXPORT_PATH_DATA, "problems.pkl"), "wb") as pickle_file:
    pickle.dump(problems, pickle_file)

# %% [markdown]
# **MAIN PART IS DONE HERE. Continue in `extend_filter_solutions.ipynb`**
#
# **BELOW IS ONLY FOR SOME ANALYSIS**

# %% [markdown]
# ## Load from file

# %%
try:
    with open(Path(EXPORT_PATH_DATA, "problems.pkl"), "rb") as pickle_file:
        problems = pickle.load(pickle_file)
        print(len(problems), "problems loaded.")
except Exception as e:
    print("Failed to load problems from all_simulations.pkl:", e)

# %% [markdown]
# ## Visualize Problem metrics

# %%
df_metrics = pd.DataFrame(
    np.array(
        [
            [problem["capacity_load_ratio"] for problem in problems],
            [problem["utility"] for problem in problems],
            [problem["iterations"] for problem in problems],
            [problem["n_optimal_solutions"] for problem in problems],
        ]
    ).T,
    columns=[
        "capa_load_ratio",
        "utility",
        "iterations",
        "n_solutions",
    ],
)

mask = np.zeros_like(df_metrics.corr())
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(6, 6))
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
plt.title("Problem metrics")
plt.savefig(Path(EXPORT_PATH_PUB, "metrics_correlations.png"), dpi=200)

# %%
# plt.rcdefaults()
plt.rc("font", size=12)
# sns.set_theme("paper")
# sns.set(style="ticks")
# sns.set(style="ticks", font_scale = 1.2)

g = sns.PairGrid(df_metrics, height=2, aspect=1)
g.map_diag(sns.histplot, bins=20)
g.map_offdiag(
    sns.regplot,
    scatter_kws=dict(color="steelblue", alpha=100 / len(df_metrics), marker="o", s=10),
    line_kws=dict(color="salmon"),
)

plt.suptitle("Problem metrics")
plt.tight_layout()
plt.savefig(Path(EXPORT_PATH_PUB, "metrics_gridplot.png"), dpi=200)

# %% [markdown]
# Next visit `GAP Solutions.ipynb` to apply solution-based filters and compute solution-based metrics.

# %% [markdown]
# ## Problems for "many optimal solutions" animation (demo3)

# %% [markdown]
# Search and pick an interesting problem based on some custom filters

# %%
from copy import deepcopy

for problem in problems[:100]:
    if len(problem["task_loads"]) == 8:
        if len(problem["worker_capacities"]) == 5:
            if problem["capacity_load_ratio"] < 0.84:
                print(problem["id"], len(problem["optimal_solutions"]))

# %% [markdown]
# Store it in a file for the experiment

# %%
picked_id = 25

a_problem_with_many_solutions = deepcopy(problems[picked_id-1])

a_problem_with_many_solutions["task_loads"] = a_problem_with_many_solutions[
    "task_loads"
].tolist()
a_problem_with_many_solutions["worker_capacities"] = a_problem_with_many_solutions[
    "worker_capacities"
].tolist()


def solution2assignment(sm):
    """
    Convert a solution matrix to an assignment array to
    be used in the frontend. Note: This function only exists here
    for the a_problem_with_many_solutions case, for all other
    cases the backend does this transformation during the respective
    API call in main.py.
    """

    return [[int(i), int(j)] for i, j in np.flip(np.flip(np.where(sm == 1)).T)]


optimal_solutions = []

for oms in a_problem_with_many_solutions["optimal_solutions"]:
    optimal_solutions.append(solution2assignment(oms))

a_problem_with_many_solutions["optimal_solutions"] = optimal_solutions

with open(
    Path(EXPORT_PATH_DATA, "a_problem_with_many_solutions.pkl"), "wb"
) as pickle_file:
    pickle.dump(a_problem_with_many_solutions, pickle_file)

# %% [markdown]
# Copy `a_problem_with_many_solutions.pkl` into the experiments backend folder and access it under `<SERVER_URL>/joptim/demo3`.
