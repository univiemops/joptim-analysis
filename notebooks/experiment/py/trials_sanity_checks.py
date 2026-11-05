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
# Sanity Checks for Generated Trials
# ===
#
# <div class="alert alert-warning">Use this notebook only for prototyping and debugging ("sanity checks", see further below). For generating trials use <tt>generate_trials.py</tt>.</div>

# %% [markdown]
# **Content**
#
# 1. Sanity Checks
# 2. Simulate Participant
# 3. Inspect Trial
# 4. Further Steps
# 5. Delete Experimental Data

# %% [markdown]
# ## 0 Imports & Setup

# %%
import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from joptim.utils.export_path import get_export_paths

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()

STUDY = "confirmatory"  # "confirmatory" or "exploratory"
PICKLE_OUTPUT_FILE = "trials.pkl"

if STUDY not in ["exploratory", "confirmatory"]:
    raise ValueError("STUDY can only `exploratory` and `confirmatory`, got:", STUDY)

with open(
    Path(EXPORT_PATH_DATA, STUDY, "optimal_parameters_cc_v2.json"),
    "r",
) as fp:
    optimal_parameters_cc_v2 = json.load(
        fp,
    )

best_params = optimal_parameters_cc_v2["best_params"]
best_combination = optimal_parameters_cc_v2["best_combination"]

with open(
    Path(EXPORT_PATH_DATA, STUDY, PICKLE_OUTPUT_FILE),
    "rb",
) as pickle_file:
    trial_data = pickle.load(pickle_file)
    trials = trial_data["trials"]
    trial_file_id = trial_data["file_id"]
    print(
        len(trials),
        "trials loaded.",
    )

# %% [markdown]
# ## 1. Sanity Checks

# %%
catch_trial_slots = [8, 17]
coherence_trial_slots = [6, 12, 19]

# %%
data = []

for index, participant in enumerate(trials):
    solving_trials = participant["solving_trials"]
    preference_trials = participant["preference_trials"]

    record = {
        "participant": index,
        "n_st": len(solving_trials),
        "difficulties_st": str([trial["difficulty"] for trial in solving_trials]),
        "n_pt": len(preference_trials),
        "catch_1": preference_trials[catch_trial_slots[0]]["use_case"],
        "catch_2": preference_trials[catch_trial_slots[1]]["use_case"],
        "coherence_1": preference_trials[coherence_trial_slots[0]]["use_case"],
        "coherence_2": preference_trials[coherence_trial_slots[1]]["use_case"],
        "coherence_3": preference_trials[coherence_trial_slots[2]]["use_case"],
    }

    data.append(record)

df_checks = pd.DataFrame.from_records(
    data,
    columns=list(data[0]),
)

# %% [markdown]
# Should only show one line as this information should be the same for all participants

# %%
df_checks.drop_duplicates(subset=list(df_checks.columns)[1:], keep="first")

# %% [markdown]
# What information will we collect with these participants?

# %%
label = "difficulty"

sns.histplot(
    [pt[label] for participant in trials for pt in participant["preference_trials"]],
    stat="percent",
)
plt.xlabel(label);

# %%
label = "visual_manipulation"

sns.histplot(
    [pt[label] for participant in trials for pt in participant["preference_trials"]],
    stat="percent",
)
plt.xlabel(label);

# %%
label = "task_loads"

sns.histplot(
    [
        str(len(pt[label]))
        for participant in trials
        for pt in participant["preference_trials"]
    ],
    stat="percent",
)
plt.xlabel(label)
plt.xlabel("n tasks");

# %%
label = "worker_capacities"

sns.histplot(
    [
        str(len(pt[label]))
        for participant in trials
        for pt in participant["preference_trials"]
    ],
    stat="percent",
    discrete=True,
)
plt.xlabel("n workers");

# %% [markdown]
# Without coherence and catch trials

# %%
# combination of 2
label1 = "task_loads"
label2 = "worker_capacities"

data = np.array(
    [
        str(len(pt[label1])) + " x " + str(len(pt[label2]))
        for participant in trials
        for pt in participant["preference_trials"]
        if (pt["use_case"] != "coherence") & ((pt["use_case"] != "catch"))
    ]
)

unique_values, counts = np.unique(data, return_counts=True)
sorted_counts = sorted(zip(unique_values, counts), key=lambda x: x[1], reverse=True)
sorted_unique_values = [val for val, _ in sorted_counts]
sorted_counts_values = [count for _, count in sorted_counts]
sorted_counts_values = 100 * (sorted_counts_values / np.sum(sorted_counts_values))

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_unique_values, y=sorted_counts_values)
plt.grid(axis="y", linewidth=0.5)
plt.xlabel("n_tasks x n_workers")
plt.ylabel("Percent")
plt.title("Histogram of Problem Sizes Sorted by Frequency");

# %%
data = np.array(
    [
        [(len(pt[label2])), len(pt[label1])]
        for participant in trials
        for pt in participant["preference_trials"]
        if (pt["use_case"] != "coherence") & ((pt["use_case"] != "catch"))
    ]
)

# Compute the 2D histogram
hist, xedges, yedges = np.histogram2d(
    data[:, 0],
    data[:, 1],
    bins=(10, 10),
    range=[
        [min(data[:, 0]) - 0.5, max(data[:, 0]) + 0.5],
        [min(data[:, 1]) - 0.5, max(data[:, 1]) + 0.5],
    ],
)

# Convert counts to percentages
total_count = hist.sum()
hist_percent = (hist / total_count) * 100

# Filter out empty rows and columns from the histogram
rows_with_data = np.any(hist_percent, axis=1)
cols_with_data = np.any(hist_percent, axis=0)
hist_filtered = hist_percent[rows_with_data][:, cols_with_data]
xedges_filtered = np.array(range(hist_filtered.shape[0]))
yedges_filtered = np.array(range(hist_filtered.shape[1]))

# Plot with Seaborn heatmap
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(
    hist_filtered.T,
    annot=True,
    fmt=".1f",
    cmap="viridis",
    cbar_kws={"label": "Percent"},
)

# Adjust labels and titles
plt.title("2D Histogram of Problem Sizes")
plt.xlabel("Number of Workers")
plt.ylabel("Number of Tasks")

# Adjust tick labels
x_tick_labels = [int(edge + min(data[:, 0]) + 0.5) for edge in xedges_filtered]
y_tick_labels = [int(edge + min(data[:, 1]) + 0.5) for edge in yedges_filtered]

plt.xticks(np.arange(len(xedges_filtered)) + 0.5, x_tick_labels)
plt.yticks(np.arange(len(yedges_filtered)) + 0.5, y_tick_labels)

# Show plot
plt.show()

# %%
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Extract the x and y values
x = data[:, 0]
y = data[:, 1]

# Create a 2D histogram
x_bins = np.arange(x.min(), x.max() + 2) - 0.5
y_bins = np.arange(y.min(), y.max() + 2) - 0.5
hist, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])

# Normalize the histogram to percentages
hist_percentage = (hist / hist.sum()) * 100

# Construct arrays for the anchor positions of the bars.
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5, yedges[:-1] + 0.5, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# Construct arrays with the dimensions for the bars.
dx = dy = 1.0 * np.ones_like(zpos)
dz = hist_percentage.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort="average", color="steelblue")

# Set ticks to be integers
ax.set_xticks(0.5 + np.arange(x.min(), x.max() + 1), np.arange(x.min(), x.max() + 1))
ax.set_yticks(0.5 + np.arange(y.min(), y.max() + 1), np.arange(y.min(), y.max() + 1))

# ax.zaxis.set_tick_params(reset=False, labelsize=10)

# Labels and title
ax.set_xlabel("Numer of Workers")
ax.set_ylabel("Number of Tasks")
ax.set_zlabel("Percentage")
ax.set_title("Relative Frequency of Problem Sizes (Percentage)")
plt.show()

# %% [markdown]
# <div class="alert alert-waring">Sampling resulted in a biases for specific problem sizes. This is because of the coherence and catch trials which are of fixed size and for evey participant the same.</div>

# %%
label = "capacity_load_ratio"

sns.histplot(
    [pt[label] for participant in trials for pt in participant["preference_trials"]],
    stat="percent",
    bins=5,
)

plt.xlabel(label);

# %%
label = "diff_cc"
use_case = "extreme_diff_cc"

n_bins = 10

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

values_all = [
    pt[label] for participant in trials for pt in participant["preference_trials"]
]

axs[0] = sns.histplot(
    values_all,
    stat="percent",
    bins=np.linspace(np.min(values_all), np.max(values_all), n_bins + 1),
    ax=axs[0],
)
axs[0].axvline(
    np.mean(values_all), color="red", ls="--", label=f"Mean={np.mean(values_all):.3f}"
)
axs[0].set_title(f"All Trials (N={len(values_all)})")
axs[0].set_xlabel(label)
axs[0].legend()

values_filtered = [
    pt[label]
    for participant in trials
    for pt in participant["preference_trials"]
    if pt["use_case"] == use_case
]

axs[1] = sns.histplot(
    values_filtered,
    stat="percent",
    bins=np.linspace(np.min(values_all), np.max(values_all), n_bins + 1),
    ax=axs[1],
)
axs[1].axvline(
    np.mean(values_filtered),
    color="red",
    ls="--",
    label=f"Mean={np.mean(values_filtered):.3f}",
)
axs[1].set_title(f"{use_case.capitalize()} Trials (N={len(values_filtered)})")
axs[1].set_xlabel(label)
axs[1].legend()

# both plots same y limits
max_ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])

axs[0].set_ylim(0, max_ylim)
axs[1].set_ylim(0, max_ylim)

# %%
label = "diff_ac"
use_case = "extreme_diff_ac"

n_bins = 10

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

values_all = [
    pt[label] for participant in trials for pt in participant["preference_trials"]
]

axs[0] = sns.histplot(
    values_all,
    stat="percent",
    bins=np.linspace(np.min(values_all), np.max(values_all), n_bins + 1),
    ax=axs[0],
)
axs[0].axvline(
    np.mean(values_all), color="red", ls="--", label=f"Mean={np.mean(values_all):.3f}"
)
axs[0].set_title(f"All Trials (N={len(values_all)})")
axs[0].set_xlabel(label)
axs[0].legend()

values_filtered = [
    pt[label]
    for participant in trials
    for pt in participant["preference_trials"]
    if pt["use_case"] == use_case
]

axs[1] = sns.histplot(
    values_filtered,
    stat="percent",
    bins=np.linspace(np.min(values_all), np.max(values_all), n_bins + 1),
    ax=axs[1],
)
axs[1].axvline(
    np.mean(values_filtered),
    color="red",
    ls="--",
    label=f"Mean={np.mean(values_filtered):.3f}",
)
axs[1].set_title(f" trials")
axs[1].set_title(f"{use_case.capitalize()} Trials (N={len(values_filtered)})")
axs[1].set_xlabel(label)
axs[1].legend()

# both plots same y limits
max_ylim = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])

axs[0].set_ylim(0, max_ylim)
axs[1].set_ylim(0, max_ylim)

# %%
label = "utility"

sns.histplot(
    [pt[label] for participant in trials for pt in participant["preference_trials"]],
    stat="percent",
    bins=5,
)
plt.xlabel(label)

# %%
label = "use_case"

fig = plt.figure(figsize=(14, 4))
ax = sns.histplot(
    [pt[label] for participant in trials for pt in participant["preference_trials"]],
    stat="percent",
)
ax.set_axisbelow(True)
ax.grid(axis="y", linewidth=0.5)
ax.tick_params(axis="x", labelrotation=90)
ax.set_xlabel(label);

# %% [markdown]
# Did the randomization work or did all participants get the same trials?

# %%
label = "id"

trial_ids = [
    pt[label] for participant in trials for pt in participant["preference_trials"]
]

unique, frequency = np.unique(trial_ids, return_counts=True)

len(frequency)

# %%
len(unique)

# %%
sns.histplot(frequency.astype(str), stat="percent")
plt.xlabel("frequencies");

# %% [markdown]
# Re-compute composition complexity and compare with original value (we do this based on only 1 participant and trial)

# %%
from joptim.mixture_models import composition_complexity_v2

participant_id = 2
trial_id = 9

precision = 9  # decimal places

cc_L = composition_complexity_v2(
    trials[participant_id]["preference_trials"][trial_id]["solution_1"],
    trials[participant_id]["preference_trials"][trial_id]["worker_capacities"],
    trials[participant_id]["preference_trials"][trial_id]["task_loads"],
    best_params[0],
    best_params[1],
    best_params[2],
    best_combination[0],
    best_combination[1],
)

cc_R = composition_complexity_v2(
    trials[participant_id]["preference_trials"][trial_id]["solution_2"],
    trials[participant_id]["preference_trials"][trial_id]["worker_capacities"],
    trials[participant_id]["preference_trials"][trial_id]["task_loads"],
    best_params[0],
    best_params[1],
    best_params[2],
    best_combination[0],
    best_combination[1],
)

diff = trials[participant_id]["preference_trials"][trial_id]["diff_cc"] - np.abs(cc_R - cc_L)
diff = np.round(diff, precision)

assert (
    diff == 0
), f"""When re-computed with given parameters composition complexity differs from the one given in the trial file. Trial file says {trials[participant_id]["preference_trials"][trial_id]["diff_cc"]:.2f}, re-computation says {np.abs(cc_R - cc_L):.3f}. Were the trials created with the same parameters as defined here in the notebook?"""

print("OK")

# %% [markdown]
# TODO: re-create trials and check rigorously how cc is computed and if there are deviations. Also stored cc params in trial file as metadata.
#
# diff_cc comes from per-solution values from `problems_filtered_incl_solution_metrics.pkl`. Look there!

# %% [markdown]
# Check if the cc values inside `problems_filtered_incl_solution_metrics.pkl` align with our optimal parameters. If yes, we can repeat generate_trials and can use this new trials file

# %%
from pathlib import Path

# %%
with open(
    Path(EXPORT_PATH_DATA, "problems_filtered_incl_solution_metrics.pkl"), "rb"
) as fp:
    problems = pickle.load(fp)

# %%
problem_id = 53
problems[problem_id]["composition_complexity_v2_confirmatory"][0]

# %%
composition_complexity_v2(
    problems[problem_id]["optimal_solutions"][0],
    problems[problem_id]["worker_capacities"],
    problems[problem_id]["task_loads"],
    best_params[0],
    best_params[1],
    best_params[2],
    best_combination[0],
    best_combination[1],
)

# %%
problems[problem_id]["composition_complexity_v2"][2]

# %% [markdown]
# They do align, but it seems all solutions to all problem have a cc around 3.2 ...

# %%
with open(
    Path(EXPORT_PATH_DATA, "optimization_results_composition_to_pilot_cc_v2.pkl"), "rb"
) as fp:
    optim_res = pickle.load(fp)

# %%
sol_id = 0

z = problems[problem_id]["task_loads"]
w = problems[problem_id]["worker_capacities"]

sm = problems[problem_id]["optimal_solutions"][sol_id]

# %%
problems[problem_id]["composition_complexity_v2_confirmatory"][sol_id]

# %%
composition_complexity_v2(
    sm,
    w,
    z,
    optim_res[0][0],
    optim_res[0][1],
    optim_res[0][2],
    optim_res[1][0],
    optim_res[1][1],
)

# %% [markdown]
# Check if the `composition_complexity_v2_confirmatory` align with re-computed cc values for the best params (from pkl file). If yes, then we can restart `generate_trial.py` and check that file if the diff_cc and re-computed values align there (in contrast to the May and June trials.pkl files)

# %%
ccs_recomputed = []  # for checking the distibutions
ccs_stored = []
for problem in problems:
    z = problem["task_loads"]
    w = problem["worker_capacities"]
    for i, sm in enumerate(problem["optimal_solutions"]):
        cc = composition_complexity_v2(
            sm,
            w,
            z,
            optim_res[0][0],
            optim_res[0][1],
            optim_res[0][2],
            optim_res[1][0],
            optim_res[1][1],
        )
        ccs_recomputed.append(cc)
        ccs_stored.append(problem["composition_complexity_v2_confirmatory"][i])

# %%
np.sum(np.array(ccs_stored) - np.array(ccs_recomputed))

# %%
plt.hist(ccs_recomputed);

# %%
plt.hist(ccs_stored);

# %% [markdown]
# Yes, they do exactly! :)
#
# But the weird thing is, that all values are around 3.2

# %% [markdown]
# Lets check again, what we had in the trial files

# %% [markdown]
# Lets check the May file:

# %%
with open(
    Path(EXPORT_PATH_DATA, "confirmatory", "trials_may.pkl"),
    "rb",
) as pickle_file:
    trials_may = pickle.load(pickle_file)
    trials_may = trials_may["trials"]
    print(
        len(trials_may),
        "trials loaded.",
    )

# %%
for t in trials_may[0]["preference_trials"]:
    print(t["diff_cc"])

# %% [markdown]
# It can not be to have values >10. `diff_cc` must be based on different calcuatio/parameters. So this trial file must be sampled from a different problems file.

# %% [markdown]
# And lets check the June file:

# %%
with open(
    Path(EXPORT_PATH_DATA, "confirmatory", "trials_june.pkl"),
    "rb",
) as pickle_file:
    trials_june = pickle.load(pickle_file)
    trials_june = trials_june["trials"]
    print(
        len(trials_june),
        "trials loaded.",
    )

# %%
for t in trials_june[0]["preference_trials"]:
    print(t["diff_cc"])

# %% [markdown]
# This pattern would match the current problems file with cc always being around 3.2

# %% [markdown]
# Let's see if diff_cc and the recomputed values differ in the junes file. If now, that would suggest that the June file was built with the current parameters (with which the problems file aligns too)

# %%
diffs = []
for user_id in range(len(trials_june)):
    for t in trials_june[user_id]["preference_trials"]:

        z = t["task_loads"]
        w = t["worker_capacities"]
        sm_1 = t["solution_1"]
        sm_2 = t["solution_2"]

        cc_1 = composition_complexity_v2(
            sm_1,
            w,
            z,
            optim_res[0][0],
            optim_res[0][1],
            optim_res[0][2],
            optim_res[1][0],
            optim_res[1][1],
        )
        cc_2 = composition_complexity_v2(
            sm_2,
            w,
            z,
            optim_res[0][0],
            optim_res[0][1],
            optim_res[0][2],
            optim_res[1][0],
            optim_res[1][1],
        )
        diffs.append(np.abs(cc_2 - cc_1) - t["diff_cc"])

# %%
sum(diffs)

# %% [markdown]
# This means the June file was created with the current problem instances file which reflects the currently best parameters. It should be the correct trials file for confirmatory analysis.

# %% [markdown]
# Now question remains, why so strange compositin complexity values all crammed around 3.2? 

# %% [markdown]
# TODO: Run sanity checks on that June file, see if composition complexity differences are really visually meaningful

# %% [markdown]
# ## 2. Simulate Participant

# %%
from joptim.utils.render_html import render_html, render_pair_html

p_i = 1  # participant id
i = 0  # set initial trial to 0

# %% [markdown]
# Execute this cell several times to iterate through all trials

# %%
k = i - len(trials[p_i]["solving_trials"])

if i < len(trials[p_i]["solving_trials"]):
    print("solving trial", i + 1)
    w = trials[p_i]["solving_trials"][i]["worker_capacities"]
    z = trials[p_i]["solving_trials"][i]["task_loads"]
    render_html(w, z, labels=True)
elif k < len(trials[p_i]["preference_trials"]):
    print("preference trial", k + 1, trials[p_i]["preference_trials"][k]["use_case"])
    w_order_1 = trials[p_i]["preference_trials"][k]["worker_order_1"]
    z_order_1 = trials[p_i]["preference_trials"][k]["task_order_1"]
    w_order_2 = trials[p_i]["preference_trials"][k]["worker_order_2"]
    z_order_2 = trials[p_i]["preference_trials"][k]["task_order_2"]

    w_1 = trials[p_i]["preference_trials"][k]["worker_capacities"][w_order_1]
    z_1 = trials[p_i]["preference_trials"][k]["task_loads"][z_order_1]
    w_2 = trials[p_i]["preference_trials"][k]["worker_capacities"][w_order_2]
    z_2 = trials[p_i]["preference_trials"][k]["task_loads"][z_order_2]

    sm_1 = trials[p_i]["preference_trials"][k]["solution_1"][z_order_1][:, w_order_1]
    sm_2 = trials[p_i]["preference_trials"][k]["solution_2"][z_order_2][:, w_order_2]

    render_pair_html(
        w_1,
        z_1,
        w_2,
        z_2,
        labels=True,
        sm_1=sm_1,
        sm_2=sm_2,
    )
else:
    print("All trials completed")

i += 1

# %% [markdown]
# TODO: Implement diff_ac check for all other trial types too! coherence etc.

# %% [markdown]
# ## 3. Inspect specific trial

# %%
participant_id = 27  # participant
i_trial = 22

# %%
for i, a in enumerate(trials[participant_id]["preference_trials"]):
    print(i, a["id"])

# %%
p_i = 9  # participant id

ks = [20, 15]

for k in ks:
    sample = trials[p_i]["preference_trials"][k]

    task_order_1 = sample["task_order_1"]
    task_order_2 = sample["task_order_2"]
    worker_order_1 = sample["worker_order_1"]
    worker_order_2 = sample["worker_order_2"]

    print(
        f"""difference in assignment complexity (problem manipulation: {sample["visual_manipulation"]}): """,
        sample["diff_ac"],
    )
    render_pair_html(
        sample["worker_capacities"][worker_order_1],
        sample["task_loads"][task_order_1],
        sample["worker_capacities"][worker_order_2],
        sample["task_loads"][task_order_2],
        labels=True,
        sm_1=sample["solution_1"][task_order_1][:, worker_order_1],
        sm_2=sample["solution_2"][task_order_2][:, worker_order_2],
    )

# %%
from joptim.functions import print_matrix
from joptim.utils.elements import find_element_by_id

# %%
k = 20  # trial

geds = []

# TODO: which one? 
stimulus = trials[p_i]["preference_trials"][k]
stimulus = find_element_by_id(stimuli, "3960_1_2")
stimulus = find_element_by_id(problems, "3960")

stimulus["solution_1"] = stimulus["optimal_solutions"][1]
stimulus["solution_2"] = stimulus["optimal_solutions"][2]
stimulus[f"task_order_1"] = np.argsort(stimulus[f"task_loads"] * -1)
stimulus[f"task_order_2"] = np.argsort(stimulus[f"task_loads"] * -1)
stimulus[f"worker_order_1"] = np.argsort(stimulus[f"worker_capacities"] * -1)
stimulus[f"worker_order_2"] = np.argsort(stimulus[f"worker_capacities"] * -1)


keep_original = False

print(stimulus["id"])

for sol_i in [1, 2]:
    task_order = stimulus[f"task_order_{sol_i}"]
    worker_order = stimulus[f"worker_order_{sol_i}"]

    if keep_original:
        task_loads = stimulus["task_loads"]
        worker_capacities = stimulus["worker_capacities"]
        solution = stimulus[f"solution_{sol_i}"]
    else:
        task_loads = stimulus["task_loads"][task_order]
        worker_capacities = stimulus["worker_capacities"][worker_order]
        solution = stimulus[f"solution_{sol_i}"][task_order][:, worker_order]

    heuristic_solution = solve_gap_heuristic(worker_capacities, task_loads)

    print(f"\nsolution {sol_i}")
    print_matrix(task_loads, worker_capacities, solution)
    print(f"\nheuristic_solution")
    print_matrix(task_loads, worker_capacities, heuristic_solution)

    geds.append(graph_edit_distance(solution, heuristic_solution))
    print(f"""\nGED {["left", "right"][sol_i-1]}: {geds[sol_i-1]}""")

print(f"""\nGED calc.: {np.abs(np.diff(geds))[0]}""")
print(f"""GED diff (in sample obj): {stimulus["diff_ac"]}""")

# %% [markdown]
# ## 4. Further Steps

# %% [markdown]
# Next you can use the trials from `trials_exploratory.pkl` or `trials_confirmatory.pkl` by renaming the respective file to `trials.pkl` and use it for the experiment `joptim-exp`.

# %% [markdown]
# Check structure of trials.pkl

# %% [markdown]
# ## 5. Delete Experimental Data
#
# <div class="alert alert-danger">Use with care!</div>
#
# ### 5.1. Delete participants

# %% [raw]
# !pip install psycopg2

# %% [raw]
# !pip install python-dotenv

# %% [raw]
# import requests
# import subprocess

# %% [raw]
# response = requests.get("http://localhost:8000/tables/user").json()

# %% [raw]
# for el in response:
#     subprocess.call(["python3", "/home/user/code/joptim-exp/backend/delete_participant.py", el["pid"]])

# %% [markdown]
# ### 5.2. Reset Database

# %% [raw]
# subprocess.call(["python3", "/home/user/code/joptim-exp/backend/reset_database.py"])
