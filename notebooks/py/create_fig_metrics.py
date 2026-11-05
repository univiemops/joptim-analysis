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
# Create Figure for Introduction of Complexity Metrics
# ===

# %% [markdown]
# ## 0. Setup

# %%
import pickle
import pandas as pd
from pathlib import Path
from joptim.utils.export_path import get_export_paths
from joptim.utils.render_html import create_stimulus, crop_image


EXPORT_PATH_DATA, EXPORT_PATH_PUB, _ = get_export_paths()

with open(Path(EXPORT_PATH_DATA, "exploratory", "trials.pkl"), "rb") as fp:
    trials = pickle.load(fp)

data = []

for p_id in range(len(trials["trials"])):
    for trial_id in range(len(trials["trials"][p_id]["preference_trials"])):
        diff_ac = trials["trials"][p_id]["preference_trials"][trial_id]["diff_ac"]
        diff_cc = trials["trials"][p_id]["preference_trials"][trial_id]["diff_cc"]
        visual_manipulation = trials["trials"][p_id]["preference_trials"][trial_id]["visual_manipulation"]
        data.append({"trial_file_id":str(trials["file_id"]), "trial_file_user_id": p_id+1, "trial_id": trial_id+1, "diff_ac":diff_ac, "diff_cc":diff_cc, "visual_manipulation": visual_manipulation})

df = pd.DataFrame(data)

# %% [markdown]
# ## 1. Example for Heuristic-Related Complexity (HC)

# %%
metric = "ac"

df.sort_values(f"diff_{metric}", ascending=False).head(10)

# %%
# p_id = 17
# trial_id = 11

p_id = 53
trial_id = 15

example_trial = trials["trials"][p_id]["preference_trials"][trial_id]

example_trial

# %%
for sol_id in (1, 2):

    create_stimulus(
        worker_capacities=example_trial["worker_capacities"],
        task_loads=example_trial["task_loads"],
        directory=EXPORT_PATH_PUB,
        out="png",
        labels=False,
        annotations=None,
        sm=example_trial[f"solution_{sol_id}"],
        order=None,
        worker_order=example_trial[f"worker_order_{sol_id}"],
        task_order=example_trial[f"task_order_{sol_id}"],
        fname_base=f"example_{metric}_{sol_id}",
    )
    
    _ = crop_image(
        Path(EXPORT_PATH_PUB, "png",f"example_{metric}_{sol_id}.png"),
        margin=1,
    )

# %% [markdown]
# ## 2. Example for Compositional Complexity (CC)

# %%
metric = "cc"

df.sort_values(f"diff_{metric}", ascending=False).head(10)

# %%
p_id = 77
trial_id = 18

example_trial = trials["trials"][p_id]["preference_trials"][trial_id]

example_trial

# %%
for sol_id in (1, 2):

    create_stimulus(
        worker_capacities=example_trial["worker_capacities"],
        task_loads=example_trial["task_loads"],
        directory=EXPORT_PATH_PUB,
        out="png",
        labels=False,
        annotations=None,
        sm=example_trial[f"solution_{sol_id}"],
        order=None,
        worker_order=example_trial[f"worker_order_{sol_id}"],
        task_order=example_trial[f"task_order_{sol_id}"],
        fname_base=f"example_{metric}_{sol_id}",
    )
    
    _ = crop_image(
        Path(EXPORT_PATH_PUB, "png",f"example_{metric}_{sol_id}.png"),
        margin=1,
    )

# %% [markdown]
# ## 3. Example for Visual-Order Complexity (VC)

# %%
metric = "disorder"

df.loc[df["visual_manipulation"]!= "none"]

# %%
p_id = 0
trial_id = 7

example_trial = trials["trials"][p_id]["preference_trials"][trial_id]

example_trial

# %%
for sol_id in (1, 2):

    create_stimulus(
        worker_capacities=example_trial["worker_capacities"],
        task_loads=example_trial["task_loads"],
        directory=EXPORT_PATH_PUB,
        out="png",
        labels=False,
        annotations=None,
        sm=example_trial[f"solution_{sol_id}"],
        order=None,
        worker_order=example_trial[f"worker_order_{sol_id}"],
        task_order=example_trial[f"task_order_{sol_id}"],
        fname_base=f"example_{metric}_{sol_id}",
    )
    
    _ = crop_image(
        Path(EXPORT_PATH_PUB, "png",f"example_{metric}_{sol_id}.png"),
        margin=1,
    )
