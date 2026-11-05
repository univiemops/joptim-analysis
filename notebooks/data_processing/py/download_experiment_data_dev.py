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

# %%
# #!/usr/bin/env python
# coding: utf-8

# Download experimental data
# ===

import os

import argparse
import pickle
from glob import glob
from copy import deepcopy
import logging
import requests
import urllib3
import numpy as np
import pandas as pd
from pathlib import Path

from joptim.download import test_solutions, test_workers_tasks, filter_by_frequency
from joptim.identifiers import matrix_to_identifier
from joptim.utils.log_config import setup_logging
from joptim.utils.export_path import get_export_paths
from joptim.utils.secrets import load_secrets


urllib3.disable_warnings()

try:
    setup_logging(os.path.basename(__file__))
except Exception as e:
    print(e)

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()

# minimum number of false negatives for exclusion
EXCLUSION_CRIT_FNEG = 2

# minimum number of false positives for exclusion
EXCLUSION_CRIT_FPOS = 2

# 1. DOWNLOAD

api_key, server_url = load_secrets()

res = requests.get(server_url + f"/joptim-api/{api_key}/tables", verify=False)
ignore_tables = ["alembic_version"]
tables = res.json()["tables"]
tables = [t for t in tables if t not in ignore_tables]
logging.info(f"Available tables {tables}")

results_raw = {}
for table in tables:
    res = requests.get(
        server_url + f"/joptim-api/{api_key}/tables/{table}", verify=False
    )
    results_raw[table] = pd.DataFrame(res.json())

# %% [markdown]
# Remove all results that have old format

# %%
# remove all results from old experiment versions
# i.e., trial_file_id == 1724688094.1341743	and trial_file_user_id < 40

crit_1 = "1724688094.1341743"
crit_2 = 40

relevant_xids = (
    results_raw["user"].loc[(results_raw["user"]["trial_file_id"] != crit_1) | (results_raw["user"]["trial_file_user_id"] >= crit_2), "xid"].tolist()
)
for table in list(results_raw):
    try:
        results_raw[table] = (
            results_raw[table]
            .loc[results_raw[table].xid.isin(relevant_xids)]
            # sort by row id in database
            .sort_values("id")
            .reset_index(drop=True)
        )
        if table != "user":
            # remove id column as it could be confused with the participant id in table user
            results_raw[table] = results_raw[table].drop("id", axis=1)
            # add id for participant to all other tables
            results_raw[table] = results_raw[table].merge(
                results_raw["user"][["xid", "id"]],
                on="xid",
                how="left",
            )
            # shift column 'id' to first position
            first_column = results_raw[table].pop("id")
            results_raw[table].insert(0, "id", first_column)
    except Exception as e:
        logging.warning("Warning table %s: %s", table, e)

# %% [raw]
# trial_file_id = "1724688094.1341743"
# trial_file_user_id < 40
#
# if min_id is not None:
#     relevant_xids = (
#         results_raw["user"].loc[results_raw["user"]["id"] >= min_id, "xid"].tolist()
#     )
#     for table in list(results_raw):
#         try:
#             results_raw[table] = (
#                 results_raw[table]
#                 .loc[results_raw[table].xid.isin(relevant_xids)]
#                 # sort by row id in database
#                 .sort_values("id")
#                 .reset_index(drop=True)
#             )
#             if table != "user":
#                 # remove id column as it could be confused with the participant id in table user
#                 results_raw[table] = results_raw[table].drop("id", axis=1)
#                 # add id for participant to all other tables
#                 results_raw[table] = results_raw[table].merge(
#                     results_raw["user"][["xid", "id"]],
#                     on="xid",
#                     how="left",
#                 )
#                 # shift column 'id' to first position
#                 first_column = results_raw[table].pop("id")
#                 results_raw[table].insert(0, "id", first_column)
#         except Exception as e:
#             logging.warning("Warning table %s: %s", table, e)

# %% [markdown]
# Tests

# %%
# 2. TESTS & DATA CLEANING
# 2.1. TESTS

logging.info("Compare data with original trials.")
trial_data = {}
trial_files = glob(Path(EXPORT_PATH_DATA, "trials*.pkl"))

for file in trial_files:
    with open(file, "rb") as pickle_file:
        trial_data_tmp = pickle.load(pickle_file)
    if "file_id" in trial_data_tmp:
        trial_data[str(trial_data_tmp["file_id"])] = trial_data_tmp["trials"]
        logging.info(
            "file_id %s: trial sets for %s participants loaded.",
            trial_data_tmp["file_id"],
            len(trial_data_tmp["trials"]),
        )
    else:
        logging.info("trials from file %s not loaded: file_id missing.", file)
    del trial_data_tmp

idx_of_first_bool_var = 3
idx_of_last_bool_var = 10

users_completed = (
    # filter to participants that have completed all parts of the experiment
    results_raw["user"]
    .loc[
        results_raw["user"].iloc[:, idx_of_first_bool_var:idx_of_last_bool_var+1].sum(axis=1)
        == len(list(results_raw["user"])[idx_of_first_bool_var:idx_of_last_bool_var+1]),
        :,
    ]
    .copy()
    .reset_index(drop=True)
)

logging.info(f"{len(users_completed)} participants completed the study.")

# Main execution loop for equality checks
result_types = ["results_trials_1", "results_trials_2"]

for trial_file_id, trial_file_user_id, xid in users_completed[
    ["trial_file_id", "trial_file_user_id", "xid"]
].itertuples(index=False, name=None):
    for result_type in result_types:
        user_results = (
            results_raw[result_type]
            .loc[results_raw[result_type]["xid"] == xid]
            .result.values.copy()
        )

        if result_type == "results_trials_1":
            sample_trials_str = "solving_trials"

            # Compare only tasks and workers
            comparison_fields = [
                ("task_loads", "tasks"),
                ("worker_capacities", "workers"),
            ]
            for i_trial, result_trial in enumerate(user_results):
                for sample_field, result_field in comparison_fields:
                    sample_ar = trial_data[trial_file_id][trial_file_user_id - 1][
                        sample_trials_str
                    ][i_trial][sample_field]
                    result_ar = result_trial[result_field]
                    iscorrect = np.array_equal(sample_ar, result_ar)
                    if not iscorrect:
                        print(
                            f"file {trial_file_id} user {trial_file_user_id} {xid} {sample_trials_str} trial {i_trial} {sample_field} == {result_field}: {iscorrect}"
                        )

        elif result_type == "results_trials_2":
            sample_trials_str = "preference_trials"

            for i_trial, result_trial in enumerate(user_results):
                sample_trial = trial_data[trial_file_id][trial_file_user_id - 1][
                    sample_trials_str
                ][i_trial]

                # Testing tasks1, tasks2, workers1, workers2 using
                # test_workers_tasks
                for field in ["tasks1", "tasks2", "workers1", "workers2"]:
                    sample_field = (
                        sample_trial["task_loads"]
                        if "tasks" in field
                        else sample_trial["worker_capacities"]
                    )
                    order_field = (
                        sample_trial["task_order_1"]
                        if "1" in field
                        else sample_trial["task_order_2"]
                    )
                    if "workers" in field:
                        order_field = sample_trial[f"worker_order_{field[-1]}"]
                    iscorrect = test_workers_tasks(
                        sample_field, order_field, result_trial[field]
                    )
                    if not iscorrect:
                        print(
                            f"file {trial_file_id} user {trial_file_user_id} {xid} {sample_trials_str} trial {i_trial} {field} differ"
                        )

                # Testing solutions
                solution_1_correct, solution_2_correct = test_solutions(
                    sample_trial, result_trial
                )
                if not solution_1_correct:
                    print(
                        f"file {trial_file_id} user {trial_file_user_id} {xid} {sample_trials_str} trial {i_trial} solution 1 differ"
                    )
                if not solution_2_correct:
                    print(
                        f"file {trial_file_id} user {trial_file_user_id} {xid} {sample_trials_str} trial {i_trial} solution 2 differ"
                    )

        del user_results

logging.info("Equality checks complete.")

# %% [markdown]
# Data Cleaning

# %%
# 2.2. DATA CLEANING

results_no_exclusion = deepcopy(results_raw)

# Rename keys to align with naming in trial files such, e.g., `trials.pkl`
results_no_exclusion["solving_trials"] = results_no_exclusion.pop(
    "results_trials_1"
)
results_no_exclusion["preference_trials"] = results_no_exclusion.pop(
    "results_trials_2"
)

# add trial_file_id and trial_file_user_id columns
df_trial_file_info = results_no_exclusion["user"][["xid", "trial_file_id", "trial_file_user_id"]]
results_no_exclusion["solving_trials"] = results_no_exclusion["solving_trials"].merge(df_trial_file_info, on="xid", how="left")
results_no_exclusion["preference_trials"] = results_no_exclusion["preference_trials"].merge(df_trial_file_info, on="xid", how="left")

# Delete Eye-Tracking and rrweb (these results we will add later)
del results_no_exclusion["eyetracking"]
del results_no_exclusion["rrweb"]

# solving trials: create column trial_id
# and shift column 'trial_id' to third position
trial_id = (
    results_no_exclusion["solving_trials"]
    .groupby(["trial_file_id", "trial_file_user_id", "xid"])
    .cumcount()
)
results_no_exclusion["solving_trials"].insert(2, "trial_id", trial_id)

# preference trials: create column trial_id
# and shift column 'trial_id' to third position
trial_id = (
    results_no_exclusion["preference_trials"]
    .groupby(["trial_file_id", "trial_file_user_id", "xid"])
    .cumcount()
)
results_no_exclusion["preference_trials"].insert(2, "trial_id", trial_id)

# keep sorting in check
results_no_exclusion["preference_trials"] = (
    results_no_exclusion["preference_trials"]
    .sort_values(["trial_file_id", "trial_file_user_id", "trial_id"])
    .reset_index(drop=True)
)
results_no_exclusion["solving_trials"] = (
    results_no_exclusion["solving_trials"]
    .sort_values(["trial_file_id", "trial_file_user_id", "trial_id"])
    .reset_index(drop=True)
)

# remove results, keep only the new information
result_fields_to_columns = [
    "assignments",
    "tileEvents",
    "start_time",
    "end_time",
]
for field in result_fields_to_columns:
    results_no_exclusion["solving_trials"][field] = results_no_exclusion[
        "solving_trials"
    ]["result"].apply(lambda x: x[field])
del results_no_exclusion["solving_trials"]["result"]

result_fields_to_columns = ["choice", "start_time", "end_time"]
for field in result_fields_to_columns:
    results_no_exclusion["preference_trials"][field] = results_no_exclusion[
        "preference_trials"
    ]["result"].apply(lambda x: x[field])
del results_no_exclusion["preference_trials"]["result"]

# add response times
results_no_exclusion["preference_trials"]["rt"] = (
    results_no_exclusion["preference_trials"]["end_time"]
    - results_no_exclusion["preference_trials"]["start_time"]
)
results_no_exclusion["solving_trials"]["rt"] = (
    results_no_exclusion["solving_trials"]["end_time"]
    - results_no_exclusion["solving_trials"]["start_time"]
)

# solving trials

columns_to_add = [
    "task_loads",
    "worker_capacities",
    "capacity_load_ratio",
    "utility",
    "iterations",
    "n_optimal_solutions",
    "greedy_solution",
    "difficulty",
    "example_solution",
]

results_no_exclusion["solving_trials"][columns_to_add] = results_no_exclusion[
    "solving_trials"
].apply(
    lambda row: pd.Series(
        trial_data[row.trial_file_id][row.trial_file_user_id - 1]["solving_trials"][
            row.trial_id
        ]
    )[columns_to_add],
    axis=1,
)

# preference trials

columns_to_add = [
    "capacity_load_ratio",
    "difficulty",
    "diff_ac",
    "diff_cc",
    "task_loads",
    "worker_capacities",
    "task_order_1",
    "task_order_2",
    "worker_order_1",
    "worker_order_2",
    "solution_1",
    "solution_2",
    "utility",
    "visual_manipulation",
    "use_case",
]

results_no_exclusion["preference_trials"][columns_to_add] = results_no_exclusion[
    "preference_trials"
].apply(
    lambda row: pd.Series(
        trial_data[row.trial_file_id][row.trial_file_user_id - 1][
            "preference_trials"
        ][row.trial_id]
    )[columns_to_add],
    axis=1,
)

# drop id (represents the row number in the user table -> not useful)
del results_no_exclusion["solving_trials"]["id"]
del results_no_exclusion["preference_trials"]["id"]

# remove incomplete users
for field in results_no_exclusion:
    results_no_exclusion[field] = (
        results_no_exclusion[field]
        .loc[results_no_exclusion[field]["xid"].isin(list(users_completed["xid"]))]
        .reset_index(drop=True)
    )

# Create human-readable and unique identifiers for binary solution
# matrices (makes it easier to compare solutions during coherence checks)

results_no_exclusion["preference_trials"]["solution_1_id"] = results_no_exclusion[
    "preference_trials"
].apply(
    lambda row: pd.Series(
        matrix_to_identifier(
            row.solution_1[row.task_order_1][:, row.worker_order_1]
        )
    ),
    axis=1,
)

results_no_exclusion["preference_trials"]["solution_2_id"] = results_no_exclusion[
    "preference_trials"
].apply(
    lambda row: pd.Series(
        matrix_to_identifier(
            row.solution_2[row.task_order_2][:, row.worker_order_2]
        )
    ),
    axis=1,
)

# %% [markdown]
# Exlclusion of participants

# %%
# 2.3. EXCLUSION

# 2.3.1. exclude participants

results = deepcopy(results_no_exclusion)

id_exclusion = []

id_catch_false_neg = list(
    results["preference_trials"].loc[
        (results["preference_trials"]["use_case"] == "catch")
        & (results["preference_trials"]["choice"] != 0),
        "xid",
    ]
)

id_catch_false_pos = list(
    results["preference_trials"].loc[
        (results["preference_trials"]["use_case"] != "catch")
        & (results["preference_trials"]["choice"] == 0),
        "xid",
    ]
)

# exclusion list for false negatives (at least 1)
id_exclusion_false_neg = filter_by_frequency(
    id_catch_false_neg, EXCLUSION_CRIT_FNEG
)
id_exclusion.extend(id_exclusion_false_neg)

# exclusion list for false positives (at least 2)
id_exclusion_false_pos = filter_by_frequency(
    id_catch_false_pos, EXCLUSION_CRIT_FPOS
)
id_exclusion.extend(id_exclusion_false_pos)

# eliminate duplicates
id_exclusion = list(set(id_exclusion))

for field in list(results):

    results[field] = (
        results[field]
        .loc[~results[field]["xid"].isin(id_exclusion)]
        .reset_index(drop=True)
    )

logging.info(f"Excluded participant IDs: {id_exclusion}")
logging.info(
    f"""Remaining participants: {len(results["user"])} ({len(results["user"]) / len(users_completed):.1%})"""
)

logging.info(f"Exclusion rate: {len(id_exclusion) / len(users_completed):.1%}")
logging.info(
    f"{len(set(id_exclusion_false_neg) - set(id_exclusion_false_pos)) / len(id_exclusion):.1%} of exclusions are due to false negatives"
)
logging.info(
    f"{len(set(id_exclusion_false_pos) - set(id_exclusion_false_neg)) / len(id_exclusion):.1%} of exclusions are due to false positives"
)
logging.info(
    f"{len(set(id_exclusion_false_pos) & set(id_exclusion_false_neg)) / len(id_exclusion):.1%} of exclusions are due to both"
)

# %% [markdown]
# Exclusion of trials

# %%
# 2.3.2. exclude trials

# Remove trials from remaining participants
#    - all catch trials
#    - all trials with choice 0

results["preference_trials"] = (
    results["preference_trials"]
    .loc[results["preference_trials"]["use_case"] != "catch"]
    .reset_index(drop=True)
)
logging.info(
    f"""Remaining trials after removing all catch trials: {len(results["preference_trials"])}"""
)
results["preference_trials"] = (
    results["preference_trials"]
    .loc[results["preference_trials"]["choice"] != 0]
    .reset_index(drop=True)
)
logging.info(
    f"""Remaining trials after removing all trials with choice 0: {len(results["preference_trials"])}"""
)

# %% [markdown]
# Save

# %%
# 2.4. Save results to disk

with open(Path(EXPORT_PATH_DATA, "results.pkl"), "wb") as pickle_file:
    pickle.dump(results, pickle_file)

with open(
    Path(EXPORT_PATH_DATA, "results_no_exclusion.pkl"), "wb"
) as pickle_file:
    pickle.dump(results_no_exclusion, pickle_file)

logging.info(
    f"""Saved results with exclusion to: {Path(EXPORT_PATH_DATA, "results.pkl")}"""
)
logging.info(
    f"""Saved results without exclusion to: {Path(EXPORT_PATH_DATA, "results_no_exclusion.pkl")}"""
)
