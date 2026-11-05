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
# # Demographics
#
# - Requirement
#     - Download the demographics CSV files for each data collection (prolific: "study") and place them in `EXPORT_PATH_DATA/<STUDY>/"prolific"/`.
#     - Run `download_eye_data.py/ipynb` to add mobile device users to csv file in `BLACKLIST_PATH`.
# - Creates `<STUDY>/prolific_demographics.csv`
# - Create publication metrics (JSON)
# - Afterwards run `data_processing/preprocess_experiment_data.ipynb/py`

# %% [markdown]
# ## 0. Setup

# %%
import os
import glob
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from joptim.utils.export_path import get_export_paths

ANALYSIS_NAME = "demographics"


EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
STUDY = "exploratory"  # "exploratory" or "confirmatory"
DEMOGRAPHICS_OUTPUT_PATH = Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, "prolific_demographics.csv")
BLACKLIST_PATH = Path(EXPORT_PATH_DATA, STUDY, "blacklist.csv")
if BLACKLIST_PATH.exists():
    DF_BLACKLIST = pd.read_csv(BLACKLIST_PATH, dtype={"trial_file_id":"string"})
else:
    DF_BLACKLIST = pd.DataFrame([])
os.makedirs(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME), exist_ok=True)
os.makedirs(Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME), exist_ok=True)


# requires to download data first in notebooks/data_processing/download_experiment_data.py
with open(
    Path(EXPORT_PATH_DATA, "results_no_exclusion.pkl"), "rb"
) as pickle_file:
    results = pickle.load(pickle_file)

with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
    results_with_exclusion = pickle.load(pickle_file)

included_participant_ids = list(results_with_exclusion["user"]["id"])
del results_with_exclusion

# filter to completed users
bool_values_start_at = 3
results["user"] = results["user"][
    np.all(results["user"].iloc[:, bool_values_start_at:], axis=1)
]

df_user = results["user"]

if len(DF_BLACKLIST)>0:
    print("The following participants were blacklisted and excluded:")
    N_BLACKLIST_REMOVED = 0
    for row in DF_BLACKLIST.itertuples():
        tfi = str(row[1])
        tfui = int(row[2])
        assert df_user["trial_file_id"].dtype == object, "Filter with tfi will only work if column is type object/str."
        assert df_user["trial_file_user_id"].dtype == int, "Filter with tfui will only work if column is type int."
        print("trial_file_id:", tfi, "| user_id:", tfui, "| reason:", row[3])
        N_BLACKLIST_REMOVED += int(
            ((df_user["trial_file_id"].astype(str) == tfi) & (df_user["trial_file_user_id"].astype(str) == str(tfui))).sum()
        )
        df_user = df_user.loc[~((df_user["trial_file_id"] == tfi) & (df_user["trial_file_user_id"] == tfui))].reset_index(drop=True)
    print(f"Blacklisted and excluded: {N_BLACKLIST_REMOVED} participants.")
else:
    N_BLACKLIST_REMOVED = 0
        

df_user["included"] = df_user.id.isin(included_participant_ids)
df_user = df_user.merge(results["survey"][["id", "result_demographics"]], on="id")

# r2 (CHBR round 2, editor point 1): keep the raw trial tables for the
# per-criterion exclusion breakdown below (they are dropped with `results`).
PREF_ALL = results["preference_trials"]
SOLVING_ALL = results["solving_trials"]
del results

print(
    f"""\nWe have data from {len(df_user)} participants of which {sum(df_user["included"] == True)} ({(df_user["included"] == True).mean():.1%}) """
    """completed the study.\n\nIn the next steps we will \n  a) see how many of them were prolific participants """
    f"""in the {STUDY} study and \n  b) store their demographics profiles in {EXPORT_PATH_DATA}/{STUDY}/prolific_demographics.csv."""
)

# %% [markdown]
# ## 1. Create Demographics Data
# - contains all participants participating in the respective study (exploratory/confirmatory) (unfiltered)
# - columns `included` and `trial_file_id` can later be used for filtering.

# %%
csv_files = Path(EXPORT_PATH_DATA, STUDY, "prolific").glob("*.csv")

df_prolific = pd.DataFrame()

# loop over the list of csv files
for f in csv_files:

    # read the csv file
    df_tmp = pd.read_csv(f)
    df_tmp["file"] = os.path.basename(f)
    df_prolific = pd.concat([df_prolific, df_tmp], sort=False)

df_prolific = (
    df_prolific.loc[df_prolific["Status"] == "APPROVED"]
    .drop("Status", axis=1)
    .reset_index(drop=True)
)

df_prolific = df_prolific[
    [
        "Participant id",
        "Time taken",
        "Total approvals",
        "Fluent languages",
        "Webcam",
        "Computer operating system",
        "Age",
        "Sex",
        "Ethnicity simplified",
        "Country of birth",
        "Country of residence",
        "Nationality",
        "Language",
        "Student status",
        "Employment status",
    ]
]

df = df_prolific.merge(
    df_user[["pid", "included", "result_demographics", "trial_file_id"]],
    left_on="Participant id",
    right_on="pid",
    how="inner",
)

demographics_dict = {
    "1_handedness": {
        "title": "Handedness",
        "options": ["Right", "Left", "Ambidextrous"],
    },
    "2_highest_edu": {
        "title": "Highest education level completed",
        "options": [
            "No formal qualifications",
            "Secondary education (e.g. GED/GCSE)",
            "High school diploma/A-levels",
            "Technical/community college",
            "Undergraduate degree (BA/BSc/other)",
            "Graduate degree (MA/MSc/MPhil/other)",
            "Doctorate degree (PhD/other)",
            "Don't know / not applicable",
        ],
    },
    "3_degree_subject": {
        "title": "Degree subject",
        "options": [
            "Arts & Humanities",
            "Education",
            "Social Sciences",
            "Journalism & Information Business",
            "Administration & Law",
            "Mathematics and statistics",
            "Information and Communication Technologies",
            "Engineering, manufacturing and construction",
            "Agriculture, forestry, fisheries and veterinary",
            "Health and welfare",
            "Services",
            "Natural sciences",
            "History",
            "Other",
        ],
    },
}

df_result_demographics = df["result_demographics"].apply(
    lambda x: pd.Series(
        {
            demographics_dict[list(x.keys())[i]]["title"]: demographics_dict[
                list(x.keys())[i]
            ]["options"][int(x[list(x.keys())[i]])]
            for i in range(3)
        }
    )
)

df = df.merge(df_result_demographics, left_index=True, right_index=True).drop(
    ["result_demographics", "pid"], axis=1
)

# some formatting
df["Time taken"] = df["Time taken"] / 60
df["Age"] = df["Age"].astype(int)


df.to_csv(
    DEMOGRAPHICS_OUTPUT_PATH, index=False
)

print(f"""Stored demographics to {DEMOGRAPHICS_OUTPUT_PATH}\n""")


# %% [markdown]
# ## 2. Inspect

# %%
def extract_demographic_statistics(df, bins=5, included_only=False):
    for trial_file_id in df["trial_file_id"].unique():
        
        df_demographics = df.loc[df["trial_file_id"] == trial_file_id].copy().drop(columns=["Participant id"])
        
        print(
            f"\ntrial_file_id {trial_file_id}:", f"""N_total = {len(df_demographics)}, N_included = {sum(df_demographics["included"])}"""
            f""" (Exclusion rate = {1 - sum(df_demographics["included"])/len(df_demographics):.1%})"""
        )

        if included_only:
            print("Statistics are for included participants only\n")
            df_demographics = df_demographics.loc[df_demographics["included"] == True]

        else:
            print("Statistics are for all participants (included + excluded)\n")        
        
        statistics = {}
        for column in df_demographics.columns:
            if df_demographics[column].dtype in ["int64", "float64"]:
                binned_data = pd.cut(df_demographics[column], bins)
                proportions = binned_data.value_counts(normalize=True)
    
                mean = df_demographics[column].mean()
                std_dev = df_demographics[column].std()
                min_val = df_demographics[column].min()
                max_val = df_demographics[column].max()
    
                statistics[column] = {
                    "proportions": proportions,
                    "mean": mean,
                    "std_dev": std_dev,
                    "min": min_val,
                    "max": max_val,
                }
            else:
                proportions = df_demographics[column].value_counts(normalize=True)
                statistics[column] = {"proportions": proportions}
    
        for column, stats in statistics.items():
            print(f"Statistics for {column}:")
            if "mean" in stats:
                print(f'  Mean: {stats["mean"]:.2f}')
                print(f'  Standard Deviation: {stats["std_dev"]:.2f}')
                print(f'  Min: {stats["min"]:.2f}')
                print(f'  Max: {stats["max"]:.2f}')
    
            print("  Proportions:")
            for value, prop in stats["proportions"].items():
                print(f"    {value}: {prop:.2%}")
            print()


extract_demographic_statistics(df, included_only=True)

# %% [markdown]
# ## 2. Publication metrics export (JSON)

# %% [markdown]
# - Here we filter further to the actual trial file used

# %%
with open(Path(EXPORT_PATH_DATA, STUDY, "trials.pkl"), "rb") as fp:
    trials = pickle.load(fp)

trial_file_id = trials["file_id"]

del trials

def _sanitize_key(label: str) -> str:
    if pd.isna(label):
        return "na"
    s = str(label).strip().lower()
    rep = {
        # countries
        "united kingdom": "uk",
        "united states": "us",
        # education (highest)
        "no formal qualifications": "no_qualifications",
        "secondary education (e.g. ged/gcse)": "secondary",
        "high school diploma/a-levels": "high_school",
        "technical/community college": "technical_college",
        "undergraduate degree (ba/bsc/other)": "undergrad",
        "graduate degree (ma/msc/mphil/other)": "graduate",
        "doctorate degree (phd/other)": "doctorate",
        "don't know / not applicable": "dont_know_na",
        # employment
        "full-time": "full_time",
        "part-time": "part_time",
        "not in paid work (e.g. homemaker', 'retired or disabled)": "not_in_paid_work",
        "unemployed (and job seeking)": "unemployed",
        "other": "other",
        # handedness
        "right": "right",
        "left": "left",
        "ambidextrous": "ambidextrous",
        # sex
        "male": "male",
        "female": "female",
        # ethnicity simplified
        "white": "white",
        "black": "black",
        "asian": "asian",
        "mixed": "mixed",
        # student
        "yes": "yes",
        "no": "no",
    }
    s2 = rep.get(s, s)
    s2 = (
        s2.replace(" ", "_")
        .replace("/", "_")
        .replace("&", "and")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace("’", "")
        .replace("'", "")
    )
    return s2

def _proportions(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)
    total = counts.sum()
    if total == 0:
        return {}
    props = (counts / total).to_dict()
    return {_sanitize_key(k): float(v) for k, v in props.items()}

def _ensure_props(d: dict, keys: list[str]) -> dict:
    out = dict(d)
    for k in keys:
        out.setdefault(k, 0.0)
    return out

def _num_stats(s: pd.Series):
    s = s.dropna()
    if s.shape[0] == 0:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "median": float(np.median(s)),
        "q25": float(np.quantile(s, 0.25)),
        "q75": float(np.quantile(s, 0.75))
    }

def compute_json(df: pd.DataFrame, study: str, trial_file_id: int, analysis_name: str):

    df = df.loc[df["trial_file_id"] == str(trial_file_id)].copy()
    
    # Included subset
    df_inc = df.loc[df["included"] == True].copy()

    # Counts (df only)
    # N_completed: participants present in df (completed + merged)
    # N_included: final after exclusions
    # Note: "N_recruited" (all approved on Prolific incl. non-completers) is not derivable from df.
    pid_col = "Participant id" if "Participant id" in df.columns else None
    if pid_col is not None:
        N_completed = int(df[pid_col].nunique())
        N_included = int(df_inc[pid_col].nunique())
    else:
        N_completed = int(df.shape[0])
        N_included = int(df_inc.shape[0])

    counts = {
        "N_completed": N_completed,
        "N_included": N_included,
        "exclusion_rate": float(1.0 - (N_included / N_completed if N_completed else 0.0)),
    }

    # Numerics (included only)
    time_taken = _num_stats(df_inc["Time taken"]) if "Time taken" in df_inc.columns else _num_stats(pd.Series([], dtype=float))
    age_stats = _num_stats(df_inc["Age"]) if "Age" in df_inc.columns else _num_stats(pd.Series([], dtype=float))

    # Categorical proportions (included only)
    sex = {"prop": _proportions(df_inc["Sex"])} if "Sex" in df_inc.columns else {"prop": {}}
    sex["prop"] = _ensure_props(sex["prop"], ["male", "female"])

    ethnicity = {"prop": _proportions(df_inc["Ethnicity simplified"])} if "Ethnicity simplified" in df_inc.columns else {"prop": {}}
    ethnicity["prop"] = _ensure_props(ethnicity["prop"], ["white", "black", "asian", "mixed"])

    residence = {"prop": _proportions(df_inc["Country of residence"])} if "Country of residence" in df_inc.columns else {"prop": {}}
    residence["prop"] = _ensure_props(residence["prop"], ["uk", "us"])

    nationality = {"prop": _proportions(df_inc["Nationality"])} if "Nationality" in df_inc.columns else {"prop": {}}

    student_status = {"prop": _proportions(df_inc["Student status"])} if "Student status" in df_inc.columns else {"prop": {}}
    student_status["prop"] = _ensure_props(student_status["prop"], ["yes", "no"])

    employment_status = {"prop": _proportions(df_inc["Employment status"])} if "Employment status" in df_inc.columns else {"prop": {}}
    employment_status["prop"] = _ensure_props(employment_status["prop"], ["full_time", "part_time", "not_in_paid_work", "unemployed", "other"])

    # Survey-derived demographics (already merged into df)
    handedness = {"prop": _proportions(df_inc.get("Handedness", pd.Series(dtype=object)))}
    handedness["prop"] = _ensure_props(handedness["prop"], ["right", "left", "ambidextrous"])

    edu_highest = {"prop": _proportions(df_inc.get("Highest education level completed", pd.Series(dtype=object)))}
    edu_highest["prop"] = _ensure_props(
        edu_highest["prop"],
        ["no_qualifications", "secondary", "high_school", "technical_college", "undergrad", "graduate", "doctorate", "dont_know_na"],
    )

    degree_subject = {"prop": _proportions(df_inc.get("Degree subject", pd.Series(dtype=object)))}

    # Optional fields
    fluent_languages = {"prop": _proportions(df_inc.get("Fluent languages", pd.Series(dtype=object)))}
    language = {"prop": _proportions(df_inc.get("Language", pd.Series(dtype=object)))}
    webcam = {"prop": _proportions(df_inc.get("Webcam", pd.Series(dtype=object)))}
    os_prop = {"prop": _proportions(df_inc.get("Computer operating system", pd.Series(dtype=object)))}


    meta= {
        "analysis_name": analysis_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    out = {
        "meta": meta,
        "study": study,
        "counts": counts,
        "time_taken": time_taken,
        "age": age_stats,
        "sex": sex,
        "ethnicity": ethnicity,
        "residence": residence,
        "nationality": nationality,
        "student_status": student_status,
        "employment_status": employment_status,
        "handedness": handedness,
        "education_highest": edu_highest,
        "degree_subject": degree_subject,
        # optional
        "fluent_languages": fluent_languages,
        "language": language,
        "webcam": webcam,
        "os": os_prop,
    }
    return out
    
def _ts_str(ms):
    """UTC ISO timestamp string from a millisecond epoch."""
    from datetime import timezone
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def compute_r2_breakdown(trial_file_id, df_user, df_prolific, pref_all, solving_all,
                         included_participant_ids, n_blacklist_removed):
    """
    CHBR round 2 (editor points 1 and 2): per-criterion exclusion breakdown
    with overlap, and data-collection dates, for the analyzed sample.

    Criteria mirror the experiment pipeline exactly (see the preprocessing
    exclusion code): (a) failed both catch trials: `use_case == "catch"` and
    `choice != 0` on >= 2 catch trials; (b) used the duplicated-solutions
    button in >= 2 non-catch trials: `use_case != "catch"` and `choice == 0`.
    Criterion (c) of the manuscript ("did not complete the study") applies
    before these; it corresponds to approved participants without a recorded
    session (n_sessions - n_completed).
    """
    tf = str(trial_file_id)

    # population: user rows for this trial file that passed the completion
    # flags and (upstream) the study blacklist; the blacklist rows were
    # already removed, so re-add them to n_sessions for the accounting
    rows = df_user.loc[df_user["trial_file_id"].astype(str) == tf].copy()
    n_sessions = len(rows) + int(n_blacklist_removed)

    xid_by_pid = dict(zip(rows["pid"].astype(str), rows["xid"].astype(str)))
    id_by_xid = dict(zip(rows["xid"].astype(str), rows["id"]))
    approved_pids = set(df_prolific["Participant id"].astype(str))
    approved_xids = {xid_by_pid[p] for p in xid_by_pid if p in approved_pids}

    pref_tf = pref_all.loc[pref_all["trial_file_id"].astype(str) == tf].copy()
    pref_tf["xid"] = pref_tf["xid"].astype(str)
    catch_fail = pref_tf.loc[
        (pref_tf["use_case"] == "catch") & (pref_tf["choice"] != 0), "xid"
    ].value_counts()
    dup_use = pref_tf.loc[
        (pref_tf["use_case"] != "catch") & (pref_tf["choice"] == 0), "xid"
    ].value_counts()
    crit_a = {x for x, k in catch_fail.items() if k >= 2 and x in approved_xids}
    crit_b = {x for x, k in dup_use.items() if k >= 2 and x in approved_xids}

    # pipeline-excluded among the approved completers, for verification only
    included_ids = set(included_participant_ids)
    excluded_xids = {
        x for x in approved_xids if id_by_xid[x] not in included_ids
    }
    assert excluded_xids == (crit_a | crit_b), (
        f"pipeline exclusions ({len(excluded_xids)}) differ from the "
        f"criteria union ({len(crit_a | crit_b)}); investigate before use"
    )

    breakdown = {
        "n_recruited": None,  # set below; approved submissions of the cohort launches
        "n_sessions": int(n_sessions),
        "n_completed": int(len(approved_xids)),
        "n_excluded_catch": int(len(crit_a)),
        "n_excluded_duplicate": int(len(crit_b)),
        "n_overlap_catch_duplicate": int(len(crit_a & crit_b)),
        "n_excluded": int(len(excluded_xids)),
        "n_included": int(len(approved_xids - excluded_xids)),
        # fixed historical facts (see the recruited comment below): approved
        # participants without a usable session record. NOTE: n_sessions
        # counts all recorded sessions of the trial file, including sessions
        # outside the recruited cohorts (exploratory: 3 additional sessions
        # in 2024-10/12 without Prolific submissions, not part of the
        # analyzed sample), so "did not complete" is not n_recruited - n_sessions.
        "n_did_not_complete": {"exploratory": 3, "confirmatory": 2}[STUDY],
        "n_excluded_mobile": int(n_blacklist_removed),
    }

    # recruited: approved Prolific submissions of this study's cohort launches.
    # Fixed historical facts, derived from the Prolific exports:
    #   exploratory: 117 = approved submissions with "Started at" >= the final
    #     trial file creation (2024-08-26 17:21 UTC; file id 1724688094.1341743):
    #     cohort launches of 3, 6, 12, 24, 36, 36 (2024-08-27 .. 2024-09-12);
    #     the two earlier procedure-piloting launches of 3 + 3 participants
    #     (2024-08-02/10) predate the final experiment version
    #   confirmatory: 90 (cohort launch 2025-11-10, Prolific export 6911e7d6e);
    #     an earlier piloting launch (2025-11-07, 21 participants, trial file
    #     1746822738) is not part of the confirmatory study
    n_recruited_r2 = {"exploratory": 117, "confirmatory": 90}
    breakdown["n_recruited"] = n_recruited_r2[STUDY]

    # data-collection dates over the analyzed participants' sessions only
    scoped = lambda dfx: dfx[dfx["xid"].astype(str).isin(approved_xids)]
    starts = [scoped(pref_tf)["start_time"].values]
    ends = [scoped(pref_tf)["end_time"].values]
    starts = [s for s in starts if len(s)]
    ends = [e for e in ends if len(e)]
    sol_tf = scoped(solving_all.loc[solving_all["trial_file_id"].astype(str) == tf])
    if len(sol_tf):
        starts.append(sol_tf["start_time"].values)
        ends.append(sol_tf["end_time"].values)
    min_start = int(np.min([s.min() for s in starts if len(s)]))
    max_end = int(np.max([e.max() for e in ends if len(e)]))
    dates = {
        "min": _ts_str(min_start),
        "max": _ts_str(max_end),
        "min_date": _ts_str(min_start)[:10],
        "max_date": _ts_str(max_end)[:10],
    }

    return breakdown, dates


out = compute_json(df, STUDY, trial_file_id, ANALYSIS_NAME)

# --- CHBR round 2 (editor points 1 and 2) ------------------------------------
# Per-criterion exclusion breakdown with overlap, and data-collection dates,
# for the analyzed sample. Criteria mirror the experiment pipeline exactly:
# (a) failed both catch trials (>= 2 false responses on catch trials, where
#     choice != 0 on a catch trial), (b) used the duplicated-solutions button
#     in >= 2 non-catch trials (choice == 0 on a non-catch trial).
out["exclusions"], out["dates"] = compute_r2_breakdown(
    trial_file_id, df_user, df_prolific, PREF_ALL, SOLVING_ALL,
    included_participant_ids, N_BLACKLIST_REMOVED,
)
assert out["exclusions"]["n_completed"] == out["counts"]["N_completed"], (
    "r2 breakdown does not agree with counts.N_completed")
assert out["exclusions"]["n_included"] == out["counts"]["N_included"], (
    "r2 breakdown does not agree with counts.N_included")
assert (out["exclusions"]["n_excluded_catch"] + out["exclusions"]["n_excluded_duplicate"]
        - out["exclusions"]["n_overlap_catch_duplicate"]) == (
        out["counts"]["N_completed"] - out["counts"]["N_included"]), (
    "criteria breakdown does not sum to the exclusion count")

out_path = Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{ANALYSIS_NAME}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {out_path}")
