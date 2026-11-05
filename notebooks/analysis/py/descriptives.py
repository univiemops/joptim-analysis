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
# # Descriptive Statistics
#
# 1. Problem-solving performance and
# 2. Self-reported problem-solving skills
#
# in the sample and how they relate.

# %% [markdown]
# **TODO: prepare publication metrics and figures**
#
# - add here descriptives about sampled probems, solutions, and trials (total sample as well as trial that participants received)?

# %% [markdown]
# Prerequisites:
# - `data_preprocessed_<TRIAL_FILE_ID>.csv` (make sure to have downloaded prolific demographics, run `demographics.ipynb/py` and `preprocess_experimental_data.ipynb/py`)

# %% [markdown]
# ## 0. Setup

# %%
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# Optional convenience: seaborn for heatmaps (style comes from figures.setup_style)
import seaborn as sns

from pub_utils.figures import mm_to_in
from joptim.utils.export_path import get_export_paths
from pub_utils.figures import (
    setup_style,
    make_fig,
    make_grid,
    hist_panel,
    save,
    hide_interior_labels,
    file_dimensions,
)



ANALYSIS_NAME = "descriptives"

STUDY = "confirmatory"  # "exploratory" or "confirmatory"

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
CC_VERSION = "v2"  # "v1" or "v2"

os.makedirs(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME), exist_ok=True)
os.makedirs(Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME), exist_ok=True)

with open(Path(EXPORT_PATH_DATA, STUDY, "trials.pkl"), "rb") as pickle_file:
    trials = pickle.load(pickle_file)

TRIAL_FILE_ID = trials["file_id"]

updated_str = ""
if STUDY == "exploratory":
    OPT = ""
    updated_str = "_updated"
elif STUDY in ["confirmatory", "exploratory_with_optimized_cc"]:
    OPT = "_opt"
    if STUDY == "exploratory_with_optimized_cc":
        updated_str = "_updated"

else:
    raise ValueError(f"STUDY can only be 'exploratory', 'confirmatory' or 'exploratory_with_optimized_cc', got: {STUDY}")

cc_variable_left = f"cc_{CC_VERSION}_L{OPT}"
cc_variable_right = f"cc_{CC_VERSION}_R{OPT}"
diff_cc_variable = f"diff_cc_{CC_VERSION}_RvL{OPT}"
cc_variable_left_z = f"cc_{CC_VERSION}_L{OPT}_z"
cc_variable_right_z = f"cc_{CC_VERSION}_R{OPT}_z"
diff_cc_variable_z = f"diff_cc_{CC_VERSION}_RvL{OPT}_z"

data = pd.read_csv(
    Path(
        EXPORT_PATH_DATA,
        STUDY,
        f"""data_preprocessed{updated_str}_{str(TRIAL_FILE_ID).replace(".", "-")}.csv""",
    ),
    sep=",",
    dtype={"trial_file_id": float},
)

data_eye = pd.read_csv(
    Path(EXPORT_PATH_DATA, STUDY, "mlm_gaze", f"""data_eye_{str(TRIAL_FILE_ID).replace(".", "-")}.csv""")
)
data = (
    data.merge(
        data_eye,
        on=["trial_file_id", "trial_file_user_id", "trial_id"],
        how="left",
    )
    # .dropna()
    .copy()
)

# filter to TRIAL_FILE_ID (loading from above file should be enough,
# but we want to be on the safe side)
data = data.loc[data["trial_file_id"] == TRIAL_FILE_ID].reset_index(drop=True)
data = data.drop(["trial_file_id", "trial_file_user_id"], axis=1)

data["choice_cat_RvL"] = pd.Categorical(
    data["choice_cat_RvL"],
    # categories=['-2', '-1', '1', '2'],
    categories=[
        "definitely left",
        "slightly left",
        "slightly right",
        "definitely right",
    ],
    ordered=True,
)

data["gaze_bias"] = (data["eye_R"] - data["eye_L"]) / (data["eye_R"] + data["eye_L"])
data["log_rt"] = np.log(data["rt"])

with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
    results = pickle.load(pickle_file)

print(len(data.participant.unique()), "participants loaded.")
print(f"""{len(data)} trials loaded ({STUDY} study / trial file: {TRIAL_FILE_ID}).""")

# -------------------
# Column names (signed / absolute)
# -------------------

CC_PREDICTORS = [f"diff_cc_{CC_VERSION}_RvL_z"]
CC_PREDICTORS_ABS = [f"abs_diff_cc_{CC_VERSION}_RvL_z"]

if STUDY == "confirmatory":
    CC_PREDICTORS.append(f"diff_cc_{CC_VERSION}_RvL_opt_z")
    CC_PREDICTORS_ABS.append(f"abs_diff_cc_{CC_VERSION}_RvL_opt_z")
    


diff_signed = [
    "diff_ac_RvL_z",
    *CC_PREDICTORS,
    "diff_disorder_RvL_z",
    "diff_diaga_RvL_z",
    "max_disorder_z",
]
diff_abs = [
    "abs_diff_ac_RvL_z",
    *CC_PREDICTORS_ABS,
    "abs_diff_disorder_RvL_z",
    "abs_diff_diaga_RvL_z",
    "max_disorder_z",
]

# Participant- and problem-level cols
col_psi = "psi"
col_eff = "efficiency"
col_diff = "load_capacity_ratio"  # problem difficulty (raw)
col_ho = "heuristic_optimality"  # raw

# Output directory
OUT_DIR = Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
setup_style(
    profile="nature",
    use_tex=False,
    base_font=7,
    title_font_delta=0,
    minor_ticks=False,
    dpi_out=300,
    dpi_out_savefig=300,
)


# Utility: add figure-level note box
def fig_note(fig, text, y=0.02, fontsize=None):
    fig.text(
        0.01,
        y,
        text,
        ha="left",
        va="bottom",
        fontsize=(fontsize or mpl.rcParams.get("font.size", 7)),
    )


with open(
    Path(EXPORT_PATH_DATA, STUDY, f"optimal_parameters_cc_{CC_VERSION}.json"), "r"
) as fp:
    cc_params = json.load(fp)

# %% [markdown]
# ---
# 4.3.2026
#
# Overwrite complexity predictor with a different standardization (divide unstandardized differences by their SD).
#
# TODO: do not use `_z` in these variable names.

# %%
from scipy.stats import zscore

data["diff_ac_RvL_z"] = data["diff_ac_RvL"] / data["diff_ac_RvL"].std(ddof=0)
data["abs_diff_ac_RvL_z"] = data["diff_ac_RvL_z"].abs() 
data["diff_diaga_RvL_z"] = data["diff_diaga_RvL"] / data["diff_diaga_RvL"].std(ddof=0)
data["abs_diff_diaga_RvL_z"] = data["diff_diaga_RvL_z"].abs() 
data["diff_cc_v2_RvL_opt_z"] = data["diff_cc_v2_RvL_opt"] / data["diff_cc_v2_RvL_opt"].std(ddof=0)
data["abs_diff_cc_v2_RvL_opt_z"] = data["diff_cc_v2_RvL_opt_z"].abs() 
data["diff_disorder_RvL_z"] = data["diff_disorder_RvL"] / data["diff_disorder_RvL"].std(ddof=0)
data["abs_diff_disorder_RvL_z"] = data["diff_disorder_RvL_z"].abs() 
data["max_disorder_z"] = zscore(data["max_disorder"], ddof=0)


# %% [markdown]
# ---

# %% [markdown]
# ## 1. Publication Metrics (JSON)

# %%
def _summary(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return {"mean": None, "sd": None, "min": None, "max": None, "median":None, "q25":None, "q75":None}
    return {
        "mean": float(s.mean()),
        "sd": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(np.median(s)),
        "q25": float(np.quantile(s, 0.25)),
        "q75": float(np.quantile(s, 0.75))
    }


def _proportions_map(d):
    tot = sum(d.values())
    return {k: (v / tot if tot else 0.0) for k, v in d.items()}


def _corr_matrix(df, cols):
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return {"matrix": [], "cols": cols, "max_abs_r": 0.0}
    C = df[cols].corr()
    iu = np.triu_indices(len(cols), 1)
    max_abs = float(np.nanmax(np.abs(C.values[iu])))
    return {"matrix": C.values.tolist(), "cols": cols, "max_abs_r": max_abs}


def compute_json(data: pd.DataFrame, cc_params):
    """Build descriptives from the existing preprocessed trial-level DataFrame `data`.
    Does not recompute metrics exported elsewhere (e.g., coherence proportion, demographics).
    If out_path is provided, write JSON; always return the dict."""
    df = data.copy()

    # Choice proportions
    choice_counts = {}
    if "choice_cat_RvL" in df.columns:
        key_map = {
            "definitely left": "def_left",
            "slightly left": "sl_left",
            "slightly right": "sl_right",
            "definitely right": "def_right",
        }
        vals = df["choice_cat_RvL"].astype(str).map(key_map)
        for k in ["def_left", "sl_left", "sl_right", "def_right"]:
            choice_counts[k] = int((vals == k).sum())

    trials_block = {
        "n_obs": int(len(df)),
        "n_participants": (
            int(df["participant"].nunique()) if "participant" in df.columns else None
        ),
        "choice_counts": choice_counts or None,
        "choice_props": _proportions_map(choice_counts) if choice_counts else None,
        "rt_raw": (
            _summary(df["rt"])
            if "rt" in df.columns
            else {"mean": None, "sd": None, "min": None, "max": None, "median":None, "q25":None, "q75":None}
        ),
        "rt_log": (
            _summary(np.log(df["rt"]))
            if "rt" in df.columns
            else {"mean": None, "sd": None, "min": None, "max": None, "median":None, "q25":None, "q75":None}
        ),
    }

    # Gaze summaries (if available)
    if {"eye_R", "eye_L"}.issubset(df.columns):
        total = df["eye_R"].fillna(0) + df["eye_L"].fillna(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            bias = (df["eye_R"] - df["eye_L"]) / total.replace(0, np.nan)
        trials_block["gaze_bias"] = {
            **_summary(bias),
            "n_no_gaze": int((total == 0).sum()),
            "prop_no_gaze": float((total == 0).mean()),
        }

    # Predictors: signed and absolute diffs; moderators
    preds = {
        "diff": {
            "ac": (
                _summary(df["diff_ac_RvL_z"]) if "diff_ac_RvL_z" in df.columns else None
            ),
            "cc": (
                _summary(df["diff_cc_v2_RvL_opt_z"])
                if "diff_cc_v2_RvL_opt_z" in df.columns
                else None
            ),
            "disorder": (
                _summary(df["diff_disorder_RvL_z"])
                if "diff_disorder_RvL_z" in df.columns
                else None
            ),
            "diaga": (
                _summary(df["diff_diaga_RvL_z"])
                if "diff_diaga_RvL_z" in df.columns
                else None
            ),
        },
        "abs_diff": {
            "ac": (
                _summary(df["abs_diff_ac_RvL_z"])
                if "abs_diff_ac_RvL_z" in df.columns
                else None
            ),
            "cc": (
                _summary(df["abs_diff_cc_v2_RvL_opt_z"])
                if "abs_diff_cc_v2_RvL_opt_z" in df.columns
                else None
            ),
            "disorder": (
                _summary(df["abs_diff_disorder_RvL_z"])
                if "abs_diff_disorder_RvL_z" in df.columns
                else None
            ),
            "diaga": (
                _summary(df["abs_diff_diaga_RvL_z"])
                if "abs_diff_diaga_RvL_z" in df.columns
                else None
            ),
        },
        "moderators": {
            "load_capacity_ratio": (
                _summary(df["load_capacity_ratio"])
                if "load_capacity_ratio" in df.columns
                else None
            ),
            "heuristic_optimality": (
                _summary(df["heuristic_optimality"])
                if "heuristic_optimality" in df.columns
                else None
            ),
            "max_disorder": (
                _summary(df["max_disorder"]) if "max_disorder" in df.columns else None
            ),
        },
    }

    # Correlations among signed diffs (no VIF)
    diff_cols = [
        "diff_ac_RvL_z",
        "diff_cc_v2_RvL_opt_z",
        "diff_disorder_RvL_z",
        "diff_diaga_RvL_z",
    ]
    corr_block = {"diff": _corr_matrix(df, diff_cols)}

    # Participant-level PSI and efficiency (using z if raw not present)
    part = (
        df[["participant"]].drop_duplicates()
        if "participant" in df.columns
        else pd.DataFrame()
    )
    if "psi_z" in df.columns:
        part = part.merge(
            df[["participant", "psi_z"]].drop_duplicates(), on="participant", how="left"
        )
    if "efficiency_z" in df.columns:
        part = part.merge(
            df[["participant", "efficiency_z"]].drop_duplicates(),
            on="participant",
            how="left",
        )
    psi_eff = {
        "psi_total": (
            _summary(part["psi_z"])
            if "psi_z" in part.columns
            else {"mean": None, "sd": None, "min": None, "max": None, "median":None, "q25":None, "q75":None}
        ),
        "efficiency": (
            _summary(part["efficiency_z"])
            if "efficiency_z" in part.columns
            else {"mean": None, "sd": None, "min": None, "max": None, "median":None, "q25":None, "q75":None}
        ),
    }
    if {"psi_z", "efficiency_z"}.issubset(part.columns):
        psi_eff["correlation"] = {
            "r": float(part[["psi_z", "efficiency_z"]].corr().iloc[0, 1])
        }

    empty_dist = {
        "cont_bern": "continuous Bernoulli",
        "normal": "truncated normal",
        "laplace": "truncated Laplace",
    }[cc_params["best_combination"][0]]

    out = {
        "meta": {
            "analysis_name": "descriptives",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        },
        "trials": trials_block,
        "predictors": preds,
        "correlations": corr_block,
        # coherence proportion remains in coherence_{c,e}.json
        "psi_efficiency": psi_eff,
        "cc_params": {
            "geom_p": cc_params["best_params"][0],
            "empty_param": cc_params["best_params"][1],
            "comp_alpha": cc_params["best_params"][2],
            "empty_dist": empty_dist,
            "dirichlet_corr": "with" if cc_params["best_combination"][1] else "without",
        },
    }

    return out


out = compute_json(data, cc_params)

out_path = Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{ANALYSIS_NAME}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {out_path}")

# %%
RELABEL = {
    "diff_ac_RvL_z": "$\Delta$HC",
    f"diff_cc_{CC_VERSION}_RvL_z": "$\Delta$CC$_{0}$",
    f"diff_cc_{CC_VERSION}_RvL_opt_z": "$\Delta$CC",
    "diff_disorder_RvL_z": "$\Delta$VC",
    "diff_diaga_RvL_z": "$\Delta$DD",
    "abs_diff_ac_RvL_z": "|$\Delta$HC|",
    f"abs_diff_cc_{CC_VERSION}_RvL_z": "|$\Delta$CC$_{0}$|",
    f"abs_diff_cc_{CC_VERSION}_RvL_opt_z": "|$\Delta$CC|",
    "abs_diff_disorder_RvL_z": "|$\Delta$VC|",
    "abs_diff_diaga_RvL_z": "|$\Delta$DD|",
    "max_disorder_z":"MD"
}


# %% [markdown]
# ## 2. Scatter matrix (Trial-level predictors): Panel A (signed), Panel B (absolute)

# %%
def scatter_matrix(ax_grid, data_frame, labels, bins=12, zero_tick_only=False):
    """
    Populate a k×k grid with:
      - diagonal: histograms (show axes/ticks)
      - lower: scatter with ticks
      - upper: Pearson r (text) with ticks visible
    """
    k = len(labels)
    # correlation
    R = data_frame[labels].corr().values
    # loop
    for i in range(k):
        for j in range(k):
            ax = ax_grid[i][j]
            ax.set_box_aspect(1.0)
            # make sure axes/ticks are visible
            ax.tick_params(labelsize=mpl.rcParams.get("xtick.labelsize", 7))
            if i == j:
                arr = data_frame[labels[i]].to_numpy()
                _ = hist_panel(
                    ax, arr, bins=bins, color="steelblue",
                    edgecolor="white", grid=True, stats=False
                )
                # Let diagonal axes show ticks (compact)
                ax.grid(True, axis="y", alpha=0.3)
            elif i > j:
                ax.scatter(
                    data_frame[labels[j]], data_frame[labels[i]],
                    s=6, color="#1f77b4", alpha=0.33
                )
                # keep ticks (compact)
                ax.grid(False)
            else:
                r = R[i, j]
                ax.text(0.5, 0.5, f"$r={r:.2f}$",
                        ha="center", va="center", transform=ax.transAxes, fontsize=7)
                # keep ticks for consistency
                ax.grid(False)
                ax.set_axis_off()
            if zero_tick_only and i != j:
                ax.set_xticks([0])
                ax.set_yticks([0])
    
            # Clean spines lightly
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

def _labels_for_axes(axs, col_labels):
    k = len(col_labels)
    # bottom row xlabels; left col ylabels
    for j in range(k):
        axs[k-1][j].set_xlabel(col_labels[j])
    for i in range(k):
        axs[i][0].set_ylabel(col_labels[i])

# -------------------
# Plot signed scatter matrix
# -------------------
def plot_scatter_matrix_signed(df_in, cols, label_map, out_base: Path,
                               width_mm=140, panel_aspect=1.0, zero_tick_only=False):
    df_s = df_in[cols].dropna()
    N = len(df_s)
    # Relabel for display
    disp_labels = [label_map[c] for c in cols]

    # Grid
    k = len(cols)
    fig, axs = make_grid(
        width_mm=width_mm, panels=(k, k), panel_aspect=panel_aspect,
        margins=(0.06, 0.08, 0.995, 0.94), gutter=(0.0, 0.0),
        constrained=True, flatten=False
    )

    scatter_matrix(axs, df_s.rename(columns=label_map), disp_labels, bins=9, zero_tick_only=zero_tick_only)
    _labels_for_axes(axs, disp_labels)

    # Figure-level title/N
    fig.text(0.5, 1.01, f"$N={N}$", ha="center", va="top")

    save(fig, out_base, formats=("pdf", "png"), dpi_png=600, tight=True)
    plt.show()
    plt.close(fig)

# -------------------
# Plot absolute scatter matrix
# -------------------
def plot_scatter_matrix_abs(df_in, cols, label_map, out_base: Path,
                            width_mm=140, panel_aspect=1.0, zero_tick_only=False):
    df_a = df_in[cols].dropna()
    N = len(df_a)
    disp_labels = [label_map[c] for c in cols]

    k = len(cols)
    fig, axs = make_grid(
        width_mm=width_mm, panels=(k, k), panel_aspect=panel_aspect,
        margins=(0.06, 0.08, 0.995, 0.94), gutter=(0.0, 0.0),
        constrained=True, flatten=False
    )

    scatter_matrix(axs, df_a.rename(columns=label_map), disp_labels, bins=12, zero_tick_only=zero_tick_only)
    _labels_for_axes(axs, disp_labels)

    fig.text(0.5, 1.01, f"$N={N}$", ha="center", va="top")

    save(fig, out_base, formats=("pdf", "png"), dpi_png=600, tight=True)
    plt.show()
    plt.close(fig)

# -------------------
# Run
# -------------------
plot_scatter_matrix_signed(
    data, diff_signed, RELABEL, Path(OUT_DIR, "fig_scatter_matrix_signed"), width_mm=180, zero_tick_only=False
)
plot_scatter_matrix_abs(
    data, diff_abs, RELABEL, Path(OUT_DIR, "fig_scatter_matrix_abs"), width_mm=180, zero_tick_only=False
)


# %% [markdown]
# ## 3. Participant-level distributions + correlation (PSI, Efficiency)

# %%
def plot_participant_corr(df, psi_col, eff_col, out_base: Path, width_mm=183):
    if psi_col.split("_")[-1] == "z":
        psi_label = "PSI (z)"
    else:
        psi_label = "PSI"
    if eff_col.split("_")[-1] == "z":
        eff_label = "Efficiency (z)"
    else:
        eff_label = "Efficiency"
    # Unique per participant
    part = (
        df[["participant", psi_col, eff_col]]
        .dropna()
        .drop_duplicates(subset=["participant"])
    )
    Np = len(part)
    x = part[psi_col].to_numpy()
    y = part[eff_col].to_numpy()
    r = (
        float(pd.DataFrame({psi_col: x, eff_col: y}).corr().iloc[0, 1])
        if Np > 1
        else np.nan
    )

    fig, axs = make_grid(
        width_mm=width_mm,
        panels=(1, 3),
        panel_aspect=0.7,
        sharex=False,
        sharey=False,
        margins=(0.06, 0.08, 0.995, 0.93),
        gutter=(0.22, 0.18),
        constrained=False,
        flatten=True,
    )

    # Hist PSI
    h1 = hist_panel(
        axs[0],
        x,
        bins=12,
        color="steelblue",
        edgecolor="white",
        stats=True,
        stats_mode="box",
        stats_fmt="$N={N:d}$, $M={mean:.2f}$, $SD={sd:.2f}$",
    )
    axs[0].set_xlabel(psi_label)
    axs[0].set_ylabel("Count")

    # Hist Efficiency
    h2 = hist_panel(
        axs[1],
        y,
        bins=12,
        color="steelblue",
        edgecolor="white",
        stats=True,
        stats_mode="box",
        stats_fmt="$N={N:d}$, $M={mean:.2f}$, $SD={sd:.2f}$",
    )
    axs[1].set_xlabel(eff_label)
    axs[1].set_ylabel("Count")
    axs[1].tick_params(labelleft=False)  # cleaner middle panel

    # Scatter with regression
    ax = axs[2]
    ax.scatter(x, y, s=10, color="#1f77b4", alpha=0.6)
    if Np >= 2 and np.isfinite(r):
        # Least-squares line
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, m * xx + b, color="black", linewidth=1.0)
    ax.set_xlabel(psi_label)
    ax.set_ylabel(eff_label)
    # Stats box
    ax.text(
        0.98,
        0.02,
        f"$N={Np},\\ r={r:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=mpl.rcParams.get("font.size", 7),
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"
        ),
    )

    save(fig, out_base, formats=("pdf", "png"), dpi_png=600, tight=True)
    plt.show()
    plt.close(fig)


plot_participant_corr(data, col_psi, col_eff, Path(OUT_DIR, "fig_corr_participant"), width_mm=180)


# %% [markdown]
# ## 4. Problem-level distributions + correlation (difficulty, heuristic optimality)

# %%
def plot_problem_corr(df, diff_col, ho_col, out_base: Path, width_mm=183):

    Nt = len(df)
    x = df[diff_col].to_numpy()
    y = df[ho_col].to_numpy()
    r = (
        float(pd.DataFrame({diff_col: x, ho_col: y}).corr().iloc[0, 1])
        if Nt > 1
        else np.nan
    )

    fig, axs = make_grid(
        width_mm=width_mm,
        panels=(1, 3),
        panel_aspect=0.7,
        sharex=False,
        sharey=False,
        margins=(0.06, 0.08, 0.995, 0.93),
        gutter=(0.3, 0.18),
        constrained=False,
        flatten=True,
    )

    # Hist Difficulty
    h1 = hist_panel(
        axs[0],
        x,
        bins=12,
        color="steelblue",
        edgecolor="white",
        stats=True,
        stats_mode="box",
        stats_fmt="$N={N:d}$, $M={mean:.2f}$, $SD={sd:.2f}$",
    )
    axs[0].set_xlabel("Difficulty (Load–Capacity Ratio)")
    axs[0].set_ylabel("Count")

    # Hist Heuristic Optimality
    h2 = hist_panel(
        axs[1],
        y,
        bins=12,
        color="steelblue",
        edgecolor="white",
        stats=True,
        stats_mode="box",
        stats_fmt="$N={N:d}$, $M={mean:.2f}$, $SD={sd:.2f}$",
    )
    axs[1].set_xlabel("Heuristic Optimality")
    axs[1].set_ylabel("Count")
    axs[1].tick_params(labelleft=False)

    # Scatter with regression
    ax = axs[2]
    ax.scatter(x, y, s=10, color="#1f77b4", alpha=0.33)
    if Nt >= 2 and np.isfinite(r):
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xx, m * xx + b, color="black", linewidth=1.0)
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Heuristic Optimality")
    ax.text(
        0.98,
        0.02,
        f"$N={Nt},\\ r={r:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=mpl.rcParams.get("font.size", 7),
        bbox=dict(
            boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"
        ),
    )

    save(fig, out_base, formats=("pdf", "png"), dpi_png=600, tight=True)
    plt.show()
    plt.close(fig)


plot_problem_corr(data, col_diff, col_ho, Path(OUT_DIR, "fig_corr_problem"), width_mm=180)

# %% [markdown]
# ## 5. Dependent Variables 

# %%
from pub_utils.figures import bar_panel, bins_fd


# %%
def plot_dv(df, choice_col, rt_col, gaze_col, out_base: Path, width_mm=180):
    # Extract arrays
    # Choice: keep as pandas Series to retain category order if present
    choice = df[choice_col]
    rt = pd.to_numeric(df[rt_col], errors="coerce").to_numpy()
    gaze = pd.to_numeric(df[gaze_col], errors="coerce").to_numpy()

    fig, axs = make_grid(
        width_mm=width_mm,
        panels=(1, 3),
        panel_aspect=0.7,
        sharex=False,
        sharey=False,
        margins=(0.06, 0.08, 0.995, 0.93),
        gutter=(0.30, 0.18),
        constrained=False,
        flatten=True,
    )

    # Panel 1: Choice (categorical bar chart)
    if pd.api.types.is_categorical_dtype(choice):
        labels = list(choice.cat.categories)
        counts = choice.value_counts().reindex(choice.cat.categories, fill_value=0).to_numpy()
    else:
        # Fallback: treat as ordered unique values
        labels = list(pd.Index(sorted(choice.dropna().unique())))
        counts = choice.value_counts().reindex(labels, fill_value=0).to_numpy()
    x = np.arange(len(labels))
    bar_panel(
        axs[0],
        x=x,
        heights=counts,
        labels=labels,
        colors="steelblue",
        ylabel="Count",
        xlabel="Choice",
        annotate={"text": f"$N={int(np.sum(counts))}$", "loc": "upper right", "fontsize": 8},
    )
    axs[0].set_xticks(axs[0].get_xticks(), axs[0].get_xticklabels(), rotation=45)


    # Panel 2: log RT (continuous histogram)
    # Choose bins via Freedman–Diaconis for sensible bin count
    rt_clean = rt[np.isfinite(rt)]
    nb_rt = max(6, min(40, bins_fd(rt_clean, max_bins=40))) if rt_clean.size else 10
    _ = hist_panel(
        axs[1],
        rt_clean,
        # bins=nb_rt,
        bins=12,
        color="steelblue",
        edgecolor="white",
        stats=True,
        stats_mode="box",
        stats_fmt="$N={N:d}$, $M={mean:.2f}$, $SD={sd:.2f}$",
    )
    axs[1].set_xlabel("log(RT in ms)")
    axs[1].set_ylabel("Count")
    axs[1].tick_params(labelleft=False)

    # Panel 3: Gaze bias (continuous histogram)
    gaze_clean = gaze[np.isfinite(gaze)]
    nb_gz = max(6, min(40, bins_fd(gaze_clean, max_bins=40))) if gaze_clean.size else 10
    _ = hist_panel(
        axs[2],
        gaze_clean,
        # bins=nb_gz,
        bins=12,
        color="steelblue",
        edgecolor="white",
        stats=True,
        stats_mode="box",
        stats_fmt="$N={N:d}$, $M={mean:.2f}$, $SD={sd:.2f}$",
    )
    axs[2].set_xlabel("Gaze Bias b = (R - L) / (R + L)")
    axs[2].set_ylabel("Count")
    axs[2].tick_params(labelleft=False)

    save(fig, out_base, formats=("pdf", "png"), dpi_png=600, tight=True)
    plt.show()
    plt.close(fig)

# Correct argument order: (choice, rt, gaze)
plot_dv(data, "choice_cat_RvL", "log_rt", "gaze_bias", Path(OUT_DIR, "fig_dv"), width_mm=180)


# %%
