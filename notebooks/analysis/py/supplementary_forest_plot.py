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
# ## Extended Data Fig. S1 — Cross-study forest plot for choice effects
# - Loads JSON from project, computes ORs and 95% CIs, and saves PDF/PNG.
#

# %% [markdown]
# TODO: tidy up (only 1 import block), pick better 2 colors, use figures.py

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joptim.utils.export_path import get_export_paths

offsets = {"Exploratory": -0.15, "Confirmatory": +0.15}
colors = {"Exploratory": "#a0a0a0", "Confirmatory": "steelblue"}

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.right": False,
    "axes.spines.top": False
})

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

paths = {
    "expl_choice": Path(EXPORT_PATH_PUB, "exploratory", "mlm_choice", "mlm_choice.json"),
    "conf_choice": Path(EXPORT_PATH_PUB, "confirmatory", "mlm_choice", "mlm_choice.json"),
}

J_ex = load_json(paths["expl_choice"])
J_co = load_json(paths["conf_choice"])

def extract_choice(df_json, study_label):
    coefs = df_json["coefs"]
    # Map our four metrics to JSON keys (note CC key differs across studies)
    keys = {
        "HC": "diff_ac_RvL_z",
        "CC": "diff_cc_v2_RvL_z" if "diff_cc_v2_RvL_z" in coefs else "diff_cc_v2_RvL_opt_z",
        "VC": "diff_disorder_RvL_z",
        "DD": "diff_diaga_RvL_z",
    }
    rows = []
    for label, key in keys.items():
        c = coefs[key]
        beta = float(c["estimate"])
        lo = float(c["conf_low"])
        hi = float(c["conf_high"])
        OR = np.exp(beta)
        OR_lo = np.exp(lo)
        OR_hi = np.exp(hi)
        rows.append({
            "metric": label, "study": study_label,
            "beta": beta, "OR": OR, "OR_lo": OR_lo, "OR_hi": OR_hi
        })
    return pd.DataFrame(rows)

df = pd.concat([
    extract_choice(J_ex, "Exploratory"),
    extract_choice(J_co, "Confirmatory")
], ignore_index=True)

# Plot
fig, ax = plt.subplots(figsize=(7.2, 4.6))
metrics = ["HC", "CC", "VC", "DD"]

ypos = np.arange(len(metrics))


for i, m in enumerate(metrics):
    for study in ["Exploratory", "Confirmatory"]:
        r = df[(df.metric == m) & (df.study == study)].iloc[0]
        x, xlo, xhi = r["OR"], r["OR_lo"], r["OR_hi"]
        ax.errorbar(
            x, i + offsets[study],
            xerr=[[x - xlo], [xhi - x]],
            fmt="o", color=colors[study], capsize=5, lw=1.75, ms=6
        )

# Reference line
ax.axvline(1.0, color="#2c2c2c", lw=1, ls="--", zorder=0)

# ax.set_yticks(ypos)
# ax.set_yticklabels(metrics)

metrics = [r"$\Delta$HC", r"$\Delta$CC", r"$\Delta$VC", r"$\Delta$DD"]
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics)

ax.set_xlabel("Odds ratio (per 1-SD increase in complexity difference)")
ax.set_title("Choice effects across studies")


# Legend
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[s], label=s)
    for s in ["Exploratory", "Confirmatory"]
]
ax.legend(handles, ["Exploratory", "Confirmatory"], frameon=False, loc="lower right")

plt.tight_layout()

outdir = Path(EXPORT_PATH_PUB, "extended")
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(outdir / f"fig_extended_forest_choice.{ext}", dpi=300)
print(outdir / "fig_extended_forest_choice.pdf")

# %% [markdown]
# ## Extended Data Fig. S2 — Cross-study forest plot for RT effects
# - Displays percent change in RT per 1-SD |Δ| in HC/CC/VC/DD (95% CI).
# - Saves: fig_extended_forest_rt.{pdf,png}

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.right": False,
    "axes.spines.top": False
})

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

paths = {
    "expl_rt": Path(EXPORT_PATH_PUB, "exploratory", "mlm_rt", "mlm_rt.json"),
    "conf_rt": Path(EXPORT_PATH_PUB, "confirmatory", "mlm_rt", "mlm_rt.json"),
}

J_ex = load_json(paths["expl_rt"])
J_co = load_json(paths["conf_rt"])

def extract_rt(df_json, study_label):
    coefs = df_json["coefs"]
    # Map the four absolute-difference predictors (note CC key name differs)
    keys = {
        "HC": "abs_diff_ac_RvL_z",
        "CC": "abs_diff_cc_v2_RvL_z" if "abs_diff_cc_v2_RvL_z" in coefs else "abs_diff_cc_v2_RvL_opt_z",
        "VC": "abs_diff_disorder_RvL_z",
        "DD": "abs_diff_diaga_RvL_z",
    }
    rows = []
    for label, key in keys.items():
        c = coefs.get(key)
        if c is None:
            continue
        beta = float(c["estimate"])
        lo = float(c["conf_low"])
        hi = float(c["conf_high"])
        # Percent change from log coefficient
        pc   = 100.0*(np.exp(beta) - 1.0)
        pc_lo= 100.0*(np.exp(lo)    - 1.0)
        pc_hi= 100.0*(np.exp(hi)    - 1.0)
        rows.append({
            "metric": label, "study": study_label,
            "pct": pc, "pct_lo": pc_lo, "pct_hi": pc_hi,
            "beta": beta
        })
    return pd.DataFrame(rows)

df = pd.concat([
    extract_rt(J_ex, "Exploratory"),
    extract_rt(J_co, "Confirmatory")
], ignore_index=True)

# Ensure all four metrics present; if not, fill missing rows as NaN (will be skipped)
metrics = ["HC", "CC", "VC", "DD"]
studies = ["Exploratory", "Confirmatory"]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
ypos = np.arange(len(metrics))

for i, m in enumerate(metrics):
    for study in studies:
        sel = df[(df.metric == m) & (df.study == study)]
        if sel.empty:
            continue
        r = sel.iloc[0]
        x, xlo, xhi = r["pct"], r["pct_lo"], r["pct_hi"]
        ax.errorbar(
            x, i + offsets[study],
            xerr=[[x - xlo], [xhi - x]],
            fmt="o", color=colors[study], capsize=5, lw=1.75, ms=6
        )

# Reference line at 0% change
ax.axvline(0.0, color="#2c2c2c", lw=1, ls="--", zorder=0)

# ax.set_yticks(ypos)
# ax.set_yticklabels(metrics)

metrics = [r"|$\Delta$HC|", r"|$\Delta$CC|", r"|$\Delta$VC|", r"|$\Delta$DD|"]
ax.set_yticks(np.arange(len(metrics)))
ax.set_yticklabels(metrics)


ax.set_xlabel("Percent change in RT (per 1-SD |Δ|)")
ax.set_title("RT effects across studies")

# Data-driven x-limits with margin
finite = df[["pct_lo","pct_hi"]].values.flatten()
finite = finite[np.isfinite(finite)]

# Legend
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[s], label=s)
    for s in studies
]
ax.legend(handles, studies, frameon=False, loc="lower right")

plt.tight_layout()

outdir = Path(EXPORT_PATH_PUB, "extended")
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(outdir / f"fig_extended_forest_rt.{ext}", dpi=300)
print(outdir / "fig_extended_forest_rt.pdf")

# %% [markdown]
# ## Extended Data Fig. S3 — Cross-study forest plot for gaze model.
# - If complexity predictors exist, plots their ORs; otherwise plots the intercept OR.
# - Saves: fig_extended_forest_gaze.{pdf,png}

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.right": False,
    "axes.spines.top": False
})

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

paths = {
    "expl_gaze": Path(EXPORT_PATH_PUB, "exploratory", "mlm_gaze", "mlm_gaze.json"),
    "conf_gaze": Path(EXPORT_PATH_PUB, "confirmatory", "mlm_gaze", "mlm_gaze.json"),
}

J_ex = load_json(paths["expl_gaze"])
J_co = load_json(paths["conf_gaze"])

def extract_gaze(df_json, study_label):
    coefs = df_json["coefs"]
    # Try complexity predictors first
    keys_candidates = [
        ("HC", "diff_ac_RvL_z"),
        ("CC", "diff_cc_v2_RvL_z"),
        ("CC", "diff_cc_v2_RvL_opt_z"),
        ("VC", "diff_disorder_RvL_z"),
        ("DD", "diff_diaga_RvL_z"),
    ]
    rows = []
    for label, key in keys_candidates:
        if key in coefs:
            c = coefs[key]
            beta = float(c["estimate"])
            lo = float(c["conf_low"])
            hi = float(c["conf_high"])
            OR   = np.exp(beta)
            OR_lo= np.exp(lo)
            OR_hi= np.exp(hi)
            rows.append({
                "metric": label, "study": study_label,
                "OR": OR, "OR_lo": OR_lo, "OR_hi": OR_hi,
                "beta": beta, "type": "predictor"
            })
    # If none found, fall back to intercept (right-gaze odds)
    if len(rows) == 0 and "(Intercept)" in coefs:
        c = coefs["(Intercept)"]
        beta = float(c["estimate"])
        lo = float(c["conf_low"])
        hi = float(c["conf_high"])
        OR   = np.exp(beta)
        OR_lo= np.exp(lo)
        OR_hi= np.exp(hi)
        rows.append({
            "metric": "Intercept (right-gaze odds)",
            "study": study_label,
            "OR": OR, "OR_lo": OR_lo, "OR_hi": OR_hi,
            "beta": beta, "type": "intercept"
        })
    return pd.DataFrame(rows)

df = pd.concat([
    extract_gaze(J_ex, "Exploratory"),
    extract_gaze(J_co, "Confirmatory")
], ignore_index=True)

# If both studies have only intercepts, plot a single line item; else plot by metrics
metrics = list(pd.unique(df["metric"]))
metrics_sorted = metrics  # keep input order
ypos = np.arange(len(metrics_sorted))
studies = ["Exploratory", "Confirmatory"]

fig, ax = plt.subplots(figsize=(7.2, 4.0))

for i, m in enumerate(metrics_sorted):
    for study in studies:
        sel = df[(df.metric == m) & (df.study == study)]
        if sel.empty:
            continue
        r = sel.iloc[0]
        x, xlo, xhi = r["OR"], r["OR_lo"], r["OR_hi"]
        ax.errorbar(
            x, i + offsets[study],
            xerr=[[x - xlo], [xhi - x]],
            fmt="o", color=colors[study], capsize=5, lw=1.75, ms=6
        )

# Reference line at OR = 1
ax.axvline(1.0, color="#2c2c2c", lw=1, ls="--", zorder=0)

ax.set_yticks(ypos)
ax.set_yticklabels(metrics_sorted)
ax.set_xlabel("Odds ratio")
title = "Gaze model across studies"
ax.set_title(title)

# Data-driven x-limits
finite = df[["OR_lo","OR_hi"]].values.flatten()
finite = finite[np.isfinite(finite)]

# Legend
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[s], label=s)
    for s in studies
]
ax.legend(handles, studies, frameon=False, loc="lower right")

# Annotation if only intercepts
if all(df["type"] == "intercept"):
    ax.text(
        0.5, -0.25,
        "Note: AIC selection retained intercept-only models; no complexity predictors.",
        transform=ax.transAxes, ha="left", va="top", fontsize=9
    )

plt.tight_layout()

outdir = Path(EXPORT_PATH_PUB, "extended")
outdir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(outdir / f"fig_extended_forest_gaze.{ext}", dpi=300)
print(outdir / "fig_extended_forest_gaze.pdf")
