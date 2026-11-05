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
# Multilevel Analysis
# ===
#
# From Appendix C
#
# C.1 Choice (Ordinal Outcome)
#
# - Outcome: choice (right vs left; clmm).
# - Main effects: signed right–left differences in heuristic-related complexity; compositional complexity; visual-order complexity; and diagonal dissimilarity.
# - Moderators: PSI, efficiency, difficulty, maximum disorder (not differenced).
# - Constraint: Do not include maximum disorder with visual-order complexity (here: its signed difference).
# - Interactions: Only complexity-difference × moderator; no diagonal × moderator.
# - Candidate set: All theoretically meaningful subsets of the above, respecting the rules above.
#

# %% [markdown]
# ## Imports and Data
#
# TODO: simplify below procedure:
#
# Make sure that `data_preprocessed_<TRIAL_FILE_ID>.csv` exists. If not, create it running first `preprocess_experiment_data.ipynb` (and then `mixture_model_fit_pilot`). If you don't need the values for the optimized composition metrics, you can simply use `data_preprocessed.csv` and remove those variables from the model. The reason to run `mixture_model_fit_pilot.ipynb` is to derive updated optimized composition metric values from the exploratory data. The reason to use `data_preprocessed_updated.csv` is if `STUDY=exploratory_with_optimized_cc`. In the case of `STUDY=confirmatory` the composition metric in `data_preprocessed_updated.csv` is already the optimized one.

# %% [markdown]
# **required `rpy2` package and R-package `ordinal`**
#
# In terminal:
#     
# `pip install rpy2`
#
# in R:
#
# `install.packages('ordinal')`

# %%
import os
import re
from pathlib import Path
import pickle
import json
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from joptim.utils.export_path import get_export_paths
from joptim.model_formulas import create_model_formulas
from joptim.model_selection import best_model
from joptim.publication_utils import translate_variables, translate_effects
from pub_utils.tables import TableSpec, write_table
from joptim.pretty_labels import map_labels, map_labels_long, RANDOM_EFFECTS_LABELS
import rpy2
from rpy2.robjects import Formula, r, pandas2ri
from rpy2.robjects.packages import importr

# Import R package
ordinal = importr("ordinal")
ggeffects = importr("ggeffects")

ANALYSIS_NAME = "mlm_choice"

# "exploratory" or "confirmatory"
STUDY = "confirmatory"
RECOMPUTE_MODEL_COMPARISON = False
RECOMPUTE_BEST_MODEL = False
RECOMPUTE_PRED_PROBS = False

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
CC_VERSION = "v2"  # "v1" or "v2"
R_FIT_FILENAME = "clmm_fit_r_object.rds"
R2_FILENAME = "clmm_fit_R2.json"
MODEL_FORMULA_FILENAME = "formula_best_model.txt"

os.makedirs(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME), exist_ok=True)
os.makedirs(Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME), exist_ok=True)

with open(Path(EXPORT_PATH_DATA, STUDY, "trials.pkl"), "rb") as pickle_file:
    trials = pickle.load(pickle_file)

TRIAL_FILE_ID = trials["file_id"]

if STUDY == "exploratory":
    OPT = ""
    UPDATED = "updated_" # we could also use "" if we do not need columns for optimized CC
elif STUDY in ["confirmatory"]:
    OPT = "_opt"
    UPDATED = ""

else:
    raise ValueError(
        f"STUDY can only be 'exploratory', 'confirmatory' got: {STUDY}"
    )

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
        f"""data_preprocessed_{UPDATED}{str(TRIAL_FILE_ID).replace(".", "-")}.csv""",
    ),
    sep=",",
    dtype={"trial_file_id": float},
)

# filter to TRIAL_FILE_ID (loading from above file should be enough,
# but we want to be on the safe side)
data = data.loc[data["trial_file_id"] == TRIAL_FILE_ID].reset_index(drop=True)
data = data.drop(["trial_file_id", "trial_file_user_id"], axis=1)

threshold_method = "symmetric"

print(len(data.participant.unique()), "participants loaded.")
print(f"""{len(data)} trials loaded ({STUDY} study / trial file: {TRIAL_FILE_ID}).""")

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
data["diff_diaga_RvL_z"] = data["diff_diaga_RvL"] / data["diff_diaga_RvL"].std(ddof=0)
data["diff_cc_v2_RvL_opt_z"] = data["diff_cc_v2_RvL_opt"] / data["diff_cc_v2_RvL_opt"].std(ddof=0)
data["diff_disorder_RvL_z"] = data["diff_disorder_RvL"] / data["diff_disorder_RvL"].std(ddof=0)
data["max_disorder_z"] = zscore(data["max_disorder"], ddof=0)

# %% [markdown]
# ---

# %% [markdown]
# Convert to be used with R

# %%
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

data["handedness"] = pd.Categorical(
    data["handedness"],
    # categories=['-2', '-1', '1', '2'],
    categories=["Ambidextrous", "Left", "Right"],
    ordered=False,
)

# data_r = pandas2ri.py2rpy(data)


def r2pandas(rdf):
    with (rpy2.robjects.default_converter + pandas2ri.converter).context():
        return rpy2.robjects.conversion.get_conversion().rpy2py(rdf)


def pandas2r(df):
    with (rpy2.robjects.default_converter + pandas2ri.converter).context():
        return rpy2.robjects.conversion.get_conversion().py2rpy(df)


data_r = pandas2r(data)

# send it to r-environment
r.assign("data_r", data_r)


# %% [markdown]
# ## 1. Model Selection

# %% [markdown]
# Skip to 2, if not recomputed (`RECOMPUTE_MODEL_COMPARISON=False`)

# %% [markdown]
# ### 1.1. Model Comparison

# %% [markdown]
# Consider allowing below interactions
#
# - `diff_ac:diff_disorder`
#
# - `diff_ac:diff_disorder + diff_diaga:diff_disorder`
#
#  
#

# %%
def get_model_formulas():

    outcome = "choice_cat_RvL"

    # groups are always added together, separate groups by making use of the inner lists

    predictor_groups = [
        [
            "diff_ac_RvL_z",
            diff_cc_variable_z,
            "diff_disorder_RvL_z",
            "diff_diaga_RvL_z",
        ]
    ]

    moderators = [
        "psi_z",
        "efficiency_z",
        "load_capacity_ratio_z",
        "max_disorder_z",
        "heuristic_optimality_z",
    ]

    no_interactions_with = ["handedness", "diff_diaga_RvL_z"]
    no_random_effects_for = ["handedness"]
    no_random_effects_for_interaction_terms = True
    group = "participant"
    further_exclusions = [
        "diff_disorder_RvL_z:max_disorder_z",
    ]

    model_formulas = create_model_formulas(
        outcome,
        predictor_groups,
        group,
        moderators,
        no_interactions_with,
        no_random_effects_for,
        no_random_effects_for_interaction_terms,
        further_exclusions,
    )
    len(model_formulas)

    cannot_be_in_same_model = ["diff_disorder_RvL_z", "max_disorder_z"]

    model_formulas_filtered = []

    for formula in model_formulas:
        if (cannot_be_in_same_model[0] in formula) and (
            cannot_be_in_same_model[1] in formula
        ):
            ...
        else:
            model_formulas_filtered.append(formula)

    model_formulas = model_formulas_filtered
    return model_formulas


def model_comparison(data, model_formulas, threshold_method):

    model_results = []

    # Fit models and collect AIC/BIC
    for i, formula in enumerate(model_formulas):
        print(f"Fitting {i+1}/{len(model_formulas)}", end=" - ")
        try:
            clmm_contr = ordinal.clmm_control(
                method="nlminb", maxIter=50
            )  # "nlminb", "ucminf", "model.frame"
            clmm_fit = ordinal.clmm(
                formula,
                data=data,
                Hess=True,
                model=True,
                link="logit",
                threshold=threshold_method,
            )
            bic = r.BIC(clmm_fit)[0]
            aic = r.AIC(clmm_fit)[0]

            llf = r.logLik(clmm_fit)[0]
            try:
                converged = ", ".join(clmm_fit[5][5])
            except TypeError as e:
                converged = ", ".join(clmm_fit[2][5])

            error = np.nan

        except Exception as e:
            aic = np.nan
            bic = np.nan
            llf = np.nan
            converged = np.nan
            error = e

        model_results.append(
            {
                "Model": formula,
                "AIC": aic,
                "BIC": bic,
                "Log-Likelihood": llf,
                "Converged": converged,
                "Error": error,
            }
        )
        print("converged:", converged, "- error:", error)

    df = pd.DataFrame(model_results)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


if RECOMPUTE_MODEL_COMPARISON:
    model_formulas = get_model_formulas()
    print("number of formulas after further exclusion: ", len(model_formulas))

# %%
if RECOMPUTE_MODEL_COMPARISON:
    df_model_comparison = model_comparison(data_r, model_formulas, threshold_method)

    df_model_comparison.to_csv(
        Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, "model_comparison.csv"),
        index=False,
    )
    print("Completed model comparison and saved to disk.")
else:
    df_model_comparison = pd.read_csv(
        Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, "model_comparison.csv")
    )
    print("Loaded existing model comparison from disk.")

# %% [markdown]
# Non-Converging Models

# %%
df_model_comparison.loc[df_model_comparison["Converged"] != "relative convergence (4)"]

# %% [markdown]
# AIC-based top Models (among the converged ones)

# %%
df_model_comparison.loc[
    df_model_comparison["Converged"] == "relative convergence (4)"
].sort_values("AIC").head(15).reset_index(drop=True)

# %% [markdown]
# ## 2. Final Model

# %% [markdown]
# Load model comparison

# %%
df_model_comparison = pd.read_csv(
    Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, "model_comparison.csv")
)

# filter to converged only
df_model_comparison_converged = df_model_comparison.loc[
    df_model_comparison["Converged"] == "relative convergence (4)"
]

# %% [markdown]
# ### 2.1. Fit
#
# Skip if `RECOMPUTE_BEST_MODEL` is set to `False`

# %%
if RECOMPUTE_BEST_MODEL:
    print("Fitting model...")
    r("ctrl <- clmm.control(method='nlminb', maxIter=50)")
    # method = "nlminb", "ucminf", "model.frame"

    formula_r = best_model(
        df_model_comparison_converged, model_col="Model", criterion="AIC"
    )

    clmm_fit = r(
        f"""fit <- clmm({formula_r},
        data=data_r,
        Hess=TRUE,
        model=TRUE,
        link='logit',
        threshold='{threshold_method}',
        control=ctrl
        )"""
    )

    r(
        f"""saveRDS(fit, file="{Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R_FIT_FILENAME).as_posix()}")"""
    )
    print(f"R results saved to {R_FIT_FILENAME}")

    pR2_results = {
        "description": "Nakagawa (pseudo) R2 values for final fit.",
        "r2_marg": float(r("performance::r2_nakagawa(fit)$R2_marginal")[0]),
        "r2_cond": float(r("performance::r2_nakagawa(fit)$R2_conditional")[0]),
    }

    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R2_FILENAME), "w") as fp:
        json.dump(pR2_results, fp)

    print(f"Pseudo-R2 results saved to {R2_FILENAME}")

    with open(
        Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, MODEL_FORMULA_FILENAME), "w"
    ) as fp:
        fp.write(formula_r)


else:
    print(f"Loading R results from {R_FIT_FILENAME}...")
    r(
        f"""fit <- readRDS(file="{Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R_FIT_FILENAME).as_posix()}")"""
    )
    clmm_fit = r["fit"]
    print(f"Loading Pseudo-R2 results from {R2_FILENAME}...")

    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R2_FILENAME), "r") as fp:
        pR2_results = json.load(fp)

    with open(
        Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, MODEL_FORMULA_FILENAME), "r"
    ) as fp:
        formula_r = fp.readline()

print(f"""\nFormula used to fit the model:\n{formula_r}\n""")
print(r("""summary(fit)"""))
print("Nakagawa conditional R2:", pR2_results["r2_cond"])
print("Nakagawa marginal R2:", pR2_results["r2_marg"])

# %% [markdown]
#
# ### 2.2. Table for Publication

# %%
REMOVE_THRESHOLDS = False

# 1) Fixed effects with CIs from R, drop thresholds, apply pretty labels
tidy_fixed_r = r(
    'broom.mixed::tidy(fit, effects="fixed", conf.int=TRUE, conf.level=.95)'
)
fixed = r2pandas(tidy_fixed_r)
fixed.columns = [c.replace(".", "_") for c in fixed.columns]

# Remove clmm threshold rows
if REMOVE_THRESHOLDS:
    fixed = fixed[~fixed["term"].isin(["central"])]
    fixed = fixed[~fixed["term"].astype(str).str.startswith("spacing")].reset_index(
        drop=True
    )
    threshold_note = "Threshold parameters are omitted. "
else:
    threshold_note = ""


def _pretty_term(term: str) -> str:
    parts = term.split(":")
    return " × ".join(map_labels(p) for p in parts)


fixed["term"] = fixed["term"].astype(str).map(_pretty_term)


# p-values to match original ($<\,0.001$)
def fmt_p(p):
    try:
        p = float(p)
        return "$<\\,0.001$" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return str(p)


fixed["p_value"] = fixed["p_value"].map(fmt_p)

# 2) Random effects SDs using VarCorr for consistency
is_clmm = bool(r('inherits(fit, "clmm")')[0])
if is_clmm:
    sds_df_r = r("""
    vc <- ordinal::VarCorr(fit)
    # Extract just the SDs from the 'participant' group
    # VarCorr returns a list of matrices with 'stddev' attributes
    sds <- attr(vc$participant, "stddev")
    data.frame(term = names(sds), estimate = as.numeric(sds))
    """)
    random = r2pandas(sds_df_r)
    # Map your labels if necessary, though VarCorr names usually match formula names
    if len(random) > 0:
        random["term"] = RANDOM_EFFECTS_LABELS[: len(random)]

else:
    random = pd.DataFrame(columns=["term", "estimate"])

# 3) Build sections, indent rows with \hspace{1em}
stat_lab = "z"
fixed_rows = pd.DataFrame(
    {
        "Term": "\\hspace{1em} " + fixed["term"].astype(str),
        "Estimate": fixed["estimate"],
        "SE": fixed["std_error"],
        stat_lab: fixed["statistic"],
        "p": fixed["p_value"],
        "CI low": fixed["conf_low"],
        "CI high": fixed["conf_high"],
    }
)

random_rows = pd.DataFrame(
    {
        "Term": "\\hspace{1em} " + random["term"].astype(str),
        "Estimate": random["estimate"],
        "SE": [""] * len(random),
        stat_lab: [""] * len(random),
        "p": [""] * len(random),
        "CI low": [""] * len(random),
        "CI high": [""] * len(random),
    }
)

# Section marker rows (single multicolumn cell; others left blank for now)
sec_fixed = pd.DataFrame(
    [
        {
            "Term": r"\multicolumn{7}{l}{\textbf{Fixed effects}}",
            "Estimate": "",
            "SE": "",
            stat_lab: "",
            "p": "",
            "CI low": "",
            "CI high": "",
        }
    ]
)

sec_random = pd.DataFrame(
    [
        {
            "Term": r"\multicolumn{7}{l}{\textbf{Random effects}}",
            "Estimate": "",
            "SE": "",
            stat_lab: "",
            "p": "",
            "CI low": "",
            "CI high": "",
        }
    ]
)

t = pd.concat([sec_fixed, fixed_rows, sec_random, random_rows], ignore_index=True)
t = t[["Term", "Estimate", "SE", stat_lab, "p", "CI low", "CI high"]]

# italic
t = t.rename(columns={"p": "$p$", "z": "$z$"})

# 4) Notes (keep wording close to original)
N_obs = int(len(data))
N_groups = int(data["participant"].nunique()) if "participant" in data.columns else None
aic = float(r("AIC(fit)")[0])
bic = float(r("BIC(fit)")[0])
llf = float(r("logLik(fit)")[0])


if ":" in formula_r:
    interactions_note = "Interactions are denoted by '$\\times$'. "
else:
    interactions_note = ""

pretty_formula = map_labels(formula_r).replace("~", "$\sim$")

notes_text = (
    pretty_formula + ". "
    # "Complexity predictors (HC = heuristic-related, CC = compositional, VC = visual-order, DD = diagonal dissimilarity) are signed right–left differences ($\Delta$) of the individual z-standardized (across the sample) solution complexities. "
    # "Negative coefficients indicate that higher complexity in one solution compared to the opposite solution decreases the odds of choosing "
    # "a higher preference category for that solution (and increases the odds of choosing the opposite solution). "
    # + interactions_note
    # + "The Random effects panel lists SDs of participant random intercepts and random slopes. "
    # + "Symmetric thresholding was used. "
    + threshold_note
    + f"$N(obs) =$ {N_obs}, $N(subj) =$ {('-' if N_groups is None else N_groups)}, logLik = {llf:.1f}. "
    # + f"$N(obs) =$ {N_obs}, $N(subj) =$ {('-' if N_groups is None else N_groups)}, logLik = {llf:.1f}, AIC = {aic:.1f}, BIC = {bic:.1f}. "
    "The odds ratio (OR) for a 1‑SD change in a predictor is $\\mathrm{OR} = \\exp(\\beta)$; "
    "the corresponding percent change in odds is $100\\cdot(\\exp(\\beta) - 1)$. "
)


# 5) TableSpec with post-processing to:
#    - remove cmidrule under header (add_cmidrule=False)
#    - avoid addlinespace after \toprule (top_addlinespace='')
#    - turn section rows into clean single \multicolumn lines
#    - insert \addlinespace[0.3em] before each section header
def _post_fix_sections(body: str, spec: TableSpec) -> str:
    lines = body.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if r"\multicolumn{7}{l}{\textbf{Fixed effects}}" in ln:
            out.append(r"\addlinespace[0.3em]")
            out.append(r"\multicolumn{7}{l}{\textbf{Fixed effects}}\\")
            continue
        if r"\multicolumn{7}{l}{\textbf{Random effects}}" in ln:
            out.append(r"\addlinespace[0.3em]")
            out.append(r"\multicolumn{7}{l}{\textbf{Random effects}}\\")
            continue
        out.append(ln)
    return "\n".join(out)


caption = "Ordinal Mixed-Effects Model Predicting Choice"
if STUDY == "exploratory":
    caption = caption + " (Exploratory Study)"

label = f"tab:choice-{STUDY}"

spec = TableSpec(
    caption=caption,
    label=label,
    float_fmt="{:.3f}",
    wrap="threeparttable",
    use_siunitx=False,
    width="1col",
    use_tabularx=False,
    placement="H",
    column_format="lrrrrrr",
    fontsize_pt=(11, 13),
    notes=notes_text,
    use_index=False,
    add_cmidrule=False,  # no cmidrule line under header
    top_addlinespace="",  # no addlinespace right after \toprule
    post_latex=_post_fix_sections,
)

out_path = Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, "tab_choice.tex")
write_table(t, out_path, spec)
print(f"Wrote {out_path}")

# %% [markdown]
# Print thresholds (in case we want to move them later into the table notes)

# %%
# 1) Fixed effects with CIs from R, drop thresholds, apply pretty labels
tidy_fixed_r = r(
    'broom.mixed::tidy(fit, effects="fixed", conf.int=TRUE, conf.level=.95)'
)
fixed = r2pandas(tidy_fixed_r)
fixed.columns = [c.replace(".", "_") for c in fixed.columns]

", ".join(
    [
        "=".join([map_labels(t), format(v, ".2f")])
        for t, v in fixed[fixed["coef_type"] == "intercept"][
            ["term", "estimate"]
        ].values.tolist()
    ]
)

# %% [markdown]
# ## 3. Figure with predicted choice probabilities

# %% [markdown]
# - Function to convert color to greyscale to see if a figure would still work as a greyscale print
# - Dictionary for cleaner titles

# %% [markdown]
# ### 3.1. Arrange main predictors for the plot

# %%
exclude_predictors = []

main_predictors = [
    x
    for x in formula_r.split(" ~ ")[1].split(" + (")[0].split(" + ")
    if ":" not in x and x not in exclude_predictors
]


def move_diag_after_ac(lst):
    """Move the element containing '_diag' right after the one containing '_ac'."""
    try:
        ac_idx = next(i for i, s in enumerate(lst) if "_ac" in s)
        diag_idx = next(i for i, s in enumerate(lst) if "_diag" in s)
    except StopIteration:  # one of the markers is absent
        return lst  # return unchanged list

    item = lst.pop(diag_idx)
    # adjust target index if the pop came from before the insert point
    if diag_idx < ac_idx:
        ac_idx -= 1
    lst.insert(ac_idx + 1, item)
    return lst


main_predictors = move_diag_after_ac(main_predictors)

# %% [markdown]
# ### 3.2. Compute/Load Predicition Probablities for all Predictors

# %% [markdown]
# **TODO:** 
# - Z-scored values for optimized CC are more contrained +/- 1-2 SDs instead of >3 
# - The get a uniform figure, decide:
#    1. cap the plot range to the observed ranges (i.e., using the CC range for the other metrics as well)
#    2. extrapolate to a fixed plot range of +/- 3 (good practice, if we have no observations there?)

# %%
# Set FORCE_RANGE to force compuations of predictions
# in a specific range, even if they lie outside the observed range
# If we are confident that the extrapolation is theoretically sound
# and you want to see the predictions over the full range of the Z-scored variable
# (e.g., [−3.0,3.0]), you can manually specify the desired grid points using the values argument within terms.

FORCE_RANGE = True
FORCED_RANGE = (-3.0, 3.0)

forced_str = "_forced_range" if FORCE_RANGE else "" 

if RECOMPUTE_PRED_PROBS:
    pred_probs_all = {}
    if FORCE_RANGE:
        r_seq = r.seq(*FORCED_RANGE, length_out=150)
    for predictor in main_predictors:
        if FORCE_RANGE:
            term_string = f'{predictor} [c({",".join(map(str, r_seq))})]'

            r_seq_str = r_seq.r_repr()
            term_string_fixed = f"{predictor} [{r_seq_str}]"
            pred_probs_all[predictor] = r2pandas(
                ggeffects.ggpredict(
                    clmm_fit, terms=term_string_fixed, bias_correction=True
                )
            )
        else:
            pred_probs_all[predictor] = r2pandas(
                ggeffects.ggpredict(
                    clmm_fit, terms=predictor + " [all]", bias_correction=True
                )
            )
    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, f"pred_prob{forced_str}.pkl"), "wb") as fp:
        pickle.dump(pred_probs_all, fp)

else:
    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, f"pred_prob{forced_str}.pkl"), "rb") as fp:
        pred_probs_all = pickle.load(fp)

# %% [markdown]
# ### 3.3. Practical Significance
#
# For reporting predicted probabilities at z=0 and z=+1 for each predictor
#

# %%
practical_sig = {}

for predictor in main_predictors:
    df_pred = pred_probs_all[predictor].copy()

    for target_x, label in [(0.0, "baseline"), (1.0, "plus_1sd")]:
        # find the row closest to target_x
        ix = (df_pred["x"] - target_x).abs().idxmin()
        closest_x = df_pred.loc[ix, "x"]
        sub = df_pred[df_pred["x"] == closest_x].copy()

        # response.level should be "1","2","3","4" as strings
        probs = sub.set_index("response.level")["predicted"].to_dict()

        # P(right) = P(slightly right) + P(definitely right)
        p_right = float(probs.get("3", 0)) + float(probs.get("4", 0))
        p_left = float(probs.get("1", 0)) + float(probs.get("2", 0))

        practical_sig[(predictor, label)] = {
            "x": float(closest_x),
            "p_right": float(p_right),
            "p_left": float(p_left),
        }

# Print summary and verify baseline consistency
print("Predicted P(choose right) at baseline (z≈0) and +1 SD:")
print("-" * 65)
baselines = []
for predictor in main_predictors:
    b = practical_sig[(predictor, "baseline")]
    p1 = practical_sig[(predictor, "plus_1sd")]
    shift = p1["p_right"] - b["p_right"]
    baselines.append(b["p_right"])
    print(
        f"{map_labels(predictor):40s}  "
        f"z≈0: {b['p_right']:.1%} (x={b['x']:.3f})  "
        f"z≈+1: {p1['p_right']:.1%} (x={p1['x']:.3f})  "
        f"shift: {shift:+.1%}"
    )

print(f"\nBaseline P(right) range across predictors: "
      f"{min(baselines):.4f} – {max(baselines):.4f}")
if max(baselines) - min(baselines) > 0.01:
    print("WARNING: baselines differ by >1pp across predictors. "
          "Report per-predictor baselines or use a single average.")


# %% [markdown]
# ### 3.4. Create Figure

# %%
out_base = "fig_choice_probability_panels"
dpi_out = 600
dpi_out_savefig = 600

# style
sns.set_theme(
    style="whitegrid",
    font_scale=1.2,
    palette=sns.color_palette("colorblind"),  # colour-blind–friendly palette
)  # alternative style: ticks
plt.rcParams.update(
    {
        "figure.dpi": dpi_out,
        "savefig.dpi": dpi_out_savefig,
        "axes.titlesize": 14,  # 16
        "axes.labelsize": 12,  # 14
        "xtick.labelsize": 11,  # 12
        "ytick.labelsize": 11,  # 12
        "legend.fontsize": 11,  # 12
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",  # default in most Matplotlib installs
            "Liberation Sans",  # Red Hat/Fedora and derivatives
            "Nimbus Sans",  # Ghostscript / many TeX installs
            "Bitstream Vera Sans",  # GNOME / old XFree86
            "FreeSans",  # GNU FreeFont
            "Helvetica",  # available on some proprietary stacks
            "Arial",  # MS core font, present on many systems via ttf-mscorefonts
        ],
        "axes.edgecolor": "black",  # "lightgray" default
        "figure.figsize": (8, 6),
        "lines.linewidth": 1.0,  # 1 default
        "axes.linewidth": 0.5,  # 1.0 default
        "grid.linewidth": 0.5,  # 1.0 default
        # "axes.spines.top": False,  # clean journal look
        # "axes.spines.bottom": False,
        # "axes.spines.left": False,
        # "axes.spines.right": False,
        # "errorbar.capsize": 3,
        # "lines.markersize": 8,
    }
)

# Optional: prevent label clipping
plt.rcParams["figure.autolayout"] = True

# ------------------------------------------------------------------
# USER INPUT (replace the “...” placeholders with your own objects)
# ------------------------------------------------------------------

choice_map = {
    "1": "definitely left",
    "2": "slightly left",
    "3": "slightly right",
    "4": "definitely right",
}

# palette = ["#003cb3", "#b3ccff", "#ffb3b3", "#b30000"] # red/blue
# palette = ["#0072B2" , "#80d0ff", "#ffd780", "#b37a00"] # orange/blue
# palette = ["#1B7837" , "#87c48c", "#b6afcf", "#542788"] # violet/green
# palette = ["#008A6E", "#33C7A6", "#F56ABF", "#A1005A"] # teal/salmon
palette = ["#1d5763", "#349db2", "#dd9688", "#c65139"]
# ------------------------------------------------------------------

# helper: rows/cols rule
layout = lambda n: (1, n) if n <= 2 else (2, int(np.ceil(n / 2)))
rows, cols = layout(len(main_predictors))

# common y-limit
pmax = max(
    df.loc[df.x.between(-3, 3), "conf.high"].max() for df in pred_probs_all.values()
)
y_max = np.ceil(min(1.0, pmax + 0.05) / 0.05) * 0.05

# figure + axes
fig, axes = plt.subplots(
    rows, cols, figsize=(4.5 * cols, 3.0 * rows), sharex=True, sharey=True
)
axes = np.atleast_1d(axes).ravel()

# ------------------------------------------------------------------
# plotting loop
# ------------------------------------------------------------------
for ax, predictor in zip(axes, main_predictors):
    df = pred_probs_all[predictor].copy()
    df["choice"] = df["response.level"].map(choice_map)

    # seaborn lineplot (no legend per-axes)
    sns.lineplot(
        data=df,
        x="x",
        y="predicted",
        hue="choice",
        palette=palette,
        ax=ax,
        legend=False,
    )

    # confidence ribbons
    for choice, col in zip(df.choice.unique(), palette):
        sub = df[df.choice == choice]
        ax.fill_between(sub.x, sub["conf.low"], sub["conf.high"], color=col, alpha=0.3)

    # give the (hidden) artists proper labels for the fig-level legend
    for line, choice in zip(ax.lines, df.choice.unique()):
        line.set_label(choice)

    # cosmetics
    ax.set_title(map_labels_long(predictor))
    ax.set_xlim(-3, 3)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x:+g}" if x else "0")
    )
    ax.grid(alpha=1.0)

    # remove per-panel axis labels (figure-level labels will be added)
    ax.set_xlabel(None)
    ax.set_ylabel(None)

# hide any empty panels (odd numbers)
for ax in axes[len(main_predictors) :]:
    ax.set_visible(False)

# ------------------------------------------------------------------
# figure-level labels
# ------------------------------------------------------------------
fig.text(0.5, 0.05, "Complexity difference (R – L)", ha="center", va="top")
fig.text(0.05, 0.5, "Choice probability", ha="right", va="center", rotation="vertical")

# ------------------------------------------------------------------
# figure-level legend (take handles/labels from last populated axis)
# ------------------------------------------------------------------
handles, labels = axes[len(main_predictors) - 1].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels),
    frameon=False,
    bbox_to_anchor=(0.5, -0.025),
)

plt.tight_layout(rect=[0.05, 0.05, 1, 1])

# ------------------------------------------------------------------
# save outputs
# ------------------------------------------------------------------
fig.savefig(
    Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{out_base}.png"),
    dpi=dpi_out_savefig,
    bbox_inches="tight",
)
fig.savefig(Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{out_base}.pdf"), bbox_inches="tight")
plt.show()
print(f"Wrote {out_base}.png  &  {out_base}.pdf  at {dpi_out_savefig} dpi")

# %% [markdown]
# ## 4. Plot Example Solutions
#
# Find 4 pairs that differ maximally (+- 3 SD) in the 4 metrics and plot them side by side to make it more understandable for readers to interpret the results incl. the last plot.

# %%
from joptim.utils.render_html import crop_image, create_stimulus


with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
    results = pickle.load(pickle_file)


cols = [
    "participant",
    "trial_id",
    "ac_R",
    "ac_L",
    "diff_ac_RvL",
    "diaga_R",
    "diaga_L",
    "diff_diaga_RvL",
    cc_variable_left,
    cc_variable_right,
    diff_cc_variable,
    "disorder_R",
    "disorder_L",
    "diff_disorder_RvL",
    "ac_R_z",
    "ac_L_z",
    "diff_ac_RvL_z",
    "diaga_R_z",
    "diaga_L_z",
    "diff_diaga_RvL_z",
    cc_variable_left_z,
    cc_variable_right_z,
    diff_cc_variable_z,
    "disorder_R_z",
    "disorder_L_z",
    "diff_disorder_RvL_z",
]

directory = Path(EXPORT_PATH_PUB, STUDY, "mlm_choice")
fname_base = "example_solution_"
sol_id_dict = {"1": "L", "2": "R"}
example_trial = {}
candidate_example_trials = {}

# %% [markdown]
# ### 4.1. Heuristic-Related Complexity (HC)
#
# - Find trials that scored high on HC, but lower on CC and VC
# - Need to tweak `threshold_min_z` a bit.

# %% [markdown]
# TODO: store png files in project output directory not in `Downloads/tmp`

# %%
np.random.seed(0)

# ---
# metric-specific variables
# ---

threshold_min_z = 2.5 # exploratory 2.65
threshold_max_z = 3.0
threshold_max_other_z = 0.2
fname_suffix = "hc"
metric = "ac"
# ---

candidate_example_trials[metric] = data.loc[
    (data["diff_ac_RvL_z"].abs() >= threshold_min_z)
    & (data["diff_ac_RvL_z"].abs() <= threshold_max_z)
    & (data[diff_cc_variable_z].abs() <= threshold_max_other_z)
    & (data["diff_disorder_RvL_z"].abs() <= threshold_max_other_z),
    cols,
].reset_index(drop=True)

if len(candidate_example_trials[metric]) == 0:
    raise ValueError("Expected at least one row, got zero.")

# Randomly sample 1 trial/pair and add the some information from the results data
i = np.random.choice(candidate_example_trials[metric].index)

example_trial[metric] = candidate_example_trials[metric].loc[[i]].reset_index(drop=True)

example_trial[metric] = example_trial[metric].merge(
    results["preference_trials"],
    left_on=["participant", "trial_id"],
    right_on=["xid", "trial_id"],
    how="left",
    validate="1:1",
)

# transform to pd.Series since we only have 1 row
example_trial[metric] = example_trial[metric].squeeze()

for sol_id in sol_id_dict:

    create_stimulus(
        worker_capacities=example_trial[metric]["worker_capacities"],
        task_loads=example_trial[metric]["task_loads"],
        directory=directory,
        out="png",
        labels=False,
        annotations=None,
        sm=example_trial[metric][f"solution_{sol_id}"],
        order=None,
        worker_order=example_trial[metric][f"worker_order_{sol_id}"],
        task_order=example_trial[metric][f"task_order_{sol_id}"],
        fname_base=fname_base + fname_suffix + "_" + sol_id_dict[sol_id],
    )

    _ = crop_image(
        Path(directory, "png", f"{fname_base}{fname_suffix}_{sol_id_dict[sol_id]}.png"),
        margin=1,
    )

# %% [markdown]
# ### 4.2. Compositional Complexity (CC)
#
# - Find trials that scored high in CC, but lower on HRC and VOC

# %%
np.random.seed(0)

# ---
# metric-specific variables
# ---

threshold_min_z = 1.68 # eploratory: 2.65
threshold_max_z = 3.0
threshold_max_other_z = 0.6
fname_suffix = "cc"
metric = "cc"

# ---

candidate_example_trials[metric] = data.loc[
    (data[diff_cc_variable_z].abs() >= threshold_min_z)
    & (data[diff_cc_variable_z].abs() <= threshold_max_z)
    & (data["diff_ac_RvL_z"].abs() <= threshold_max_other_z)
    & (data["diff_disorder_RvL_z"].abs() <= threshold_max_other_z),
    cols,
].reset_index(drop=True)

if len(candidate_example_trials[metric]) == 0:
    raise ValueError("Expected at least one row, got zero.")

# Randomly sample 1 trial/pair and add the some information from the results data
i = np.random.choice(candidate_example_trials[metric].index)

example_trial[metric] = candidate_example_trials[metric].loc[[i]].reset_index(drop=True)

example_trial[metric] = example_trial[metric].merge(
    results["preference_trials"],
    left_on=["participant", "trial_id"],
    right_on=["xid", "trial_id"],
    how="left",
    validate="1:1",
)

# transform to pd.Series since we only have 1 row
example_trial[metric] = example_trial[metric].squeeze()

for sol_id in sol_id_dict:

    create_stimulus(
        worker_capacities=example_trial[metric]["worker_capacities"],
        task_loads=example_trial[metric]["task_loads"],
        directory=directory,
        out="png",
        labels=False,
        annotations=None,
        sm=example_trial[metric][f"solution_{sol_id}"],
        order=None,
        worker_order=example_trial[metric][f"worker_order_{sol_id}"],
        task_order=example_trial[metric][f"task_order_{sol_id}"],
        fname_base=fname_base + fname_suffix + "_" + sol_id_dict[sol_id],
    )

    _ = crop_image(
        Path(directory, "png", f"{fname_base}{fname_suffix}_{sol_id_dict[sol_id]}.png"),
        margin=1,
    )

# %% [markdown]
# ### 4.3. Visual-Order Complexity (VC)
#
# - Find trials that scored high in VC, but lower on HC and CC

# %%
np.random.seed(5)

# ---
# metric-specific variables
# ---
threshold_min_z = 2.65
threshold_max_z = 3.0
threshold_max_other_z = 0.2
fname_suffix = "vc"
metric = "disorder"

# ---

candidate_example_trials[metric] = data.loc[
    (data["diff_disorder_RvL_z"].abs() >= threshold_min_z)
    & (data["diff_disorder_RvL_z"].abs() <= threshold_max_z)
    & (data["diff_ac_RvL_z"].abs() <= threshold_max_other_z)
    & (data[diff_cc_variable_z].abs() <= threshold_max_other_z),
    cols,
].reset_index(drop=True)


if len(candidate_example_trials[metric]) == 0:
    raise ValueError("Expected at least one row, got zero.")

# Randomly sample 1 trial/pair and add the some information from the results data
i = np.random.choice(candidate_example_trials[metric].index)

example_trial[metric] = candidate_example_trials[metric].loc[[i]].reset_index(drop=True)

example_trial[metric] = example_trial[metric].merge(
    results["preference_trials"],
    left_on=["participant", "trial_id"],
    right_on=["xid", "trial_id"],
    how="left",
    validate="1:1",
)

# transform to pd.Series since we only have 1 row
example_trial[metric] = example_trial[metric].squeeze()

for sol_id in sol_id_dict:

    create_stimulus(
        worker_capacities=example_trial[metric]["worker_capacities"],
        task_loads=example_trial[metric]["task_loads"],
        directory=directory,
        out="png",
        labels=False,
        annotations=None,
        sm=example_trial[metric][f"solution_{sol_id}"],
        order=None,
        worker_order=example_trial[metric][f"worker_order_{sol_id}"],
        task_order=example_trial[metric][f"task_order_{sol_id}"],
        fname_base=fname_base + fname_suffix + "_" + sol_id_dict[sol_id],
    )

    _ = crop_image(
        Path(directory, "png", f"{fname_base}{fname_suffix}_{sol_id_dict[sol_id]}.png"),
        margin=1,
    )

# %% [markdown]
# ### 4.4. Diagonality
#
# - Find trials that scored high on Diagonality, but lower on HRC, CC and VOC

# %%
np.random.seed(5)

# ---
# metric-specific variables
# ---

threshold_min_z = 2.65
threshold_max_z = 4.0
threshold_max_other_z = 2.0
fname_suffix = "dd"
metric = "diag"

# ---

candidate_example_trials[metric] = data.loc[
    (data["diff_diaga_RvL_z"].abs() >= threshold_min_z)
    & (data["diff_diaga_RvL_z"].abs() <= threshold_max_z)
    & (data["diff_ac_RvL_z"].abs() <= threshold_max_other_z)
    & (data[diff_cc_variable_z].abs() <= threshold_max_other_z)
    & (data["diff_disorder_RvL_z"].abs() <= threshold_max_other_z),
    cols,
].reset_index(drop=True)


if len(candidate_example_trials[metric]) == 0:
    raise ValueError("Expected at least one row, got zero.")

# Randomly sample 1 trial/pair and add the some information from the results data
i = np.random.choice(candidate_example_trials[metric].index)

example_trial[metric] = candidate_example_trials[metric].loc[[i]].reset_index(drop=True)

example_trial[metric] = example_trial[metric].merge(
    results["preference_trials"],
    left_on=["participant", "trial_id"],
    right_on=["xid", "trial_id"],
    how="left",
    validate="1:1",
)

# transform to pd.Series since we only have 1 row
example_trial[metric] = example_trial[metric].squeeze()

for sol_id in sol_id_dict:

    create_stimulus(
        worker_capacities=example_trial[metric]["worker_capacities"],
        task_loads=example_trial[metric]["task_loads"],
        directory=directory,
        out="png",
        labels=False,
        annotations=None,
        sm=example_trial[metric][f"solution_{sol_id}"],
        order=None,
        worker_order=example_trial[metric][f"worker_order_{sol_id}"],
        task_order=example_trial[metric][f"task_order_{sol_id}"],
        fname_base=fname_base + fname_suffix + "_" + sol_id_dict[sol_id],
    )

    _ = crop_image(
        Path(directory, "png", f"{fname_base}{fname_suffix}_{sol_id_dict[sol_id]}.png"),
        margin=1,
    )

# %% [markdown]
# ### 4.5. Integrate all 8 solutions into 1 Figure

# %%
dpi_out = 600
dpi_out_savefig = 600

sns.set_theme(
    style="whitegrid",
    font_scale=1.2,
    palette=sns.color_palette("colorblind"),  # colour-blind–friendly palette
)  # alternative style: ticks
plt.rcParams.update(
    {
        "figure.dpi": dpi_out,
        "savefig.dpi": dpi_out_savefig,
        "axes.titlesize": 14,  # 16
        "axes.labelsize": 12,  # 14
        "xtick.labelsize": 11,  # 12
        "ytick.labelsize": 11,  # 12
        "legend.fontsize": 11,  # 12
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",  # default in most Matplotlib installs
            "Liberation Sans",  # Red Hat/Fedora and derivatives
            "Nimbus Sans",  # Ghostscript / many TeX installs
            "Bitstream Vera Sans",  # GNOME / old XFree86
            "FreeSans",  # GNU FreeFont
            "Helvetica",  # available on some proprietary stacks
            "Arial",  # MS core font, present on many systems via ttf-mscorefonts
        ],
        "axes.edgecolor": "black",  # "lightgray" default
        "figure.figsize": (8, 6),
        "lines.linewidth": 1.0,  # 1 default
        "axes.linewidth": 0.5,  # 1.0 default
        "grid.linewidth": 0.5,  # 1.0 default
        # "axes.spines.top": False,  # clean journal look
        # "axes.spines.bottom": False,
        # "axes.spines.left": False,
        # "axes.spines.right": False,
        # "errorbar.capsize": 3,
        # "lines.markersize": 8,
    }
)

# Optional: prevent label clipping
plt.rcParams["figure.autolayout"] = True


stat_text = []

for metric in [
    "ac",
    "diag",
    "cc",
    "disorder",
]:  # same order as the files are added to the plot
    for side in ["L", "R"]:
        if metric == "cc":
            stat_text.append(
                [
                    example_trial[metric][f"{metric}_{CC_VERSION}_{side}{OPT}_z"],
                    example_trial[metric][f"{metric}_{CC_VERSION}_{side}{OPT}"],
                    f"diff_{metric}_{CC_VERSION}_RvL{OPT}_z",  # will be needed to find the probs
                ]
            )
        elif metric == "diag":
            stat_text.append(
                [
                    example_trial[metric][f"{metric}a_{side}_z"],
                    example_trial[metric][f"{metric}a_{side}"],
                    f"diff_{metric}a_RvL_z",
                ]
            )
        else:
            stat_text.append(
                [
                    example_trial[metric][f"{metric}_{side}_z"],
                    example_trial[metric][f"{metric}_{side}"],
                    f"diff_{metric}_RvL_z",
                ]
            )


def get_probs(x_val, df, group_val="1"):

    ix = (df["x"] - x_val).abs().idxmin()
    closest_x = df.loc[ix, "x"]

    out = (
        df.loc[
            (df["x"] == closest_x) & (df["group"] == group_val),
            ["response.level", "predicted"],
        ]
        .set_index("response.level")  # 1–4 as index
        .sort_index()["predicted"]
    )
    return out.to_numpy()  # or return the Series if preferred


# %%
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image, ImageOps
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fout_base = "fig_example_solutions_per_metric"

# ------------------------------------------------------------------
# parameters
nrows, ncols, block_size = 2, 4, 2
# For extra block padding, add a dummy column between blocks
n_blocks = ncols // block_size
dx_px, dy_px = 0, 0
alignment = "SE"
dpi = 600
figure_width = 8.5
wspace = 0.05
hspace = 0.02
dummy_col_width = 0.21  # adjust for desired extra space between blocks
logical_col_width = 0.8
row_scale = 1.55
image_scale = 0.85
block_title_offset = 0.0  # increase to move up
bbox_inches = "tight"
pad_inches = 0.1
pref_bar_height = 0.05
pref_bar_alpha = 0.8
pref_bar_label_offset = -0.12  # decrease to pull closer to bar
pref_bar_offset = 0.024  # + move bar down
block_subtitle_offset = 0.022  # + move z-scores down
gap_below_pref_bar = 0.00
delta_label_offset = 0.19  # decrease to pull closer to bar
delta_marker_size = 9
delta_strip_height = 0.03
show_letters = False
extra_space_at_bottom = 0.10  # important for legend
marker_color = "#4d4d4d"  # "#4d4d4d", "#e60000"
label_fs = 9
legend_y_pos = -0.07
palette = [
    "#1d5763",
    "#349db2",
    "#dd9688",
    "#c65139",
]  # make sure it's the same as in previous figure
show_totals = False
image_greyscale = False
# ------------------------------------------------------------------
img_files = [
    Path(directory, "png", f"example_solution_{p1}_{p2}.png")
    for p1 in ["hc", "dd", "cc", "vc"]  # same order as in stat_text
    for p2 in ["L", "R"]
]  # 8 paths
row_titles = [
    "Heuristic-related complexity ($\Delta$HC)",
    "Diagonal dissimilarity ($\Delta$DD)",
    "Compositional complexity ($\Delta$CC)",
    "Visual-order complexity ($\Delta$VC)",
]  # 4 strings

# complexity strip

from matplotlib import transforms
from matplotlib.ticker import FixedLocator, FuncFormatter

def _plot_top_aligned_hat(ax, x, ms_pt=9, color="#d24a44", z=5):
    x0d, y0d = ax.transData.transform((x, 0.0))
    half_px = 0.5 * ms_pt * ax.figure.dpi / 72.0
    xc, yc = ax.transData.inverted().transform((x0d, y0d - half_px))
    ax.plot(
        [xc],
        [yc],
        marker="^",
        markersize=ms_pt,
        markerfacecolor=color,
        markeredgecolor=color,
        linestyle="None",
        zorder=z,
        clip_on=False,
    )


def _plot_bottom_aligned_v(ax, x, ms_pt=9, color="#d24a44", z=5):
    # compute display coords of (x, 0)
    x0d, y0d = ax.transData.transform((x, 0.0))
    # shift center up by half the marker height (points -> pixels)
    half_px = 0.5 * ms_pt * ax.figure.dpi / 72.0
    xc, yc = ax.transData.inverted().transform((x0d, y0d + half_px))
    ax.plot(
        [xc],
        [yc],
        marker="v",
        markersize=ms_pt,
        markerfacecolor=color,
        markeredgecolor=color,
        linestyle="None",
        zorder=z,
        clip_on=False,
    )


def _draw_complexity_axis_strip(
    fig,
    ax_left,
    ax_right,
    z_value,
    y_center,
    delta_max=3.1,
    pad_frac=0.06,
    height=0.055,
    zorder=2,
    axis_color="#6e6e6e",
    tick_w=0.8,
    tick_fs=8.5,
    grid=True,
    grid_style="-",
    grid_alpha=0.6,
    grid_w=0.8,
    marker_color="#d24a44",
    marker_size_pt=9,
    label_text="Complexity difference (R – L)",
    label_fs=8.5,
    delta_label_offset=0.03,
):
    # axes placement
    pos0, pos1 = ax_left.get_position(), ax_right.get_position()
    x0, x1 = pos0.x0, pos1.x1
    x0i = x0 + pad_frac * (x1 - x0)
    x1i = x1 - pad_frac * (x1 - x0)
    ax = fig.add_axes([x0i, y_center - height / 2.0, x1i - x0i, height], zorder=zorder)
    ax.patch.set_alpha(0)

    # ranges and baseline at y=0
    pad_px = 1
    ax_w_px = ax.get_window_extent().width
    pad_data = pad_px * (2 * delta_max) / ax_w_px
    ax.set_xlim(-delta_max - pad_data, delta_max + pad_data)
    ax.set_ylim(-1, 1)
    for s in ("left", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["top"].set_visible(True)
    ax.spines["top"].set_position(("data", 0))
    ax.spines["top"].set_color(axis_color)
    ax.spines["top"].set_linewidth(tick_w)
    ax.yaxis.set_visible(False)

    # ticks at integers; labels with + for positives, plain 0
    ticks = list(range(-int(delta_max), int(delta_max) + 1))
    ax.xaxis.set_major_locator(FixedLocator(ticks))

    def _fmt(x, pos):
        xi = int(round(x))
        if xi != x:
            return ""
        if xi == 0:
            return "0"
        return f"+{xi}" if xi > 0 else f"{xi}"

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(
        axis="x",
        which="major",
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        length=0,
        width=0,
        color=axis_color,
        direction="in",
        labelsize=tick_fs,
        pad=2,
    )

    ax.minorticks_off()

    # ensure default grid is off, then draw custom gridlines only above the axis
    ax.grid(False)
    if grid:
        trans_blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        y0a = ax.transAxes.inverted().transform(ax.transData.transform((0, 0)))[1]
        for x in ticks:
            ax.vlines(
                x,
                0.0,
                y0a,  # bottom -> axis (was y0a -> 1.0)
                transform=trans_blend,
                colors=axis_color,
                linestyles=grid_style,
                linewidth=grid_w,
                alpha=grid_alpha,
                zorder=zorder - 1,
            )

    # red bottom-aligned “v” marker at z_value (clamped)
    xv = max(-delta_max, min(float(z_value), delta_max))
    _plot_top_aligned_hat(
        ax, xv, ms_pt=marker_size_pt, color=marker_color, z=zorder + 3
    )

    # label above/below (match preference bar style/offset)
    if label_text:
        ax.text(
            0.5,
            1.0 + delta_label_offset,
            label_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=label_fs,
            color="black",
            clip_on=False,
        )

    return ax


# ------------------------------------------------------------------
# preference bar helpers
def _place_seg_label(ax, x0, w, y, text, dark=False, fs=8.5):
    inside = abs(w) >= 8
    if inside:
        x = x0 + w / 2.0
        ha = "center"
        color = "white" if dark else "black"
    else:
        pad = 1.0 if w > 0 else -1.0
        x = x0 + w + pad
        ha = "left" if w > 0 else "right"
        color = "black"
    ax.text(x, y, text, va="center", ha=ha, fontsize=fs, color=color)


def _draw_pref_bar(
    fig,
    ax_left,
    ax_right,
    probs,
    y_center,
    palette,
    bar_height=0.04,
    show_totals=True,
    pref_bar_alpha=1.0,
    pref_bar_label_offset=0.03,
    label_fs=8.5,
):
    # probs: [def_left, slight_left, slight_right, def_right] (fractions summing to 1)
    pos0 = ax_left.get_position()
    pos1 = ax_right.get_position()
    x0 = pos0.x0
    x1 = pos1.x1
    h = bar_height
    y0 = y_center - h / 2.0
    bar_ax = fig.add_axes([x0, y0, x1 - x0, h], zorder=10)
    bar_ax.text(
        0.5,
        1.0 + pref_bar_label_offset,  # above bar
        # -pref_bar_label_offset,  # below bar
        # r"$P(choice)$",
        "Choice probability",
        transform=bar_ax.transAxes,
        ha="center",
        va="bottom",  # above/below bar
        # va="bottom",  # above bar
        fontsize=label_fs,
        color="black",
        clip_on=False,
    )

    # Styles (in order: def-left, slight-left, slight-right, def-right)
    col_dl, col_sl, col_sr, col_dr = palette

    dl, sl, sr, dr = [100.0 * float(p) for p in probs]
    y = 0

    # Left (negative)
    bar_ax.barh(
        y,
        -sl,
        left=0,
        color=col_sl,
        edgecolor="white",
        linewidth=1.0,
        alpha=pref_bar_alpha,
    )
    _place_seg_label(bar_ax, 0, -sl, y, f"{sl:.0f}%", dark=False, fs=11)
    bar_ax.barh(
        y,
        -dl,
        left=-sl,
        color=col_dl,
        edgecolor="white",
        linewidth=1.0,
        alpha=pref_bar_alpha,
    )
    _place_seg_label(bar_ax, -sl, -dl, y, f"{dl:.0f}%", dark=True, fs=11)

    # Right (positive)
    bar_ax.barh(
        y,
        sr,
        left=0,
        color=col_sr,
        edgecolor="white",
        linewidth=1.0,
        alpha=pref_bar_alpha,
    )
    _place_seg_label(bar_ax, 0, sr, y, f"{sr:.0f}%", dark=False, fs=11)
    bar_ax.barh(
        y,
        dr,
        left=sr,
        color=col_dr,
        edgecolor="white",
        linewidth=1.0,
        alpha=pref_bar_alpha,
    )
    _place_seg_label(bar_ax, sr, dr, y, f"{dr:.0f}%", dark=True, fs=11)

    # Center line and totals
    tl = dl + sl
    tr = dr + sr
    # bar_ax.axvline(0, color="black", linewidth=1)
    if show_totals:
        bar_ax.text(
            -tl - 2,
            y,
            f"Left ${tl:.0f}\%$",
            va="center",
            ha="right",
            fontsize=1,
            fontweight="bold",
        )
        bar_ax.text(
            tr + 2,
            y,
            f"Right ${tr:.0f}\%$",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )

    # Clean axis
    bar_ax.set_xlim(-100, 100)
    bar_ax.set_ylim(-0.8, 0.8)
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for sp in bar_ax.spines.values():
        sp.set_visible(False)

    # transparent bg
    bar_ax.patch.set_alpha(0)

    return bar_ax


# ------------------------------------------------------------------
def crop_and_pad_images(img_paths, nrows, ncols, dx_px=0, dy_px=0):
    cropped, sizes = [], []
    for fp in img_paths:
        im = Image.open(fp).convert("RGBA")
        arr = np.asarray(im)
        mask = ~(np.all(arr[:, :, :3] == 255, axis=2) | (arr[:, :, 3] == 0))
        if mask.any():
            ys, xs = np.where(mask)
            x0, x1 = xs.min() - 1, xs.max() + 2
            y0, y1 = ys.min() - 1, ys.max() + 2
            im = im.crop(
                (max(x0, 0), max(y0, 0), min(x1, im.width), min(y1, im.height))
            )
        cropped.append(im)
        sizes.append((im.width, im.height))
    row_heights = []
    for r in range(nrows):
        row_imgs = sizes[r * ncols : (r + 1) * ncols]
        row_h = max(h for w, h in row_imgs)
        row_heights.append(row_h)
    max_w = max(w for w, _ in sizes)
    imgs = []
    for idx, im in enumerate(cropped):
        w, h = im.width, im.height
        row = idx // ncols
        pad_top = row_heights[row] - h
        pad_left = max_w - w
        padded = ImageOps.expand(
            im, border=(pad_left, pad_top, dx_px, dy_px), fill="white"
        )
        imgs.append(np.asarray(padded))
    return imgs, sizes


def create_grid_figure(
    imgs,
    stat_text,
    row_titles,
    palette,
    figure_width=8.5,
    dpi=600,
    nrows=2,
    ncols=4,
    block_size=2,
    wspace=0.35,
    hspace=0.0,
    row_scale=1.2,
    alignment="SE",
    block_title_offset=0.01,
    dummy_col_width=0.4,
    logical_col_width=1.0,
    block_subtitle_offset=0.01,
    row_letters=["A", "B", "C", "D", "E", "F", "G", "H"],
    show_totals=False,
    pref_bar_height=0.04,
    pref_bar_alpha=1.0,
    gap_below_pref_bar=0.0,
    show_letters=False,
    image_scale=1.0,
    extra_space_at_bottom=0.0,
    pref_bar_label_offset=0.0,
    delta_label_offset=0.0,
    pref_bar_offset=0.005,
    delta_marker_size=11,
    marker_color="red",
    image_greyscale=False,
    label_fs=9.5,
    delta_strip_height=0.05,
    legend_y_pos=0.02,
):
    total_cols = ncols + 1  # 5 columns: 4 real + 1 dummy
    dummy_col_index = block_size  # place dummy after col 2

    col_widths = [max(img.shape[1] for img in imgs[c::ncols]) for c in range(ncols)]

    # Build width ratios with a well-defined dummy column width based on average real col width
    avg_col_w = float(np.mean(col_widths))
    width_ratios = []
    for gcol in range(total_cols):
        if gcol == dummy_col_index:
            width_ratios.append(dummy_col_width * avg_col_w)
        else:
            logical_col = gcol if gcol < dummy_col_index else gcol - 1
            width_ratios.append(logical_col_width * col_widths[logical_col])

    h_px = [im.shape[0] for im in imgs]
    row_heights = [max(h_px[r * ncols : (r + 1) * ncols]) for r in range(nrows)]
    h_w_ratio = sum(row_heights) * row_scale / sum(width_ratios)
    figure_height = figure_width * h_w_ratio
    fig = plt.figure(figsize=(figure_width, figure_height), dpi=dpi)

    gs = gridspec.GridSpec(
        nrows=nrows,
        ncols=total_cols,
        height_ratios=row_heights,
        width_ratios=width_ratios,
        wspace=wspace,
        hspace=hspace,
    )

    axes = []
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, ncols)
        grid_col = c if c < dummy_col_index else c + 1
        ax = fig.add_subplot(gs[r, grid_col])

        # allow scaling
        # compute a centered extent so the image scales within the same axes box
        # ax.imshow(
        #     im,
        #     interpolation="none",
        # )

        arr = im
        if image_greyscale:
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                rgb = arr[..., :3].astype(np.float32) / 255.0
                # sRGB luminance
                L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
                if arr.shape[2] == 4:
                    alpha = arr[..., 3].astype(np.float32) / 255.0
                    gray_rgba = np.dstack([L, L, L, alpha])
                    ax.imshow(gray_rgba, interpolation="none")
                else:
                    ax.imshow(L, cmap="gray", vmin=0.0, vmax=1.0, interpolation="none")
            else:
                ax.imshow(arr, cmap="gray", interpolation="none")
        else:
            ax.imshow(arr, interpolation="none")

        h, w = im.shape[:2]
        s = image_scale
        if s != 1.0:
            cx = (w - 1) / 2.0
            cy = (h - 1) / 2.0
            rx = w / (2.0 * s)
            ry = h / (2.0 * s)
            ax.set_xlim(cx - rx, cx + rx)
            ax.set_ylim(cy + ry, cy - ry)  # preserves origin='upper' orientation

        ax.set_anchor(alignment)
        ax.set_xticks([]), ax.set_yticks([])
        [spine.set_visible(False) for spine in ax.spines.values()]
        axes.append(ax)

    fig.subplots_adjust(
        left=0, right=1, top=1, bottom=extra_space_at_bottom, wspace=None, hspace=hspace
    )
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    n_blocks = ncols // block_size

    for r in range(nrows):
        for b in range(n_blocks):
            start_col = b * block_size + (1 if b > 0 else 0)
            end_col = start_col + block_size - 1
            idx0 = r * ncols + b * block_size
            idx1 = idx0 + block_size - 1
            ax0, ax1 = axes[idx0], axes[idx1]
            pos0 = ax0.get_position()
            pos1 = ax1.get_position()
            x_center = (pos0.x0 + pos1.x1) / 2
            y_block_title = max(pos0.y1, pos1.y1) + block_title_offset

            fig.text(
                x_center,
                y_block_title,
                row_titles[r * n_blocks + b],
                ha="center",
                va="bottom",
            )
            if show_letters:
                fig.text(
                    pos0.x0,
                    y_block_title,
                    row_letters[r * n_blocks + b],
                    ha="left",
                    weight="bold",
                    va="bottom",
                )

            # content for bar and z-scores
            delta_z = stat_text[idx1][0] - stat_text[idx0][0]

            probs = get_probs(
                delta_z, pred_probs_all[stat_text[idx0][2]]
            )  # [defL, slightL, slightR, defR]

            # z-scores
            badge_fs = 11  # or whatever you prefer

            zL = float(stat_text[idx0][0])
            zR = float(stat_text[idx1][0])
            delta = zR - zL

            # display complexity delta

            y_img_bot = min(pos0.y0, pos1.y0)
            y_strip = y_img_bot - block_subtitle_offset - delta_strip_height / 2.0

            _draw_complexity_axis_strip(
                fig,
                ax0,
                ax1,
                z_value=delta,
                y_center=y_strip,
                delta_max=3,
                pad_frac=0.06,
                height=delta_strip_height,
                zorder=2,
                axis_color="#6e6e6e",
                tick_w=0.8,
                tick_fs=8.5,
                grid=True,
                grid_style="-",
                grid_alpha=0.6,
                grid_w=0.5,
                marker_color=marker_color,
                marker_size_pt=delta_marker_size,
                label_text="Complexity difference (R – L)",
                label_fs=label_fs,
                delta_label_offset=delta_label_offset,
            )

            # draw preference bar
            y_strip_bottom = y_strip - delta_strip_height / 2.0
            y_bar_center = (
                y_strip_bottom
                - gap_below_pref_bar
                - pref_bar_offset
                - pref_bar_height / 2.0
            )

            _draw_pref_bar(
                fig,
                ax0,
                ax1,
                probs,
                y_bar_center,
                palette,
                bar_height=pref_bar_height,
                show_totals=show_totals,
                pref_bar_alpha=pref_bar_alpha,
                pref_bar_label_offset=pref_bar_label_offset,
                label_fs=label_fs,
            )

    fig.canvas.draw()

    # legend for probabilities
    legend_handles = [
        Patch(facecolor=palette[0], edgecolor=palette[0], label="definitely left"),
        Patch(facecolor=palette[1], edgecolor=palette[1], label="slightly left"),
        Patch(facecolor=palette[2], edgecolor=palette[2], label="slightly right"),
        Patch(facecolor=palette[3], edgecolor=palette[3], label="definitely right"),
    ]

    # fig.tight_layout(rect=[0, 0.08, 1, 1], pad=0.02)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y_pos),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    return fig


# ------------------------------------------------------------------
imgs, sizes = crop_and_pad_images(img_files, nrows, ncols)
fig = create_grid_figure(
    imgs,
    stat_text,
    row_titles,
    palette=palette,
    figure_width=figure_width,
    dpi=dpi,
    nrows=nrows,
    ncols=ncols,
    block_size=block_size,
    wspace=wspace,
    hspace=hspace,
    row_scale=row_scale,
    alignment=alignment,
    block_title_offset=block_title_offset,
    dummy_col_width=dummy_col_width,
    logical_col_width=logical_col_width,
    block_subtitle_offset=block_subtitle_offset,
    show_totals=show_totals,
    pref_bar_height=pref_bar_height,
    pref_bar_alpha=pref_bar_alpha,
    gap_below_pref_bar=gap_below_pref_bar,
    show_letters=show_letters,
    image_scale=image_scale,
    extra_space_at_bottom=extra_space_at_bottom,
    pref_bar_label_offset=pref_bar_label_offset,
    delta_label_offset=delta_label_offset,
    pref_bar_offset=pref_bar_offset,
    delta_marker_size=delta_marker_size,
    marker_color=marker_color,
    image_greyscale=image_greyscale,
    label_fs=label_fs,
    delta_strip_height=delta_strip_height,
    legend_y_pos=legend_y_pos,
)
# ------------------------------------------------------------------
# save outputs
# ------------------------------------------------------------------
fig.savefig(
    Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{fout_base}.png"),
    dpi=dpi,
    bbox_inches=bbox_inches,
    pad_inches=pad_inches,
)

fig.savefig(
    Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{fout_base}.pdf"),
    bbox_inches=bbox_inches,
    pad_inches=pad_inches,
)

print(f"Wrote {fout_base}.png  &  {fout_base}.pdf  at {dpi} dpi")


# %% [markdown]
# ## 5. Publication Metrics (JSON)

# %%
def compute_json(
    data_df, formula_r, fit_r_obj, r2_marg, r2_cond, practical_sig
):

    # pull tidy fixed effects from R
    tidy_fixed = r("broom.mixed::tidy(fit, effects='fixed', conf.int=TRUE, conf.level=.95)")
    df_fixed = r2pandas(tidy_fixed)
    # normalize column names
    df_fixed.columns = [c.replace(".", "_").lower() for c in df_fixed.columns]

    coefs = {}
    for _, row in df_fixed.iterrows():
        term = str(row["term"])
        beta = float(row["estimate"])
        exp_pct = np.exp(beta) - 1.0  # percent change in RT per 1 SD increase
        
        # odds ratios and their CIs (for clmm/logit models)
        or_val = float(np.exp(beta))
        or_low = float(np.exp(row["conf_low"]))
        or_high = float(np.exp(row["conf_high"]))

        coefs[term] = {
            "estimate": float(row["estimate"]),
            "std_error": float(row["std_error"]),
            "statistic": float(row["statistic"]),
            "p_value": float(row["p_value"]),
            "conf_low": float(row["conf_low"]),
            "conf_high": float(row["conf_high"]),
            "exp_pct": float(exp_pct),
            "exp_pct_abs": float(abs(exp_pct)),
            "exp_direction": "reduced" if exp_pct < 0 else "increased",
            "or": or_val,
            "or_low": or_low,
            "or_high": or_high,

        }

    r("""
    get_random_sds_df <- function(fit) {
      if (inherits(fit, "clmm")) {
        st_list <- fit$ST
        sds <- c(); terms <- c()
        # 'participant' is the grouping factor in your formula
        grp <- "participant"
        for (i in seq_along(st_list)) {
          M <- st_list[[i]]
          vals <- if (is.null(dim(M))) abs(M) else abs(diag(M))
          for (k in seq_along(vals)) {
            sds   <- c(sds, vals[k])
            terms <- c(terms, paste0("sd__component", k, "|", grp))
          }
        }
        return(data.frame(term = terms, sd = sds))
      } else if (inherits(fit, "lmerMod") || inherits(fit, "glmerMod")) {
        vc <- as.data.frame(VarCorr(fit))
        term <- ifelse(is.na(vc$var2),
                       paste0("sd__", vc$var1, "|", vc$grp),
                       paste0("sd__", vc$var1, ":", vc$var2, "|", vc$grp))
        return(data.frame(term = term, sd = vc$sdcor))
      } else {
        return(data.frame(term = character(0), sd = numeric(0)))
      }
    }
    """)

    df_sds = r2pandas(r("get_random_sds_df(fit)"))
    random_sds = {}
    sd_vals = []
    if df_sds.shape[0] > 0:
        for _, row in df_sds.iterrows():
            random_sds[str(row["term"])] = float(row["sd"])
            sd_vals.append(float(row["sd"]))
    random_sds_min = float(np.min(sd_vals)) if sd_vals else None
    random_sds_max = float(np.max(sd_vals)) if sd_vals else None


    # overall model stats from R
    aic = float(r("AIC(fit)")[0])
    bic = float(r("BIC(fit)")[0])
    llf = float(r("logLik(fit)")[0])
    
    meta= {
        "analysis_name": ANALYSIS_NAME,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "study":STUDY,
    }
    
    out = {
        "meta": meta,
        "n_obs": int(len(data_df)),
        "n_participants": int(data_df["participant"].nunique()) if "participant" in data_df.columns else None,
        "aic": aic,
        "bic": bic,
        "logLik": llf,
        "r2": {"marg": float(r2_marg), "cond": float(r2_cond)},
        "formula": str(formula_r),
        "coefs": coefs,
        "random_sds": random_sds,
        "random_sds_min": random_sds_min,
        "random_sds_max": random_sds_max,
    }

    # Add practical significance values to JSON output
    out["practical_significance"] = {}
    for predictor in main_predictors:
        b = practical_sig[(predictor, "baseline")]
        p1 = practical_sig[(predictor, "plus_1sd")]
        shift = p1["p_right"] - b["p_right"]
        out["practical_significance"][predictor] = {
            "p_right_baseline": b["p_right"],           # 0-1 scale
            "p_right_plus1sd": p1["p_right"],            # 0-1 scale
            "p_right_shift": shift,                      # 0-1 scale
            "p_right_shift_pp": 100.0 * shift,           # percentage points
            "p_right_shift_pp_abs": 100.0 * abs(shift),  # absolute pp
        }
    
    baseline_vals = [
        practical_sig[(p, "baseline")]["p_right"] for p in main_predictors
    ]
    out["practical_significance"]["_mean_baseline_p_right"] = float(
        np.mean(baseline_vals)
    )
    #

    return out


out = compute_json(data, formula_r, clmm_fit, pR2_results["r2_marg"], pR2_results["r2_cond"], practical_sig)



out_path = Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{ANALYSIS_NAME}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {out_path}")

# %% [markdown]
# **Consistency check**

# %%
with open(Path(EXPORT_PATH_PUB, STUDY, "demographics", "demographics.json"), "r", encoding="utf-8") as f:
    out_demo = json.load(f)

assert out_demo["counts"]["N_included"] == out["n_participants"], f"""Different number of participants demographics.json file: {out["n_participants"]} (here: {out_demo["counts"]["N_included"]})!"""

# %% [markdown]
# ## 6. Citations for R packages

# %% [markdown]
# Example: "All analyses were performed using R Statistical Software (v4.1.2; R Core Team 2021)."

# %%
print(r("""citation()"""))


# %%
print(r("""version$version.string"""))
print(r("""citation()"""))
print(r("""citation('lme4')"""))
print(r("""citation('ordinal')"""))
print(r("""citation('performance')"""))

