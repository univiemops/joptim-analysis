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
# From Appendix C (*may be outdated*)
#
# C.3 Gaze Bias (Binomial Outcome)
# - Outcome: `cbind(R, L)`, binomial(logit), `glmer`.
# - Main effects: signed right–left differences in heuristic-related complexity;
# compositional complexity; visual-order complexity; and diagonal dissimilarity.
# - Moderators: self-reported problem-solving skills, problem-solving efficiency, difficulty,
# maximum disorder (not differenced).
# - Constraint: Do not include maximum disorder with visual-order complexity (here: its
# signed difference).
# - Interactions: Only complexity-difference × moderator; no diagonal × moderator.
# - Candidate set: All theoretically meaningful subsets of the above, respecting the rules
# above.
#
# Reproducibility Notes:
#
# All predictors z-standardized prior to model fitting. Random-effects structure is held
# constant across candidates within each analysis as described in C.0. AIC comparison
# and tie-breaking follow the rule in C.0.

# %% [markdown]
# ## Imports and Data
#
# Prerequisites:
# - `data_preprocessed.csv_<TRIAL_FILE_ID>` (run `/preprocessing/preprocess_experiment_data.py.ipynb` to create it)
# - `data_eye__<TRIAL_FILE_ID>` (run `/preprocessing/download_eye_data.py/pynb` to create it)

# %% [markdown]
# **TODO: continue here and use the the model fitting from mlm_rt and the predictor specs from mlm_choice**

# %%
import os
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import zscore

from joptim.utils.export_path import get_export_paths
from joptim.model_formulas import create_model_formulas
from joptim.model_selection import best_model

from rpy2.robjects import r
from rpy2.robjects.packages import importr
from joptim.r_pandas import pandas2r, r2pandas
from joptim.pretty_labels import map_labels, RANDOM_EFFECTS_LABELS
from joptim.publication_utils import translate_variables, translate_effects
from pub_utils.tables import TableSpec, write_table

# Import R package
lme4 = importr("lme4")
lmerTest = importr("lmerTest")

ANALYSIS_NAME = "mlm_gaze"

STUDY = "confirmatory"  #"exploratory" or "confirmatory" or "exploratory_with_optimized_cc"
RECOMPUTE_MODEL_COMPARISON = False
RECOMPUTE_BEST_MODEL = True
# RECOMPUTE_PRED_PROBS = True # TODO: implement?
USE_OLRE = True # (observation level random effects)

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
CC_VERSION = "v2"  # "v1" or "v2"
R_FIT_FILENAME = "fit_r_object.rds"
R2_FILENAME = "fit_R2.json" # TODO: implement?
MODEL_FORMULA_FILENAME = "formula_best_model.txt"

os.makedirs(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME), exist_ok=True)
os.makedirs(Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME), exist_ok=True)



EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()

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


data = pd.read_csv(
    Path(
        EXPORT_PATH_DATA,
        STUDY,
        f"""data_preprocessed{updated_str}_{str(TRIAL_FILE_ID).replace(".", "-")}.csv""",
    ),
    sep=",",
    dtype={"trial_file_id": float},
)

# filter to TRIAL_FILE_ID (loading from above file should be enough,
# but we want to be on the safe side)
data = data.loc[data["trial_file_id"] == TRIAL_FILE_ID].reset_index(drop=True)

df_eye_RvL = pd.read_csv(
    Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, f"""data_eye_{str(TRIAL_FILE_ID).replace(".", "-")}.csv""")
)
data = (
    data.merge(
        df_eye_RvL,
        on=["trial_file_id", "trial_file_user_id", "trial_id"],
        how="left",
    )
    .dropna()
    .copy()
)

data = data.drop(["trial_file_id", "trial_file_user_id"], axis=1)

# Exclude trials with no usable gaze
data = data.loc[(data["eye_R"] + data["eye_L"]) > 0].copy()
data["obs_id"] = np.arange(len(data))  # OLRE key


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
# Compute proportion and bias, and convert data to be used with R

# %%
data["eye_prop_R"] = data["eye_R"] / (data["eye_R"] + data["eye_L"])
data["eye_bias_RvL"] = (data["eye_R"] - data["eye_L"]) / (data["eye_R"] + data["eye_L"])


data_r = pandas2r(data)
_ = r.assign("data_r", data_r)


# %% [markdown]
# ## 1. Model Selection

# %% [markdown]
# ### 1.1. Model Comparison

# %%
def get_model_formulas():

    outcome = "cbind(eye_R, eye_L)"
    predictor_groups = [[
        "diff_ac_RvL_z",
        f"diff_cc_{CC_VERSION}_RvL{OPT}_z",
        "diff_disorder_RvL_z",
        "diff_diaga_RvL_z",
    ]]

    moderators = [
        "psi_z",
        "efficiency_z",
        "load_capacity_ratio_z",  # PD
        "max_disorder_z",
        "heuristic_optimality_z",
    ]

    no_interactions_with = ["handedness", "diff_diaga_RvL_z"]
    no_random_effects_for = ["handedness"]
    no_random_effects_for_interaction_terms = True
    group = "participant"

    # a) forbid diagonal × moderator
    # b) forbid disorder × max disorder, as in your code
    further_exclusions = [
        "diff_disorder_RvL_z:max_disorder_z"
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

    # Do not include both VC (diff_disorder) and MD (max_disorder) together
    cannot_be_in_same_model = ["diff_disorder_RvL_z", "max_disorder_z"]

    model_formulas_filtered = []
    for formula in model_formulas:
        both = all(tok in formula for tok in cannot_be_in_same_model)
        if both:
            continue
        model_formulas_filtered.append(formula)

    return model_formulas_filtered


def _augment_with_olre(formula: str) -> str:
    return f"{formula} + (1 | obs_id)" if USE_OLRE else formula

def model_comparison(data_r_obj, formulas):
    r.assign("data_for_comparison", data_r_obj)
    results = []
    for i, base_formula in enumerate(formulas):
        formula = _augment_with_olre(base_formula)
        print(f"Fitting {i+1}/{len(formulas)}", end=" - ")
        try:
            r(f"fit <- glmer({formula}, data = data_for_comparison, family=binomial)")
            bic = float(r("BIC(fit)")[0])
            aic = float(r("AIC(fit)")[0])
            llf = float(r("logLik(fit)")[0])
            r("summary_fit <- summary(fit)")
            converged = bool(r("summary_fit$optinfo$conv$opt == 0")[0])
            err = np.nan
        except Exception as e:
            aic = bic = llf = np.nan
            converged = np.nan
            err = e
        results.append({
            "Model": base_formula,  # store the base formula (without OLRE) for readability
            "AIC": aic, "BIC": bic, "Log-Likelihood": llf,
            "Converged": converged, "Error": err
        })
        print("converged:", converged, "- error:", err)
    return pd.DataFrame(results).replace([np.inf, -np.inf], np.nan)

if RECOMPUTE_MODEL_COMPARISON:
    model_formulas = get_model_formulas()
    print("number of formulas after further exclusion: ", len(model_formulas))

# %%
if RECOMPUTE_MODEL_COMPARISON:    
    df_model_comparison = model_comparison(data_r, model_formulas)
    
    
    df_model_comparison.to_csv(
        Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, "model_comparison.csv"), index=False
    )
    print("Completed model comparison and saved to disk.")
else:
    df_model_comparison = pd.read_csv(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, "model_comparison.csv"))
    print("Loaded existing model comparison from disk.")

# %% [markdown]
# Non-Converging Models

# %%
df_model_comparison.loc[df_model_comparison["Converged"] == False]

# %% [markdown]
# AIC-based top Models (among the converged ones)

# %%
df_model_comparison.loc[
    df_model_comparison["Converged"] == True
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
    df_model_comparison["Converged"] == True
]

# %% [markdown]
# ### 2.1. Fit
#
# Skip if `RECOMPUTE_BEST_MODEL` is set to `False`

# %%
if RECOMPUTE_BEST_MODEL:
    print("Fitting model...")
    formula_r = best_model(df_model_comparison_converged, model_col="Model", criterion="AIC")
    formula_r_aug = _augment_with_olre(formula_r)
    r(f"fit <- glmer({formula_r_aug}, data = data_r, family=binomial)")
    r(f"""saveRDS(fit, file="{Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R_FIT_FILENAME).as_posix()}")""")
    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, MODEL_FORMULA_FILENAME), "w") as fp:
        fp.write(formula_r_aug)
    # R2
    r2_marg = float(r("performance::r2_nakagawa(fit)$R2_marginal")[0])
    r2_cond = float(r("performance::r2_nakagawa(fit)$R2_conditional")[0])
    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R2_FILENAME), "w") as fp:
        json.dump({"description": "Nakagawa R2 (GLMM, binomial).",
                   "r2_marg": r2_marg, "r2_cond": r2_cond}, fp)
else:
    r(f"""fit <- readRDS(file="{Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R_FIT_FILENAME).as_posix()}")""")
    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, MODEL_FORMULA_FILENAME), "r") as fp:
        formula_r_aug = fp.readline()
    formula_r = formula_r_aug  # for printing

print(f"\nFormula used to fit the model:\n{formula_r_aug}\n")
print(r("summary(fit)"))
fit = r["fit"]
# Overdispersion diagnostic
try:
    od = r("performance::check_overdispersion(fit)")
    print(r("print(performance::check_overdispersion(fit))"))
except Exception as e:
    print("Overdispersion check failed:", e)


# %% [markdown]
# ### 2.2. Table for Publication

# %%
# 1) Fixed effects (with CIs) from R
fixed_r = r('broom.mixed::tidy(fit, effects="fixed", conf.int=TRUE, conf.level=.95)')
fixed = r2pandas(fixed_r)
fixed.columns = [c.replace(".", "_") for c in fixed.columns]



def pretty_term(term: str) -> str:
    parts = term.split(":")
    return " × ".join(map_labels(p) for p in parts)

fixed["term"] = fixed["term"].astype(str).map(pretty_term)

# p-values
def fmt_p(p):
    try:
        p = float(p)
        return "$<\\,0.001$" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return str(p)

fixed["p_value"] = fixed["p_value"].map(fmt_p)

# 2) Random effects SDs (glmer)
ran_r = r('try(broom.mixed::tidy(fit, effects="ran_pars"), silent=TRUE)')
if "try-error" in str(ran_r.rclass):
    random = pd.DataFrame(columns=["term", "estimate"])
else:
    rp = r2pandas(ran_r)
    random = pd.DataFrame({"term": rp["term"], "estimate": rp["estimate"]})
    # Normalize labels: sd__(Intercept)|participant -> SD (Intercept), sd__Observation -> SD (residual)
    random["term"] = (
        random["term"]
        .astype(str)
        .str.replace(r"^sd__Observation$", "SD (residual)", regex=True)
        .str.replace(r"^sd__\(", "SD (", regex=True)
        .str.replace(r"\|.*$", ")", regex=True)
    )

# 3) Build sectioned table with indentation
stat_lab = "z"
fixed_rows = pd.DataFrame({
    "Term": "\\hspace{1em} " + fixed["term"].astype(str),
    "Estimate": fixed["estimate"],
    "SE": fixed["std_error"],
    stat_lab: fixed["statistic"],
    "p": fixed["p_value"],
    "CI low": fixed["conf_low"],
    "CI high": fixed["conf_high"],
})

random_rows = pd.DataFrame({
    "Term": "\\hspace{1em} " + random["term"].astype(str),
    "Estimate": random["estimate"],
    "SE": [""] * len(random),
    stat_lab: [""] * len(random),
    "p": [""] * len(random),
    "CI low": [""] * len(random),
    "CI high": [""] * len(random),
})

sec_fixed = pd.DataFrame([{
    "Term": r"\multicolumn{7}{l}{\textbf{Fixed effects}}",
    "Estimate": "", "SE": "", stat_lab: "", "p": "", "CI low": "", "CI high": ""
}])

sec_random = pd.DataFrame([{
    "Term": r"\multicolumn{7}{l}{\textbf{Random effects}}",
    "Estimate": "", "SE": "", stat_lab: "", "p": "", "CI low": "", "CI high": ""
}])

t = pd.concat([sec_fixed, fixed_rows, sec_random, random_rows], ignore_index=True)
t = t[["Term", "Estimate", "SE", stat_lab, "p", "CI low", "CI high"]]

# italic
t = t.rename(columns={"p": "$p$"})

# 4) Footnote text (aligned in detail to choice table style)
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
    "Outcome: binomial logit for right vs left gaze events ($\\mathrm{logit}\\,P(\\text{right})$). "
    # "Complexity predictors (HC = heuristic-related, CC = compositional, VC = visual-order, DD = diagonal dissimilarity) are signed right–left differences ($\Delta$) of the individual z-standardized (across the sample) solution complexities. "
    # + interactions_note
    # "The Random effects panel lists SDs of participant random intercepts and random slopes where applicable; residual SD is not defined for binomial models. "
    + f"$N(obs) =$ {N_obs}, $N(subj) =$ {('-' if N_groups is None else N_groups)}, logLik = {llf:.1f}. "
    # + f"$N(obs) =$ {N_obs}, $N(subj) =$ {('-' if N_groups is None else N_groups)}, logLik = {llf:.1f}, AIC = {aic:.1f}, BIC = {bic:.1f}. "

)


# 5) Spec and post-processing to insert addlinespace before section headers
def _post_sections(body: str, spec: TableSpec) -> str:
    lines = body.splitlines()
    out = []
    for ln in lines:
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

caption = "Binomial Mixed-Effects Model Predicting Gaze Bias"
label = f"tab:gaze-{STUDY}"

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
    add_cmidrule=False,
    top_addlinespace="",
    post_latex=_post_sections,
)

out_path = Path(EXPORT_PATH_PUB, STUDY, "mlm_gaze", "tab_gaze.tex")
write_table(t, out_path, spec)
print(f"Wrote {out_path}")


# %% [markdown]
# In case of an **intercept-only model** it makes no sense to create an own table, just use text like below.

# %%
from joptim.publication_utils import one_sentence

if len(fixed) == 1:

    print(
        "Based on the AIC-based model selection procedure, the intercept-only model was selected as the most parsimonious model. "
        + one_sentence(fit)
        + f"This indicates no evidence that between-solution absolute differences in complexity or diagonal dissimilarity affected gaze in the {STUDY} study."
    )
else:
    print(t)


# %% [markdown]
# ## 3. Figures

# %%
#

# %% [markdown]
# ## 4. Publication Metrics (JSON)

# %%
def compute_json(data_df, formula_r, fit_r_obj):
    tidy_fixed = r("broom.mixed::tidy(fit, effects='fixed', conf.int=TRUE, conf.level=.95)")
    df_fixed = r2pandas(tidy_fixed); df_fixed.columns = [c.replace(".", "_").lower() for c in df_fixed.columns]
    coefs = {}
    for _, row in df_fixed.iterrows():
        beta = float(row["estimate"])
        or_val = float(np.exp(beta))
        or_pct = (or_val - 1.0) * 100.0
        coefs[str(row["term"])] = {
            "estimate": float(row["estimate"]),
            "std_error": float(row["std_error"]),
            "statistic": float(row["statistic"]),
            "p_value": float(row["p_value"]),
            "conf_low": float(row["conf_low"]),
            "conf_high": float(row["conf_high"]),
            "odds_ratio": or_val,
            "odds_pct_change": or_pct,
            "direction": "decrease" if or_pct < 0 else "increase",
        }
    aic = float(r("AIC(fit)")[0]); bic = float(r("BIC(fit)")[0]); llf = float(r("logLik(fit)")[0])
    r2_marg = float(r("performance::r2_nakagawa(fit)$R2_marginal")[0])
    r2_cond = float(r("performance::r2_nakagawa(fit)$R2_conditional")[0])
    out = {
        "meta": {"analysis_name": ANALYSIS_NAME, "timestamp": datetime.utcnow().isoformat() + "Z", "study": STUDY},
        "n_obs": int(len(data_df)),
        "n_participants": int(data_df["participant"].nunique()) if "participant" in data_df.columns else None,
        "aic": aic, "bic": bic, "logLik": llf,
        "formula": str(formula_r),
        "r2": {"marg": r2_marg, "cond": r2_cond},
        "coefs": coefs,
        "overdispersion": r("try(performance::check_overdispersion(fit)$dispersion_ratio, silent=TRUE)")[0] \
            if "try-error" not in str(r("try(performance::check_overdispersion(fit), silent=TRUE)").rclass) else None,
    }
    return out

out = compute_json(data, formula_r_aug, r["fit"])
out_path = Path(EXPORT_PATH_PUB, STUDY, ANALYSIS_NAME, f"{ANALYSIS_NAME}.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {out_path}")

