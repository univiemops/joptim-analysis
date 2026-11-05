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
# C.2 Reaction Time (Continuous Outcome)
# - Outcome: z-standardized log reaction time (lmer; REML = FALSE).
# - Main effects: absolute differences between the two solutions in heuristic-related complexity; compositional complexity; visual-order complexity;  and diagonal dissimilarity.
# - Additional covariates (no interactions): problem difficulty; problem-solving efficiency; heuristic optimality; maximum disorder.
# - Constraint: Do not include maximum disorder in any model that includes visual-order complexity (here: its absolute difference) because of redundancy.
# - Random effects: Participant random intercepts in all models; random slopes for included complexity absolute-difference main effects (and diagonal dissimilarity, if present) where convergence allows; interactions are not given random slopes.
# - Candidate set: All combinations of the main effects and any subset of covariates consistent with the constraint; no interactions.
# - Model selection: AIC-based selection as specified in C.0 (ΔAIC > 2 to retain a single best model; tie-break by parsimony).
#

# %% [markdown]
# ## Imports and Data
#
# Make sure that `data_preprocessed.csv` exists. If not, create it running `preprocess_experiment_data.ipynb`.

# %%
import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.stats import zscore
from pathlib import Path
from datetime import datetime
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

ANALYSIS_NAME = "mlm_rt"

STUDY = "confirmatory"  #"exploratory" or "confirmatory" or "exploratory_with_optimized_cc"
RECOMPUTE_MODEL_COMPARISON = False
RECOMPUTE_BEST_MODEL = True
# RECOMPUTE_PRED_PROBS = True # TODO: implement?


EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
CC_VERSION = "v2"  # "v1" or "v2"
R_FIT_FILENAME = "fit_r_object.rds"
R2_FILENAME = "fit_R2.json" # TODO: implement?
MODEL_FORMULA_FILENAME = "formula_best_model.txt"

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
data = data.drop(["trial_file_id", "trial_file_user_id"], axis=1)


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
# Convert to be used with R

# %%
data["log_rt"] = np.log(data["rt"])

data_r = pandas2r(data)

# send it to r-environment
_ = r.assign("data_r", data_r)


# %% [markdown]
# ## 0. Prepare Data
#
# TODO: move this part into the data preprocessing notebook

# %% [markdown]
# <div class="alert alert-warning">Inspect participants first to see if there were some problems/lags that impacted response times. Also look at UK and US separated.</div>

# %% [markdown]
# Still need to define how to model this.

# %% [markdown]
# ### 0.1. Absolute Difference in Complexity

# %% [markdown]
# *Reasons for not z-scoring absolute difference:*
# - To preserve the zero point's interpretability (indicates no difference between left and right values).
# - Original signed difference not standardized either, derived from standardized right and left values.

# %%
# done in preprocessing

# %% [markdown]
# ### 0.2. Outcome Variable Response Time 
#
# *Reasons for Log Transformation:*
# - Normalizes positively skewed response times.
# - Stabilizes variance across predictor levels.
# - Mitigates impact of extreme values.
# - Facilitates analysis of multiplicative effects.
#
# *Reasons for Z-Scoring Log-Transformed Data:*
# - Standardizes for easier comparison with other variables.
# - Facilitates interpretability in terms of standard deviations.
# - Improves numerical stability in the model.
#

# %% [markdown]
# ## 1. Model Selection

# %% [markdown]
# Skip to 2, if not recomputed (`RECOMPUTE_MODEL_COMPARISON=False`)

# %%
def get_model_formulas():

    outcome = "log_rt"

    # groups are always added together, separate groups by making use of the inner lists

    predictor_groups = [
        [
            "abs_diff_ac_RvL_z",
            f"abs_diff_cc_{CC_VERSION}_RvL{OPT}_z",
            "abs_diff_disorder_RvL_z",
            "abs_diff_diaga_RvL_z",
        ],
        ["load_capacity_ratio_z"],
        ["heuristic_optimality_z"],
        ["efficiency_z"],
        ["psi_z"],
        ["max_disorder_z"],
    ]
    
    moderators = []

    no_interactions_with = []
    no_random_effects_for = []
    no_random_effects_for_interaction_terms = True
    group = "participant"
    further_exclusions = [
        "abs_diff_disorder_RvL_z:max_disorder_z"
    ]  # specific interaction terms, e.g.
    
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
    print("number of formulas:", len(model_formulas))
    
    cannot_be_in_same_model = ["abs_diff_disorder_RvL_z", "max_disorder_z"]
    
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


def model_comparison(data, formulas):

    r.assign("data_for_comparison", data)

    model_results = []

    # Fit models and collect AIC/BIC
    for i, formula in enumerate(formulas):
        print(f"Fitting {i+1}/{len(formulas)}", end=" - ")
        try:
            lme_fit = r(
                f"fit <- lmer({formula}, data = data_for_comparison, REML = FALSE)"
            )

            bic = r("BIC(fit)")[0]
            aic = r("AIC(fit)")[0]
            llf = r("logLik(fit)")[0]
            summary_fit = r("summary_fit <- summary(fit)")
            coefficients = r2pandas(r("as.data.frame(coefficients(summary_fit))"))
            converged = r("""converged <- summary_fit$optinfo$conv$opt == 0""")[0]

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
    r(f"fit <- lmer({formula_r}, data = data_r, REML = FALSE)")
    r(f"""saveRDS(fit, file="{Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R_FIT_FILENAME).as_posix()}")""")
    print(f"R results saved to {R_FIT_FILENAME}")
    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, MODEL_FORMULA_FILENAME), "w") as fp:
        fp.write(formula_r)
    # Nakagawa R2
    r2_marg = float(r("performance::r2_nakagawa(fit)$R2_marginal")[0])
    r2_cond = float(r("performance::r2_nakagawa(fit)$R2_conditional")[0])
    pR2_results = {"description": "Nakagawa R2 (LMM).",
                   "r2_marg": r2_marg, "r2_cond": r2_cond}
    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R2_FILENAME), "w") as fp:
        json.dump(pR2_results, fp)
    print(f"Pseudo-R2 results saved to {R2_FILENAME}")
else:
    print(f"Loading R results from {R_FIT_FILENAME}...")
    r(f"""fit <- readRDS(file="{Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R_FIT_FILENAME).as_posix()}")""")
    with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, MODEL_FORMULA_FILENAME), "r") as fp:
        formula_r = fp.readline()
    # Load R2 if present
    if Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R2_FILENAME).exists():
        with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R2_FILENAME), "r") as fp:
            pR2_results = json.load(fp)
    else:
        pR2_results = None

fit = r["fit"]

print(f"\nFormula used to fit the model:\n{formula_r}\n")
print(r("summary(fit)"))
print(pR2_results)

# %%
# if RECOMPUTE_BEST_MODEL:
#     print("Fitting model...")

#     formula_r = best_model(
#         df_model_comparison_converged, model_col="Model", criterion="AIC"
#     )

#     fit = r(f"fit <- lmer({formula_r}, data = data_r, REML = FALSE)")


#     r(
#         f"""saveRDS(fit, file="{Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R_FIT_FILENAME).as_posix()}")"""
#     )
#     print(f"R results saved to {R_FIT_FILENAME}")


#     with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, MODEL_FORMULA_FILENAME), "w") as fp:
#         fp.write(formula_r)


# else:
#     print(f"Loading R results from {R_FIT_FILENAME}...")
#     r(
#         f"""fit <- readRDS(file="{Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, R_FIT_FILENAME).as_posix()}")"""
#     )
#     fit = r["fit"]
#     with open(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, MODEL_FORMULA_FILENAME), "r") as fp:
#         formula_r = fp.readline()

# print(f"""\nFormula used to fit the model:\n{formula_r}\n""")
# print(r("""summary(fit)"""))

# %% [markdown]
# ### 2.2. Table for Publication

# %%
# 1) Fixed effects (with CIs)
fixed_r = r('broom.mixed::tidy(fit, effects="fixed", conf.int=TRUE, conf.level=.95)')
fixed = r2pandas(fixed_r)
fixed.columns = [c.replace(".", "_") for c in fixed.columns]



fixed["term"] = fixed["term"].astype(str).map(map_labels)

def fmt_p(p):
    try:
        p = float(p)
        return "$<\\,0.001$" if p < 0.001 else f"{p:.3f}"
    except Exception:
        return str(p)

fixed["p_value"] = fixed["p_value"].map(fmt_p)

# 2) Random effects SDs (lmer)
ran_r = r('try(broom.mixed::tidy(fit, effects="ran_pars"), silent=TRUE)')
if "try-error" in str(ran_r.rclass):
    random = pd.DataFrame(columns=["term", "estimate"])
else:
    rp = r2pandas(ran_r)
    random = pd.DataFrame({"term": rp["term"], "estimate": rp["estimate"]})
    random["term"] = (
        random["term"]
        .astype(str)
        .str.replace(r"^sd__Observation$", "SD (residual)", regex=True)
        .str.replace(r"^sd__\(", "SD (", regex=True)
        .str.replace(r"\|.*$", ")", regex=True)
    )

# 3) Build sectioned table with indentation
stat_lab = "t"
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
t = t.rename(columns={"p": "$p$", "t": "$t$"})

# 4) Footnote text
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
    "Outcome: log reaction time ($\\log\\,\\mathrm{RT}$). "
    # "Complexity predictors (HC = heuristic-related, CC = compositional, VC = visual-order, DD = diagonal dissimilarity) are absolute right–left differences (|$\Delta$|) of the individual z-standardized (across the sample) solution complexities. "
    # + interactions_note
    # "The Random effects panel lists SDs of participant random intercepts and residual. "
    + f"$N(obs) =$ {N_obs}, $N(subj) =$ {('-' if N_groups is None else N_groups)}, logLik = {llf:.1f}. "
    # + f"$N(obs) =$ {N_obs}, $N(subj) =$ {('-' if N_groups is None else N_groups)}, logLik = {llf:.1f}, AIC = {aic:.1f}, BIC = {bic:.1f}. "
)


# 5) Spec and post-processing to insert addlinespace before sections
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

caption = "Linear Mixed-Effects Model Predicting Response Time"
label = f"tab:rt-{STUDY}"

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

out_path = Path(EXPORT_PATH_PUB, STUDY, "mlm_rt", "tab_rt.tex")
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
def compute_json(data_df, fit_r_obj, formula_r):
    tidy_fixed = r("broom.mixed::tidy(fit, effects='fixed', conf.int=TRUE, conf.level=.95)")
    df_fixed = r2pandas(tidy_fixed)
    df_fixed.columns = [c.replace(".", "_").lower() for c in df_fixed.columns]

    coefs = {}
    for _, row in df_fixed.iterrows():
        term = str(row["term"])
        beta = float(row["estimate"])
        # interpret exp_pct only for predictors (not intercept)
        is_intercept = term.strip() in ("(Intercept)", "Intercept")
        exp_pct = None
        exp_pct_abs = None
        exp_direction = None
        if not is_intercept:
            exp_pct = np.exp(beta) - 1.0
            exp_pct_abs = abs(exp_pct)
            exp_direction = "reduction" if exp_pct < 0 else "increase"
        coefs[term] = {
            "estimate": float(row["estimate"]),
            "std_error": float(row["std_error"]),
            "statistic": float(row["statistic"]),
            "p_value": float(row["p_value"]),
            "conf_low": float(row["conf_low"]),
            "conf_high": float(row["conf_high"]),
            "exp_pct": (None if exp_pct is None else float(exp_pct)),
            "exp_pct_abs": (None if exp_pct_abs is None else float(exp_pct_abs)),
            "exp_direction": (None if exp_direction is None else exp_direction),
        }

    aic = float(r("AIC(fit)")[0])
    bic = float(r("BIC(fit)")[0])
    llf = float(r("logLik(fit)")[0])
    # Nakagawa R2
    r2_marg = float(r("performance::r2_nakagawa(fit)$R2_marginal")[0])
    r2_cond = float(r("performance::r2_nakagawa(fit)$R2_conditional")[0])

    meta = {"analysis_name": ANALYSIS_NAME, "timestamp": datetime.utcnow().isoformat() + "Z", "study": STUDY}
    out = {
        "meta": meta,
        "n_obs": int(len(data_df)),
        "n_participants": int(data_df["participant"].nunique()) if "participant" in data_df.columns else None,
        "aic": aic, "bic": bic, "logLik": llf,
        "formula": str(formula_r),
        "r2": {"marg": r2_marg, "cond": r2_cond},
        "coefs": coefs,
    }
    return out



out = compute_json(data, fit, formula_r)

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
