import re
from rpy2 import robjects
import pypandoc
import tempfile
import pathlib

translate_effects = {
    "abs_diff_ac_RvL_z": {
        "model": "abs_diff_heur_c",
        "short": "Heuristic",
        "long": "Heuristic-related complexity",
    },
    "abs_diff_cc_v2_RvL_z": {
        "model": "abs_diff_comp_c",
        "short": "Composition",
        "long": "Compositional complexity",
    },
    "abs_diff_cc_v1_RvL_z": {
        "model": "abs_diff_comp_c",
        "short": "Composition",
        "long": "Compositional complexity",
    },
    "abs_diff_cc_v2_RvL_opt_z": {
        "model": "abs_diff_comp_c",
        "short": "Composition",
        "long": "Compositional complexity",
    },
    "abs_diff_cc_v1_RvL_opt_z": {
        "model": "abs_diff_comp_c",
        "short": "Composition",
        "long": "Compositional complexity",
    },
    "abs_diff_disorder_RvL_z": {
        "model": "abs_diff_visu_c",
        "short": "Visual Disorder",
        "long": "Visual-order complexity",
    },
    "abs_diff_diaga_RvL_z": {
        "model": "abs_diff_diag",
        "short": "Diagonal",
        "long": "Diagonal dissimilarity",
    },
    "diff_ac_RvL_z": {
        "model": "diff_heur_c",
        "short": "Heuristic",
        "long": "Heuristic-related complexity",
    },
    "diff_cc_v2_RvL_z": {
        "model": "diff_comp_c",
        "short": "Composition",
        "long": "Compositional complexity",
    },
    "diff_cc_v1_RvL_z": {
        "model": "diff_comp_c",
        "short": "Composition",
        "long": "Compositional complexity",
    },
    "diff_cc_v2_RvL_opt_z": {
        "model": "diff_comp_c",
        "short": "Composition",
        "long": "Compositional complexity",
    },
    "diff_cc_v1_RvL_opt_z": {
        "model": "diff_comp_c",
        "short": "Composition",
        "long": "Compositional complexity",
    },
    "diff_disorder_RvL_z": {
        "model": "diff_visu_c",
        "short": "Visual Disorder",
        "long": "Visual-order complexity",
    },
    "diff_diaga_RvL_z": {
        "model": "diff_diag",
        "short": "Diagonal",
        "long": "Diagonal dissimilarity",
    },
}


def translate_variables(formula: str, mapping: dict, counterpart: str = 'model') -> str:
    """
    Replace every variable name that occurs in *mapping* by its
    mapping[var][counterpart] ('model', 'short', 'long') counterpart.
    """
    # build one regular-expression that matches any old variable name
    pattern = r'\b(' + '|'.join(map(re.escape, mapping.keys())) + r')\b'

    def repl(match):
        old = match.group(0)
        return mapping[old][counterpart]

    return re.sub(pattern, repl, formula)


one_sentence_R = robjects.r(
    """
  function(fit, conf = 0.95) {

    if (!requireNamespace("broom.mixed", quietly = TRUE))
      stop("Install the ‘broom.mixed’ package.")
    if (!requireNamespace("lme4", quietly = TRUE))
      stop("Install the ‘lme4’ package (for nobs / ngrps helpers).")

    # -------- 1.  Intercept statistics ---------------------------------------
    fx  <- broom.mixed::tidy(fit, effects = "fixed",
                             conf.int = TRUE, conf.level = conf)
    int <- fx[fx$term == "(Intercept)", ]
    if (nrow(int) == 0) stop("No (Intercept) term found.")

    est <- round(int$estimate,  3)
    se  <- round(int$std.error, 3)
    lo  <- round(int$conf.low,  3)
    hi  <- round(int$conf.high, 3)

    p    <- int$p.value
    ptxt <- if (is.na(p)) "" else if (p < .001) ", p < .001"
            else sprintf(", p = %.3f", p)

    # -------- 2. Random-effect variance / SD ---------------------------------
    ran          <- broom.mixed::tidy(fit, effects = "ran_pars")
    ran_sentence <- ""
    if (nrow(ran) > 0) {
      ri <- ran[grepl("Intercept", ran$term), ]
      if (nrow(ri) > 0) {
        val <- paste(round(ri$estimate, 3), collapse = ", ")
        if (grepl("^sd__", ri$term[1]))
          ran_sentence <- paste0(" The random-intercept SD was ", val, ".")
        else
          ran_sentence <- paste0(" The random-intercept variance was ", val, ".")
      }
    }

    # -------- 3. Sample sizes -------------------------------------------------
    tot_n     <- formatC(stats::nobs(fit), format = "d", big.mark = ",")
    grp_clause <- ""
    if (inherits(fit, "merMod")) {          # lme4 mixed model
      ngr  <- lme4::ngrps(fit)              # named integer vector
      fmt  <- function(x) formatC(x, format = "d", big.mark = ",")
      parts <- paste0(fmt(ngr), " ", names(ngr))
      if (length(parts) == 1) {
        grp_clause <- paste(parts)
      } else if (length(parts) == 2) {
        grp_clause <- paste(parts, collapse = " and ")
      } else {                              # > 2 grouping factors
        grp_clause <- paste(paste(parts[-length(parts)], collapse = ", "),
                            "and", parts[length(parts)])
      }
      grp_sentence <- paste0(
        "The analysis was based on ", tot_n,
        " observations from ", grp_clause, "s.")
    } else {                                # simple lm/glm
      grp_sentence <- paste0(
        "The analysis was based on ", tot_n, " observations.")
    }

    # -------- 4. Glue everything together ------------------------------------
    sprintf(
      "The intercept was %.3f (SE = %.3f, %.0f%% CI [%.3f, %.3f]%s).%s %s",
      est, se, conf * 100, lo, hi, ptxt, ran_sentence, grp_sentence)
  }
"""
)


def one_sentence(fit, conf=0.95):
    """Return a one-sentence summary (Python str) for an intercept-only model."""
    return str(one_sentence_R(fit, conf)[0])


def latex_to_docx(latex: str, docx_path: str | pathlib.Path) -> pathlib.Path:
    """
    Convert `latex` (string) to a DOCX written to `docx_path`.

    Returns the pathlib.Path to the generated file.
    """
    docx_path = pathlib.Path(docx_path)

    # Direct conversion from text (no temp files needed)
    pypandoc.convert_text(latex, to="docx", format="latex", outputfile=str(docx_path))
    return docx_path



