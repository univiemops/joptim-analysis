import pandas as pd
import re
from typing import Literal


def best_model(df: pd.DataFrame,
               criterion: Literal["AIC", "BIC"] = "AIC",
               model_col: str = "Model",
               tie_threshold: float = 2.0,
               prefer_re_model: bool = True) -> str:
    """
    Select the 'best' model under the following rules

    1. The model with the smallest information-criterion value wins
       if it is > tie_threshold units better than any competitor.

    2. Otherwise (Δ ≤ tie_threshold), all tied models are considered
       equivalent.  Break the tie by
          a. preferring models that *do* (or *do not*) contain random
             effects, depending on `prefer_re_model`;
          b. if a tie still remains, choosing the model with the fewest
             fixed-effect predictors;
          c. if still tied, the one with the lowest IC (or the first row).

    Parameters
    ----------
    df : DataFrame
        Must contain `model_col` plus columns named 'AIC' and/or 'BIC'.
    criterion : {"AIC", "BIC"}, default "AIC"
        Column used for selection.
    model_col : str, default "model"
        Column holding the model-formula strings.
    tie_threshold : float, default 2.0
        Width of the indistinguishable zone (Δ ≤ threshold).
    prefer_re_model : bool, default True
        If True, prefer models *with* random effects when Δ ≤ threshold.
        If False, prefer models *without* random effects.

    Returns
    -------
    str
        The winning model formula string.
    """

    # --- basic checks ----------------------------------------------------
    if criterion not in df.columns:
        raise ValueError(f"Column '{criterion}' not found in DataFrame.")
    if model_col not in df.columns:
        raise ValueError(f"Column '{model_col}' not found in DataFrame.")

    # --- helper functions -------------------------------------------------
    def _has_random_effects(formula: str) -> bool:
        """Detect presence of a ' | ' term typical for (.. | group)."""
        return "|" in formula

    def _n_fixed_predictors(formula: str) -> int:
        """
        Count predictors on the fixed-effects side of a mixed-model formula.
        Intercept ('1') is not counted.
        """
        rhs = formula.split("~", 1)[1]                 # after '~'
        rhs_fixed = rhs.split("(", 1)[0]               # discard random part
        tokens = re.split(r"\s*\+\s*", rhs_fixed)
        tokens = [t.strip() for t in tokens if t.strip() and t.strip() != "1"]
        return len(tokens)

    # --- 1. find the minimum IC ------------------------------------------
    min_ic = df[criterion].min()

    # --- 2. models within tie_threshold ----------------------------------
    tied = df[df[criterion] - min_ic <= tie_threshold].copy()

    if len(tied) == 1:                 # unique winner
        return tied.iloc[0][model_col]

    # --- 3. apply random-effects preference ------------------------------
    tied["has_re"] = tied[model_col].apply(_has_random_effects)

    desired_re = prefer_re_model        # True → want RE; False → want no RE
    subset = tied[tied["has_re"] == desired_re]

    # If preference eliminates everything, fall back to full tied set
    if subset.empty:
        subset = tied

    # --- 4. choose the simplest model(s) ---------------------------------
    subset = subset.copy()
    subset["simplicity"] = subset[model_col].apply(_n_fixed_predictors)
    simplest = subset[subset["simplicity"] == subset["simplicity"].min()]

    # --- 5. final tie-break: lowest IC (then first row) ------------------
    best_row = simplest.sort_values(by=criterion).iloc[0]

    return best_row[model_col]
