import numpy as np


def mlm_r2(fit, data):
    """
    Calculate the marginal and conditional R-squared (R²) values for a given model fit and observed true response values.

    The marginal R² describes the proportion of variance explained by the fixed effects in the model, while the conditional
    R² describes the proportion of variance explained by both fixed and random effects.

    Parameters:
    fit (statsmodels object): A fitted model object from statsmodels containing fixed and potentially random effects.
    data (DataFrame): The original data the model was fit to.

    Returns:
    tuple: A tuple containing two float values:
           - marginal_r2 (float): The marginal R-squared value.
           - conditional_r2 (float): The conditional R-squared value.
    
    Implementation of the method of calculating marginal and conditional R² suggested by
    https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/j.2041-210x.2012.00261.x
    """

    var_resid = fit.scale
    var_random_effect = np.sum(np.diag(fit.cov_re))
    var_fixed_effect = fit.predict(data).var()
    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var

    return marginal_r2, conditional_r2


def mlm_r2_alternative(fit, y_true):
    """
    Calculate the marginal and conditional R-squared (R²) values for a given model fit and observed true response values.

    The marginal R² describes the proportion of variance explained by the fixed effects in the model, while the conditional
    R² describes the proportion of variance explained by both fixed and random effects.

    Parameters:
    fit (statsmodels object): A fitted model object from statsmodels containing fixed and potentially random effects.
    y_true (array-like): The true response values.

    Returns:
    tuple: A tuple containing two float values:
           - marginal_r2 (float): The marginal R-squared value.
           - conditional_r2 (float): The conditional R-squared value.

    Example:
    >>> fit_model = sm.MixedLM.from_formula("response ~ predictor", data, groups=data["group"]).fit()
    >>> y_true = data["response"]
    >>> marginal_r2, conditional_r2 = calculate_r2(fit_model, y_true)
    >>> print(f"Marginal R^2: {marginal_r2}")
    >>> print(f"Conditional R^2: {conditional_r2}")
    """

    def ssq(x):
        return np.sum((x - np.mean(x)) ** 2)

    # fitted values fitted using both the fixed and random effects
    # https://www.statsmodels.org/v0.12.2/generated/statsmodels.regression.mixed_linear_model.MixedLMResults.fittedvalues.html?highlight=fittedvalues#statsmodels.regression.mixed_linear_model.MixedLMResults.fittedvalues
    y_pred_mixed = fit.fittedvalues
    # predict() uses only the fixed effects
    # https://www.statsmodels.org/v0.12.2/generated/statsmodels.regression.mixed_linear_model.MixedLM.predict.html
    y_pred_fixed = fit.predict()

    marginal_r2 = 1 - ssq(y_true - y_pred_fixed) / ssq(y_true)
    conditional_r2 = 1 - ssq(y_true - y_pred_mixed) / ssq(y_true)

    return marginal_r2, conditional_r2
