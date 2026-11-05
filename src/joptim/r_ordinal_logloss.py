import pandas as pd
import numpy as np
from rpy2.robjects import Formula, r, pandas2ri
from rpy2.robjects.packages import importr

# import R package 'ordinal'
ordinal = importr("ordinal")

pandas2ri.activate()


def compute_logloss(outcome, predictor):
    """
    Compute the average log loss for a non-hierarchical model using ordinal logistic regression.

    Parameters:
    - outcome (pandas.Series): The observed categorical outcomes.
    - predictor (pandas.Series): The predictor variable.

    Returns:
    - float: The average log loss value between the predicted and actual outcomes.
    """

    df_r = pd.DataFrame.from_dict(
        {"outcome": outcome, "predictor": predictor},
    )
    r.assign("df", df_r)
    r.assign("df_predict", df_r.drop("outcome", axis=1))

    clmm_fit = r(
        "fit <- clm(outcome ~ predictor, data = df, link = 'logit', Hess = TRUE, model = TRUE, threshold = 'symmetric')"
    )
    predicted_probs = r(
        """pred_probs <- as.data.frame(predict(fit, newdata=df_predict))"""
    )
    actual_outcomes = pd.get_dummies(outcome)

    log_loss_values = -np.sum(
        actual_outcomes.to_numpy() * np.log(predicted_probs), axis=1
    )
    average_log_loss = np.mean(log_loss_values)

    return average_log_loss


def compute_logloss_hierarchical(outcome, predictor, group):
    """
    Compute the average log loss for a hierarchical model using ordinal logistic regression.

    Parameters:
    - outcome (pandas.Series): The observed categorical outcomes.
    - predictor (pandas.Series): The predictor variable.
    - group (pandas.Series): Grouping factor for hierarchical modeling.

    Returns:
    - float: The average log loss value between the predicted and actual outcomes.
    """

    df_r = pd.DataFrame.from_dict(
        {"outcome": outcome, "predictor": predictor, "group": group},
    )
    r.assign("df", df_r)
    r.assign("df_predict", df_r.drop("outcome", axis=1))

    clmm_fit = r(
        "fit <- clmm(outcome ~ predictor + (1 | group), data = df, link = 'logit', Hess = TRUE, model = TRUE, threshold = 'symmetric')"
    )

    # Extract model parameters
    beta = r("beta <- coef(fit)")
    theta = r("theta <- fit$Theta")  # Thresholds

    # Define data points to predict
    r("new_data_values <- df$predictor")

    # Initialize a matrix to store probabilities
    r(
        "category_probabilities <- matrix(nrow = length(new_data_values), ncol = length(theta) + 1)"
    )
    r("colnames(category_probabilities) <- levels(df$outcome)")

    # Function to calculate probabilities
    r("logit_inverse <- function(x) 1 / (1 + exp(-x))")

    # Compute probabilities for each new predictor value
    r(
        """for (i in seq_along(new_data_values)) {
      linear_predictor_fixed <- beta['predictor'] * new_data_values[i]
      cumulative_probs <- sapply(theta, function(th) logit_inverse(th - linear_predictor_fixed))
      category_probabilities[i, ] <- c(cumulative_probs[1], diff(cumulative_probs), 1 - tail(cumulative_probs, 1))
    }"""
    )

    # Convert matrix to data frame for easier viewing
    category_probabilities = r(
        "category_probabilities <- data.frame(category_probabilities)"
    )

    # rename columns to match columns in actual_outcomes
    category_probabilities = category_probabilities.rename(
        columns={col: col.replace(".", " ") for col in category_probabilities.columns}
    )

    actual_outcomes = pd.get_dummies(outcome)

    log_loss_values = -np.sum(
        actual_outcomes.to_numpy() * np.log(category_probabilities), axis=1
    )
    average_log_loss = np.mean(log_loss_values)

    return average_log_loss
