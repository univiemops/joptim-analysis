import numpy as np
import pandas as pd


def zscore_LR(data, left_col, right_col):
    """Compute z-scores for left and right values based on the same
    means and standard deviations."""
    
    all_values = data[[left_col, right_col]].values.flatten()
    combined_mean = np.mean(all_values)
    combined_std = np.std(all_values)
    left_z = (data[left_col] - combined_mean) / combined_std
    right_z = (data[right_col] - combined_mean) / combined_std
    left_right_z = pd.concat([left_z, right_z], axis=1)
    return left_right_z