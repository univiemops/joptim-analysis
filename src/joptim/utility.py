import numpy as np


def compute_utility(task_sizes, assign):
    return np.sum(assign.sum(axis=1) * task_sizes)