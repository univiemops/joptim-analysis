import numpy as np


def avg_discrepancy_index(game, sm=None):
    """
    Worker_capacity - max(assigned_task_loads).

    Only those workers that have some tasks
    assigned would contribute to the average discrepancy.

    Accepts an alternative solution matrix `sm`.

    w ... worker capacities
    z ... task loads
    x ... binary decision matrix
    m ... number of workers
    """

    z = game["task_loads"]
    x = game["solution_matrix"] if type(sm) == type(None) else sm
    w = game["worker_capacities"]
    m = len(w)

    return np.mean(
        [(w[i] - max(z * x[:, i])) for i in range(m) if sum(z * x[:, i]) != 0]
    )


def avg_ratio_index(game, sm=None):
    """
    Average of mean(assigned_task_loads)/worker_capacity.

    Only those workers that have some tasks assigned
    contribute to the average ratio.

    Accepts an alternative solution matrix `sm`.

    w ... worker capacities
    z ... task loads
    x ... binary decision matrix
    m ... number of workers
    """

    z = game["task_loads"]
    x = game["solution_matrix"] if type(sm) == type(None) else sm
    w = game["worker_capacities"]
    m = len(w)

    return 1 / np.mean(
        [(np.mean(z * x[:, i]) / w[i]) for i in range(m) if sum(z * x[:, i]) != 0]
    )  # invert it to make it comparable with other measures


def assignment_variance_index(game, sm=None):
    """
    Variance (standard deviation) across workers' number of assigned tasks

    Accepts an alternative solution matrix `sm`.

    w ... worker capacities
    z ... task loads
    x ... binary decision matrix
    m ... number of workers
    """

    z = game["task_loads"]
    x = game["solution_matrix"] if type(sm) == type(None) else sm
    w = game["worker_capacities"]
    m = len(w)

    return np.std([(sum(x[:, i])) for i in range(m)])
