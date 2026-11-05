import numpy as np


def assignments2solution(assignments, tasks, workers):
    """
    Create and compare the solution matrix from the given solution list.

    :param assignments: List of tuples
    :param tasks: List of tasks
    :param workers: List of workers
    :return: Boolean indicating whether solutions match
    """
    solution_matrix = np.zeros((len(tasks), len(workers)), dtype=int)
    for one in assignments:
        solution_matrix[one[0], one[1]] = 1
    return solution_matrix