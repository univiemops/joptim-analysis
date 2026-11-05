import numpy as np
from collections import Counter
from joptim.utils.representations import assignments2solution


def test_workers_tasks(ar_sample, order_sample, ar_result):
    """
    Compare workers' tasks arrays to check for equality.

    :param ar_sample: Array of sample tasks/workers
    :param order_sample: Array of sample order indices
    :param ar_result: Result array from user data
    :return: Boolean indicating whether arrays are equal
    """
    isequal = np.array_equal(ar_sample[order_sample], ar_result)
    return isequal


def test_solutions(sample_trial, result_trial):
    """
    Test if sample solutions match with the result solutions.

    :param sample_trial: Sample trial data
    :param result_trial: Result trial data from user
    :return: Tuple of booleans indicating equality for two solutions
    """
    solution1_matrix_sample = sample_trial["solution_1"][sample_trial["task_order_1"]][
        :, sample_trial["worker_order_1"]
    ]
    solution1_matrix = assignments2solution(
        result_trial["solution1"], result_trial["tasks1"], result_trial["workers1"]
    )
    isequal_1 = np.array_equal(solution1_matrix, solution1_matrix_sample)

    solution2_matrix_sample = sample_trial["solution_2"][sample_trial["task_order_2"]][
        :, sample_trial["worker_order_2"]
    ]
    solution2_matrix = assignments2solution(
        result_trial["solution2"], result_trial["tasks2"], result_trial["workers2"]
    )
    isequal_2 = np.array_equal(solution2_matrix, solution2_matrix_sample)

    return isequal_1, isequal_2


def filter_by_frequency(id_list, k):
    # Count the frequency of each integer in the list
    frequency = Counter(id_list)

    # Filter the list to only include integers that appear at least k times
    result = [number for number in id_list if frequency[number] >= k]

    # Optional: Remove duplicates to get unique numbers that meet the criteria
    result = list(set(result))

    return result
