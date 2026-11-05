import numpy as np
from collections import defaultdict
from joptim.utils.elements import find_element_by_id


def sample_problem_size(problems):
    unique_problem_sizes = list(
        set(
            [
                str(len(problem["task_loads"]))
                + "x"
                + str(len(problem["worker_capacities"]))
                for problem in problems
            ]
        )
    )
    return np.random.choice(unique_problem_sizes)


def sample_stimulus(ids, elements, stratified=True):
    """
    Samples from a list of IDs (stratified or non-stratified by class).

    Parameters:
    ids (list): List of strings.
    elements (list): List of dictionaries where each dictionary contains an
    'id' key.
    stratified (bool): Whether to stratify by class of element (the first part
    of id).

    Returns:
    dict: The sample.
    """
    if stratified:
        classes = np.unique(np.array([id.split("_")[0] for id in ids]))
        class_id = np.random.choice(classes)
        ids_subset = [id for id in ids if id.split("_")[0] == class_id]
        id = np.random.choice(ids_subset)
    else:
        id = np.random.choice(ids)

    sample = find_element_by_id(elements, id)
    if sample is None:
        raise LookupError(f"id {id} not found in elements.")
    else:
        return sample
