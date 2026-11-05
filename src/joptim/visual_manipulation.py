import numpy as np


def unsort(arr):
    """
    Rearrange the array such that:
    - If the array has only one element, do nothing.
    - If the array has exactly two elements, return elements in ascending order.
    - If the array has three or more elements, rearrange elements to follow a pattern where
      the second element is larger than the first, the third element is smaller than the second, the fourth
      element is larger than the third, etc., creating a "bouncing" up and down pattern.

    Parameters:
    arr (numpy.ndarray): Input array of integers.

    Returns:
    numpy.ndarray: Rearranged array in the specified "unsorted" pattern.

    Examples:
    >>> arr = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
    >>> unsort(arr)
    array([0, 4, 1, 5, 2, 6, 3, 7, 4, 8])
    """
    n = len(arr)
    if n <= 1:
        return arr
    elif n == 2:
        return arr[np.argsort(arr)]
    else:
        # Sort the array
        sorted_arr = np.sort(arr)
        # Create a new array to store the 'unsorted' version
        unsorted_arr = np.zeros(n, dtype=arr.dtype)

        mid = n // 2
        for i in range(mid + 1):
            if 2 * i < n:
                unsorted_arr[2 * i] = sorted_arr[i]
            if 2 * i + 1 < n:
                unsorted_arr[2 * i + 1] = sorted_arr[-(i + 1)]

        return unsorted_arr


def find_indices_for_unsorted(original, unsorted):
    """
    Given an original array and an unsorted array, return the indices of the original array
    that can be used to obtain the unsorted array.

    Parameters:
    original (numpy.ndarray): The original array.
    unsorted   (numpy.ndarray): The array that is a rearranged version of the original array.

    Returns:
    numpy.ndarray: Array of indices from the original array that lead to the unsorted array.

    Examples:
    >>> original = np.array([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
    >>> unsorted = np.array([0, 1, 8, 3, 7, 5, 6, 9, 2, 4])
    >>> find_indices_for_unsorted(original, unsorted)
    array([9, 0, 8, 1, 3, 2, 7, 4, 5, 6])
    """
    # Get the sorted indices of the original array
    original_idx = np.argsort(original)
    # Get the sorted indices of the unsorted array
    unsorted_idx = np.argsort(unsorted)

    # Map the indices using the sorted positions
    index_map = np.zeros_like(original_idx)
    index_map[unsorted_idx] = original_idx

    return index_map


def shuffle_array(arr):
    """
    Permute an array while preserving the relative order between duplicate elements.

    Parameters:
    arr (np.ndarray): The input array to be shuffled.

    Returns:
    np.ndarray: An array of shuffled indices, maintaining the original order among duplicate elements.
    """
    arr = arr.copy()
    unique, counts = np.unique(arr, return_counts=True)
    multiple_values = dict(i for i in zip(unique, counts) if i[1] > 1)
    multiple_values_indices = {i: np.where(arr == i)[0] for i in multiple_values}

    # initial shuffle
    shuffled_indices = np.random.permutation(len(arr))

    # correct the shuffle by making sure that
    # the order among same-value element is stable
    for val in list(multiple_values_indices):
        control_array = multiple_values_indices[val]
        control_i = 0
        for i, el in enumerate(shuffled_indices):
            if el in control_array:
                shuffled_indices[i] = control_array[control_i]
                control_i += 1
    return shuffled_indices
