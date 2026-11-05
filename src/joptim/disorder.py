import numpy as np
from scipy.stats import kendalltau


def disorder(arr):
    """
    Modified kendall's tau (bubble-sort-distance)
    to quantify to what extent the values in an array
    deviate from perfect order.
    """

    arr = np.array(arr)

    rank_asc = np.array(range(len(arr)))
    rank_desc = np.flip(rank_asc)

    # add small number to make sure equal numbers are
    # not penalized by rank correlation
    correction_arr_asc = rank_asc / 100_000
    correction_arr_desc = rank_desc / 100_000

    tau_asc, _ = np.abs(kendalltau(arr + correction_arr_asc, rank_asc))
    tau_desc, _ = np.abs(kendalltau(arr + correction_arr_desc, rank_desc))

    d = np.round(1 - max(tau_asc, tau_desc), 5)

    return d


def disorder_index(w, z):
    """
    Calculates the weighted disorder index of tasks and capacity sizes.

    Parameters:
    w (list): List representing bin sizes.
    z (list): List representing item sizes.

    Returns:
    float: The weighted disorder index.
    """
    d_w = disorder(w)
    d_z = disorder(z)
    total_length = len(w) + len(z)
    return (len(w) / total_length) * d_w + (len(z) / total_length) * d_z
