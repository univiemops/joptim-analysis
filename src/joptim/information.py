import zlib
import numpy as np


def compress_string(s):
    compressed_data = zlib.compress(s.encode())
    return len(compressed_data)


def estimate_complexity(backpacks):
    composition_string = str(backpacks)
    compressed_size = compress_string(composition_string)

    return compressed_size


def compute_compressed_information(x, w, z, weighted=None):
    """
    w ... worker capacities
    z ... task loads
    x ... binary decision matrix
    """
    compositions = np.where(x == 0, 0, z[:, np.newaxis])
    w_rest = w - compositions.sum(axis=0)
    compressed_size = 0

    if weighted == None:
        weights = np.ones(len(w))
    elif weighted == "absolute":
        weights = w / 100
    elif weighted == "relative":
        weights = w / np.sum(w)
    else:
        raise ("weighted argument can only be None, 'relative' or 'absolute'.")

    for i, comp in enumerate(compositions.T):
        comp = comp[comp != 0]
        if w_rest[i] != 0:
            comp = np.append(comp, [w_rest[i]])
        else:
            comp = np.append(comp, [0])

        compressed_size += weights[i] * estimate_complexity(comp)

    return compressed_size
