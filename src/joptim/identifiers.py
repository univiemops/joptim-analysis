import numpy as np


def binary_to_base26(binary_str):
    """
    Converts a binary string to a base-26 string using letters A-Z.

    Args:
        binary_str (str): The binary string to convert.

    Returns:
        str: The base-26 encoded string.
    """
    # Convert the binary string to an integer
    num = int(binary_str, 2)
    base26_str = ""
    while num > 0:
        num, remainder = divmod(num, 26)
        base26_str = chr(65 + remainder) + base26_str
    return base26_str or "A"  # 'A' represents 0


def base26_to_binary(base26_str, binary_len):
    """
    Converts a base-26 string using letters A-Z back to a binary string.

    Args:
        base26_str (str): The base-26 string to convert.
        binary_len (int): The length of the binary string.

    Returns:
        str: The binary string.
    """
    num = 0
    for char in base26_str:
        num = num * 26 + (ord(char) - 65)
    return bin(num)[2:].zfill(binary_len)


def matrix_to_identifier(matrix):
    """
    Converts a binary matrix to a human-readable identifier using base-26.

    Args:
        matrix (np.ndarray): A 2D numpy array with binary values (0s and 1s).

    Returns:
        str: A human-readable string identifier for the given matrix.
    """
    row_len = matrix.shape[1]
    identifier = ""
    for row in matrix:
        # Convert each row to a binary string
        binary_str = "".join(str(int(bit)) for bit in row)
        # Convert the binary string to a base-26 string
        base26_str = binary_to_base26(binary_str)
        # Append the base-26 string to the identifier
        identifier += base26_str
    return identifier + "-" + str(row_len)


def identifier_to_matrix(identifier):
    """
    Reconstructs a binary matrix from its human-readable identifier using base-26.

    Args:
        identifier (str): The string identifier to be converted back to a matrix.

    Returns:
        np.ndarray: The reconstructed binary matrix.
    """
    matrix = []
    identifier, row_len = identifier.split("-")
    row_len = int(row_len)
    # Calculate the length of each base-26 encoded row
    base26_row_length = len(binary_to_base26("0" * row_len))
    for i in range(0, len(identifier), base26_row_length):
        base26_str = identifier[i : i + base26_row_length]
        # Convert the base-26 string to a binary string
        binary_str = base26_to_binary(base26_str, row_len)
        row = [int(bit) for bit in binary_str]
        matrix.append(row)
    return np.array(matrix)


def encode_problem_id(w, z):
    """
    Generates a unique identifier for a problem based on worker capacities and task loads.

    Args:
        w (Union[np.ndarray, list]): A 1D array or list containing integer values representing worker capacities.
        z (Union[np.ndarray, list]): A 1D array or list containing integer values representing task loads.

    Returns:
        str: A string identifier composed of concatenated worker capacities and task loads, separated by '-' and '|'.
    """
    concat_str = "-".join(map(str, w)) + "|" + "-".join(map(str, z))
    return concat_str


def decode_problem_id(problem_id):
    """
    Decodes a string identifier into worker capacities and task loads.

    Args:
        problem_id (str): A string identifier consisting of concatenated worker capacities and task loads,
                          separated by '-' and '|'.

    Returns:
        tuple: A tuple containing two lists:
            - w (list): A list of integers representing worker capacities.
            - z (list): A list of integers representing task loads.
    """
    w_part, z_part = problem_id.split("|")
    w = list(map(int, w_part.split("-")))
    z = list(map(int, z_part.split("-")))
    return w, z
