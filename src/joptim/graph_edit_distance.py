import numpy as np
import networkx as nx


def match(a, b):
    return a["label"] == b["label"]


def matrix_to_graph(binary_matrix):
    G = nx.Graph()

    num_rows, num_cols = binary_matrix.shape

    # Add nodes
    for i in range(1, num_cols+1):
        G.add_node("w"+str(i), label=f"worker {i}")
    for j in range(1, num_rows+1):
        G.add_node("t"+str(j), label=f"task {j}")

    # Add edges
    for i in range(1, num_cols+1):
        for j in range(1, num_rows+1):
            if binary_matrix[j-1][i-1] == 1:
                G.add_edge("w"+str(i), "t"+str(j), label=f"w{i}-t{j}")

    return G


def graph_edit_distance(a, b, simple=True):
    """
    In simple cases the difference
    of the matrices gives us the same result
    as the graph edit distance, but at lower
    computational costs.
    """
    if simple:
        return np.sum(a != b)
    else:
        return nx.graph_edit_distance(
            matrix_to_graph(a),
            matrix_to_graph(b),
            node_match=match,
            edge_match=match
            )


def graph_edit_distance_diagonal(x, simple=True):
    """
    Compute graph edit distance based on the diagonal
    for square matrices, and based on two diagonals
    for rectangular matrices (ged1 = upper, ged2 = lower).
    """
    rows = x.shape[0]
    cols = x.shape[1]
    x1 = np.eye(rows, cols, k=0)
    ged1 = graph_edit_distance(x1, x, simple)
    if rows == cols:
        return ged1, ged1
    else:
        x2 = np.flip(np.eye(rows, cols, k=0))
        ged2 = graph_edit_distance(x2, x, simple)
        return ged1, ged2


def create_diagonal_matrix_approximated(shape):
    rows, cols = shape
    matrix = np.zeros((rows, cols), dtype=int)
    
    for i in range(rows):
        col_index = round(i * (cols - 1) / (rows - 1))
        matrix[i, col_index] = 1
    
    return matrix


def graph_edit_distance_diagonal_approximated(x, simple=True):
    """
    Compute graph edit distance based on the 
    approximated diagonal.
    """
    x1 = create_diagonal_matrix_approximated(x.shape)
    ged = graph_edit_distance(x1, x, simple)
    return ged
