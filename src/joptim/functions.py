import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from joptim.utility import compute_utility


def get_task_matrix(agent_sizes, task_sizes, assign):
    matrix = agent_sizes.copy()
    for _ in range(len(task_sizes) - 1):
        matrix = np.vstack([matrix, agent_sizes])

    matrix = matrix * assign

    return matrix


def get_agent_matrix(task_sizes, agent_sizes, assign):
    matrix = task_sizes.copy()
    for _ in range(len(agent_sizes) - 1):
        matrix = np.vstack([matrix, task_sizes])

    matrix = matrix.T * assign

    return matrix


def get_point(num):
    # return "   " if num == 0 else " ● "
    return "   " if num == 0 else "▐█▌"


def print_matrix(task_sizes, agent_sizes, assign):
    print("    ", end="")
    for i in range(len(agent_sizes)):
        print("|", str(agent_sizes[i]).rjust(3, " "), end=" ")
    print("|")
    print((6 * len(agent_sizes) + 5) * "-")

    for j in range(len(task_sizes)):
        print(str(task_sizes[j]).rjust(3, " ") + " ", end="")
        for i in range(len(agent_sizes)):
            print("|", get_point(assign[j, i]), end=" ")
        print("|")
        print((6 * len(agent_sizes) + 5) * "-")

    print("utility =", compute_utility(task_sizes, assign))

    return None


def print_two_matrices(
    task_sizes1, agent_sizes1, assign1, task_sizes2, agent_sizes2, assign2
):

    # Print the header for both matrices
    print("    ", end="")
    for i in range(len(agent_sizes1)):
        print("|", str(agent_sizes1[i]).rjust(3, " "), end=" ")
    print("|        ", end="")  # Space between two matrices
    for i in range(len(agent_sizes2)):
        print("|", str(agent_sizes2[i]).rjust(3, " "), end=" ")
    print("|")

    # Print the divider line between header and matrix content for both matrices
    print(
        (6 * len(agent_sizes1) + 5) * "-" + "    " + (6 * len(agent_sizes2) + 5) * "-"
    )

    # Print the rows of both matrices
    for j in range(len(task_sizes1)):
        print(
            str(task_sizes1[j]).rjust(3, " ") + " ", end=""
        )  # Task size for first matrix
        for i in range(len(agent_sizes1)):
            print("|", get_point(assign1[j, i]), end=" ")
        print("|", end="")  # End of first matrix row

        print(
            "    " + str(task_sizes2[j]).rjust(3, " ") + " ", end=""
        )  # Task size for second matrix
        for i in range(len(agent_sizes2)):
            print("|", get_point(assign2[j, i]), end=" ")
        print("|")  # End of second matrix row

        print(
            (6 * len(agent_sizes1) + 5) * "-"
            + "    "
            + (6 * len(agent_sizes2) + 5) * "-"
        )

    # Print utility for the first matrix
    print("utility =", compute_utility(task_sizes1, assign1), end=" ")

    for i in range(len(agent_sizes1)):
        print("    ", end=" ")

    # Print utility for the second matrix
    print("utility =", compute_utility(task_sizes2, assign2))

    return None


def validate(agent_sizes, task_sizes, assign, verbose=False):
    # for 1: task_size <= agent_size
    agent_matrix = get_agent_matrix(task_sizes, agent_sizes, assign)

    # for 2: one task can be assigned to only 1 agent
    task_matrix = get_task_matrix(agent_sizes, task_sizes, assign)

    # 1
    if np.sum(agent_sizes - agent_matrix.sum(axis=0) < 0) != 0:
        if verbose is True:
            print("agent overload!")
        return False

    # 2.a: one task can be assigned to 1 or 0 agents
    elif len(set(assign.sum(axis=1)) - {0, 1}) != 0:
        if verbose is True:
            print("tasks were assigned more than once!")
        return False

    # 2.b: sum has to be 0 or the tasks size
    elif np.sum((agent_matrix.sum(axis=1) - task_sizes) * task_matrix.sum(axis=1)) != 0:
        if verbose is True:
            print("tasks were partly assigned!")

        return False

    else:
        return True


def plot_problem_ax(problem, ax, row=None, col=None):
    """
    Takes in ax object, adds plot and return ax. Useful for plots
    with multiple axes.
    """

    fontsize_ticklabel = 11
    fontsize_axislabel = 15
    fontsize_note = 10
    x_tick_labels = problem["worker_capacities"]
    y_tick_labels = problem["task_loads"]
    ax.imshow(problem["solution_matrix"], cmap="binary", interpolation="none")
    ax.set_xticks(
        np.arange(len(x_tick_labels)), x_tick_labels, fontsize=fontsize_ticklabel
    )
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(
        np.arange(len(y_tick_labels)), y_tick_labels, fontsize=fontsize_ticklabel
    )
    ax.tick_params(axis="both", which="both", length=0)
    title = f"s={problem['n_optimal_solutions']}, r={problem['capacity_load_ratio']:.2f},\ni={problem['iterations']}, u={problem['utility']}"

    ax.text(
        0,
        -0.05,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        # fontweight="bold",
        fontsize=fontsize_note,
    )
    # Add gridlines around the individual cells (pixels)
    for i in range(len(x_tick_labels)):
        for j in range(len(y_tick_labels)):
            ax.axvline(x=i - 0.5, color="gray", linewidth=0.5)
            ax.axhline(y=j - 0.5, color="gray", linewidth=0.5)

    ax.xaxis.set_label_position("top")
    if (row == 0) & (col == 1):
        ax.set_xlabel("Workers".upper(), fontsize=fontsize_axislabel, labelpad=12)
    if (col == 0) & (row == 1):
        ax.set_ylabel("Tasks".upper(), fontsize=fontsize_axislabel, labelpad=5)

    return ax


def plot_problem(problem):
    """
    Wrapper around plot_problem_ax. For single plots.
    """

    ax = plt.subplot()
    plot_problem_ax(problem, ax, 0, 0)

    return None


def plot_problem_experiment_ax(
    ax,
    problem,
    directory,
    index,
    annotated=True,
    solved=True,
    solution=None,
    row=None,
    col=None,
    title=None,
):
    """
    Takes in ax object, adds plot and return ax. Useful for plots
    with multiple axes.
    """
    a = "_annotated" if annotated else ""
    prefix = f"{len(problem['task_loads'])}x{len(problem['worker_capacities'])}"
    suffix = "solved" if solved else "unsolved"
    solution_str = f"_{solution}" if solution is not None else ""
    suffix += solution_str
    filepath = Path(
        directory, f"png{a}", f"gap_{prefix}_{str(index).zfill(3)}_{suffix}.png"
    )
    ax.imshow((plt.imread(filepath)))
    ax.axis("off")
    if title is not None:
        ax.set_title(title)

    return ax


def get_bin_labels(data, n_bins):
    bin_step = (data.max() - data.min() + 1) / n_bins
    bin_boundaries = [round((1 + i) * bin_step, 2) for i in range(n_bins)]
    return np.digitize(data, bin_boundaries)


def negative_to_positive_index(negative_index, list_length):
    return negative_index + list_length
