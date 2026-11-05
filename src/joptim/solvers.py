from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
import numpy as np


def mip_solve(worker_capacities, task_loads):
    """
    Run MIP and get status, solution, iterations and costs.
    """

    num_workers = worker_capacities.shape[0]
    num_tasks = task_loads.shape[0]
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Binary decision variables
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[(i, j)] = solver.IntVar(0, 1, f"Assignment_{i}_{j}")

    # Ojective function
    objective_expr = 0
    # Unassigned task costs
    for j in range(num_tasks):
        load_expr = solver.Sum(x[(i, j)] for i in range(num_workers))
        objective_expr += task_loads[j] * (1 - load_expr)

    solver.Minimize(objective_expr)

    # Constraints
    for j in range(num_tasks):
        solver.Add(solver.Sum(x[(i, j)] for i in range(num_workers)) <= 1)
    for i in range(num_workers):
        solver.Add(
            solver.Sum(task_loads[j] * x[(i, j)] for j in range(num_tasks))
            <= worker_capacities[i]
        )

    # Solve
    solver_status = solver.Solve()
    costs = round(solver.Objective().Value())
    iterations = solver.iterations()
    solution_matrix = np.abs(
        np.array(
            [
                [x[(i, j)].solution_value() for j in range(num_tasks)]
                for i in range(num_workers)
            ]
        )
    ).T
    if solver_status == pywraplp.Solver.OPTIMAL:
        status = "OPTIMAL"
    elif solver_status == pywraplp.Solver.FEASIBLE:
        status = "FEASIBLE"
    elif solver_status == pywraplp.Solver.INFEASIBLE:
        status = "INFEASIBLE"
    elif solver_status == pywraplp.Solver.UNBOUNDED:
        status = "UNBOUNDED"
    else:
        status = "Unknown or other"

    return status, solution_matrix, costs, iterations


def mip_solve_max(worker_capacities, task_loads):
    """
    Run MIP and get status, solution, iterations and costs.
    """

    num_workers = worker_capacities.shape[0]
    num_tasks = task_loads.shape[0]
    solver = pywraplp.Solver.CreateSolver("SCIP")

    # Binary decision variables
    x = {}
    for i in range(num_workers):
        for j in range(num_tasks):
            x[(i, j)] = solver.IntVar(0, 1, f"Assignment_{i}_{j}")

    # Ojective function
    objective_expr = 0
    # Assigned task utility
    for j in range(num_tasks):
        load_expr = solver.Sum(x[(i, j)] for i in range(num_workers))
        objective_expr += task_loads[j] * load_expr

    solver.Maximize(objective_expr)

    # Constraints
    for j in range(num_tasks):
        solver.Add(solver.Sum(x[(i, j)] for i in range(num_workers)) <= 1)
    for i in range(num_workers):
        solver.Add(
            solver.Sum(task_loads[j] * x[(i, j)] for j in range(num_tasks))
            <= worker_capacities[i]
        )

    # Solve
    solver_status = solver.Solve()
    utility = round(solver.Objective().Value())
    iterations = solver.iterations()
    solution_matrix = np.abs(
        np.array(
            [
                [x[(i, j)].solution_value() for j in range(num_tasks)]
                for i in range(num_workers)
            ]
        )
    ).T
    if solver_status == pywraplp.Solver.OPTIMAL:
        status = "OPTIMAL"
    elif solver_status == pywraplp.Solver.FEASIBLE:
        status = "FEASIBLE"
    elif solver_status == pywraplp.Solver.INFEASIBLE:
        status = "INFEASIBLE"
    elif solver_status == pywraplp.Solver.UNBOUNDED:
        status = "UNBOUNDED"
    else:
        status = "Unknown or other"

    return status, solution_matrix, utility, iterations


def find_all_optimal_solutions_cp_max(worker_capacities, task_loads, limit=None):
    """
    Find all optimal solutions using CP-SAT solver.
    """

    num_workers = worker_capacities.shape[0]
    num_tasks = task_loads.shape[0]
    previous_solution_matrices = []
    previous_utility = None
    optimal = True
    loop_i = 0
    while optimal:

        model = cp_model.CpModel()

        x = {}
        for i in range(num_workers):
            for j in range(num_tasks):
                x[(i, j)] = model.NewIntVar(0, 1, f"Assignment_{i}_{j}")

        y = {}
        for j in range(num_tasks):
            for k in range(j + 1, num_tasks):
                y[(j, k)] = model.NewIntVar(0, 1, f"Dependency_{j}_{k}")

        # Objective function
        objective_expr = 0
        for j in range(num_tasks):
            load_expr = sum(x[(i, j)] for i in range(num_workers))
            objective_expr += task_loads[j] * load_expr

        model.Maximize(objective_expr)

        # Constraints
        for j in range(num_tasks):
            model.Add(sum(x[(i, j)] for i in range(num_workers)) <= 1)
        for i in range(num_workers):
            model.Add(
                sum(task_loads[j] * x[(i, j)] for j in range(num_tasks))
                <= worker_capacities[i]
            )

        # No previous solutions
        for s in previous_solution_matrices:
            model.Add(
                sum([x[j, i] for i, j in np.array(np.where(s == 1)).T]) != s.sum()
            )

        solver = cp_model.CpSolver()
        solver_status = solver.Solve(model)

        utility = round(solver.ObjectiveValue())

        try:
            solution_matrix = np.abs(
                np.array(
                    [
                        [solver.Value(x[(i, j)]) for j in range(num_tasks)]
                        for i in range(num_workers)
                    ]
                )
            ).T
        except Exception as e:
            print(e)
            solution_matrix = np.zeros((len(task_loads), len(worker_capacities)))

        if solver_status == cp_model.OPTIMAL:
            status = "OPTIMAL"
        elif solver_status == cp_model.FEASIBLE:
            status = "FEASIBLE"
        elif solver_status == cp_model.INFEASIBLE:
            status = "INFEASIBLE"
        elif solver_status == cp_model.UNBOUNDED:
            status = "UNBOUNDED"
        else:
            status = "Unknown or other"

        if previous_utility == None:
            previous_utility = utility  # first solution
            previous_solution_matrices.append(solution_matrix)
        elif previous_utility > utility:  # found a worse solution
            optimal = False
        elif utility > previous_utility:  # found an even better solution
            previous_solution_matrices = [solution_matrix]
            previous_utility = utility
        else:  # found an equally good solution
            previous_solution_matrices.append(solution_matrix)

        # stop loop if the optimal solution is to assign nothing
        if solution_matrix.sum() == 0:
            optimal = False

        if limit != None:
            if loop_i == limit:
                optimal = False

        loop_i += 1

    return previous_solution_matrices


def find_all_optimal_solutions_cp(worker_capacities, task_loads, limit=None):
    """
    Find all optimal solutions using CP-SAT solver.
    """

    num_workers = worker_capacities.shape[0]
    num_tasks = task_loads.shape[0]
    previous_solution_matrices = []
    previous_costs = None
    optimal = True
    loop_i = 0
    while optimal:

        model = cp_model.CpModel()

        x = {}
        for i in range(num_workers):
            for j in range(num_tasks):
                x[(i, j)] = model.NewIntVar(0, 1, f"Assignment_{i}_{j}")

        y = {}
        for j in range(num_tasks):
            for k in range(j + 1, num_tasks):
                y[(j, k)] = model.NewIntVar(0, 1, f"Dependency_{j}_{k}")

        # Objective function
        objective_expr = 0
        for j in range(num_tasks):
            load_expr = sum(x[(i, j)] for i in range(num_workers))
            objective_expr += task_loads[j] * (1 - load_expr)

        model.Minimize(objective_expr)

        # Constraints
        for j in range(num_tasks):
            model.Add(sum(x[(i, j)] for i in range(num_workers)) <= 1)
        for i in range(num_workers):
            model.Add(
                sum(task_loads[j] * x[(i, j)] for j in range(num_tasks))
                <= worker_capacities[i]
            )

        # No previous solutions
        for s in previous_solution_matrices:
            model.Add(
                sum([x[j, i] for i, j in np.array(np.where(s == 1)).T]) != s.sum()
            )

        solver = cp_model.CpSolver()
        solver_status = solver.Solve(model)

        costs = round(solver.ObjectiveValue())

        try:
            solution_matrix = np.abs(
                np.array(
                    [
                        [solver.Value(x[(i, j)]) for j in range(num_tasks)]
                        for i in range(num_workers)
                    ]
                )
            ).T
        except Exception as e:
            print(e)
            solution_matrix = np.zeros((len(task_loads), len(worker_capacities)))

        if solver_status == cp_model.OPTIMAL:
            status = "OPTIMAL"
        elif solver_status == cp_model.FEASIBLE:
            status = "FEASIBLE"
        elif solver_status == cp_model.INFEASIBLE:
            status = "INFEASIBLE"
        elif solver_status == cp_model.UNBOUNDED:
            status = "UNBOUNDED"
        else:
            status = "Unknown or other"

        if previous_costs == None:
            previous_costs = costs  # first solution
            previous_solution_matrices.append(solution_matrix)
        elif previous_costs < costs:  # found a worse solution
            optimal = False
        elif costs < previous_costs:  # found an even better solution
            previous_solution_matrices = [solution_matrix]
            previous_costs = costs
        else:  # found an equally good solution
            previous_solution_matrices.append(solution_matrix)

        # stop loop if the optimal solution is to assign nothing
        if solution_matrix.sum() == 0:
            optimal = False

        if limit != None:
            if loop_i == limit:
                optimal = False

        loop_i += 1

    return previous_solution_matrices


def solve_gap_heuristic_first(w, z):
    """
    Heuristic based on the assumption that people try to assign to
    first task to the first worker first;
    After cycling through all tasks they move to the next worker.

    Returns binary decision matrix.
    """
    x = np.zeros((len(z), len(w)))
    remaining_capacities = w.copy()
    for i, worker in enumerate(w):
        for j, task in enumerate(z):
            task_not_yet_assigned = bool(x[j].sum()) == False
            task_fits_in = task <= remaining_capacities[i]
            if task_fits_in & task_not_yet_assigned:
                x[j, i] = 1
                remaining_capacities[i] -= task

    return x.astype(int)


def solve_gap_heuristic_diag(w, z):
    """
    Heuristic based on the assumption that people try to assign in
    sorted GAPs the largest task to the largest worker first;
    if this is not possible, they move to the next task. If they can
    assign it, they move to the next worker. This creates assignment
    patterns that resemble a diagonal.

    Returns binary decision matrix.
    """
    x = np.zeros((len(z), len(w)))
    for i, worker in enumerate(w):
        for j, task in enumerate(z):
            task_not_yet_assigned = bool(x[j].sum()) == False
            task_fits_in = False if worker - task < 0 else True
            if task_fits_in & task_not_yet_assigned:
                x[j, i] = 1
                break

    return x.astype(int)


def solve_gap_heuristic(w, z):
    """
    Heuristic based on the assumption that people try to assign
    the largest task to the largest worker first;
    After cycling through all tasks they move to the next worker.

    version=1 uses np.argsort(w)[::-1] to compute the temporary order,
    version=2 uses (-1 * w).argsort(). This only matters in cases
    with duplicate values. Use version 1 for unordered problems.

    Returns binary decision matrix.
    """
    x = np.zeros((len(z), len(w)))

    # sort in descending order, but remember the indices

    w_indices = (-1 * w).argsort()
    z_indices = (-1 * z).argsort()

    w = w[w_indices]
    z = z[z_indices]

    remaining_capacities = w.copy()

    # iterate over workers
    for i, worker in enumerate(w):
        # iterate over tasks
        for j, task in enumerate(z):
            task_not_yet_assigned = bool(x[j].sum()) == False
            task_fits_in = task <= remaining_capacities[i]
            if task_fits_in & task_not_yet_assigned:
                x[j, i] = 1
                remaining_capacities[i] -= task

    # use indices and sort back to original
    w_revert_indices = np.argsort(w_indices)
    z_revert_indices = np.argsort(z_indices)

    x = x[:, w_revert_indices][z_revert_indices, :]
    return x.astype(int)
