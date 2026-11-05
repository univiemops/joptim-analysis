#!/usr/bin/env python
# coding: utf-8

# Create GAP Pairs
# ===
#
# Note: This requires the pickle file `problems_filtered_incl_solution_metrics`
# which is created in `filter_extend_solutions.ipynb`

# This Notebook/Script creates the *preference trials* and the
# *solving trials* for the experiment.

import os

import argparse
import pickle
import copy
from datetime import datetime as dt
from collections import defaultdict
import logging
from datetime import datetime

import numpy as np
from joptim.visual_manipulation import shuffle_array
from joptim.sampling import sample_stimulus, sample_problem_size
from joptim.functions import negative_to_positive_index
from joptim.disorder import disorder_index
from joptim.graph_edit_distance import graph_edit_distance
from joptim.solvers import solve_gap_heuristic
from joptim.utils.log_config import setup_logging
from joptim.utils.export_path import get_export_paths


def load_trial_data_from_pickle(file_path):
    try:
        with open(file_path, "rb") as file:
            trial_data = pickle.load(file)
            array = trial_data["trials"]
            file_id = trial_data["file_id"]
            return array, file_id
    except FileNotFoundError:
        return []  # If the file doesn't exist, return an empty list


def save_trial_data_to_pickle(file_path, array, file_id):
    with open(file_path, "wb") as file:
        trial_data = {"file_id": file_id, "trials": array}
        pickle.dump(trial_data, file)


def main(args):

    setup_logging(os.path.basename(__file__))

    # define constants
    EXPORT_PATH_DATA, _, _ = get_export_paths()

    logging.info("Trial generation started")
    logging.info("Data will be stored in: %s", EXPORT_PATH_DATA)
    PICKLE_INPUT_FILE = "problems_filtered_incl_solution_metrics.pkl"

    if args.confirmatory:
        PICKLE_OUTPUT_FILE = "trials_confirmatory.pkl"
        if args.simple:
            C_COMPLEX = "composition_complexity_v1_confirmatory"
        else:
            C_COMPLEX = "composition_complexity_v2_confirmatory"
    else:
        PICKLE_OUTPUT_FILE = "trials_exploratory.pkl"
        if args.simple:
            C_COMPLEX = "composition_complexity_v1"
        else:
            C_COMPLEX = "composition_complexity_v2"

    A_COMPLEX = "assignment_complexity"
    DIFFICULTY_METRIC = "capacity_load_ratio"
    N_SAMPLES_PER_DIFFICULTY = 2  # extreme trials
    N_RANDOM_TRIALS = 3
    N_RANDOM_SAME_TRIALS = 2
    N_COHERENCE_TRIALS = 3
    N_CATCH_TRIALS = 2
    N_SOLVING_TRIALS = 7
    n_participants = args.n_participants[0]

    # prepare files

    with open(
        Path(EXPORT_PATH_DATA, PICKLE_INPUT_FILE),
        "rb",
    ) as pickle_file:
        problems = pickle.load(pickle_file)
        logging.info(f"{len(problems)} problem instances loaded.")

    if args.new:
        logging.info(
            f"New pickle file '{PICKLE_OUTPUT_FILE}' to create trials for n_participants: {n_participants}"
        )
        trials = []
        file_id = datetime.now().timestamp()

    elif args.append:
        # Append n_participants to the existing pickle file

        logging.info(
            f"Existing pickle file '{PICKLE_OUTPUT_FILE}' will be used to add trials for n_participants: {n_participants}"
        )
        trials, file_id = load_trial_data_from_pickle(
            Path(EXPORT_PATH_DATA, PICKLE_OUTPUT_FILE)
        )

        logging.info(f"Existing trials: {len(trials)}")

    # prepare difficulty data

    log_transform = False
    offset = 1  # if log_transform and min value is 0

    difficulty_data = np.array([problem[DIFFICULTY_METRIC] for problem in problems])

    if log_transform:
        if difficulty_data.min() == 0:
            difficulty_data += offset
        bin_edges = np.logspace(
            np.log10(difficulty_data.min()),
            np.log10(difficulty_data.max()),
            num=4,
        )
    else:
        bin_edges = np.quantile(difficulty_data, [0, 1 / 3, 2 / 3, 1])

    labels = ["low", "medium", "high"]

    # Assign bins to problems

    for problem in problems:
        for k, label in enumerate(labels):
            if bin_edges[k] <= problem[DIFFICULTY_METRIC] <= bin_edges[k + 1]:
                problem["difficulty"] = label

    # Make sure every problem got assigned a difficulty label

    for i, problem in enumerate(problems):
        try:
            _ = problem["difficulty"]
        except Exception as e:
            logging.warning(f"Problem {i} is missing difficulty label: {e}")

    # ### 1.3. Create All Possible Pairs (= Stimuli)
    #
    # Iterate over all problems and create stimuli (=pairs) right there,
    # add them to a stimuli array.

    def generate_stimuli(problems):
        """
        Generate stimuli from a list of problems in a memory-efficient manner.
        """

        for problem in problems:
            # Pre-compute task and worker orders outside of the loops
            task_order = np.argsort(-1 * problem["task_loads"])
            worker_order = np.argsort(-1 * problem["worker_capacities"])

            for sol1 in range(problem["n_optimal_solutions"]):
                for sol2 in range(sol1 + 1, problem["n_optimal_solutions"]):
                    diff_ac = np.abs(
                        problem[A_COMPLEX][sol1] - problem[A_COMPLEX][sol2]
                    )
                    diff_cc = np.abs(
                        problem[C_COMPLEX][sol1] - problem[C_COMPLEX][sol2]
                    )
                    yield {
                        "id": f"{problem['id']}_{sol1}_{sol2}",
                        DIFFICULTY_METRIC: problem[DIFFICULTY_METRIC],
                        "difficulty": problem["difficulty"],
                        "diff_ac": diff_ac,
                        "diff_cc": diff_cc,
                        "task_loads": problem["task_loads"],
                        "worker_capacities": problem["worker_capacities"],
                        "task_order_1": task_order,
                        "task_order_2": task_order,  # same as task_order_1
                        "worker_order_1": worker_order,
                        "worker_order_2": worker_order,  # same as worker_order_1
                        "solution_1": problem["optimal_solutions"][sol1],
                        "solution_2": problem["optimal_solutions"][sol2],
                        "utility": problem["utility"],
                    }

    logging.info("Creating all possible stimuli ...")
    stimuli = list(generate_stimuli(problems))
    logging.info("%s stimuli created.", len(stimuli))

    # ### 1.4. Create Buckets to sample one random stimulus from each bucket
    # for each metric

    difficulty_labels = ["low", "medium", "high"]

    quantile_ids_all = {}

    for metric in ["diff_cc", "diff_ac"]:
        quantile_ids_all[metric] = {}
        for k, difficulty_label in enumerate(difficulty_labels):

            ids = np.array(
                [
                    stimulus["id"]
                    for stimulus in stimuli
                    if stimulus["difficulty"] == difficulty_label
                ]
            )

            data = np.array(
                [
                    stimulus[metric]
                    for stimulus in stimuli
                    if stimulus["difficulty"] == difficulty_label
                ]
            )
            quantile = np.quantile(data, q=0.9)
            quantile_idx = data >= quantile
            ids_filtered = ids[quantile_idx]
            quantile_ids_all[metric][difficulty_label] = ids_filtered

    # ## 0. Create trials that are the same for all participants

    # ### 0.3. Coherence Trials

    diff_level = "medium"
    metric = C_COMPLEX
    quantile = 0.95

    # 1. decide which problems

    problems_coherence = [
        problem
        for problem in problems
        if (problem["difficulty"] == diff_level) & (problem["n_optimal_solutions"] >= 3)
    ]

    dist = []

    for sample in problems_coherence:
        metric_range = max(sample[metric]) - min(sample[metric])
        sample[metric + "_range"] = metric_range
        sample[metric + "_asymmetry"] = np.abs(
            np.median(sample[metric]) - np.mean(sample[metric])
        )
        dist.append(metric_range)

    # Top 5% in range

    cutoff = np.quantile(dist, quantile)
    bool_idx = [val >= cutoff for val in dist]

    problems_coherence = list(np.array(problems_coherence)[bool_idx])

    # Pick the one most symmetric (mean close to median)

    dist = [sample[metric + "_asymmetry"] for sample in problems_coherence]
    problem_coherence = problems_coherence[np.argmin(dist)]

    del problems_coherence

    assert isinstance(problem_coherence, dict)

    # 2.  decide which solutions

    # Pick the max, min and median solution in terms of the chosen metric

    indices_solution = (
        np.argmin(problem_coherence[metric]),
        np.argmax(problem_coherence[metric]),
        np.argsort(problem_coherence[metric])[len(problem_coherence[metric]) // 2],
    )

    samples_coherence = []

    for i, sol1 in enumerate(indices_solution):
        for j, sol2 in enumerate(indices_solution):
            if i < j:
                solution_1 = problem_coherence["optimal_solutions"][sol1]
                solution_2 = problem_coherence["optimal_solutions"][sol2]
                id = problem_coherence["id"] + "_" + str(sol1) + "_" + str(sol2)

                diff_ac = np.abs(
                    problem_coherence[A_COMPLEX][sol1]
                    - problem_coherence[A_COMPLEX][sol2]
                )
                diff_cc = np.abs(
                    problem_coherence[C_COMPLEX][sol1]
                    - problem_coherence[C_COMPLEX][sol2]
                )

                samples_coherence.append(
                    {
                        "id": id,
                        "capacity_load_ratio": problem_coherence["capacity_load_ratio"],
                        "difficulty": problem_coherence["difficulty"],
                        "diff_ac": diff_ac,
                        "diff_cc": diff_cc,
                        "use_case": "coherence",
                        "task_loads": problem_coherence["task_loads"],
                        "worker_capacities": problem_coherence["worker_capacities"],
                        "task_order_1": np.argsort(
                            -1 * problem_coherence["task_loads"]
                        ),
                        "task_order_2": np.argsort(
                            -1 * problem_coherence["task_loads"]
                        ),
                        "worker_order_1": np.argsort(
                            -1 * problem_coherence["worker_capacities"]
                        ),
                        "worker_order_2": np.argsort(
                            -1 * problem_coherence["worker_capacities"]
                        ),
                        "solution_1": solution_1,
                        "solution_2": solution_2,
                        "visual_manipulation": "none",
                        "utility": problem_coherence["utility"],
                    }
                )

    # ### 0.4. Catch Trials

    # arbitrary indices. predefined to keep it fixed for every participant
    indices_problem = [163, 10]
    indices_solution = [0, -1]

    assert N_CATCH_TRIALS == len(indices_problem)
    assert N_CATCH_TRIALS == len(indices_solution)

    samples_catch = []

    # 1. decide which problems

    for i, problem in enumerate(problems):
        if problem["difficulty"] == "low":
            samples_catch.append(problem)
        samples_catch = [
            problem for problem in problems if problem["difficulty"] == "low"
        ]

    # reduce to N_CATCH_TRIALS
    samples_catch = list(np.array(samples_catch)[indices_problem])

    # 2. decide which solutions

    for i, sample in enumerate(samples_catch):

        sol_id = indices_solution[i]
        if sol_id < 0:
            sol_id = negative_to_positive_index(
                sol_id, len(sample["optimal_solutions"])
            )
        solution_1 = sample["optimal_solutions"][sol_id]
        solution_2 = sample["optimal_solutions"][sol_id]
        id = sample["id"] + "_" + str(sol_id) + "_" + str(sol_id)

        # redefine sample

        samples_catch[i] = {
            "id": id,
            "capacity_load_ratio": sample["capacity_load_ratio"],
            "difficulty": sample["difficulty"],
            "diff_ac": 0.0,
            "diff_cc": 0.0,
            "use_case": "catch",
            "task_loads": sample["task_loads"],
            "worker_capacities": sample["worker_capacities"],
            "task_order_1": np.argsort(-1 * sample["task_loads"]),
            "task_order_2": np.argsort(-1 * sample["task_loads"]),
            "worker_order_1": np.argsort(-1 * sample["worker_capacities"]),
            "worker_order_2": np.argsort(-1 * sample["worker_capacities"]),
            "solution_1": solution_1,
            "solution_2": solution_2,
            "visual_manipulation": "none",
            "utility": sample["utility"],
        }

    # ## 0.5. Solving Trials

    dist = [problem[DIFFICULTY_METRIC] for problem in problems]

    step_size = 1 / N_SOLVING_TRIALS
    quantiles = (
        np.array([(i + 1) * step_size for i in range(N_SOLVING_TRIALS)]) - step_size / 2
    )
    quantile_vals = np.quantile(
        dist,
        quantiles,
    )
    quantile_indices = [(np.abs(dist - i)).argmin() for i in quantile_vals]
    samples_solving = copy.deepcopy(list(np.array(problems)[quantile_indices]))

    # pick an arbitrary example solution, e.g. [0]
    # and remove the optimal_solutions

    for i, _ in enumerate(samples_solving):
        samples_solving[i]["example_solution"] = samples_solving[i][
            "optimal_solutions"
        ][0]
        # remove everything that relates to individual other optimal
        # solutions
        del samples_solving[i]["assignment_complexity"]
        del samples_solving[i]["assignment_complexity_diag1"]
        del samples_solving[i]["assignment_complexity_diag2"]
        del samples_solving[i]["assignment_complexity_diag3"]
        del samples_solving[i]["composition_complexity_v1"]
        del samples_solving[i]["composition_complexity_v2"]
        del samples_solving[i]["optimal_solutions"]

    for participant_i in range(n_participants):

        starttime = dt.now()

        # log info
        logging.info(f"Participant {participant_i+1:4}/{n_participants}")

        # ## 1. Extreme Trials

        # #### 1.4.2. Draw the Samples
        #
        # In order not to introduce a bias towards certain problem instances
        # that have many optimal solutions and are therefore over-represented
        # in the individual bins, we will ensure that each problem instance has
        # the same chance of being sampled. That is, the problem is chosen
        # first, then the stimulus.

        samples_extreme = []

        for metric in list(quantile_ids_all):
            for difficulty in list(quantile_ids_all[metric]):
                for _ in range(N_SAMPLES_PER_DIFFICULTY):
                    # sample a size
                    problem_size = sample_problem_size(problems)
                    # filter to stimuli of specific size
                    stimuli_subset = [
                        stimulus
                        for stimulus in stimuli
                        if str(len(stimulus["task_loads"]))
                        + "x"
                        + str(len(stimulus["worker_capacities"]))
                        == problem_size
                    ]
                    # get the ids of all samples for specific metric and
                    # specific bin
                    ids = quantile_ids_all[metric][difficulty]
                    # filter ids to specific size
                    stimuli_subset_ids = set(
                        [stimulus["id"] for stimulus in stimuli_subset]
                    )
                    ids_subset = list(set(ids) & stimuli_subset_ids)  # intersection
                    # sample the stimulus
                    sample = sample_stimulus(
                        ids_subset, stimuli_subset, stratified=True
                    )
                    sample["visual_manipulation"] = "none"
                    sample["use_case"] = "extreme_" + metric
                    samples_extreme.append(sample)

        # add extreme samples to the more general samples collector
        samples_preference = samples_extreme
        del samples_extreme

        # ## 2. Random Trials
        #
        # 2x 3 trials.
        #
        # ### 2.1. Implement Sampling Stratified by Problem Difficulty

        # 1. define a random uniform distribution of difficulty levels
        # 2. filter stimuli to difficulty
        # 3. sample a problem size
        # 4. sample a problem
        # 5. sample a stimulus

        # 1.

        difficulty_labels_ = difficulty_labels.copy()

        np.random.shuffle(difficulty_labels_)
        factor = int(np.ceil(N_RANDOM_TRIALS / len(difficulty_labels_)))
        cutoff = factor * len(difficulty_labels_) - N_RANDOM_TRIALS
        if cutoff > 0:
            sampled_difficulties = (difficulty_labels_ * factor)[:-cutoff]
        else:
            sampled_difficulties = difficulty_labels_ * factor

        samples_random = []

        for difficulty in sampled_difficulties:
            # 2.
            stimuli_subset = [
                stimulus for stimulus in stimuli if stimulus["difficulty"] == difficulty
            ]
            # 3.
            problem_size = sample_problem_size(stimuli_subset)
            # 4.
            stimuli_subset = [
                stimulus
                for stimulus in stimuli_subset
                if str(len(stimulus["task_loads"]))
                + "x"
                + str(len(stimulus["worker_capacities"]))
                == problem_size
            ]
            problem_ids = list(
                set([stimulus["id"].split("_")[0] for stimulus in stimuli_subset])
            )
            problem_id = np.random.choice(problem_ids)

            # filter to problem id
            stimuli_subset = [
                stimulus
                for stimulus in stimuli_subset
                if stimulus["id"].split("_")[0] == problem_id
            ]
            # 5. sample a stimulus

            samples_random.append(np.random.choice(stimuli_subset))

        for sample in samples_random:
            sample["use_case"] = "random"
            sample["visual_manipulation"] = "none"

        # ### 2.2. Duplicate Trials and Apply Visual Manipulation Randomly

        visual_manipulation_types = ["both", "left", "right"]
        visual_manipulations_to_apply = np.random.choice(
            visual_manipulation_types, N_RANDOM_TRIALS, replace=False
        )

        # Create a separate sample for the manipulated problems and solutions
        # and assign the manipulation type

        samples_random_cloned = copy.deepcopy(samples_random)

        for i, sample in enumerate(samples_random_cloned):
            sample["visual_manipulation"] = visual_manipulations_to_apply[i]
            if sample["visual_manipulation"] == "both":
                # left and right get the same visual manipulation/same indices
                samples_random_cloned[i]["task_order_1"] = shuffle_array(
                    samples_random_cloned[i]["task_loads"]
                )
                samples_random_cloned[i]["task_order_2"] = samples_random_cloned[i][
                    "task_order_1"
                ]
                samples_random_cloned[i]["worker_order_1"] = shuffle_array(
                    samples_random_cloned[i]["worker_capacities"]
                )
                samples_random_cloned[i]["worker_order_2"] = samples_random_cloned[i][
                    "worker_order_1"
                ]
            elif sample["visual_manipulation"] == "left":
                # only left side gets shuffle
                samples_random_cloned[i]["task_order_1"] = shuffle_array(
                    samples_random_cloned[i]["task_loads"]
                )
                samples_random_cloned[i]["worker_order_1"] = shuffle_array(
                    samples_random_cloned[i]["worker_capacities"]
                )
            elif sample["visual_manipulation"] == "right":
                # only right side gets shuffle
                samples_random_cloned[i]["task_order_2"] = shuffle_array(
                    samples_random_cloned[i]["task_loads"]
                )
                samples_random_cloned[i]["worker_order_2"] = shuffle_array(
                    samples_random_cloned[i]["worker_capacities"]
                )
            else:
                raise NotImplementedError(
                    """Visual manipulation not implemented:"""
                    f"""{sample["visual_manipulation"]}"""
                )

            # re-calculate assignment complexity
            heuristic_solution_1 = solve_gap_heuristic(
                sample["worker_capacities"][sample["worker_order_1"]],
                sample["task_loads"][sample["task_order_1"]],
            )
            heuristic_solution_2 = solve_gap_heuristic(
                sample["worker_capacities"][sample["worker_order_2"]],
                sample["task_loads"][sample["task_order_2"]],
            )

            solution_1_visual = sample["solution_1"][sample["task_order_1"]][
                :, sample["worker_order_1"]
            ]
            solution_2_visual = sample["solution_2"][sample["task_order_2"]][
                :, sample["worker_order_2"]
            ]

            ged_1 = graph_edit_distance(solution_1_visual, heuristic_solution_1)
            ged_2 = graph_edit_distance(solution_2_visual, heuristic_solution_2)

            old_diff_ac = sample["diff_ac"]

            samples_random_cloned[i]["diff_ac"] = np.abs(ged_1 - ged_2)
            if old_diff_ac != samples_random_cloned[i]["diff_ac"]:
                logging.warning(
                    f"""problem for {sample["id"]}"""
                    f""" ({sample["use_case"]}): diff_ac changed: """
                    f"""{old_diff_ac} -> """
                    f"""{samples_random_cloned[i]["diff_ac"]}"""
                )
        # ### 2.3. Add to `samples_random`

        samples_random.extend(samples_random_cloned)
        del samples_random_cloned

        # ### 2.4. Random Trials with Same Solutions and Visual Manipulation

        # 1. sample a problem size
        # 2. sample a problem
        # 3. sample a stimulus
        # 4. apply visual manipulation (odd numbers left, even numbers right)

        # Preprocessing step to categorize stimuli
        categorized_stimuli = defaultdict(list)
        for stimulus in stimuli:
            key = (
                str(len(stimulus["task_loads"]))
                + "x"
                + str(len(stimulus["worker_capacities"]))
            )
            categorized_stimuli[key].append(stimulus)

        samples_random_same = []

        for i in range(N_RANDOM_SAME_TRIALS):
            problem_size = sample_problem_size(stimuli)
            stimuli_subset = (
                categorized_stimuli[problem_size]
                if problem_size in categorized_stimuli
                else []
            )

            if not stimuli_subset:
                continue

            problem_ids = list(
                set(stimulus["id"].split("_")[0] for stimulus in stimuli_subset)
            )
            problem_id = np.random.choice(problem_ids)

            filtered_subset = [
                stimulus
                for stimulus in stimuli_subset
                if stimulus["id"].split("_")[0] == problem_id
            ]

            if not filtered_subset:
                continue

            stimulus = np.random.choice(filtered_subset)

            sol_key = np.random.choice([1, 2])
            sol_id = stimulus["id"].split("_")[sol_key]
            stimulus["id"] = problem_id + "_" + sol_id + "_" + sol_id

            stimulus["solution_1"] = stimulus[f"solution_{str(sol_key)}"]
            stimulus["solution_2"] = stimulus[f"solution_{str(sol_key)}"]

            stimulus["use_case"] = "random_same"

            if i % 2 == 1:
                stimulus["task_order_1"] = shuffle_array(stimulus["task_loads"][:])
                stimulus["worker_order_1"] = shuffle_array(
                    stimulus["worker_capacities"][:]
                )
                stimulus["visual_manipulation"] = "left"
            else:
                stimulus["task_order_2"] = shuffle_array(stimulus["task_loads"][:])
                stimulus["worker_order_2"] = shuffle_array(
                    stimulus["worker_capacities"][:]
                )
                stimulus["visual_manipulation"] = "right"

            # since left and right are same, ac and cc need to be 0.0 sample

            stimulus["diff_ac"] = 0.0
            stimulus["diff_cc"] = 0.0

            samples_random_same.append(stimulus)

        # Add same-solution random trials to other random trials
        samples_random.extend(samples_random_same)

        # Clean up
        del samples_random_same

        # ### 2.5. Add to `samples_preference`

        samples_preference.extend(samples_random)
        del samples_random

        # ## 3. Coherence Trials

        samples_preference.extend(samples_coherence)

        # ## 4. Catch Trials

        samples_preference.extend(samples_catch)

        # ## 5. Solving Trials

        # ...

        # ## Sequence

        # Only preference trials need to be sequenced, solving trials are
        # already in correct order.

        slots_catch_trials = [
            round((len(samples_preference) / (N_CATCH_TRIALS + 1)) * k)
            for k in range(1, N_CATCH_TRIALS + 1)
        ]
        slots_coherence_trials = [
            round((len(samples_preference) / (N_COHERENCE_TRIALS + 1)) * k)
            for k in range(1, N_COHERENCE_TRIALS + 1)
        ]

        slots_other_trials = list(
            set(list(range(len(samples_preference))))
            - set(slots_catch_trials)
            - set(slots_coherence_trials)
        )

        indices = []

        for sample in samples_preference:
            if sample["use_case"] == "catch":
                slot = slots_catch_trials.pop(0)
            elif sample["use_case"] == "coherence":
                slot = slots_coherence_trials.pop(0)
            else:
                slot = slots_other_trials.pop(
                    np.random.randint(len(slots_other_trials))
                )

            indices.append(slot)

        samples_preference_seq = np.empty_like(samples_preference)

        for i, pos in enumerate(indices):
            samples_preference_seq[pos] = samples_preference[i]

        # ## Combine solving and preference trials

        samples_participant = {
            "solving_trials": samples_solving,
            "preference_trials": samples_preference_seq,
        }

        # log the trials for inspection

        logging.info("""Solving trials:""")
        for i_st, st in enumerate(samples_solving):
            logging.info(f"""Trial {i_st:3} {st["id"]:>15}   {st["difficulty"]}""")
        logging.info("""Preference trials:""")
        for i_pt, pt in enumerate(samples_preference_seq):
            logging.info(
                f"""Trial {i_pt:3} """
                f"""{pt["use_case"]:>15}   {pt["difficulty"]:>6}   """
                f"""{pt["visual_manipulation"]}"""
            )

            # logging the problem ID (currently too long)
            # logging.info(
            #     f"""Trial {i_pt:3} {pt["id"]:>15}   """
            #     f"""{pt["use_case"]:>15}   {pt["difficulty"]:>6}   """
            #     f"""{pt["visual_manipulation"]}"""
            # )

        # Add to trials

        trials.append(samples_participant)

        # Save

        save_trial_data_to_pickle(
            Path(EXPORT_PATH_DATA, PICKLE_OUTPUT_FILE), trials, file_id
        )

        logging.info(f"Participant time: {(dt.now() - starttime).total_seconds()}")

    logging.info(f"Completed generating trials for {n_participants} participants.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pickle file manager")
    parser.add_argument(
        "n_participants",
        type=int,
        nargs="+",
        help="New participants to be added",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-n",
        "--new",
        action="store_true",
        help="Create a new pickle file with the provided number of participants",
    )
    group.add_argument(
        "-a",
        "--append",
        action="store_true",
        help="Append new participants to the existing pickle file",
    )

    group2 = parser.add_argument_group()
    group2.add_argument(
        "-c",
        "--confirmatory",
        action="store_true",
        help="Use confirmatory metrics for confirmatory study and output to trials_confirmatory.pkl",
    )
    group2.add_argument(
        "-s",
        "--simple",
        action="store_true",
        help="Use simple mixture model for composition complexity that treats empty space like an item (v1).",
    )

    args = parser.parse_args()
    main(args)
