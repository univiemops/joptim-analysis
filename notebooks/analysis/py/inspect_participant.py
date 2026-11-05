# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py//py:percent
#     notebook_metadata_filter: kernelspec,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.0
# ---

# %%
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from joptim.utils.export_path import get_export_paths
import ipywidgets as widgets
from IPython.display import display, Markdown
from copy import deepcopy
from datetime import datetime
from scipy.stats import percentileofscore
from joptim.functions import print_matrix, print_two_matrices
from joptim.utils.representations import assignments2solution
from joptim.demographics import get_prolific_participant_ids


EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
STUDY = "confirmatory"  # "exploratory" or "confirmatory"
FILTER_TO_PROLIFIC = (
    True  # True, filters out all participants that were not part of STUDY,
)
# False shows all (participants not taking part it STUDY are flagged with "NO" in section "Prolific"


# requires to download data first with script download_experiment_data.py
with open(
    Path(EXPORT_PATH_DATA, "results_no_exclusion.pkl"), "rb"
) as pickle_file:
    results = pickle.load(pickle_file)

with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
    results_with_exclusion = pickle.load(pickle_file)

included_participant_xids = list(results_with_exclusion["user"]["xid"])
del results_with_exclusion

prolific_participant_ids = get_prolific_participant_ids(EXPORT_PATH_DATA, STUDY)

# filter to completed users
bool_values_start_at = 3
results["user"] = results["user"][
    np.all(results["user"].iloc[:, bool_values_start_at:], axis=1)
]

# convert to dictionaries
results_dict = deepcopy(results)

if FILTER_TO_PROLIFIC:
    results_dict["user"] = (
        results_dict["user"]
        .loc[results_dict["user"]["pid"].isin(prolific_participant_ids)]
        .reset_index(drop=True)
    )

results_dict["user"] = results_dict["user"].set_index("xid").T.to_dict()
results_dict["survey"] = results_dict["survey"].set_index("xid").T.to_dict()
results_dict["solving_trials"] = {}

for i, row in results["solving_trials"].iterrows():
    if row["xid"] not in results_dict["solving_trials"]:
        results_dict["solving_trials"][row["xid"]] = {}
    results_dict["solving_trials"][row["xid"]][row["trial_id"]] = row.drop(
        ["xid", "trial_id"]
    ).to_dict()

results_dict["preference_trials"] = {}

for i, row in results["preference_trials"].iterrows():
    if row["xid"] not in results_dict["preference_trials"]:
        results_dict["preference_trials"][row["xid"]] = {}
    results_dict["preference_trials"][row["xid"]][row["trial_id"]] = row.drop(
        ["xid", "trial_id"]
    ).to_dict()

# problem-solving inventory reversal mask
reverse = np.array(
    [
        7,
        7,
        7,
        7,
        0,
        0,
        0,
        0,
        0,
        7,
        0,
        7,
        7,
        0,
        7,
        0,
        0,
        0,
        7,
        0,
        0,
        7,
        7,
        0,
        0,
        7,
        0,
        7,
        0,
        7,
        0,
    ]
)

psi_scores_raw_all = np.array(
    [
        list(results_dict["survey"][id]["result_psi"].values())
        for id in list(results_dict["survey"])
    ],
    dtype=int,
)

psi_scores_all = np.sum(np.abs(reverse - psi_scores_raw_all), axis=1)


debrief_dict = {
    "1": {
        "title": "Enjoyment",
        "options": {
            1: "0/4 Stars",
            2: "1/4 Stars",
            3: "2/4 Stars",
            4: "3/4 Stars",
            5: "4/4 Stars",
        },
    },
    "2": {
        "title": "Interest",
        "options": {
            1: "0/4 Stars",
            2: "1/4 Stars",
            3: "2/4 Stars",
            4: "3/4 Stars",
            5: "4/4 Stars",
        },
    },
    "3": {
        "title": "Understanding",
        "options": {
            1: "0/4 Stars",
            2: "1/4 Stars",
            3: "2/4 Stars",
            4: "3/4 Stars",
            5: "4/4 Stars",
        },
    },
    "4": {
        "title": "Study Purpose Explained",
        "options": {
            1: "0/4 Stars",
            2: "1/4 Stars",
            3: "2/4 Stars",
            4: "3/4 Stars",
            5: "4/4 Stars",
        },
    },
    "5": {
        "title": "Length",
        "options": {
            1: "Too long",
            2: "Rather long",
            3: "Adequate",
            4: "Rather short",
            5: "Too short",
        },
    },
    "6": {
        "title": "Any Typos or Grammatical Mistakes?",
        "options": [],
    },
    "7": {
        "title": "Any Display Errors?",
        "options": [],
    },
    "8": {
        "title": "Additional Feedback",
        "options": [],
    },
}

demographics_dict = {
    "1_handedness": {
        "title": "Handedness",
        "options": ["Right", "Left", "Ambidextrous"],
    },
    "2_highest_edu": {
        "title": "Highest education level completed",
        "options": [
            "No formal qualifications",
            "Secondary education (e.g. GED/GCSE)",
            "High school diploma/A-levels",
            "Technical/community college",
            "Undergraduate degree (BA/BSc/other)",
            "Graduate degree (MA/MSc/MPhil/other)",
            "Doctorate degree (PhD/other)",
            "Don't know / not applicable",
        ],
    },
    "3_degree_subject": {
        "title": "Degree subject",
        "options": [
            "Arts & Humanities",
            "Education",
            "Social Sciences",
            "Journalism & Information Business",
            "Administration & Law",
            "Mathematics and statistics",
            "Information and Communication Technologies",
            "Engineering, manufacturing and construction",
            "Agriculture, forestry, fisheries and veterinary",
            "Health and welfare",
            "Services",
            "Natural sciences",
            "History",
            "Other",
        ],
    },
}

preference_text = {
    0: "identical",
    1: "strongly left",
    2: "weakly left",
    3: "weakly right",
    4: "strongly right",
}

# Create a dropdown for selecting a participant
participant_dropdown = widgets.Dropdown(
    options=list(results_dict["user"].keys()),
    # value='Alice',
    description="",
    disabled=False,
)

# Create output widgets to display participant information
output_participant = widgets.Output()
output_strategies = widgets.Output()
output_psi = widgets.Output()
experiment_details = widgets.Output()
output_demographics = widgets.Output()
output_debrief = widgets.Output()
output_solving = widgets.Output()
output_preference = widgets.Output()
output_inclusion = widgets.Output()

# global variables
solving_trial_id = 0
preference_trial_id = 0

with output_participant:
    display(widgets.HTML("<h2>Participant</h2>"))


# Define a function to update the outputs based on the selected participant
def update_dashboard(change):
    global id
    id = change["new"]
    survey_data = results_dict["survey"][id]

    psi_scores_raw = np.array(
        list(results_dict["survey"][id]["result_psi"].values()), dtype=int
    )
    psi_scores = np.abs(reverse - psi_scores_raw)
    psi_score = psi_scores.sum()
    psi_score_percentile = percentileofscore(psi_scores_all, psi_score)

    with experiment_details:
        experiment_details.clear_output()
        last_preference_trial = sorted(
            list(results_dict["preference_trials"][id].keys())
        )[-1]
        last_preference_trial_end = (
            results_dict["preference_trials"][id][last_preference_trial]["end_time"]
            / 1_000
        )

        last_preference_trial_end_str = datetime.utcfromtimestamp(
            last_preference_trial_end
        ).strftime("%Y-%m-%d %H:%M:%S")
        trial_file_id = results_dict["preference_trials"][id][last_preference_trial][
            "trial_file_id"
        ]
        trial_file_user_id = results_dict["preference_trials"][id][
            last_preference_trial
        ]["trial_file_user_id"]
        display(widgets.HTML("<h2>Experiment Details</h2>"))
        display(Markdown(f"""**Trial File ID**: {trial_file_id}"""))
        display(Markdown(f"""**Trial File User ID**: {trial_file_user_id}"""))
        display(Markdown(f"""**Completed**: {last_preference_trial_end_str}"""))

    with output_demographics:
        output_demographics.clear_output()
        display(widgets.HTML("<h2>Demographics</h2>"))
        for field in survey_data["result_demographics"]:
            title = demographics_dict[field]["title"]
            option = demographics_dict[field]["options"][
                int(survey_data["result_demographics"][field])
            ]
            display(Markdown(f"**{title}**: {option}"))

    with output_inclusion:
        output_inclusion.clear_output()
        display(widgets.HTML("<h2>Inclusion</h2>"))
        if id in included_participant_xids:
            included = "YES"
        else:
            included = "NO"
        display(Markdown(f"""**{included}**"""))

        display(widgets.HTML("<h2>Prolific</h2>"))
        display(Markdown(f"""**Prolific ID**: {results_dict["user"][id]["pid"]}"""))
        if results_dict["user"][id]["pid"] in prolific_participant_ids:
            included = "YES"
        else:
            included = "NO"
        display(Markdown(f"""**{included}**"""))

    with output_psi:
        output_psi.clear_output()
        display(widgets.HTML("<h2>Questionnaire</h2>"))
        display(
            Markdown(
                f"**PSI Score**: {psi_score} (Percentile: {psi_score_percentile:.2f}%)"
            )
        )

    with output_strategies:
        output_strategies.clear_output()
        display(widgets.HTML("<h2>Strategies</h2>"))
        display(Markdown(f"**Solving**: {survey_data['result_strategies_1']['1']}"))
        display(Markdown(f"**Preferences**: {survey_data['result_strategies_2']['1']}"))

    with output_debrief:
        output_debrief.clear_output()
        display(widgets.HTML("<h2>Debrief</h2>"))
        for field in survey_data["result_debrief"]:
            title = debrief_dict[field]["title"]
            if debrief_dict[field]["options"] == []:
                option = survey_data["result_debrief"][field]
                if option == "0":
                    option = "No"
            else:
                option = debrief_dict[field]["options"][
                    int(survey_data["result_debrief"][field])
                ]
            display(Markdown(f"**{title}**: {option}"))

    with output_solving:
        global solving_trial_id
        solving_trial_id = 0
        output_solving.clear_output()
        display(
            Markdown(
                f"""Trial No. {solving_trial_id+1}/{len(results_dict["solving_trials"][id])}"""
            )
        )
        solution = assignments2solution(
            results_dict["solving_trials"][id][solving_trial_id]["assignments"],
            results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
            results_dict["solving_trials"][id][solving_trial_id]["worker_capacities"],
        )
        print_two_matrices(
            results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
            results_dict["solving_trials"][id][solving_trial_id]["worker_capacities"],
            solution,
            results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
            results_dict["solving_trials"][id][solving_trial_id]["worker_capacities"],
            results_dict["solving_trials"][id][solving_trial_id]["example_solution"],
        )

    with output_preference:
        global preference_trial_id
        preference_trial_id = 0
        output_preference.clear_output()
        display(
            Markdown(
                f"""Trial No. {preference_trial_id+1}/{len(results_dict["preference_trials"][id])}"""
            )
        )

        display(
            Markdown(
                f"""**Use Case**: {results_dict["preference_trials"][id][preference_trial_id]["use_case"]}"""
            )
        )

        display(
            Markdown(
                f"""**Visual Manipulation**: {results_dict["preference_trials"][id][preference_trial_id]["visual_manipulation"]}"""
            )
        )

        # display(
        #     Markdown(
        #         f"""**Disorder Solution 1**: {results_dict["preference_trials"][id][preference_trial_id]["disorder_1"]}"""
        #     )
        # )

        # display(
        #     Markdown(
        #         f"""**Disorder Solution 2**: {results_dict["preference_trials"][id][preference_trial_id]["disorder_2"]}"""
        #     )
        # )

        display(
            Markdown(
                f"""**Choice**: {preference_text[results_dict["preference_trials"][id][preference_trial_id]["choice"]]}"""
            )
        )

        print_two_matrices(
            results_dict["preference_trials"][id][preference_trial_id]["task_loads"][
                results_dict["preference_trials"][id][preference_trial_id][
                    "task_order_1"
                ]
            ],
            results_dict["preference_trials"][id][preference_trial_id][
                "worker_capacities"
            ][
                results_dict["preference_trials"][id][preference_trial_id][
                    "worker_order_1"
                ]
            ],
            results_dict["preference_trials"][id][preference_trial_id]["solution_1"][
                results_dict["preference_trials"][id][preference_trial_id][
                    "task_order_1"
                ]
            ][
                :,
                results_dict["preference_trials"][id][preference_trial_id][
                    "worker_order_1"
                ],
            ],
            results_dict["preference_trials"][id][preference_trial_id]["task_loads"][
                results_dict["preference_trials"][id][preference_trial_id][
                    "task_order_2"
                ]
            ],
            results_dict["preference_trials"][id][preference_trial_id][
                "worker_capacities"
            ][
                results_dict["preference_trials"][id][preference_trial_id][
                    "worker_order_2"
                ]
            ],
            results_dict["preference_trials"][id][preference_trial_id]["solution_2"][
                results_dict["preference_trials"][id][preference_trial_id][
                    "task_order_2"
                ]
            ][
                :,
                results_dict["preference_trials"][id][preference_trial_id][
                    "worker_order_2"
                ],
            ],
        )


def next_solving_trial(change):
    with output_solving:
        global solving_trial_id
        if len(results_dict["solving_trials"][id]) > solving_trial_id + 1:
            solving_trial_id += 1
            output_solving.clear_output()
            display(
                Markdown(
                    f"""Trial No. {solving_trial_id+1}/{len(results_dict["solving_trials"][id])}"""
                )
            )
            solution = assignments2solution(
                results_dict["solving_trials"][id][solving_trial_id]["assignments"],
                results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
                results_dict["solving_trials"][id][solving_trial_id][
                    "worker_capacities"
                ],
            )
            print_two_matrices(
                results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
                results_dict["solving_trials"][id][solving_trial_id][
                    "worker_capacities"
                ],
                solution,
                results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
                results_dict["solving_trials"][id][solving_trial_id][
                    "worker_capacities"
                ],
                results_dict["solving_trials"][id][solving_trial_id][
                    "example_solution"
                ],
            )


def prev_solving_trial(change):
    with output_solving:
        global solving_trial_id
        if 0 <= solving_trial_id - 1:
            solving_trial_id -= 1
            output_solving.clear_output()
            display(
                Markdown(
                    f"""Trial No. {solving_trial_id+1}/{len(results_dict["solving_trials"][id])}"""
                )
            )
            solution = assignments2solution(
                results_dict["solving_trials"][id][solving_trial_id]["assignments"],
                results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
                results_dict["solving_trials"][id][solving_trial_id][
                    "worker_capacities"
                ],
            )
            print_two_matrices(
                results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
                results_dict["solving_trials"][id][solving_trial_id][
                    "worker_capacities"
                ],
                solution,
                results_dict["solving_trials"][id][solving_trial_id]["task_loads"],
                results_dict["solving_trials"][id][solving_trial_id][
                    "worker_capacities"
                ],
                results_dict["solving_trials"][id][solving_trial_id][
                    "example_solution"
                ],
            )


def next_preference_trial(change):
    with output_preference:
        global preference_trial_id
        if len(results_dict["preference_trials"][id]) > preference_trial_id + 1:
            preference_trial_id += 1
            output_preference.clear_output()
            display(
                Markdown(
                    f"""Trial No. {preference_trial_id+1}/{len(results_dict["preference_trials"][id])}"""
                )
            )

            display(
                Markdown(
                    f"""**Use Case**: {results_dict["preference_trials"][id][preference_trial_id]["use_case"]}"""
                )
            )

            display(
                Markdown(
                    f"""**Visual Manipulation**: {results_dict["preference_trials"][id][preference_trial_id]["visual_manipulation"]}"""
                )
            )

            # display(
            #     Markdown(
            #         f"""**Disorder Solution 1**: {results_dict["preference_trials"][id][preference_trial_id]["disorder_1"]}"""
            #     )
            # )

            # display(
            #     Markdown(
            #         f"""**Disorder Solution 2**: {results_dict["preference_trials"][id][preference_trial_id]["disorder_2"]}"""
            #     )
            # )

            display(
                Markdown(
                    f"""**Choice**: {preference_text[results_dict["preference_trials"][id][preference_trial_id]["choice"]]}"""
                )
            )

            print_two_matrices(
                results_dict["preference_trials"][id][preference_trial_id][
                    "task_loads"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "task_order_1"
                    ]
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "worker_capacities"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "worker_order_1"
                    ]
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "solution_1"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "task_order_1"
                    ]
                ][
                    :,
                    results_dict["preference_trials"][id][preference_trial_id][
                        "worker_order_1"
                    ],
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "task_loads"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "task_order_2"
                    ]
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "worker_capacities"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "worker_order_2"
                    ]
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "solution_2"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "task_order_2"
                    ]
                ][
                    :,
                    results_dict["preference_trials"][id][preference_trial_id][
                        "worker_order_2"
                    ],
                ],
            )


def prev_preference_trial(change):
    with output_preference:
        global preference_trial_id
        if 0 <= preference_trial_id - 1:
            preference_trial_id -= 1
            output_preference.clear_output()
            display(
                Markdown(
                    f"""Trial No. {preference_trial_id+1}/{len(results_dict["preference_trials"][id])}"""
                )
            )

            display(
                Markdown(
                    f"""**Use Case**: {results_dict["preference_trials"][id][preference_trial_id]["use_case"]}"""
                )
            )

            display(
                Markdown(
                    f"""**Visual Manipulation**: {results_dict["preference_trials"][id][preference_trial_id]["visual_manipulation"]}"""
                )
            )

            # display(
            #     Markdown(
            #         f"""**Disorder Solution 1**: {results_dict["preference_trials"][id][preference_trial_id]["disorder_1"]}"""
            #     )
            # )

            # display(
            #     Markdown(
            #         f"""**Disorder Solution 2**: {results_dict["preference_trials"][id][preference_trial_id]["disorder_2"]}"""
            #     )
            # )

            display(
                Markdown(
                    f"""**Choice**: {preference_text[results_dict["preference_trials"][id][preference_trial_id]["choice"]]}"""
                )
            )

            print_two_matrices(
                results_dict["preference_trials"][id][preference_trial_id][
                    "task_loads"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "task_order_1"
                    ]
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "worker_capacities"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "worker_order_1"
                    ]
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "solution_1"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "task_order_1"
                    ]
                ][
                    :,
                    results_dict["preference_trials"][id][preference_trial_id][
                        "worker_order_1"
                    ],
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "task_loads"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "task_order_2"
                    ]
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "worker_capacities"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "worker_order_2"
                    ]
                ],
                results_dict["preference_trials"][id][preference_trial_id][
                    "solution_2"
                ][
                    results_dict["preference_trials"][id][preference_trial_id][
                        "task_order_2"
                    ]
                ][
                    :,
                    results_dict["preference_trials"][id][preference_trial_id][
                        "worker_order_2"
                    ],
                ],
            )


# Initial display
update_dashboard({"new": participant_dropdown.value})

# buttons for solving trials
button_solving_prev = widgets.Button(
    description="",
    disabled=False,
    button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    icon="arrow-left",
)
button_solving_next = widgets.Button(
    description="", disabled=False, button_style="", icon="arrow-right"
)

# buttons for solving trials
button_preference_prev = widgets.Button(
    description="",
    disabled=False,
    button_style="",  # 'success', 'info', 'warning', 'danger' or ''
    icon="arrow-left",
)
button_preference_next = widgets.Button(
    description="", disabled=False, button_style="", icon="arrow-right"
)

# Callbacks
participant_dropdown.observe(update_dashboard, names="value")
button_solving_next.on_click(next_solving_trial)
button_solving_prev.on_click(prev_solving_trial)
button_preference_next.on_click(next_preference_trial)
button_preference_prev.on_click(prev_preference_trial)

solving_box = widgets.VBox(
    [
        widgets.HTML("<h2>Solving Trials</h2>"),
        widgets.HTML(
            "<b>Left: Participant's solution. Right: Example of an optimal solution.</b>"
        ),
        widgets.HBox([button_solving_prev, button_solving_next]),
        output_solving,
    ]
)

preference_box = widgets.VBox(
    [
        widgets.HTML("<h2>Preference Trials</h2>"),
        widgets.HBox([button_preference_prev, button_preference_next]),
        output_preference,
    ]
)

# tabs

tab = widgets.Tab()
tab_survey = widgets.VBox(
    [
        experiment_details,
        output_demographics,
        output_inclusion,
        output_psi,
        output_strategies,
        output_debrief,
    ]
)

tab.children = [tab_survey, solving_box, preference_box]
tab.titles = ["Survey Data", "Solving", "Preference"]

# Display the widgets in a layout
dashboard = widgets.VBox([output_participant, participant_dropdown, tab])
display(dashboard)
