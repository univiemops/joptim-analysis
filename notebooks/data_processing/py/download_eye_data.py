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

# %% [markdown]
# # Eye-Tracking
#
# - This notebook downloads the eye-tracking data and computes the dwell time difference right vs. left (`eye_RvL`) for each trial and stores it in `data_eye_<TRIAL_FILE_ID>.csv`.
# - TODO: column `viewports_used` could be used to exclude participans that used a mobile device (e.g., cutoff viewport width $\le 992\text{px}$)

# %% [markdown]
# ## 0. Imports

# %%
import json
import requests
from scipy.ndimage import zoom
import pickle
from joptim.utils.export_path import get_export_paths
from joptim.utils.secrets import load_secrets
import urllib3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from titlecase import titlecase
from pathlib import Path


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

EXPORT_PATH_DATA, EXPORT_PATH_PUB, EXPORT_PATH_LOGS = get_export_paths()
STUDY = "confirmatory"  # "exploratory" or "confirmatory"
ANALYSIS_NAME = "mlm_gaze"
TRIAL_TYPE = "preference_trials"

BLACKLIST_PATH = Path(EXPORT_PATH_DATA, STUDY, "blacklist.csv")

if STUDY not in ["exploratory", "confirmatory"]:
    raise ValueError(f"STUDY can only be 'exploratory' or 'confirmatory', got: {STUDY}")

with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
    results = pickle.load(pickle_file)

with open(Path(EXPORT_PATH_DATA, STUDY, "trials.pkl"), "rb") as pickle_file:
    trials = pickle.load(pickle_file)
TRIAL_FILE_ID = trials["file_id"]
del trials

with open(Path(EXPORT_PATH_DATA, "results.pkl"), "rb") as pickle_file:
    results = pickle.load(pickle_file)

api_key, server_url = load_secrets()

trial_file_id = str(TRIAL_FILE_ID)

fout = f"""data_eye_{trial_file_id.replace(".", "-")}.csv"""

print(f"Data will be saved to {fout}.")


# %% [markdown]
# ## 1. Download gaze data and get the variable `eye_RvL` for all participants and all trials
#
# This involves for each coordinate to compute the timedelta and location (left box, right box, outside left, outside right).

# %%
def get_location(x, y, box):
    box = eval(box)
    left = box["left"]
    right = box["right"]
    top = box["top"]
    bottom = box["bottom"]
    middle = (left + right) / 2
    if (x < right) and (x > left) and (y > top) and (y < bottom):
        box = True
    else:
        box = False

    if x < middle:
        if box:
            return "box left"
        else:
            return "outside left"
    else:
        if box:
            return "box right"
        else:
            return "outside right"


data_eye = []

trial_file_user_ids = (
    results[TRIAL_TYPE]
    .loc[results[TRIAL_TYPE]["trial_file_id"] == trial_file_id]["trial_file_user_id"]
    .unique()
)

for trial_file_user_id in trial_file_user_ids:

    user_id = (
        results["user"]
        .loc[
            (results["user"]["trial_file_id"] == trial_file_id)
            & (results["user"]["trial_file_user_id"] == trial_file_user_id),
            "pid",
        ]
        .values[0]
    )

    res = requests.get(
        server_url + f"/joptim-api/{api_key}/eyetracking/{user_id}", verify=False
    )

    if len(res.json()) > 0:
        df_eye = pd.DataFrame(res.json()[1:], columns=res.json()[0])
        df_eye["timestamp"] = df_eye["timestamp"].astype(int)
        df_eye["x"] = df_eye["x"].astype(float)
        df_eye["y"] = df_eye["y"].astype(float)

        # human-readable timestamps
        df_eye["timestamp_h"] = pd.to_datetime(df_eye["timestamp"], unit="ms")

        for trial_id in list(
            results[TRIAL_TYPE].loc[
                (results[TRIAL_TYPE]["trial_file_user_id"] == trial_file_user_id)
                & (results[TRIAL_TYPE]["trial_file_id"] == trial_file_id),
                "trial_id",
            ]
        ):

            eye_trial = (
                results[TRIAL_TYPE]
                .loc[
                    (results[TRIAL_TYPE]["trial_id"] == trial_id)
                    & (results[TRIAL_TYPE]["trial_file_user_id"] == trial_file_user_id)
                    & (results[TRIAL_TYPE]["trial_file_id"] == trial_file_id)
                ]
                .copy()
            )

            # 2-step filter

            # 1. start and end times

            eye_trial_coords = df_eye.loc[
                (df_eye["timestamp"] > eye_trial.start_time.values[0])
                & (df_eye["timestamp"] < eye_trial.end_time.values[0])
            ].reset_index(drop=True)

            # 2. keep only the timestamps for 1 (the most frequent)
            # stimulus box position/size (there are few other box position/sizes as a result of
            # how the html elements are loaded in the browser

            if len(eye_trial_coords) > 0:
                eye_trial_coords = eye_trial_coords.loc[
                    eye_trial_coords["stimulus2box"]
                    == eye_trial_coords["stimulus2box"].mode().values[0]
                ].reset_index(drop=True)

                assert len(eye_trial_coords["viewport"].unique()) == 1
                assert len(eye_trial_coords["stimulus2box"].unique()) == 1
                # there should be only 1 viewport size and 1 stimulus box position/size (if there are more then
                # participant changed screen size during the experiment
                # -> need to write a solution to this.

                eye_trial_coords["timedelta"] = (
                    eye_trial_coords["timestamp"].shift(-1)
                    - eye_trial_coords["timestamp"]
                )

                eye_trial_coords["location"] = eye_trial_coords.apply(
                    lambda row: get_location(row.x, row.y, row.stimulus2box), axis=1
                )

                eye_RvL = (
                    eye_trial_coords[["location", "timedelta"]]
                    .groupby("location")
                    .sum()
                    .T
                )

                if "box right" in eye_RvL:
                    box_right = eye_RvL["box right"].values[0]
                else:
                    box_right = 0
                if "box left" in eye_RvL:
                    box_left = eye_RvL["box left"].values[0]
                else:
                    box_right = 0

                # add viewport information '|'-separated in case the user used more than 1
                viewports_used = "|".join(df_eye["viewport"].unique())
                
                data_eye.append(
                    {
                        "trial_file_id": trial_file_id,
                        "trial_file_user_id": trial_file_user_id,
                        "trial_id": trial_id,
                        "eye_L": box_left,
                        "eye_R": box_right,
                        "viewports_used":viewports_used
                    }
                )
            else:
                print("trial_file_id:", trial_file_id, "– trial_file_user_id:", trial_file_user_id, "- trial_id:", trial_id, "– no stimulus2box information")

    else:
        print("trial_file_id:", trial_file_id, "– trial_file_user_id:", trial_file_user_id, "– no eye-tracking information available")

data_eye = pd.DataFrame(data_eye)
# filter to the relevant study
data_eye.loc[data_eye["trial_file_id"] == trial_file_id].reset_index(drop=True)

data_eye.to_csv(Path(EXPORT_PATH_DATA, STUDY, ANALYSIS_NAME, fout), index=False)


# %% [markdown]
# Participants to be excluded due to not using a desktop computer

# %%
def flag_mobile(vp_used, width_threshold=992):
    """Flag a set of viewports (string) if their minimal width is <= width_threshold"""
    vp_width_min = min([int(r.split("x")[0]) for r in vp_used.split("|")])
    return vp_width_min<=width_threshold


suggested_for_blacklist = data_eye.loc[data_eye.viewports_used.apply(lambda x: flag_mobile(x, 992))]
suggested_for_blacklist

# %% [markdown]
# Add participants to blacklist `<BLACKLIST_PATH>`
#
# Will be picked up by `demographics.py/ipynb` where participants get excluded for all further analyses

# %%
blacklist_new = suggested_for_blacklist.drop_duplicates(["trial_file_id", "trial_file_user_id"])[["trial_file_id", "trial_file_user_id", "viewports_used"]].rename(columns={"viewports_used":"reason"})
blacklist_new["reason"] = "used mobile viewports: " + blacklist_new["reason"]


if not BLACKLIST_PATH.exists():
    blacklist_new.to_csv(BLACKLIST_PATH, index=False)

else:
    blacklist_base = pd.read_csv(BLACKLIST_PATH, dtype={"trial_file_id":"string"})
    blacklist_new = pd.concat([blacklist_base, blacklist_new], sort=False)
    blacklist_new.drop_duplicates(["trial_file_id", "trial_file_user_id", "reason"], inplace=True)
    blacklist_new.to_csv(BLACKLIST_PATH, index=False)

print(f"Updated blacklist at {BLACKLIST_PATH}")

# %% [markdown]
# Inspect an indiividual trial

# %%
trial_id = 15
trial_file_user_id = 6

data_eye.loc[
    (data_eye["trial_file_id"] == trial_file_id)
    & (data_eye["trial_file_user_id"] == trial_file_user_id)
    & (data_eye["trial_id"] == trial_id)
]

# %%
results["user"].loc[
    (results["user"]["trial_file_id"] == trial_file_id)
    & (results["user"]["trial_file_user_id"] == trial_file_user_id)
]

# %% [markdown]
# ## 2. Inspect single Participant, single trial

# %% [markdown]
# ### 2.1. Recode choice

# %%
choice_recode_dict = {1: -2, 2: -1, 3: 1, 4: 2}
results["preference_trials"]["choice_RvL"] = results["preference_trials"][
    "choice"
].apply(lambda x: choice_recode_dict[x])

choice_label_dict = {
    -2: "definitely left",
    -1: "slightly left",
    1: "slightly right",
    2: "definitely right",
}

trial_type = "preference_trials"

max(results[trial_type]["trial_id"])

# %%
results[trial_type].loc[results[trial_type]["trial_file_id"] == trial_file_id][
    "trial_file_user_id"
].unique()

# %% [markdown]
# ### 2.2. Create heatmap for example participant

# %%
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle


def create_heatmap(
    x_coords,
    y_coords,
    viewport,
    stimulus_box,
    bin_factor=66,  # kept for API compatibility; unused
    choice_label=None,
    include_viewport=False,
    sigma_px=24,  # smoothing in pixels; set 0 for none
    dpi=600,
    label_lr=True,
    lr_offset_px=5,  # pixel offset from edges
    lr_fontsize=10,
    white_bg=0.8,
):

    x_pos = stimulus_box["left"]
    y_pos = stimulus_box["top"]
    width = stimulus_box["width"]
    height = stimulus_box["height"]

    scaling_factor_figsize = 2.4
    W, H = int(viewport[0]), int(viewport[1])
    x = np.asarray(x_coords)
    y = np.asarray(y_coords)

    # Accumulate at pixel resolution (no upsampling)
    xedges = np.arange(W + 1)
    yedges = np.arange(H + 1)
    hist, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    img = hist.T.astype(np.float32)  # rows=y, cols=x

    # Optional smoothing
    if sigma_px and sigma_px > 0:
        img = gaussian_filter(img, sigma=sigma_px, mode="constant")

    # Robust normalization
    vmax = np.percentile(img, 99) if img.max() > 0 else 1.0
    if vmax > 0:
        img = np.clip(img / vmax, 0, 1)

    # One data pixel -> one output pixel at dpi
    fig, ax = plt.subplots(
        figsize=(scaling_factor_figsize * W / dpi, scaling_factor_figsize * H / dpi),
        dpi=dpi,
    )

    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Align integer coords to pixel centers; y downward
    extent = (-0.5, W - 0.5, H - 0.5, -0.5)
    # ax.imshow(img, cmap="hot_r", interpolation="nearest", extent=extent)

    # grey background
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5), W, H, facecolor=str(white_bg), edgecolor="none", zorder=0
        )
    )
    ax.add_patch(
        Rectangle(
            (x_pos, y_pos), width, height, facecolor="white", edgecolor="none", zorder=0
        )
    )
    # ax.set_facecolor("1.0")  # light-grey background
    ax.imshow(
        img,  # colour values
        cmap="hot_r",  # black→red
        extent=extent,
        interpolation="nearest",
        # vmin=0,
        # vmax=1,
        alpha=img ** (1 / 3),  # per-pixel transparency
        zorder=1,
    )  # above grey background

    # Title/labels
    title_parts = []
    if include_viewport:
        title_parts.append(f"Heatmap of Gaze Patterns (viewport {W}x{H})")
    if choice_label:
        title_parts.append(f"Preference: {choice_label}")
    if title_parts:
        ax.set_title(titlecase("\n".join(title_parts)))
    # ax.set_xlabel("X coordinate")
    # ax.set_ylabel("Y coordinate")

    # Thin overlay lines (explicit kwargs, no unpacking)
    ax.axvline(0, color="black", lw=0.5)
    ax.axvline(W - 1, color="black", lw=0.5)
    ax.axhline(1, color="black", lw=0.5)
    ax.axhline(H - 1, color="black", lw=0.5)

    # Stimulus container
    ax.hlines(y_pos, x_pos, x_pos + width, colors="black", linewidths=0.5)
    ax.hlines(y_pos + height, x_pos, x_pos + width, colors="black", linewidths=0.5)
    ax.vlines(x_pos, y_pos, y_pos + height, colors="black", linewidths=0.5)
    ax.vlines(x_pos + width, y_pos, y_pos + height, colors="black", linewidths=0.5)
    ax.vlines(W / 2, y_pos, y_pos + height, colors="black", linewidths=0.5)

    # Add L/R labels just inside the stimulus box, above its bottom edge
    if label_lr:
        xL = x_pos + lr_offset_px
        xR = x_pos + width - lr_offset_px
        y_base = y_pos + height - lr_offset_px  # above bottom edge (y smaller = up)
        ax.text(
            xL,
            y_base,
            "L",
            ha="left",
            va="bottom",
            color="black",
            fontsize=lr_fontsize,
            zorder=3,
        )
        ax.text(
            xR,
            y_base,
            "R",
            ha="right",
            va="bottom",
            color="black",
            fontsize=lr_fontsize,
            zorder=3,
        )

    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal", "box")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(pad=0)
    fig.savefig(
        Path(EXPORT_PATH_PUB, "eye_example.png"),
        dpi=dpi,
        transparent=False,
        facecolor="white",
        edgecolor="white",
        pad_inches=0.01,
        bbox_inches="tight",
    )


create_heatmap(
    eye_trial_coords["x"],
    eye_trial_coords["y"],
    viewport,
    stimulus_box,
    sigma_px=16,
    choice_label=choice_label,
    white_bg=0.85,
)

# %%
# user_id = results["user"]["xid"][0]
trial_id = 15
trial_file_user_id = 12
try:
    user_id = (
        results["user"]
        .loc[
            (results["user"]["trial_file_id"] == trial_file_id)
            & (results["user"]["trial_file_user_id"] == trial_file_user_id),
            "pid",
        ]
        .values[0]
    )
except IndexError as e:
    raise ValueError(f"Try with a different trial_file_user_id and/or trial_file_user_id (IndexError: {e}.)")
trial_type = "preference_trials"

res = requests.get(
    server_url + f"/joptim-api/{api_key}/eyetracking/{user_id}", verify=False
)
# res = requests.get(f"http://localhost:8000/eyetracking/{user_id}", verify=False)

df_eye = pd.DataFrame(res.json()[1:], columns=res.json()[0])
df_eye["timestamp"] = df_eye["timestamp"].astype(int)
df_eye["x"] = df_eye["x"].astype(float)
df_eye["y"] = df_eye["y"].astype(float)

# human-readable timestamps
df_eye["timestamp_h"] = pd.to_datetime(df_eye["timestamp"], unit="ms")

eye_trial = (
    results[trial_type]
    .loc[
        (results[trial_type]["trial_id"] == trial_id)
        & (results[trial_type]["trial_file_user_id"] == trial_file_user_id)
    ]
    .copy()
)

# 2-step filter

# 1. start and end times

eye_trial_coords = df_eye.loc[
    (df_eye["timestamp"] > eye_trial.start_time.values[0])
    & (df_eye["timestamp"] < eye_trial.end_time.values[0])
].reset_index(drop=True)

# 2. keep only the timestamps for 1 (the most frequent)
# stimulus box position/size (there are few other box position/sizes as a result of
# how the html elements are loaded in the browser
eye_trial_coords = eye_trial_coords.loc[
    eye_trial_coords["stimulus2box"]
    == eye_trial_coords["stimulus2box"].mode().values[0]
].reset_index(drop=True)

assert len(eye_trial_coords["viewport"].unique()) == 1
assert len(eye_trial_coords["stimulus2box"].unique()) == 1
# there should be only 1 viewport size and 1 stimulus box position/size (if there are more then
# participant changed screen size during the experiment
# -> need to write a solution to this.


# def create_heatmap(
#     x_coords,
#     y_coords,
#     viewport,
#     stimulus_box,
#     bin_factor=66,
#     choice_label=None,
#     include_viewport=False,
# ):
#     """
#     Create a heatmap from eye-tracking data.

#     Parameters:
#     x_coords (list or np.array): Array of x-coordinates of gaze points.
#     y_coords (list or np.array): Array of y-coordinates of gaze points.
#     viewport (tuple): Size of the screen as (width, height).
#     stimulus_box (dict): Dictionary with values for top, left, width, height
#     bins (tuple): Number of bins for the heatmap in x and y dimensions.

#     Returns:
#     None
#     """

#     # Ensure the coordinates are numpy arrays
#     x_coords = np.array(x_coords)
#     y_coords = np.array(y_coords)

#     bins = (int(viewport[0] / bin_factor), int(viewport[1] / bin_factor))

#     # Create a 2D histogram of the gaze points
#     heatmap, _, _ = np.histogram2d(
#         x_coords, y_coords, bins=bins, range=[[0, viewport[0]], [0, viewport[1]]]
#     )

#     # Normalize the heatmap
#     heatmap /= np.max(heatmap)

#     scale_x = viewport[0] / heatmap.shape[0]
#     scale_y = viewport[1] / heatmap.shape[1]
#     heatmap_rescaled = zoom(
#         heatmap, (scale_x, scale_y), order=1
#     )  # order=1 for bilinear interpolation

#     # Plot the heatmap
#     aspect_ratio = viewport[0] / viewport[1]
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4 * aspect_ratio, 4))
#     fig.patch.set_alpha(0.0)

#     # Create the heatmap with a transparent background
#     ax = sns.heatmap(
#         heatmap_rescaled.T,
#         cmap="hot_r",
#         xticklabels=False,
#         yticklabels=False,
#         ax=ax,
#         # alpha=0.5,
#     )
#     ax.set_aspect("equal", "box")
#     title = ""
#     if include_viewport:
#         title = f"Heatmap of Gaze Patterns (viewport {viewport[0]}x{viewport[1]})\n"

#     if choice_label != None:
#         title += "Preference: " + choice_label

#     title = titlecase(title)

#     ax.set_title(title)
#     ax.set_xlabel("X coordinate")
#     ax.set_ylabel("Y coordinate")

#     # viewport
#     ax.axvline(0, color="black", lw=1)
#     ax.axvline(viewport[0] - 1, color="black", lw=1)
#     ax.axhline(1, color="black", lw=1)
#     ax.axhline(viewport[1] - 1, color="black", lw=1)

#     # stimulus container
#     x_pos = stimulus_box["left"]
#     y_pos = stimulus_box["top"]
#     width = stimulus_box["width"]
#     height = stimulus_box["height"]

#     ax.hlines(y_pos, x_pos, x_pos + width, color="black", lw=1)
#     ax.hlines(y_pos + height, x_pos, x_pos + width, color="black", lw=1)

#     ax.vlines(x_pos, y_pos, y_pos + height, color="black", lw=1)
#     ax.vlines(x_pos + width, y_pos, y_pos + height, color="black", lw=1)
#     ax.vlines(viewport[0] / 2, y_pos, y_pos + height, color="black", lw=1)

#     # fig.gca().invert_yaxis()  # Invert y axis to match screen coordinate system
#     # fig.show()
#     fig.tight_layout()
#     fig.savefig(
#         Path(EXPORT_PATH_PUB, "eye_example.png"),
#         dpi=600,
#         transparent=False,
#         facecolor="white",
#         pad_inches=0.0,
#     )


viewport = np.array(eye_trial_coords.loc[0, "viewport"].split("x"), dtype=int)
stimulus_box = json.loads(eye_trial_coords.loc[0, "stimulus2box"])
choice_label = choice_label_dict[eye_trial.choice_RvL.values[0]]

create_heatmap(
    eye_trial_coords["x"],
    eye_trial_coords["y"],
    viewport,
    stimulus_box,
    bin_factor=66,
    choice_label=choice_label,
)

if viewport[1] < stimulus_box["y"] + stimulus_box["height"]:
    print(
        f"""Viewport is too small for stimulus box!\nViewport height: {viewport[1]},\nStimulus y-position + height: {stimulus_box["y"] + stimulus_box["height"]}"""
    )

# %% [markdown]
# **TODO: What's the difference between the two heatmaps and do we need both?**
