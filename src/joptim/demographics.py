import pandas as pd
from pathlib import Path


def get_prolific_participant_ids(dir, study, analysis_name="demographics"):
    df_demographics = pd.read_csv(
        Path(dir, study, analysis_name, "prolific_demographics.csv")
    )
    return list(df_demographics["Participant id"].values.ravel())
