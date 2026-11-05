import os
from pathlib import Path
import subprocess


def get_export_paths():
    # change this to file or ENV var that contains your base bath for exports
    with open(os.path.expanduser("~/code/data_export_dir")) as fo:
        export_base_path = fo.readline().strip()
    repo_name = (
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
        )
        .strip()
        .split("/")[-1]
    )
    ep = os.path.expanduser(Path(export_base_path, repo_name))
    ep_data = Path(ep, "data")
    ep_pub = Path(ep, "pub")
    ep_logs = Path(ep, "logs")
    os.makedirs(ep_data, exist_ok=True)
    os.makedirs(ep_pub, exist_ok=True)
    os.makedirs(ep_logs, exist_ok=True)
    return ep_data, ep_pub, ep_logs
