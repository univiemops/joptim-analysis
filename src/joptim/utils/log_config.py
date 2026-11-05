import logging
from pathlib import Path
from joptim.utils.export_path import get_export_paths


def setup_logging(scriptname):
    _, _, log_dir = get_export_paths()
    log_file_path = Path(log_dir, scriptname + ".log")
    print(f"Log file: {log_file_path}.")
    logging.basicConfig(
        # this will disable logging to screen
        # filename=log_file_path,
        # filemode='a',
        encoding="utf-8",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # this part is needed to have both logging to file and to screen
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
    )
    logging.debug("Logging config finished.")
