from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
import os


CLEAN_DATA_SUFFIXES = {".csv", ".json", ".txt"}
SKIP_DATA_SUFFIXES = {".tex", ".svg", ".html", ".log", ".md"}
RAW_DATA_SUFFIXES = {".pkl"}
DROP_COLUMNS = {"pid", "Participant id", "Submission id"}
REPLACEMENTS = [
    (re.compile(r"https?://psycyb-portia\.psy\.univie\.ac\.at(?::\d+)?"), "[REDACTED_URL]"),
    (re.compile(r"psycyb-portia\.psy\.univie\.ac\.at(?::\d+)?"), "[REDACTED_HOST]"),
    (re.compile(r"/joptim-api/[^\s\"'<>]+"), "/joptim-api/[REDACTED]"),
    (re.compile(r"/home/user(?:/[^\s\"'<>]*)?"), "[REDACTED_PATH]"),
    (re.compile(r"Dropbox/data_export/joptim-analysis(?:/[^\s\"'<>]*)?"), "[REDACTED_PATH]"),
    (re.compile(r"\b(?:University of Vienna|Universit(?:y|a)t Wien|Univie)\b", re.IGNORECASE), "[REDACTED_AFFILIATION]"),
    (re.compile(r"\bFilip\b"), "[REDACTED_NAME]"),
]


def load_data_dir() -> Path:
    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path)
    if os.environ.get("DATA_DIR"):
        return Path(os.environ["DATA_DIR"]).expanduser().resolve()
    export_base_file = Path("~/code/data_export_dir").expanduser()
    if export_base_file.exists():
        export_base = Path(export_base_file.read_text(encoding="utf-8").splitlines()[0].strip()).expanduser()
        return (export_base / Path(__file__).resolve().parent.name).resolve()
    raise RuntimeError(f"Set DATA_DIR in {env_path} to the export root containing data/ and pub/.")


def should_skip(path: Path) -> bool:
    return "archive" in path.parts or path.name == "OSF"


def clean_text(text: str) -> str:
    for pattern, replacement in REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def copy_raw(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    shutil.copy2(src, dst)


def clean_copy_text(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    dst.write_text(clean_text(src.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")


def clean_copy_csv(src: Path, dst: Path) -> None:
    ensure_parent(dst)
    with src.open("r", encoding="utf-8-sig", newline="") as f_in, dst.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        header = next(reader, None)
        if header is None:
            return
        keep = [i for i, col in enumerate(header) if col not in DROP_COLUMNS]
        writer.writerow([header[i] for i in keep])
        for row in reader:
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            writer.writerow([clean_text(row[i]) for i in keep])


def export_data(data_root: Path, out_root: Path) -> None:
    for src in data_root.rglob("*"):
        if not src.is_file() or should_skip(src.relative_to(data_root)):
            continue
        dst = out_root / src.relative_to(data_root.parent)
        if src.suffix == ".csv":
            clean_copy_csv(src, dst)
        elif src.suffix in CLEAN_DATA_SUFFIXES:
            clean_copy_text(src, dst)
        elif src.suffix in RAW_DATA_SUFFIXES:
            copy_raw(src, dst)
        elif src.suffix in SKIP_DATA_SUFFIXES:
            continue


def export_pub(pub_root: Path, out_root: Path) -> None:
    for src in pub_root.rglob("*"):
        if not src.is_file() or should_skip(src.relative_to(pub_root)):
            continue
        copy_raw(src, out_root / src.relative_to(pub_root.parent))


def main() -> None:
    data_dir = load_data_dir()
    out_root = data_dir / "OSF"
    if out_root.exists():
        shutil.rmtree(out_root)
    export_data(data_dir / "data", out_root)
    export_pub(data_dir / "pub", out_root)
    print(f"Created OSF export at {out_root}")


if __name__ == "__main__":
    main()
