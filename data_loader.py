"""Data loading and normalisation for all five Reply Mirror data sources.

Each loader accepts a zip file path and returns a clean Python object ready
for downstream processing. No LLM calls, no feature computation — raw data only.
"""

import json
import zipfile
from pathlib import Path
from typing import Union

import pandas as pd

# ── Level / split name mapping ────────────────────────────────────────────────

LEVEL_MAP: dict[str, str] = {
    "brave-new-world": "Brave New World",
    "deus-ex": "Deus Ex",
    "the-truman-show": "The Truman Show",
}

SPLIT_MAP: dict[str, str] = {
    "train": "train",
    "validation": "validation",
}


def resolve_zip_path(data_dir: str, level: str, split: str) -> Path:
    """Return the zip file path for a given level and split.

    Args:
        data_dir: Directory containing the zip files (e.g. 'train-validation').
        level: CLI-style level name (e.g. 'brave-new-world').
        split: 'train' or 'validation'.
    """
    if level not in LEVEL_MAP:
        raise ValueError(f"Unknown level '{level}'. Valid: {list(LEVEL_MAP)}")
    if split not in SPLIT_MAP:
        raise ValueError(f"Unknown split '{split}'. Valid: {list(SPLIT_MAP)}")
    level_name = LEVEL_MAP[level]
    # Zip files use '+' as space separator in file names
    zip_name = f"{level_name.replace(' ', '+')}+-+{split}.zip"
    return Path(data_dir) / zip_name


def _inner_prefix(level: str, split: str) -> str:
    """Return the directory prefix used inside the zip (e.g. 'Brave New World - train/')."""
    return f"{LEVEL_MAP[level]} - {split}/"


def _open_entry(zf: zipfile.ZipFile, prefix: str, filename: str):
    """Open a file entry inside the zip, ignoring macOS metadata."""
    candidates = [
        n for n in zf.namelist()
        if filename in n and not n.startswith("__MACOSX") and not n.endswith(".DS_Store")
    ]
    if not candidates:
        raise FileNotFoundError(f"'{filename}' not found inside zip.")
    # Prefer the entry that starts with the expected prefix
    preferred = [c for c in candidates if c.startswith(prefix)]
    entry = preferred[0] if preferred else candidates[0]
    return zf.open(entry)


# ── Public loaders ────────────────────────────────────────────────────────────

def load_transactions(zip_path: Union[str, Path], level: str, split: str) -> pd.DataFrame:
    """Load transactions.csv and return a normalised DataFrame.

    Columns returned (all present, NaN for optional missing values):
        transaction_id, sender_id, recipient_id, transaction_type, amount,
        location, payment_method, sender_iban, recipient_iban, balance_after,
        description, timestamp
    """
    prefix = _inner_prefix(level, split)
    with zipfile.ZipFile(zip_path) as zf:
        with _open_entry(zf, prefix, "transactions.csv") as f:
            df = pd.read_csv(f, dtype=str)

    # Coerce numeric columns; keep NaN for truly missing values
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    return df


def load_users(zip_path: Union[str, Path], level: str, split: str) -> list[dict]:
    """Load users.json and return a list of user profile dicts.

    Each dict has: first_name, last_name, birth_year, salary, job, iban,
    residence {city, lat, lng}, description.
    """
    prefix = _inner_prefix(level, split)
    with zipfile.ZipFile(zip_path) as zf:
        with _open_entry(zf, prefix, "users.json") as f:
            users = json.load(f)

    # Normalise residence lat/lng to float
    for u in users:
        res = u.get("residence", {})
        try:
            res["lat"] = float(res.get("lat", 0))
            res["lng"] = float(res.get("lng", 0))
        except (TypeError, ValueError):
            res["lat"] = 0.0
            res["lng"] = 0.0

    return users


def load_locations(zip_path: Union[str, Path], level: str, split: str) -> list[dict]:
    """Load locations.json and return a list of GPS ping dicts.

    Each dict has: biotag (str), timestamp (str ISO-8601), lat (float),
    lng (float), city (str).
    """
    prefix = _inner_prefix(level, split)
    with zipfile.ZipFile(zip_path) as zf:
        with _open_entry(zf, prefix, "locations.json") as f:
            locations = json.load(f)

    for loc in locations:
        loc["lat"] = float(loc.get("lat", 0))
        loc["lng"] = float(loc.get("lng", 0))

    return locations


def load_sms(zip_path: Union[str, Path], level: str, split: str) -> list[str]:
    """Load sms.json and return a list of raw SMS thread strings."""
    prefix = _inner_prefix(level, split)
    with zipfile.ZipFile(zip_path) as zf:
        with _open_entry(zf, prefix, "sms.json") as f:
            raw = json.load(f)

    # Each element is {"sms": "<thread text>"} — extract the string
    return [entry["sms"] for entry in raw if isinstance(entry, dict) and "sms" in entry]


def load_mails(zip_path: Union[str, Path], level: str, split: str) -> list[str]:
    """Load mails.json and return a list of raw email/HTML thread strings."""
    prefix = _inner_prefix(level, split)
    with zipfile.ZipFile(zip_path) as zf:
        with _open_entry(zf, prefix, "mails.json") as f:
            raw = json.load(f)

    return [entry["mail"] for entry in raw if isinstance(entry, dict) and "mail" in entry]


def load_dataset(data_dir: str, level: str, split: str) -> dict:
    """Load all five data sources for a level/split and return them as a dict.

    Returns:
        {
            "transactions": pd.DataFrame,
            "users": list[dict],
            "locations": list[dict],
            "sms": list[str],
            "mails": list[str],
        }
    """
    zip_path = resolve_zip_path(data_dir, level, split)
    return {
        "transactions": load_transactions(zip_path, level, split),
        "users": load_users(zip_path, level, split),
        "locations": load_locations(zip_path, level, split),
        "sms": load_sms(zip_path, level, split),
        "mails": load_mails(zip_path, level, split),
    }
