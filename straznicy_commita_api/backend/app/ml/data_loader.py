from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError("Unsupported dataset format")
