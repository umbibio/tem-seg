from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table_from_file(filepath: str | Path) -> pd.DataFrame:
    """Return a DataFrame from a CSV, TSV, or XLSX file.

    Args:
        filepath: Path to the table file.

    Returns:
        pandas.DataFrame: DataFrame containing the table data.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(filepath)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".xlsx":
        return pd.read_excel(path)

    raise ValueError("Unsupported file type. Expected .csv, .tsv or .xlsx")
