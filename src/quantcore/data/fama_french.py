"""Fama-French factor data fetcher.

Downloads monthly factor returns from Kenneth French's data library.
Returns are converted to decimal form (percent ÷ 100).

Usage::

    fetcher = FamaFrenchFetcher()
    df = fetcher.fetch_3_factor()   # Monthly Mkt_RF, SMB, HML, RF
"""
from __future__ import annotations

import io

import pandas as pd
import requests

_FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_CSV.zip"
)


class FamaFrenchFetcher:
    """Downloads and parses Kenneth French Fama-French factor CSV files.

    Parameters
    ----------
    url_3factor:
        Override URL for the 3-factor monthly CSV zip. Defaults to the
        official Kenneth French data library URL.
    """

    def __init__(self, url_3factor: str = _FF3_URL) -> None:
        self._url_3factor = url_3factor

    def fetch_3_factor(self) -> pd.DataFrame:
        """Download and parse monthly Fama-French 3-factor data.

        Returns
        -------
        pd.DataFrame
            Monthly factor returns with columns ``Mkt_RF``, ``SMB``,
            ``HML``, ``RF`` in decimal form. Index is ``DatetimeIndex``
            (month-end frequency).
        """
        resp = requests.get(self._url_3factor, timeout=30)
        resp.raise_for_status()
        return _parse_ff3_csv(resp.text)


def _parse_ff3_csv(text: str) -> pd.DataFrame:
    """Parse the raw CSV text from French's data library.

    The format has several header lines before the data, and may include
    annual summary rows at the end.  We detect the data block by looking
    for lines where the date field is a 6-digit YYYYMM integer.
    """
    lines = text.splitlines()
    data_lines: list[str] = []
    header: list[str] | None = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # Header row: first token contains column names
        if header is None and "Mkt-RF" in parts[0] or (header is None and "Mkt-RF" in line):
            header = [p.strip().replace("-", "_") for p in parts]
            continue
        # Data row: first token is a 6-digit YYYYMM date
        if header is not None and len(parts) >= 5:
            try:
                date_int = int(parts[0])
                if 190001 <= date_int <= 210012:
                    data_lines.append(line)
            except ValueError:
                pass

    if not data_lines or header is None:
        raise ValueError("Could not parse Fama-French CSV — unexpected format")

    csv_text = ",".join(header) + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_text))

    # Rename date column (first column) and set as index
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index(date_col)
    df.index.name = "date"
    df.index = pd.DatetimeIndex(df.index)

    # Convert percent to decimal and return
    return df.astype(float) / 100.0
