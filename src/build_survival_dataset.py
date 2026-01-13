"""\
Build a survival-analysis-ready dataset from raw Port of Barcelona CKAN extracts.

Why this script exists
----------------------
- Raw data is downloaded via CKAN (src/extract_ckan.py)
- This step turns it into a *survival-ready* table with:
    - T : time-to-event (duration)
    - E : event indicator (1=event observed, 0=right-censored)

Input
-----
- data/raw/port_calls_raw.parquet
- data/raw/port_calls_raw.metadata.json (optional; used to set an "as of" cutoff)

Output
------
- data/processed/port_calls_survival.parquet
- data/processed/port_calls_survival.metadata.json

Survival framing (planning data)
-------------------------------
- Unit of analysis: port call (ESCALANUM)
- Start time: ETAUTC
- Planned event time: ETDUTC
- Cutoff: analysis timestamp (defaults to input metadata generated_at_epoch)

Definitions
-----------
- We exclude future calls that have not started yet (ETAUTC > cutoff)
  because they are not "at risk" at the analysis time.

- Event indicator:
    E = 1 if ESCALAESTAT == 'Finalitzada' AND ETDUTC is not null AND ETDUTC <= cutoff
    E = 0 otherwise  (right-censored at cutoff)

- Time-to-event:
    end_time = ETDUTC if ETDUTC <= cutoff else cutoff
    T = end_time - ETAUTC   (in hours by default)

Notes on Polars implementation
------------------------------
This script uses native columnar operations (no per-row loops).
For duration, we rely on the fact that (Datetime - Datetime) yields a Duration
in microseconds; we cast to Int64 and scale to seconds/minutes/hours.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger("build_survival_dataset")

# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_INPUT_PARQUET = Path("data/raw/port_calls_raw.parquet")
DEFAULT_INPUT_META = Path("data/raw/port_calls_raw.metadata.json")

DEFAULT_OUTPUT_PARQUET = Path("data/processed/port_calls_survival.parquet")
DEFAULT_OUTPUT_META = Path("data/processed/port_calls_survival.metadata.json")

DEFAULT_FINAL_STATUS = "Finalitzada"
DEFAULT_TIME_UNIT = "hours"  # hours | minutes | seconds


# ---------------------------------------------------------------------
# CLI / Logging
# ---------------------------------------------------------------------

def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a survival-ready dataset from raw CKAN extract.")

    p.add_argument("--input", default=str(DEFAULT_INPUT_PARQUET), help="Path to raw parquet.")
    p.add_argument("--input-meta", default=str(DEFAULT_INPUT_META), help="Path to raw metadata.json (optional).")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_PARQUET), help="Path to output survival parquet.")
    p.add_argument("--output-meta", default=str(DEFAULT_OUTPUT_META), help="Path to output metadata.json.")

    p.add_argument(
        "--cutoff",
        default=None,
        help=(
            "Analysis cutoff time (UTC). If not provided, uses generated_at_epoch from input-meta. "
            "Accepted formats: ISO8601 (e.g. 2026-01-13T12:00:00Z) or epoch seconds."
        ),
    )

    p.add_argument("--final-status", default=DEFAULT_FINAL_STATUS, help="Value of ESCALAESTAT indicating completion.")
    p.add_argument(
        "--time-unit",
        choices=["hours", "minutes", "seconds"],
        default=DEFAULT_TIME_UNIT,
        help="Duration unit for T.",
    )

    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    return p.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _parse_cutoff(cutoff_str: str | None, input_meta_path: Path) -> datetime:
    """Return cutoff as an aware UTC datetime."""

    # 1) Explicit cutoff from CLI
    if cutoff_str:
        s = cutoff_str.strip()

        # epoch seconds?
        if s.isdigit():
            return datetime.fromtimestamp(int(s), tz=timezone.utc)

        # ISO8601 (allow trailing Z)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"

        try:
            dt = datetime.fromisoformat(s)
        except ValueError as e:
            raise ValueError(
                "Invalid --cutoff. Use epoch seconds or ISO8601 like 2026-01-13T12:00:00Z"
            ) from e

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    # 2) Metadata generated_at_epoch
    if input_meta_path.exists():
        meta = json.loads(input_meta_path.read_text(encoding="utf-8"))
        epoch = meta.get("generated_at_epoch")
        if isinstance(epoch, int):
            return datetime.fromtimestamp(epoch, tz=timezone.utc)

    # 3) Fallback: now (UTC)
    logger.warning("No cutoff provided and no valid input metadata found. Falling back to now().")
    return datetime.now(tz=timezone.utc)


def _ensure_datetime_utc(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Ensure a column is Datetime('us', 'UTC') using native Polars parsing/casting."""
    dtype = df.schema.get(col)

    # If strings, parse. If already datetime, cast to UTC timezone if needed.
    if dtype == pl.Utf8:
        return df.with_columns(
            pl.col(col)
            .str.strptime(pl.Datetime, strict=False).dt.replace_time_zone("UTC")
            .alias(col)
        )

    # If already datetime-like, cast (strict=False avoids hard failures)
    return df.with_columns(pl.col(col).cast(pl.Datetime(time_zone="UTC"), strict=False).alias(col))


def _cast_optional_float(df: pl.DataFrame, col: str) -> pl.DataFrame:
    if col in df.columns:
        return df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col))
    return df


def _duration_in_unit(end_col: str, start_col: str, unit: str) -> pl.Expr:
    """Compute duration using Polars-native duration arithmetic.

    (Datetime - Datetime) -> Duration (microseconds). Cast to Int64 and scale.
    """
    dur_us = (pl.col(end_col) - pl.col(start_col)).cast(pl.Int64)

    seconds = dur_us / 1_000_000.0
    if unit == "seconds":
        return seconds
    if unit == "minutes":
        return seconds / 60.0
    # hours
    return seconds / 3600.0


def write_metadata(
    output_meta_path: Path,
    *,
    input_path: Path,
    input_meta_path: Path,
    cutoff: datetime,
    row_count: int,
    excluded_future_calls: int,
    invalid_rows_dropped: int,
) -> None:
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "input_path": str(input_path),
        "input_meta_path": str(input_meta_path) if input_meta_path.exists() else None,
        "cutoff_utc": cutoff.isoformat(),
        "row_count": row_count,
        "excluded_future_calls": excluded_future_calls,
        "invalid_rows_dropped": invalid_rows_dropped,
        "generated_at_epoch": int(datetime.now(tz=timezone.utc).timestamp()),
    }

    output_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    input_path = Path(args.input)
    input_meta_path = Path(args.input_meta)
    output_path = Path(args.output)
    output_meta_path = Path(args.output_meta)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw parquet not found: {input_path}")

    cutoff = _parse_cutoff(args.cutoff, input_meta_path)
    logger.info("Using cutoff (UTC): %s", cutoff.isoformat())

    df = pl.read_parquet(input_path)

    required_cols = ["ESCALANUM", "ETAUTC", "ETDUTC", "ESCALAESTAT"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw data: {missing}")

    # Ensure datetimes are properly typed
    df = _ensure_datetime_utc(df, "ETAUTC")
    df = _ensure_datetime_utc(df, "ETDUTC")

    # Cast numeric ship dimensions when present (useful for later Cox)
    for c in ["ESLORA_METRES", "MANEGA_METRES", "CALAT_METRES"]:
        df = _cast_optional_float(df, c)

    cutoff_lit = pl.lit(cutoff)

    # Exclude future calls that have not started yet (ETAUTC > cutoff)
    df_started = df.filter(pl.col("ETAUTC").is_not_null() & (pl.col("ETAUTC") <= cutoff_lit))
    excluded_future_calls = df.height - df_started.height

    # end_time = ETDUTC (if it exists and is <= cutoff) else cutoff
    df_started = df_started.with_columns(
        start_time=pl.col("ETAUTC"),
        end_time=pl.when(pl.col("ETDUTC").is_not_null() & (pl.col("ETDUTC") <= cutoff_lit))
        .then(pl.col("ETDUTC"))
        .otherwise(cutoff_lit),
    )

    # Event indicator
    df_started = df_started.with_columns(
        E=pl.when(
            (pl.col("ESCALAESTAT") == pl.lit(args.final_status))
            & pl.col("ETDUTC").is_not_null()
            & (pl.col("ETDUTC") <= cutoff_lit)
        )
        .then(1)
        .otherwise(0)
        .cast(pl.Int8)
    )

    # Time-to-event
    df_started = df_started.with_columns(
        T=_duration_in_unit("end_time", "start_time", args.time_unit)
    )

    # Drop invalid/negative durations
    valid = df_started.filter(pl.col("T").is_not_null() & (pl.col("T") >= 0))
    invalid_rows_dropped = df_started.height - valid.height

    # Keep a clean set of columns (raw + survival)
    keep_cols = [
        "ESCALANUM",
        "ETAUTC",
        "ETDUTC",
        "ESCALAESTAT",
        "TERMINALCODI",
        "TERMINALNOM",
        "VAIXELLTIPUS",
        "PORTORIGENCODI",
        "PORTORIGENNOM",
        "PORTDESTICODI",
        "PORTDESTINOM",
        "MOLLCODI",
        "MOLLMODULS",
        "CONSIGNATARI",
        "VAIXELLBANDERACODI",
        "VAIXELLBANDERANOM",
        "MMSI",
        "IMO",
        "CALLSIGN",
        "ESLORA_METRES",
        "MANEGA_METRES",
        "CALAT_METRES",
        "E",
        "T",
    ]

    keep_cols = [c for c in keep_cols if c in valid.columns]
    out = valid.select(keep_cols)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(output_path)

    write_metadata(
        output_meta_path,
        input_path=input_path,
        input_meta_path=input_meta_path,
        cutoff=cutoff,
        row_count=out.height,
        excluded_future_calls=excluded_future_calls,
        invalid_rows_dropped=invalid_rows_dropped,
    )

    logger.info("Wrote survival dataset: %s", output_path.resolve())
    logger.info(
        "Rows=%d | excluded_future_calls=%d | invalid_rows_dropped=%d",
        out.height,
        excluded_future_calls,
        invalid_rows_dropped,
    )


if __name__ == "__main__":
    main()
