"""
Extract vessel port call data from Port of Barcelona CKAN API and store it locally
in Parquet format.


Data source:
Port of Barcelona Open Data (CKAN DataStore API)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import polars as pl
import requests


# ---------------------------------------------------------------------
# Defaults / Configuration
# ---------------------------------------------------------------------

DEFAULT_BASE_URL = "https://opendata.portdebarcelona.cat/ca/api/3/action/datastore_search"
DEFAULT_RESOURCE_ID = "c9d5fdb7-851b-45fb-b40d-5e1a76be8011"
DEFAULT_PAGE_SIZE = 1000  # CKAN DataStore commonly supports up to 1000
DEFAULT_SLEEP_SECONDS = 0.2  # polite delay between requests
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_SECONDS = 30

DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "port_calls_raw.parquet"


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger("extract_ckan")


def configure_logging(verbosity: int) -> None:
    """
    Configure logging level:
      -v   -> INFO
      -vv  -> DEBUG
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------
# CKAN helpers
# ---------------------------------------------------------------------


def _ckan_get_with_retries(
    session: requests.Session,
    url: str,
    params: dict[str, Any],
    *,
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    """
    Perform a GET request with retries + exponential backoff and return parsed JSON.
    Raises on permanent failure.
    """
    backoff = 1.0
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.debug("GET %s params=%s (attempt %d/%d)", url, params, attempt, max_retries)
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()

            payload = resp.json()
            if not isinstance(payload, dict):
                raise ValueError("Unexpected JSON payload type (expected object).")

            # CKAN convention: {"success": true/false, "result": {...}, "error": {...}}
            success = payload.get("success", True)  # some CKAN instances omit this
            if success is False:
                raise RuntimeError(f"CKAN API returned success=false: {payload.get('error')}")

            return payload

        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_exc = exc
            logger.warning("Request failed: %s", exc)

            if attempt == max_retries:
                break

            logger.info("Retrying in %.1fs...", backoff)
            time.sleep(backoff)
            backoff *= 2.0  # exponential backoff

    raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {last_exc}") from last_exc


def fetch_all_records(
    *,
    base_url: str,
    resource_id: str,
    page_size: int,
    sleep_seconds: float,
    timeout_seconds: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    """
    Fetch all records from the CKAN datastore using pagination.
    """
    records: list[dict[str, Any]] = []
    offset = 0

    with requests.Session() as session:
        while True:
            params = {
                "resource_id": resource_id,
                "limit": page_size,
                "offset": offset,
            }

            payload = _ckan_get_with_retries(
                session,
                base_url,
                params,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )

            result = payload.get("result")
            if not isinstance(result, dict):
                raise ValueError("CKAN response missing 'result' object.")

            batch = result.get("records")
            if not isinstance(batch, list):
                raise ValueError("CKAN response missing 'records' list inside 'result'.")

            # Defensive: ensure each item is a dict
            batch_dicts = [r for r in batch if isinstance(r, dict)]
            if len(batch_dicts) != len(batch):
                logger.warning("Some records were not objects; kept %d/%d", len(batch_dicts), len(batch))

            records.extend(batch_dicts)
            logger.info("Fetched batch: %d records (offset=%d). Total=%d", len(batch_dicts), offset, len(records))

            if len(batch_dicts) < page_size:
                break

            offset += page_size
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    return records


def atomic_write_parquet(df: pl.DataFrame, output_path: Path) -> None:
    """
    Write parquet to a temp file and then rename, to avoid partial/corrupt outputs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(output_path)


def write_metadata(output_path: Path, *, base_url: str, resource_id: str, row_count: int) -> None:
    """
    Write a small metadata JSON next to the parquet, useful for reproducibility/audits.
    """
    meta = {
        "base_url": base_url,
        "resource_id": resource_id,
        "row_count": row_count,
        "generated_at_epoch": int(time.time()),
    }
    meta_path = output_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract CKAN DataStore resource to Parquet.")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="CKAN datastore_search endpoint URL.")
    p.add_argument("--resource-id", default=DEFAULT_RESOURCE_ID, help="CKAN resource_id to extract.")
    p.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="Pagination page size (limit).")
    p.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Sleep between pages (seconds).")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout (seconds).")
    p.add_argument("--retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries per request.")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE), help="Output parquet path.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    return p.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    output_path = Path(args.output)

    logger.info("Starting CKAN extraction")
    logger.info("Resource: %s", args.resource_id)
    logger.info("Output: %s", output_path)

    records = fetch_all_records(
        base_url=args.base_url,
        resource_id=args.resource_id,
        page_size=args.page_size,
        sleep_seconds=args.sleep,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
    )

    if not records:
        logger.warning("No records returned. Writing an empty parquet file.")
        df = pl.DataFrame()
    else:
        df = pl.from_dicts(records)

    atomic_write_parquet(df, output_path)
    write_metadata(output_path, base_url=args.base_url, resource_id=args.resource_id, row_count=df.height)

    logger.info("Done. Rows=%d", df.height)
    logger.info("Wrote: %s", output_path.resolve())


if __name__ == "__main__":
    main()