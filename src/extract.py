"""Extract stage: stream the openFDA device-event corpus and filter to a cohort.

The full corpus is ~22.9M MDRs across 334 quarterly partitions (~16 GB compressed,
several hundred GB expanded). Partitions are therefore downloaded, filtered, and
discarded one at a time; the full dataset is never materialised on disk.

Progress is checkpointed per partition so an interrupted run resumes rather than
restarting a multi-hour scan.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

from .config import ExtractConfig, FieldFilter

log = logging.getLogger(__name__)

CATALOG_URL = "https://api.fda.gov/download.json"


def load_catalog(cache_path: Path | None = None) -> dict[str, Any]:
    """Fetch the openFDA download catalog, caching it locally.

    The catalog carries per-partition record counts, which is where the corpus
    totals quoted in the README come from -- no full scan required.
    """
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())
    log.info("Fetching openFDA catalog from %s", CATALOG_URL)
    resp = requests.get(CATALOG_URL, timeout=60)
    resp.raise_for_status()
    catalog = resp.json()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(catalog))
    return catalog


def device_event_partitions(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return catalog["results"]["device"]["event"]["partitions"]


def corpus_summary(catalog: dict[str, Any]) -> dict[str, Any]:
    ev = catalog["results"]["device"]["event"]
    parts = ev["partitions"]
    return {
        "export_date": ev.get("export_date"),
        "total_records": ev.get("total_records"),
        "partitions": len(parts),
        "size_mb": round(sum(float(p.get("size_mb", 0)) for p in parts), 1),
    }


QUARTER_RE = re.compile(r"\b((?:19|20)\d{2})\s*Q([1-4])\b", re.IGNORECASE)


def partition_quarter(display_name: str) -> str | None:
    """Normalise a partition label to 'YYYY QN'.

    openFDA splits busy quarters across several files -- a quarter appears either
    as '2004 Q3 (all)' or as '2021 Q1 (part 1 of 6)' through '(part 6 of 6)'.
    Selecting a quarter must therefore match every one of its parts.
    """
    m = QUARTER_RE.search(display_name)
    return f"{m.group(1)} Q{m.group(2)}" if m else None


def config_fingerprint(cfg: ExtractConfig) -> str:
    """Stable hash of everything that determines which records match.

    The checkpoint is keyed on this. Resuming a run whose filters have changed
    would silently mix records selected under two different configs -- and the
    resulting cohort would look like a completed run, which is worse than a
    crash.
    """
    payload = {
        "partitions": sorted(cfg.partitions) if cfg.partitions else None,
        "product_codes": sorted(cfg.product_codes),
        "device_filters": sorted((f.field, f.pattern, f.ignore_case)
                                 for f in cfg.device_filters),
        "openfda_filters": sorted((f.field, f.pattern, f.ignore_case)
                                  for f in cfg.openfda_filters),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def select_partitions(partitions: list[dict], cfg: ExtractConfig) -> list[dict]:
    """Narrow the partition list before downloading anything.

    Skipping partitions is by far the cheapest filter available: a study scoped
    to one quarter reads 6 of 334 files instead of all of them.
    """
    if cfg.partitions:
        quarters, years = set(), set()
        for spec in cfg.partitions:
            spec = str(spec).strip()
            if q := partition_quarter(spec):
                quarters.add(q)
            elif re.fullmatch(r"(?:19|20)\d{2}", spec):
                years.add(spec)
            else:
                raise ValueError(
                    f"partition {spec!r} is not a year or quarter; "
                    'use e.g. "2012" or "2021 Q1"'
                )

        selected, seen = [], set()
        for p in partitions:
            q = partition_quarter(p["display_name"])
            if q and (q in quarters or q[:4] in years):
                selected.append(p)
                seen.add(q)
                seen.add(q[:4])

        missing = (quarters | years) - seen
        if missing:
            log.warning("No partitions matched: %s", sorted(missing))
        return selected

    return list(partitions)


def _block_matches(block: dict, filters: list[FieldFilter], subkey: str | None) -> bool:
    if subkey:
        block = block.get(subkey) or {}
    for f in filters:
        value = block.get(f.field, "")
        if isinstance(value, list):
            value = " ".join(str(v) for v in value)
        if not f.compiled().search(str(value or "")):
            return False
    return True


def _matches(record: dict, filters: list[FieldFilter], subkey: str | None,
             any_device: bool = False) -> bool:
    """Apply regex filters to the first device block of an MDR.

    Returns False on malformed records rather than raising: the corpus spans
    30+ years and older partitions have inconsistent device blocks.
    """
    if not filters:
        return True
    devices = record.get("device") or []
    if not devices:
        return False
    blocks = devices if any_device else devices[:1]
    return any(_block_matches(b, filters, subkey) for b in blocks)


def has_device_data(record: dict) -> bool:
    """Whether the record carries any device block at all.

    Device-field filters cannot see a report with an empty device array, so such
    reports drop out of every device-keyed cohort silently -- the record is still
    published, it just stops matching. Counting them makes that visible in the
    run stats.
    """
    return bool(record.get("device"))


def record_matches(record: dict, cfg: ExtractConfig) -> bool:
    """Device-field filters only.

    There is deliberately no date test here. The study period is set by which
    quarters are downloaded; a date filter layered on top would make a
    filtered-out quarter look identical to an empty one in the logs.
    """
    devices = record.get("device") or []
    if cfg.product_codes:
        if not devices:
            return False
        blocks = devices if cfg.match_any_device else devices[:1]
        if not any(b.get("device_report_product_code", "") in cfg.product_codes
                   for b in blocks):
            return False
    if not _matches(record, cfg.device_filters, None, cfg.match_any_device):
        return False
    if not _matches(record, cfg.openfda_filters, "openfda", cfg.match_any_device):
        return False
    return True


def _stream_partition(url: str) -> Iterator[dict]:
    """Download one partition, yield its records, and delete it.

    Note: the original implementation issued two GET requests per partition
    (one to check the status code, one to save), doubling ~16 GB of transfer.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "partition.zip"

        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

        with zipfile.ZipFile(zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json")]
            if not names:
                log.warning("No JSON payload in %s", url)
                return
            with zf.open(names[0]) as fh:
                payload = json.load(fh)

        yield from payload.get("results", [])


def run_extract(
    cfg: ExtractConfig,
    out_path: Path,
    catalog_cache: Path | None = None,
    resume: bool = True,
) -> dict[str, int]:
    """Filter the corpus to a cohort, written as newline-delimited JSON.

    JSONL rather than a single JSON array so that a partial run remains a valid,
    readable file and matched records survive an interruption.
    """
    catalog = load_catalog(catalog_cache)
    summary = corpus_summary(catalog)
    partitions = select_partitions(device_event_partitions(catalog), cfg)
    # A MAUDE query is not reproducible without the date it was run: the FDA
    # adds reports to quarters that have already closed. Stamping the cohort
    # with the catalog's export date is what makes two runs comparable.
    log.info(
        "Selected %d partitions | openFDA export_date %s",
        len(partitions), summary.get("export_date"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(".state.json")

    fingerprint = config_fingerprint(cfg)
    done: set[str] = set()
    if not resume:
        # An explicit re-run must not append to, or silently reuse, the previous
        # cohort -- that is exactly the failure mode the fingerprint guards
        # against when a config changes.
        if out_path.exists() or state_path.exists():
            log.info("Re-running from scratch; discarding the previous cohort.")
        out_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
    elif state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("fingerprint") == fingerprint:
            done = set(state["completed"])
            log.info("Resuming: %d partitions already done", len(done))
        else:
            log.warning(
                "Filters changed since the last run (%s -> %s); discarding the "
                "previous cohort and starting over.",
                state.get("fingerprint", "none"), fingerprint,
            )
            out_path.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)

    deviceless = 0
    scanned = sum(
        p.get("records", 0) for p in partitions if p["display_name"] in done
    )
    matched = 0
    if done and out_path.exists():
        matched = sum(1 for _ in out_path.open())

    mode = "a" if done else "w"
    with out_path.open(mode, encoding="utf-8") as out:
        for i, part in enumerate(partitions, 1):
            name = part["display_name"]
            if name in done:
                continue
            log.info(
                "[%d/%d] %s (%s records, %s MB)",
                i, len(partitions), name,
                part.get("records", "?"), part.get("size_mb", "?"),
            )
            try:
                hits = 0
                for record in _stream_partition(part["file"]):
                    scanned += 1
                    if not has_device_data(record):
                        deviceless += 1
                    if record_matches(record, cfg):
                        out.write(json.dumps(record) + "\n")
                        hits += 1
                matched += hits
                log.info("  matched %d (cohort total %d)", hits, matched)
            except Exception as exc:
                # One bad partition should not lose hours of completed work.
                log.error("  FAILED %s: %s", name, exc)
                continue

            out.flush()
            done.add(name)
            state_path.write_text(json.dumps(
                {"fingerprint": fingerprint, "completed": sorted(done)}))

    stats = {
        "scanned": scanned,
        "matched": matched,
        "partitions": len(partitions),
        "records_without_device_data": deviceless,
        "export_date": summary.get("export_date"),
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(stats, indent=2))
    log.info("Extract complete: %s", stats)
    return stats
