"""Dump the raw JSON for specific MDRs, from both sources, and diff them.

    python dump_reports.py --keys <key> <key> --partitions "2018 Q3"

The API is live; the bulk partitions are what this pipeline actually reads. When
a report's device data has been edited, the two can disagree, and the pipeline
follows the bulk file. Writing both out makes that visible.

Output lands in data/inspect/ as <key>.api.json and <key>.bulk.json.
Omit --partitions to skip the bulk lookup and only query the API.
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extract import (  # noqa: E402
    _stream_partition, device_event_partitions, load_catalog, partition_quarter,
)

API = "https://api.fda.gov/device/event.json?search=mdr_report_key:{}"


def from_api(key: str) -> dict | None:
    try:
        with urllib.request.urlopen(API.format(key), timeout=30) as resp:
            results = json.load(resp).get("results") or []
        return results[0] if results else None
    except Exception as exc:
        print(f"  API lookup failed for {key}: {exc}")
        return None


def summarise(label: str, rec: dict | None) -> None:
    if rec is None:
        print(f"  {label:5s} not found")
        return
    devices = rec.get("device") or []
    print(f"  {label:5s} {len(devices)} device block(s), "
          f"received {rec.get('date_received')}, changed {rec.get('date_changed')}")
    for i, dev in enumerate(devices):
        print(f"        device[{i}] {dev.get('manufacturer_d_name')!r} / "
              f"{dev.get('brand_name')!r} / {dev.get('device_report_product_code')!r}")


ap = argparse.ArgumentParser()
ap.add_argument("--keys", nargs="+", required=True)
ap.add_argument("--partitions", nargs="*", default=[],
                help='Quarters to scan for the bulk copy, e.g. "2018 Q3" "2019 Q2"')
ap.add_argument("--out", default="data/inspect", type=Path)
args = ap.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
wanted = {str(k) for k in args.keys}

# --- live API ---------------------------------------------------------------
api_records: dict[str, dict] = {}
for key in wanted:
    rec = from_api(key)
    if rec:
        api_records[key] = rec
        (args.out / f"{key}.api.json").write_text(json.dumps(rec, indent=2))
    time.sleep(0.3)

# --- bulk partitions --------------------------------------------------------
bulk_records: dict[str, dict] = {}
if args.partitions:
    catalog = load_catalog(Path("data/raw/download.json"))
    quarters = {partition_quarter(q) or q for q in args.partitions}
    targets = [
        p for p in device_event_partitions(catalog)
        if partition_quarter(p["display_name"]) in quarters
    ]
    print(f"Scanning {len(targets)} bulk partition(s) for {len(wanted)} key(s)...")
    for part in targets:
        print(f"  {part['display_name']}")
        for rec in _stream_partition(part["file"]):
            key = str(rec.get("mdr_report_key", ""))
            if key in wanted:
                bulk_records[key] = rec
                (args.out / f"{key}.bulk.json").write_text(json.dumps(rec, indent=2))
        if len(bulk_records) == len(wanted):
            break

# --- report -----------------------------------------------------------------
print()
for key in sorted(wanted):
    print(f"{key}:")
    summarise("api", api_records.get(key))
    if args.partitions:
        summarise("bulk", bulk_records.get(key))
        a, b = api_records.get(key), bulk_records.get(key)
        if a and b and len(a.get("device") or []) != len(b.get("device") or []):
            print("        -> SOURCES DISAGREE on device blocks")

print(f"\nRaw JSON written to {args.out}/")
