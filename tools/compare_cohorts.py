"""Which reports differ between an old result and a re-run?

    python compare_cohorts.py OLD NEW

OLD may be a CSV with an mdr_report_key column (an earlier analysis output) or a
JSONL cohort. NEW is normally a cohort from data/interim/, or the cleaned CSV
from data/processed/ if the study has date bounds.

    python compare_cohorts.py old_2479.csv data/processed/shti2024_endoscopic_clips_clean.csv

Prints the keys present in one and not the other, so a small discrepancy can be
looked up in MAUDE directly instead of guessed at.
"""

import json
import sys
from pathlib import Path


def load_keys(path: Path) -> set[str]:
    """Accept a JSONL cohort or any CSV carrying an mdr_report_key column."""
    if path.suffix == ".jsonl":
        return {
            str(json.loads(line)["mdr_report_key"])
            for line in path.open(encoding="utf-8") if line.strip()
        }

    import pandas as pd

    df = pd.read_csv(path, dtype=str)
    for col in ("mdr_report_key", "MDR_REPORT_KEY", "mdr report key"):
        if col in df.columns:
            return set(df[col].dropna().astype(str).str.strip())
    raise SystemExit(
        f"{path.name} has no mdr_report_key column. Columns are: {list(df.columns)}"
    )


if len(sys.argv) != 3:
    raise SystemExit(__doc__)

old_path, new_path = Path(sys.argv[1]), Path(sys.argv[2])
old, new = load_keys(old_path), load_keys(new_path)

print(f"{old_path.name}: {len(old)} keys")
print(f"{new_path.name}: {len(new)} keys")
print(f"in both      : {len(old & new)}")

missing = sorted(old - new)
added = sorted(new - old)

print(f"\nlost since the original run ({len(missing)}):")
for k in missing[:50]:
    print(f"  {k}   https://api.fda.gov/device/event.json?search=mdr_report_key:{k}")
if len(missing) > 50:
    print(f"  ... and {len(missing) - 50} more")

print(f"\nnew since the original run ({len(added)}):")
for k in added[:50]:
    print(f"  {k}   https://api.fda.gov/device/event.json?search=mdr_report_key:{k}")
if len(added) > 50:
    print(f"  ... and {len(added) - 50} more")

print(
    "\nOpen a lost key's URL: an empty result means the report was withdrawn or "
    "merged; a result that no longer matches the filters means a field was edited."
)
