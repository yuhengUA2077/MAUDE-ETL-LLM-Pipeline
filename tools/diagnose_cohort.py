"""Why does this cohort differ from the number reported in the paper?

Run against a cohort produced by the extract stage:

    python tools/diagnose_cohort.py data/interim/<study>_cohort.jsonl [published_N]

Checks the three things that most often explain a small discrepancy, in order of
how cheap they are to rule out.
"""

import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else \
    "data/interim/shti2025_gu_pkl_narratives_cohort.jsonl"
expected = int(sys.argv[2]) if len(sys.argv) > 2 else None

records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
print(f"rows written by extract : {len(records)}")

# 1. Duplicate report keys. openFDA partitions a report by the quarter it was
#    received, but an amended report can be re-published, so the same
#    mdr_report_key can appear more than once across files. The published
#    analyses deduplicated (the working CSVs were named *_nodup.csv); the
#    extract stage does not.
keys = [r.get("mdr_report_key") for r in records]
unique = set(keys)
print(f"unique mdr_report_key   : {len(unique)}")
dupes = {k: n for k, n in Counter(keys).items() if n > 1}
if dupes:
    print(f"  -> {len(dupes)} keys appear more than once, {len(keys) - len(unique)} extra rows")
    for k, n in list(dupes.items())[:10]:
        print(f"     {k} x{n}")
else:
    print("  -> no duplicates; this is not the explanation")

# 2. Date spread. Confirms the date filter did what the config says, and shows
#    whether anything sits near the boundary.
dates = sorted(str(r.get("date_received", "")) for r in records if r.get("date_received"))
if dates:
    print(f"\ndate_received range     : {dates[0]} .. {dates[-1]}")
    by_month = Counter(d[:6] for d in dates)
    for month in sorted(by_month):
        print(f"  {month}: {by_month[month]}")

# 3. What the filters actually let through. If a value here looks unintended,
#    the filter is broader than the paper's description.
# The fields the filters actually keyed on. If the brand list contains products
# the study never meant to cover, the regex is looser than the original search.
print("\nmanufacturer_d_name:")
for name, n in Counter(
    (r.get("device") or [{}])[0].get("manufacturer_d_name", "(none)")
    for r in records
).most_common(15):
    print(f"  {n:4d}  {name}")

print("\nbrand_name:")
for brand, n in Counter(
    (r.get("device") or [{}])[0].get("brand_name", "(none)")
    for r in records
).most_common(30):
    print(f"  {n:4d}  {brand}")

print("\nby year:")
for year, n in sorted(Counter(
    str(r.get("date_received", ""))[:4] for r in records
).items()):
    print(f"  {year}: {n}")

print("\nproduct codes:")
for code, n in Counter(
    (r.get("device") or [{}])[0].get("device_report_product_code", "(none)")
    for r in records
).most_common():
    print(f"  {n:4d}  {code}")

print("\nmedical specialty:")
for spec, n in Counter(
    ((r.get("device") or [{}])[0].get("openfda") or {}).get(
        "medical_specialty_description", "(none)")
    for r in records
).most_common():
    print(f"  {n:4d}  {spec}")

print("\ntype_of_report:")
for kind, n in Counter(
    "|".join(r.get("type_of_report") or ["(none)"]) for r in records
).most_common():
    print(f"  {n:4d}  {kind}")

# 4. Reports listing more than one device. The extract filters on device[0]
#    only, matching the original analyses; a report whose matching device is not
#    first would be missed.
multi = sum(1 for r in records if len(r.get("device") or []) > 1)
print(f"\nreports with >1 device  : {multi}")

if expected is not None:
    print(f"\nafter dedup: {len(unique)}   paper reports: {expected}   "
          f"difference: {len(unique) - expected:+d}")
else:
    print(f"\nafter dedup: {len(unique)}   (pass the paper's N as a second "
          "argument to compare)")
