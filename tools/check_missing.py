"""What happened to reports that were in an earlier cohort and are gone now?

    python check_missing.py <key> <key> <key>

For each key, asks openFDA whether the report still exists and prints the fields
this pipeline filters on, so the disappearance can be classified rather than
guessed at:

  * not found            -> withdrawn, or merged into another report
  * found, fields differ -> the record was edited after publication
  * found, fields match   -> still matching; the miss is ours, not the FDA's
"""

import json
import sys
import time
import urllib.error
import urllib.request

API = "https://api.fda.gov/device/event.json?search=mdr_report_key:{}"

keys = sys.argv[1:]
if not keys:
    raise SystemExit(__doc__)

for key in keys:
    try:
        with urllib.request.urlopen(API.format(key), timeout=30) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"{key}  NOT FOUND — withdrawn or merged into another report")
        else:
            print(f"{key}  HTTP {exc.code}")
        time.sleep(0.3)
        continue
    except Exception as exc:
        print(f"{key}  request failed: {exc}")
        time.sleep(0.3)
        continue

    results = payload.get("results") or []
    if not results:
        print(f"{key}  NOT FOUND — withdrawn or merged into another report")
        time.sleep(0.3)
        continue

    rec = results[0]
    devices = rec.get("device") or []
    print(f"{key}  STILL PRESENT — {len(devices)} device block(s)")
    print(f"    date_received : {rec.get('date_received')}")
    print(f"    date_changed  : {rec.get('date_changed')}")
    print(f"    type_of_report: {rec.get('type_of_report')}")
    for i, dev in enumerate(devices):
        openfda = dev.get("openfda") or {}
        print(f"    device[{i}] manufacturer: {dev.get('manufacturer_d_name')!r}")
        print(f"              brand       : {dev.get('brand_name')!r}")
        print(f"              generic     : {dev.get('generic_name')!r}")
        print(f"              product_code: {dev.get('device_report_product_code')!r}")
        print(f"              specialty   : {openfda.get('medical_specialty_description')!r}")
    time.sleep(0.3)

print(
    "\nA report that is still present with unchanged fields means the pipeline "
    "missed it.\nOne whose device order or names changed explains the miss. "
    "One that is gone was\nwithdrawn or merged — MAUDE amends downward as well "
    "as upward."
)
