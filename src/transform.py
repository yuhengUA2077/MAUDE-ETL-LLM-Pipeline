"""Transform stage: normalise matched MDRs into analysis-ready tables.

Two known defects in the original implementation are fixed here:

1. Multi-valued fields (product_problems, patient_problems) were joined with
   ", ". MAUDE category names contain commas of their own: "Activation,
   Positioning or Separation Problem" is one category, not three. The join was
   therefore not reversible, and downstream code had to carry a hand-maintained list
   of exceptions to split it back. A pipe delimiter is used instead, and the
   long-form tables keep one row per value so no un-splitting is needed at all.

2. Rows were accumulated with pd.concat inside the record loop, rebuilding the
   whole frame on every iteration. Rows are collected in lists and framed once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DELIM = "|"

GENERAL_COLUMNS = [
    "mdr_report_key", "event_type", "report_number", "type_of_report",
    "product_problem_flag", "date_received", "date_of_event",
    "reporter_occupation_code", "date_facility_aware", "date_added",
    "date_changed", "date_report", "report_source_code", "adverse_event_flag",
]

DEVICE_COLUMNS = [
    "mdr_report_key", "brand_name", "generic_name", "manufacturer_d_name",
    "device_report_product_code", "device_name", "medical_specialty_description",
    "manufacturer_d_city", "manufacturer_d_country", "catalog_number", "lot_number",
]


def read_cohort(path: Path) -> list[dict]:
    """Read the JSONL cohort produced by the extract stage."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _join(values) -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values
    return DELIM.join(str(v) for v in values)


def normalise(records: list[dict]) -> dict[str, pd.DataFrame]:
    """Split MDRs into five related tables keyed on mdr_report_key.

    The two *_problems tables are long-form (one row per report per category)
    because MAUDE category labels contain commas; keeping them long removes any
    need to re-split a delimited string downstream.
    """
    general, devices, patients, narratives = [], [], [], []
    product_problems, patient_problems = [], []

    for rec in records:
        key = rec.get("mdr_report_key")

        general.append({c: rec.get(c) for c in GENERAL_COLUMNS})

        for prob in rec.get("product_problems") or []:
            product_problems.append({"mdr_report_key": key, "product_problem": prob})

        for dev in rec.get("device") or []:
            openfda = dev.get("openfda") or {}
            devices.append({
                "mdr_report_key": key,
                "brand_name": dev.get("brand_name"),
                "generic_name": dev.get("generic_name"),
                "manufacturer_d_name": dev.get("manufacturer_d_name"),
                "device_report_product_code": dev.get("device_report_product_code"),
                "device_name": openfda.get("device_name"),
                "medical_specialty_description": openfda.get("medical_specialty_description"),
                "manufacturer_d_city": dev.get("manufacturer_d_city"),
                "manufacturer_d_country": dev.get("manufacturer_d_country"),
                "catalog_number": dev.get("catalog_number"),
                "lot_number": dev.get("lot_number"),
            })

        for pat in rec.get("patient") or []:
            patients.append({
                "mdr_report_key": key,
                "sequence_number_outcome": _join(pat.get("sequence_number_outcome")),
            })
            for prob in pat.get("patient_problems") or []:
                patient_problems.append({"mdr_report_key": key, "patient_problem": prob})

        for txt in rec.get("mdr_text") or []:
            narratives.append({
                "mdr_report_key": key,
                "mdr_text_key": txt.get("mdr_text_key"),
                "text_type_code": txt.get("text_type_code"),
                "patient_sequence_number": txt.get("patient_sequence_number"),
                "text": txt.get("text", ""),
            })

    return {
        "general": pd.DataFrame(general, columns=GENERAL_COLUMNS),
        "device": pd.DataFrame(devices, columns=DEVICE_COLUMNS),
        "patient": pd.DataFrame(patients).drop_duplicates(),
        "product_problems": pd.DataFrame(
            product_problems, columns=["mdr_report_key", "product_problem"]
        ).drop_duplicates(),
        "patient_problems": pd.DataFrame(
            patient_problems, columns=["mdr_report_key", "patient_problem"]
        ).drop_duplicates(),
        "narrative": pd.DataFrame(
            narratives,
            columns=["mdr_report_key", "mdr_text_key", "text_type_code",
                     "patient_sequence_number", "text"],
        ),
    }


def flatten(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per report: the wide table the LLM stage consumes.

    Narrative segments are concatenated per report; category columns are
    pipe-delimited, which is safe because MAUDE labels contain commas but not
    pipes.
    """
    general = tables["general"]
    device = tables["device"].drop_duplicates("mdr_report_key")

    narrative = (
        tables["narrative"]
        .sort_values(["mdr_report_key", "mdr_text_key"])
        .groupby("mdr_report_key")["text"]
        .apply(lambda s: " ".join(
            x for x in s.fillna("").astype(str).str.strip() if x))
        .rename("mdr_text")
        .reset_index()
    )

    def _agg(table: str, col: str) -> pd.DataFrame:
        df = tables[table]
        if df.empty:
            return pd.DataFrame(columns=["mdr_report_key", col])
        # Real MAUDE records carry nulls and non-string values in these columns,
        # which str.join rejects. Coerce and drop empties rather than letting a
        # single missing category abort a 2500-report run.
        clean = df.assign(**{col: df[col].fillna("").astype(str).str.strip()})
        clean = clean[clean[col] != ""]
        if clean.empty:
            return pd.DataFrame(columns=["mdr_report_key", col])
        return clean.groupby("mdr_report_key")[col].apply(DELIM.join).reset_index()

    flat = general.merge(device, on="mdr_report_key", how="left")
    for table, col in [("product_problems", "product_problem"),
                       ("patient_problems", "patient_problem")]:
        flat = flat.merge(_agg(table, col), on="mdr_report_key", how="left")
    flat = flat.merge(narrative, on="mdr_report_key", how="left")
    return flat.drop_duplicates("mdr_report_key")


def write_tables(tables: dict[str, pd.DataFrame], outdir: Path, prefix: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        path = outdir / f"{prefix}_{name}.csv"
        df.to_csv(path, index=False)
        log.info("Wrote %s (%d rows)", path.name, len(df))
