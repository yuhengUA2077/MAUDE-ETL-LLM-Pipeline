"""Tests for the two defects that motivated this refactor, plus filter behaviour.

Run with: python -m pytest tests/ -v
"""

import pandas as pd
import pytest

from src.clean import apply_clean
from src.config import CleanConfig, ExtractConfig, FieldFilter
from src.extract import record_matches, select_partitions
from src.transform import flatten, normalise
from src.visualize import problem_flows

# A MAUDE category label that contains commas. Joining these with ", " is not
# reversible, which is the bug this repo's long-form tables remove.
COMMA_CATEGORY = "Activation, Positioning or Separation Problem"


def make_record(key="1", product_problems=None, patient_problems=None, text="short text"):
    return {
        "mdr_report_key": key,
        "event_type": "Injury",
        "date_received": "20210115",
        "product_problems": product_problems or [],
        "device": [{
            "brand_name": "OTSC SYSTEM",
            "generic_name": "clip",
            "manufacturer_d_name": "OVESCO ENDOSCOPY AG",
            "device_report_product_code": "PKL",
            "openfda": {
                "device_name": "Clip, Implantable",
                "medical_specialty_description": "Gastroenterology, Urology",
            },
        }],
        "patient": [{"patient_problems": patient_problems or []}],
        "mdr_text": [{"mdr_text_key": "1", "text_type_code": "D", "text": text}],
    }


class TestCommaCategories:
    """The original transform joined multi-valued fields with ', ', which could
    not be split back because the values themselves contain commas."""

    def test_categories_survive_as_discrete_rows(self):
        rec = make_record(product_problems=[COMMA_CATEGORY, "Break"])
        tables = normalise([rec])
        got = tables["product_problems"]["product_problem"].tolist()
        assert got == [COMMA_CATEGORY, "Break"], "comma-containing label was split"

    def test_flat_delimiter_is_reversible(self):
        rec = make_record(product_problems=[COMMA_CATEGORY, "Break"])
        flat = flatten(normalise([rec]))
        assert flat.loc[0, "product_problem"].split("|") == [COMMA_CATEGORY, "Break"]

    def test_sankey_needs_no_special_case_list(self):
        rec = make_record(
            product_problems=[COMMA_CATEGORY], patient_problems=["Perforation"]
        )
        tables = normalise([rec])
        flows = problem_flows(tables["product_problems"], tables["patient_problems"])
        assert flows.loc[0, "source"] == COMMA_CATEGORY
        assert flows.loc[0, "count"] == 1


class TestProblemFlows:
    def test_cross_product_and_uninformative_drop(self):
        recs = [
            make_record("1", ["Break", "Misfire"], ["Perforation", "Hemorrhage"]),
            make_record("2", ["Break"], ["No Information"]),
        ]
        tables = normalise(recs)
        flows = problem_flows(tables["product_problems"], tables["patient_problems"])
        assert flows["count"].sum() == 4, "report 1 should yield 2x2 pairs"
        assert "No Information" not in flows["target"].values


class TestExtractFilters:
    def test_manufacturer_and_brand_must_both_match(self):
        cfg = ExtractConfig(device_filters=[
            FieldFilter(field="manufacturer_d_name", pattern=r"\bOVESCO\b"),
            FieldFilter(field="brand_name", pattern=r"(clip|otsc|cutter|ftrd)"),
        ])
        assert record_matches(make_record(), cfg)

        rec = make_record()
        rec["device"][0]["manufacturer_d_name"] = "BOSTON SCIENTIFIC"
        assert not record_matches(rec, cfg)

    def test_malformed_record_is_skipped_not_raised(self):
        cfg = ExtractConfig(device_filters=[
            FieldFilter(field="brand_name", pattern="clip")
        ])
        assert not record_matches({"mdr_report_key": "x", "device": []}, cfg)
        assert not record_matches({"mdr_report_key": "x"}, cfg)


class TestClean:
    def test_word_threshold_and_reason_recorded(self):
        flat = pd.DataFrame({
            "mdr_report_key": ["1", "2"],
            "mdr_text": ["word " * 150, "too short"],
        })
        kept, excluded = apply_clean(flat, CleanConfig(min_narrative_words=100))
        assert kept["mdr_report_key"].tolist() == ["1"]
        assert excluded.loc[1, "exclusion_reason"] == "under_100_words"


class TestRetry:
    """Rate limits are the normal failure mode of a few hundred sequential calls."""

    def test_retries_then_succeeds(self, monkeypatch):
        from src import llm_extract

        monkeypatch.setattr(llm_extract.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 rate limit exceeded")
            return "ok"

        assert llm_extract.with_retry(flaky)() == "ok"
        assert calls["n"] == 3

    def test_non_retryable_fails_immediately(self, monkeypatch):
        from src import llm_extract

        monkeypatch.setattr(llm_extract.time, "sleep", lambda s: None)
        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise ValueError("schema field missing")

        with pytest.raises(ValueError):
            llm_extract.with_retry(broken)()
        assert calls["n"] == 1, "a schema error should not be retried six times"


class TestCleanDateBounds:
    """Partitions are quarterly; a study that stops mid-quarter is bounded here,
    after extraction, not by filtering what gets downloaded."""

    def _flat(self):
        return pd.DataFrame({
            "mdr_report_key": ["1", "2", "3", "4"],
            "date_received": ["20201231", "20210115", "20210201", "20210331"],
            "mdr_text": ["narrative"] * 4,
        })

    def test_end_bound_excludes_later_records(self):
        kept, excluded = apply_clean(
            self._flat(), CleanConfig(date_start="20120101", date_end="20210131")
        )
        assert kept["mdr_report_key"].tolist() == ["1", "2"]
        assert set(excluded["exclusion_reason"]) == {"after_20210131"}

    def test_start_bound_excludes_earlier_records(self):
        kept, excluded = apply_clean(self._flat(), CleanConfig(date_start="20210101"))
        assert kept["mdr_report_key"].tolist() == ["2", "3", "4"]
        assert excluded["exclusion_reason"].tolist() == ["before_20210101"]

    def test_no_bounds_keeps_everything(self):
        kept, _ = apply_clean(self._flat(), CleanConfig())
        assert len(kept) == 4


class TestDirtyRealWorldValues:
    """The synthetic fixtures above are cleaner than MAUDE is.

    A 2500-report run aborted in flatten() because a category column held NaN
    and str.join rejects non-strings. Nulls, blanks and stray numerics are all
    normal in the live corpus, so they are pinned here.
    """

    def test_flatten_survives_null_and_numeric_categories(self):
        tables = normalise([
            make_record("1", product_problems=[COMMA_CATEGORY], patient_problems=["Perforation"]),
            make_record("2", product_problems=[], patient_problems=[]),
        ])
        # Inject the shapes the live corpus actually produces.
        tables["product_problems"] = pd.concat([
            tables["product_problems"],
            pd.DataFrame([
                {"mdr_report_key": "2", "product_problem": None},
                {"mdr_report_key": "2", "product_problem": float("nan")},
                {"mdr_report_key": "2", "product_problem": "  "},
                {"mdr_report_key": "2", "product_problem": 1234},
            ]),
        ], ignore_index=True)

        flat = flatten(tables)
        assert len(flat) == 2
        assert flat.set_index("mdr_report_key").loc["1", "product_problem"] == COMMA_CATEGORY
        assert flat.set_index("mdr_report_key").loc["2", "product_problem"] == "1234"

    def test_flatten_survives_null_narrative_segments(self):
        tables = normalise([make_record("1", text="first segment")])
        tables["narrative"] = pd.concat([
            tables["narrative"],
            pd.DataFrame([
                {"mdr_report_key": "1", "mdr_text_key": "2", "text_type_code": "D",
                 "patient_sequence_number": "1", "text": None},
                {"mdr_report_key": "1", "mdr_text_key": "3", "text_type_code": "D",
                 "patient_sequence_number": "1", "text": float("nan")},
            ]),
        ], ignore_index=True)

        flat = flatten(tables)
        assert flat.loc[0, "mdr_text"] == "first segment"

    def test_all_null_category_column_yields_empty_not_crash(self):
        tables = normalise([make_record("1")])
        tables["patient_problems"] = pd.DataFrame([
            {"mdr_report_key": "1", "patient_problem": None},
        ])
        flat = flatten(tables)
        assert len(flat) == 1
