"""Extract-stage tests using a locally served fake openFDA partition.

The download/unzip/filter/checkpoint path is the part of the pipeline most likely
to break silently, so it is exercised end to end here against a real HTTP server
and a real zip file rather than mocked out.
"""

import json
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from src.config import ExtractConfig, FieldFilter
from src.extract import corpus_summary, run_extract, select_partitions


def make_record(key, manufacturer, brand, product_code="PKL", date_received="20210215"):
    return {
        "mdr_report_key": str(key),
        "event_type": "Injury",
        "date_received": date_received,
        "product_problems": ["Activation, Positioning or Separation Problem"],
        "device": [{
            "brand_name": brand,
            "generic_name": "clip",
            "manufacturer_d_name": manufacturer,
            "device_report_product_code": product_code,
            "openfda": {"medical_specialty_description": "Gastroenterology, Urology"},
        }],
        "patient": [{"patient_problems": ["Perforation"]}],
        "mdr_text": [{"mdr_text_key": str(key), "text_type_code": "D", "text": "narrative"}],
    }


@pytest.fixture
def fake_openfda(tmp_path):
    """Serve a two-partition catalog and matching zipped payloads over HTTP."""
    served = tmp_path / "served"
    served.mkdir()

    partitions = {
        "2021 Q1 (part 1 of 2)": [
            make_record(1, "OVESCO ENDOSCOPY AG", "OTSC SYSTEM"),
            make_record(2, "BOSTON SCIENTIFIC", "RESOLUTION CLIP"),   # wrong manufacturer
            make_record(3, "OVESCO ENDOSCOPY AG", "FTRD SYSTEM"),
            {"mdr_report_key": "4", "device": []},                    # malformed
        ],
        "2021 Q1 (part 2 of 2)": [
            make_record(5, "OVESCO ENDOSCOPY AG", "OTSC CLIP"),
            make_record(6, "OLYMPUS", "SCOPE"),                       # wrong manufacturer
        ],
    }

    for i, (name, records) in enumerate(partitions.items(), 1):
        payload = served / f"part{i}.json"
        payload.write_text(json.dumps({"results": records}))
        with zipfile.ZipFile(served / f"part{i}.zip", "w") as zf:
            zf.write(payload, arcname=f"device-event-000{i}.json")
        payload.unlink()

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(SimpleHTTPRequestHandler, directory=str(served)),
    )
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    catalog = {"results": {"device": {"event": {
        "export_date": "2025-10-21",
        "total_records": 6,
        "partitions": [
            {"display_name": name,
             "file": f"http://127.0.0.1:{port}/part{i}.zip",
             "size_mb": "0.01",
             "records": len(records)}
            for i, (name, records) in enumerate(partitions.items(), 1)
        ],
    }}}}
    catalog_path = tmp_path / "download.json"
    catalog_path.write_text(json.dumps(catalog))

    yield catalog_path
    server.shutdown()


OVESCO_CFG = ExtractConfig(device_filters=[
    FieldFilter(field="manufacturer_d_name", pattern=r"\bOVESCO\b"),
    FieldFilter(field="brand_name", pattern=r"(clip|otsc|cutter|ftrd)"),
])


def test_full_extract_downloads_unzips_and_filters(fake_openfda, tmp_path):
    out = tmp_path / "cohort.jsonl"
    stats = run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda)

    assert stats["partitions"] == 2
    assert stats["scanned"] == 6
    assert stats["matched"] == 3, "should keep only the three OVESCO clip reports"

    keys = [json.loads(line)["mdr_report_key"] for line in out.open()]
    assert keys == ["1", "3", "5"]


def test_quarter_selects_all_its_parts(fake_openfda, tmp_path):
    cfg = ExtractConfig(
        partitions=["2021 Q1"],
        device_filters=OVESCO_CFG.device_filters,
    )
    stats = run_extract(cfg, tmp_path / "cohort.jsonl", catalog_cache=fake_openfda)
    assert stats["partitions"] == 2, "a quarter split across files must select every part"
    assert stats["matched"] == 3


def test_resume_skips_completed_partitions(fake_openfda, tmp_path):
    out = tmp_path / "cohort.jsonl"
    run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda)
    first = out.read_text()

    # A second run with resume on must not re-download or duplicate rows.
    stats = run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda, resume=True)
    assert out.read_text() == first
    assert stats["matched"] == 3


def test_corpus_summary_reads_catalog_totals(fake_openfda):
    summary = corpus_summary(json.loads(Path(fake_openfda).read_text()))
    assert summary["total_records"] == 6
    assert summary["partitions"] == 2


def test_quarter_naming_variants():
    """openFDA labels a quarter as '(all)' when small and '(part N of M)' when not.

    The first version of this fixture used '(all)' throughout, so the selection
    bug that missed every multi-part quarter passed the tests. Both forms are
    pinned here now.
    """
    from src.extract import partition_quarter, select_partitions

    assert partition_quarter("2004 Q3 (all)") == "2004 Q3"
    assert partition_quarter("2021 Q1 (part 4 of 6)") == "2021 Q1"
    assert partition_quarter("not a quarter") is None

    parts = [
        {"display_name": "2021 Q1 (part 1 of 6)"},
        {"display_name": "2021 Q1 (part 6 of 6)"},
        {"display_name": "2021 Q2 (part 1 of 6)"},
        {"display_name": "2004 Q3 (all)"},
    ]
    got = select_partitions(parts, ExtractConfig(partitions=["2021 Q1", "2004 Q3"]))
    assert len(got) == 3


class TestYearPartitions:
    """A bare year selects all four of its quarters, however they are split."""

    def test_year_selects_every_quarter_and_part(self):
        parts = [
            {"display_name": "2011 Q4 (all)"},
            {"display_name": "2012 Q1 (part 1 of 3)"},
            {"display_name": "2012 Q1 (part 2 of 3)"},
            {"display_name": "2012 Q4 (all)"},
            {"display_name": "2013 Q1 (all)"},
        ]
        got = [p["display_name"] for p in
               select_partitions(parts, ExtractConfig(partitions=["2012"]))]
        assert got == ["2012 Q1 (part 1 of 3)", "2012 Q1 (part 2 of 3)", "2012 Q4 (all)"]

    def test_years_and_quarters_combine(self):
        """The 2024 study covers whole years plus one extra quarter."""
        parts = [
            {"display_name": "2012 Q2 (all)"},
            {"display_name": "2020 Q4 (all)"},
            {"display_name": "2021 Q1 (part 1 of 6)"},
            {"display_name": "2021 Q2 (all)"},
        ]
        got = [p["display_name"] for p in select_partitions(
            parts, ExtractConfig(partitions=["2012", "2020", "2021 Q1"]))]
        assert got == ["2012 Q2 (all)", "2020 Q4 (all)", "2021 Q1 (part 1 of 6)"]

    def test_bad_spec_is_rejected(self):
        with pytest.raises(ValueError, match="year or quarter"):
            select_partitions([], ExtractConfig(partitions=["Q1 2021ish"]))

    def test_removed_date_keys_fail_loudly(self, tmp_path):
        """An old config must not silently lose its date bounds."""
        import yaml
        from src.config import StudyConfig

        cfg_path = tmp_path / "s.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "name": "t", "extract": {"date_start": "20120101"},
        }))
        with pytest.raises(ValueError, match="no longer supported"):
            StudyConfig.from_yaml(cfg_path)


class TestCheckpointInvalidation:
    """A checkpoint records which partitions are done, not what they were
    filtered by. Resuming after a config change silently produced a cohort that
    looked complete but was assembled under two different sets of filters."""

    def test_changed_filters_discard_the_old_cohort(self, fake_openfda, tmp_path):
        out = tmp_path / "cohort.jsonl"

        first = run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda)
        assert first["matched"] == 3

        # Same partitions, narrower filter: the previous run must not be reused.
        narrower = ExtractConfig(device_filters=[
            FieldFilter(field="manufacturer_d_name", pattern=r"\bOVESCO\b"),
            FieldFilter(field="brand_name", pattern=r"\bFTRD\b"),
        ])
        second = run_extract(narrower, out, catalog_cache=fake_openfda, resume=True)

        assert second["matched"] == 1, "stale checkpoint was reused"
        keys = [json.loads(line)["mdr_report_key"] for line in out.open()]
        assert keys == ["3"], "old cohort rows survived the filter change"

    def test_identical_config_still_resumes(self, fake_openfda, tmp_path):
        out = tmp_path / "cohort.jsonl"
        run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda)
        before = out.read_text()
        again = run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda, resume=True)
        assert out.read_text() == before
        assert again["matched"] == 3

    def test_fingerprint_ignores_irrelevant_ordering(self):
        from src.extract import config_fingerprint

        a = ExtractConfig(partitions=["2021 Q1", "2012"], product_codes=["PKL", "GEH"])
        b = ExtractConfig(partitions=["2012", "2021 Q1"], product_codes=["GEH", "PKL"])
        assert config_fingerprint(a) == config_fingerprint(b)


def test_cohort_is_stamped_with_export_date(fake_openfda, tmp_path):
    """Two runs of the same query differ if the corpus changed between them, so
    a cohort without a corpus date cannot be compared to anything."""
    out = tmp_path / "cohort.jsonl"
    stats = run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda)
    assert stats["export_date"] == "2025-10-21"

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["export_date"] == "2025-10-21"
    assert meta["matched"] == 3
    assert meta["fingerprint"] and meta["extracted_at"]


def test_no_resume_starts_clean(fake_openfda, tmp_path):
    """--no-resume must discard the old cohort, not append a second copy to it."""
    out = tmp_path / "cohort.jsonl"
    run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda)
    assert sum(1 for _ in out.open()) == 3

    run_extract(OVESCO_CFG, out, catalog_cache=fake_openfda, resume=False)
    keys = [json.loads(line)["mdr_report_key"] for line in out.open()]
    assert keys == ["1", "3", "5"], "re-run duplicated the previous cohort"


class TestMultiDeviceReports:
    """A report can list several devices. Testing only the first is what the
    original analyses did, and is preserved as the default."""

    @staticmethod
    def _two_devices():
        rec = make_record(9, "SOME OTHER MAKER", "UNRELATED PROBE",
                          product_code="ZZZ")
        rec["device"].append({
            "brand_name": "OTSC SYSTEM",
            "generic_name": "clip",
            "manufacturer_d_name": "OVESCO ENDOSCOPY AG",
            "device_report_product_code": "PKL",
            "openfda": {"medical_specialty_description": "Gastroenterology, Urology"},
        })
        return rec

    def test_default_misses_a_match_in_the_second_block(self):
        from src.extract import record_matches
        assert not record_matches(self._two_devices(), OVESCO_CFG)

    def test_match_any_device_finds_it(self):
        from src.extract import record_matches
        cfg = ExtractConfig(
            device_filters=OVESCO_CFG.device_filters, match_any_device=True
        )
        assert record_matches(self._two_devices(), cfg)

    def test_product_code_also_respects_the_flag(self):
        from src.extract import record_matches
        rec = self._two_devices()
        strict = ExtractConfig(product_codes=["PKL"])
        loose = ExtractConfig(product_codes=["PKL"], match_any_device=True)
        assert not record_matches(rec, strict)
        assert record_matches(rec, loose)


def test_deviceless_records_are_counted_not_just_skipped(fake_openfda, tmp_path):
    """A report whose device array is empty cannot match a device-field filter.

    Such reports leave a cohort silently even though they are still published,
    so the count is surfaced in the run stats.
    """
    from src.extract import has_device_data

    assert not has_device_data({"mdr_report_key": "1", "device": []})
    assert not has_device_data({"mdr_report_key": "1"})
    assert has_device_data({"mdr_report_key": "1", "device": [{"brand_name": "X"}]})

    stats = run_extract(OVESCO_CFG, tmp_path / "c.jsonl", catalog_cache=fake_openfda)
    # The fixture includes one malformed record with an empty device list.
    assert stats["records_without_device_data"] == 1
