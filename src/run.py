"""Pipeline entry point.

    python -m src.run --config configs/shti2025_ovesco_etl_llm.yaml
    python -m src.run --config configs/... --stages transform,clean,visualize

Stages are separable because extract is the expensive one: a full-corpus scan
should not be repeated to re-render a figure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import clean as clean_stage
from . import extract as extract_stage
from . import llm_extract, transform, visualize
from .config import StudyConfig

ALL_STAGES = ["extract", "transform", "clean", "llm", "visualize"]


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _compare(log, cohort: int, expected: int) -> None:
    delta = cohort - expected
    verdict = "MATCH" if delta == 0 else f"DIFFERS by {delta:+d}"
    log.info("Cohort %d vs %d reported in the paper -> %s", cohort, expected, verdict)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MAUDE ETL + LLM pipeline")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--data-dir", default=Path("data"), type=Path)
    ap.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any checkpoint and re-download from scratch. Only needed to "
             "re-measure a cohort against a newer corpus: a config change already "
             "invalidates the checkpoint on its own.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(args.verbose)
    log = logging.getLogger("run")

    cfg = StudyConfig.from_yaml(args.config)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    log.info("Study: %s", cfg.name)
    log.info("Citation: %s", cfg.citation)
    log.info("Stages: %s", ", ".join(stages))

    raw = args.data_dir / "raw"
    interim = args.data_dir / "interim"
    processed = args.data_dir / "processed"
    cohort_path = interim / f"{cfg.name}_cohort.jsonl"

    # -- extract ----------------------------------------------------------
    if "extract" in stages:
        stats = extract_stage.run_extract(
            cfg.extract, cohort_path,
            catalog_cache=raw / "download.json",
            resume=not args.no_resume,
        )
        if cfg.clean.date_start or cfg.clean.date_end:
            # Partitions are quarterly, so extraction overshoots a study whose
            # period ends mid-quarter. Comparing here would be comparing the
            # wrong number; the comparison happens after clean.
            log.info(
                "%d records extracted; date bounds are applied in the clean stage",
                stats["matched"],
            )
        elif cfg.expected_reports is not None:
            _compare(log, stats["matched"], cfg.expected_reports)

    # -- transform --------------------------------------------------------
    flat_path = processed / f"{cfg.name}_flat.csv"
    if "transform" in stages:
        records = transform.read_cohort(cohort_path)
        tables = transform.normalise(records)
        transform.write_tables(tables, processed, cfg.name)
        flat = transform.flatten(tables)
        flat.to_csv(flat_path, index=False)
        log.info("Wrote %s (%d reports)", flat_path.name, len(flat))

    # -- clean ------------------------------------------------------------
    clean_path = processed / f"{cfg.name}_clean.csv"
    if "clean" in stages:
        flat = pd.read_csv(flat_path)
        kept, excluded = clean_stage.apply_clean(flat, cfg.clean)
        kept.to_csv(clean_path, index=False)
        if len(excluded):
            excluded.to_csv(processed / f"{cfg.name}_excluded.csv", index=False)
        if cfg.expected_reports is not None:
            _compare(log, len(kept), cfg.expected_reports)

    # -- llm --------------------------------------------------------------
    if "llm" in stages and cfg.llm.enabled:
        source = clean_path if clean_path.exists() else flat_path
        df = pd.read_csv(source)
        for task in cfg.llm.tasks:
            if task == "consolidate_terms":
                # Pass 2: derive a controlled vocabulary from pass 1's free text.
                src_path = processed / f"{cfg.name}_narrative_extraction.csv"
                if not src_path.exists():
                    log.warning("consolidate_terms needs narrative_extraction first; skipping")
                    continue
                mapping = llm_extract.consolidate_terms(
                    pd.read_csv(src_path)["procedures"].dropna().tolist(),
                    provider=cfg.llm.provider,
                    model=cfg.llm.model,
                    temperature=cfg.llm.temperature,
                )
                mapping.to_csv(processed / f"{cfg.name}_term_mapping.csv", index=False)
                continue

            results, failures = llm_extract.run_task(
                df, task,
                provider=cfg.llm.provider,
                model=cfg.llm.model,
                temperature=cfg.llm.temperature,
                max_workers=cfg.llm.max_workers,
            )
            results.to_csv(processed / f"{cfg.name}_{task}.csv", index=False)
            if len(failures):
                failures.to_csv(processed / f"{cfg.name}_{task}_failures.csv", index=False)

    # -- visualize --------------------------------------------------------
    if "visualize" in stages:
        figures = processed / "figures"
        figures.mkdir(parents=True, exist_ok=True)

        prod = processed / f"{cfg.name}_product_problems.csv"
        pat = processed / f"{cfg.name}_patient_problems.csv"
        if prod.exists() and pat.exists():
            counts = visualize.problem_flows(pd.read_csv(prod), pd.read_csv(pat))
            counts.to_csv(processed / f"{cfg.name}_problem_flows.csv", index=False)
            visualize.save(
                visualize.sankey(counts, f"{cfg.name}: device problems to patient outcomes"),
                figures / f"{cfg.name}_sankey.html",
            )

        use_case_path = processed / f"{cfg.name}_use_case.csv"
        narrative_path = processed / f"{cfg.name}_narrative_extraction.csv"
        if use_case_path.exists() and narrative_path.exists():
            merged = pd.read_csv(narrative_path).merge(
                pd.read_csv(use_case_path), on="mdr_report_key", how="inner"
            )
            visualize.save(
                visualize.treemap(
                    merged, ["event_timing", "use_case"],
                    f"{cfg.name}: device use case by event timing",
                ),
                figures / f"{cfg.name}_treemap_use_case.html",
            )

    log.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
