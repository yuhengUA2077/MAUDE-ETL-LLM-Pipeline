# MAUDE ETL + LLM Pipeline

A reproducible pipeline over the FDA's Manufacturer and User Facility Device
Experience (MAUDE) database: streams the full public corpus, filters it to a
study cohort, normalises it, extracts structured facts from free-text adverse
event narratives with an LLM, and renders the figures.

Three published studies are included as configuration files. Each one re-runs
end to end and reports whether its cohort size matches the number printed in the
paper.

```bash
python -m src.run --config configs/shti2025_ovesco_etl_llm.yaml
```

---

## Why this exists

MAUDE is the FDA's public record of medical device adverse events — 22.9 million
reports going back to 1993. It is used heavily in patient safety research, and
that research is mostly not reproducible. A 2024 audit found that **11.5% of
MAUDE-based studies published an executable query, and only 23.3% of those
queries could be re-run.**

Most such studies describe their extraction as a sentence of prose: some keywords,
a date range, a device name. Re-running one means guessing.

The three studies in `configs/` were originally written the way most research
code is written: one script, edited in place for each new project, with the
previous project's filters commented out above the current ones. That code
produced published results but could not answer the question "which version made
Table 2?" This repo is that code rebuilt so it can.

## The corpus

Figures below are read from openFDA's own catalog (`api.fda.gov/download.json`),
not estimated:

| | |
|---|---|
| Total MDRs | **25,711,469** |
| Quarterly partitions | **365** |
| Compressed size | **17.8 GB** |
| Catalog export date | 2026-08-27 |

Ten months earlier the same catalog reported 22,881,433 records across 334
partitions. The corpus is not static, which is the point of the section below.

Busy quarters are split across several files -- 2021 Q1 is six -- so a study
scoped to one quarter still reads six partitions, not one.

Which quarters get downloaded is what defines the study period — a config names
years and quarters (`["2012", "2013", "2021 Q1"]`), and records are filtered on
device fields only. There is deliberately no date filter in the download stage:
layered on top of partition selection it makes a filtered-out quarter look
identical to an empty one in the logs, which is exactly the kind of wrong answer
that doesn't announce itself. Bounding a cohort by date belongs downstream.

Partitions are downloaded, filtered, and deleted one at a time. The full corpus
is never held on disk — expanded, it is several hundred GB, and a cohort is
typically tens to thousands of reports. Progress is checkpointed per partition,
so an interrupted multi-hour scan resumes where it stopped rather than starting
over. The checkpoint is keyed on a hash of the filters, so editing a config
discards the old cohort instead of resuming into it.

## Pipeline

```
                    openFDA catalog (334 partitions, 22.9M MDRs)
                                    │
   extract      stream ─▶ filter ─▶ discard partition        cohort.jsonl
                                    │
   transform    ├─ general ─┬─ device ─┬─ patient                (5 tables)
                ├─ product_problems ───┴─ patient_problems   (long form)
                └─ narrative                                  ─▶ flat.csv
                                    │
   clean        narrative quality filter          ─▶ clean.csv + excluded.csv
                                    │
   llm          Pydantic schema ─▶ OpenAI or Gemini ─▶ validated rows
                  ├─ narrative_extraction  (open-ended)
                  └─ use_case              (closed set)      + failures.csv
                                    │
   visualize    Sankey (problem flows) · Treemap (procedural context)
```

Stages are separable, because extract is the expensive one:

```bash
# Re-render figures without re-scanning the corpus
python -m src.run --config configs/... --stages visualize

# Re-measure a cohort against a newer corpus
python -m src.run --config configs/... --stages extract --no-resume
```

Editing a config already invalidates its checkpoint, so `--no-resume` is only
needed when the corpus has changed underneath an unchanged query — which, as
below, it does.

### The LLM stage

MAUDE's structured fields do not record which surgical procedure was underway,
when during it the event occurred, or how the device was being used. Those facts
exist only in the narrative text. Extraction runs in three passes, deliberately
ordered:

1. **Open-ended extraction** — GPT-4 fills a Pydantic schema per report. Free
   text, because the vocabulary is not known in advance.
2. **Term consolidation** — the resulting free-text procedure descriptions are
   grouped into standard terms. This is where the label set comes from.
3. **Closed-set classification** — each report is assigned one label from that
   consolidated set.

Starting at step 3 would have required knowing the answer beforehand.

The same schema runs against both OpenAI and Gemini. Each API attaches a schema
differently, but both constrain generation to the same contract, so outputs are
directly comparable across models. Switch providers with one line in the config.

`temperature=0` throughout. Both providers retry with exponential backoff on rate
limits and transient errors, and fail immediately on everything else so a schema
mistake isn't hidden behind six retries. Failed reports are written to
`*_failures.csv` with their report key and error, never silently replaced with an
empty row.

## Reproducing the published studies

| Config | Study | Reported N | Scan | Re-run | Result |
|---|---|---|---|---|---|
| `shti2025_ovesco_etl_llm` | [2025;328:96-100](https://doi.org/10.3233/SHTI250680) | 42 | 209 partitions · 17.4M records | **42** | **MATCH** |
| `shti2024_endoscopic_clips` | [2024;316:1214-1218](https://doi.org/10.3233/SHTI240629) | 2479 | 117 partitions · 9.1M records | **2475** | −4 |
| `shti2025_gu_pkl_narratives` | [2025;323:194-198](https://doi.org/10.3233/SHTI250076) | 95 | 6 partitions · 0.5M records | **102** | +7 |

All three re-run against the openFDA catalog of 2026-08-27, at roughly 24,000
records/second on one machine. Scan sizes are read from the catalog, so the cost of a config is known before it
runs.

Every boundary in a config is quoted from the paper it reproduces, because
getting them approximately right is the same as getting them wrong. Two of the
three studies end mid-quarter — one in January 2021, one at Q3 2024 — and
rounding either to a whole year moves the answer a long way.

The pipeline compares each cohort against `expected_reports` and logs `MATCH` or
the difference. One study reproduces exactly; the other two differ by −4 and +7.
Both differences have been traced to their cause, report by report, using the
tools described below. That analysis is being written up separately and is not
reproduced here.

The general point stands without it: **a MAUDE query is not reproducible without
the date it was run.** The corpus is amended after publication, so the same
query returns different answers at different times, and nothing in a published
paper usually says when the extraction happened. Every cohort produced here is
stamped with the catalog's `export_date` in a `.meta.json` beside it, and the
comparison against the published N runs as part of the pipeline rather than by
hand afterwards.

## Bugs worth describing

All of these were found while rebuilding, and all are the kind that produce a
plausible wrong answer rather than a crash.

**A delimiter that appears in the data.** The original transform joined
multi-valued fields with `", "`. MAUDE category names contain commas —
`"Activation, Positioning or Separation Problem"` is one category, not three — so
the join was not reversible. The visualisation script downstream had to carry a
hand-maintained list of comma-containing category names to un-split it, which
silently mis-split any category not on that list. Fixed by keeping the problem
tables in long form (one row per report per category), so nothing is ever joined
and re-split. `tests/test_pipeline.py` pins this.

**A download issued twice.** Each partition was fetched once to check the status
code and again to save, doubling roughly 16 GB of transfer across a full scan.

**Fixtures cleaner than the corpus.** Two bugs shipped with a green test suite
because the synthetic records used to test them had no nulls: partition
selection assumed every quarter was named `(all)` when busy quarters are split
into `(part N of M)`, and the long-to-wide join called `str.join` on columns
that hold `NaN` in real MAUDE data. Both passed every test and failed on the
first live run.

A test is only worth the realism of its fixtures. A suite built on data cleaner
than production doesn't just miss bugs — it certifies them, and the certificate
is believed. The fixtures here now carry what the corpus actually contains:
nulls, blank strings, stray numerics, missing device blocks, and both partition
naming conventions.

**A checkpoint that outlived its config.** Resume state recorded which partitions
were finished but not what they had been filtered by, so editing a config and
re-running reported the previous run's cohort in under a second — a completed-
looking result assembled under filters that no longer existed. The checkpoint now
carries a hash of the filters and invalidates itself when they change.

Also fixed: rows accumulated with `pd.concat` inside the record loop, rebuilding
the entire frame on every iteration; hardcoded absolute paths; API keys in
source; `print` instead of logging; a `requirements.txt` that was a whole-machine
`pip freeze` including `pywin32`, which made the project uninstallable on macOS
and Linux.

## Auditing a difference

When a re-run disagrees with a published number, the useful question is which
reports differ and what changed about them. Four tools answer that in order, and
together they are how every difference in the table above was traced:

```bash
# 1. Is the cohort itself sound? Duplicate keys, date spread, what the
#    filters actually admitted, reports carrying more than one device.
python tools/diagnose_cohort.py data/interim/<study>_cohort.jsonl 2479

# 2. Which specific reports differ from an earlier result?
python tools/compare_cohorts.py old_result.csv data/processed/<study>_clean.csv

# 3. Do those reports still exist, and when were they last changed?
python tools/check_missing.py <key> <key> <key>          # keys from step 2

# 4. What do they look like now -- and does the live API agree with the
#    bulk download this pipeline reads?
python tools/dump_reports.py --keys <key> --partitions "2018 Q3"
```

Step 1 rules out the explanations that are your own fault. Step 2 turns "four
reports short" into four report keys. Step 3 turns four keys into four
timestamps. Step 4 gets the raw JSON.

Applied to the two studies whose numbers moved, this chain identified every
differing report and the date each was last edited. The scripts are generic —
they take report keys and file paths, not anything specific to these three
studies.

## Setup

```bash
git clone <repo> && cd maude-etl-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env    # add OPENAI_API_KEY and/or GOOGLE_API_KEY
python -m pytest tests/ -v
```

On Windows PowerShell, run each line separately -- `&&` is not a statement
separator in PowerShell 5.1. Tested on Python 3.11-3.13.

Keys are read from the environment. Nothing under `data/` is committed.

If you contribute notebooks, strip their outputs first — `pip install nbstripout
&& nbstripout --install` — so saved model responses don't end up in the history.

## Layout

```
configs/     one YAML per study; the only thing that changes between analyses
tools/       audit scripts for tracing a difference (see above)
src/
  config.py      study config schema
  extract.py     corpus streaming, filtering, checkpointing
  transform.py   normalisation into five related tables
  clean.py       narrative quality filtering
  llm_extract.py structured extraction, OpenAI + Gemini
  visualize.py   Sankey and treemap
  run.py         CLI
tests/       covers both bugs above and the filter semantics
```

## Limitations

- The 2024 study's four-report shortfall is unexplained. Set
  `match_any_device: true` in that config to test the leading hypothesis — that
  the missing reports list the clip as their second device — or use
  `compare_cohorts.py` to diff report keys against the original run.
- Cohorts are bounded to whole quarters. A study whose period starts or ends
  mid-quarter would need a date filter applied after extraction; none is
  implemented.
- Only the extract stage has been re-run against the live corpus. The LLM stage
  has not, so no cohort here has been through narrative extraction end to end.
- Manual review of LLM output was done for the published studies but is not yet
  scripted here; agreement rates are therefore not reported. That belongs in this
  repo and isn't in it.
- Extract filters apply to `device[0]` by default, preserving the behaviour of
  the original analyses. `match_any_device: true` tests every block instead.
- Treemaps in the published papers were produced in Excel. The version here is a
  reimplementation, not the exact published figure.
- Extraction is bulk-partition only. openFDA also exposes a query API, which is
  faster for small cohorts in a known date range but caps result size; the
  original exploratory work used it and this repo does not.
- No keyword baseline. The earlier keyword-matching approach to procedure
  identification isn't implemented here, so the LLM stage has nothing to be
  measured against. Adding it back would make the comparison quantitative.
- Sampling for manual review isn't scripted (see above).

## Citation

If this pipeline is useful in your work:

> Shi Y, Yu Y, Feng Y, Gong Y. A Data Pipeline for Enhancing Quality of
> MAUDE-Based Studies. Stud Health Technol Inform. 2024;316:1214-1218.
> doi:10.3233/SHTI240629
