# MAUDE ETL + LLM Pipeline

A reproducible pipeline over the FDA's Manufacturer and User Facility Device
Experience (MAUDE) database: streams the full public corpus, filters it to a
study cohort, normalises it, extracts structured facts from free-text adverse
event narratives with an LLM, and renders the figures.

Three published studies are included as configuration files, each reproducible
through the same pipeline. Every run reports whether its cohort matches the
number printed in the paper. Validation so far covers cohort extraction against
the live corpus; the LLM stage has not been re-run.

```bash
python -m src.run --config configs/shti2025_ovesco_etl_llm.yaml
```

---

## Why this exists

MAUDE is the FDA's public record of medical device adverse events, 25 million
reports going back to 1993. It is used heavily in patient safety research, and
that research is mostly not reproducible. A 2024 audit found that **11.5% of
523 MAUDE-based studies published an executable query, and only 23.3% of those
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

Partitions are downloaded, filtered, and deleted one at a time, so the full
corpus is never held on disk. Progress is checkpointed per partition, and the
checkpoint is keyed on a hash of the filters, so an interrupted scan resumes
while an edited config starts over instead of resuming into a stale cohort.

Which quarters get downloaded is what defines the study period. A config names
years and quarters (`["2012", "2013", "2021 Q1"]`), and records are filtered on
device fields only; bounding a cohort by date is a `clean`-stage concern. A date
filter in the download stage would make a filtered-out quarter look identical to
an empty one in the logs.

## Pipeline

```
                         openFDA bulk catalog
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
needed to re-measure an unchanged query against a newer corpus.

### The LLM stage

MAUDE's structured fields do not record which surgical procedure was underway,
when during it the event occurred, or how the device was being used. Those facts
exist only in the narrative text. Extraction runs in three passes, deliberately
ordered:

1. **Open-ended extraction.** GPT-4 fills a Pydantic schema per report. Free
   text, because the vocabulary is not known in advance.
2. **Term consolidation.** The resulting free-text procedure descriptions are
   grouped into standard terms. This is where the label set comes from.
3. **Closed-set classification.** Each report is assigned one label from that
   consolidated set.

Starting at step 3 would have required knowing the answer beforehand.

OpenAI and Gemini share the same Pydantic output schema, so extraction is
provider-independent and results are comparable across models; switching takes
one line in the config. `temperature=0` throughout. Failed reports are written to
`*_failures.csv` with their key and error rather than silently becoming empty
rows.

## Reproducing the published studies

| Config | Study | Reported N | Scan |
|---|---|---|---|
| `shti2025_gu_pkl_narratives` | [2025;323:194-198](https://doi.org/10.3233/SHTI250076) | 95 | 6 partitions · 0.5M records |
| `shti2024_endoscopic_clips` | [2024;316:1214-1218](https://doi.org/10.3233/SHTI240629) | 2479 | 117 partitions · 9.1M records |
| `shti2025_ovesco_etl_llm` | [2025;328:96-100](https://doi.org/10.3233/SHTI250680) | 42 | 209 partitions · 17.4M records |

Scan sizes come from the openFDA catalog, so the cost of a config is known before
it runs; throughput is roughly 24,000 records/second on one machine.

Every boundary in a config is quoted from the paper it reproduces, because
getting them approximately right is the same as getting them wrong. Two of these
studies end mid-quarter, one in January 2021 and one at Q3 2024, while
partitions are quarterly. The period is therefore enforced in two places: which
quarters get downloaded, and a date bound applied afterwards in `clean`.

Each run compares its cohort against `expected_reports` and logs `MATCH` or the
difference. Differences are expected, because MAUDE is amended after publication
and the same query returns different answers at different times. That is why
every cohort is stamped with the catalog's `export_date` in a `.meta.json`
beside it.
**A MAUDE query is not reproducible without the date it was run.**

## A delimiter that appears in the data

Worth describing because it produced a plausible wrong answer rather than a
crash, and because the fix is structural.

The original transform joined multi-valued fields with `", "`. MAUDE category
names contain commas of their own: `"Activation, Positioning or Separation
Problem"` is one category, not three. The join was therefore not reversible, and
the visualisation script downstream carried a hand-maintained list of
comma-containing category names to split it back, which silently mis-split any
category not on that list.

The fix is not a better delimiter. The problem tables are kept in long form, one
row per report per category, so nothing is ever joined and re-split.
`tests/test_pipeline.py` pins this.

## Auditing a difference

When a re-run disagrees with a published number, the useful question is which
reports differ and what changed about them. `tools/` answers that in four steps,
from an aggregate count down to raw JSON:

```bash
# 1. Is the cohort itself sound? Duplicate keys, date spread, what the
#    filters actually admitted, reports carrying more than one device.
python tools/diagnose_cohort.py data/interim/<study>_cohort.jsonl <published_N>

# 2. Which specific reports differ from an earlier result?
python tools/compare_cohorts.py old_result.csv data/processed/<study>_clean.csv

# 3. Do those reports still exist, and when were they last changed?
python tools/check_missing.py <key> <key> <key>

# 4. What do they look like now, and does the live API agree with the
#    bulk download this pipeline reads?
python tools/dump_reports.py --keys <key> --partitions "<YYYY Qn>"
```

The scripts take report keys and file paths, so they work on any MAUDE cohort,
not just the three studies here.

## Setup

```bash
git clone <repo>
cd maude-etl-llm
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt

cp .env.example .env    # add OPENAI_API_KEY and/or GOOGLE_API_KEY
python -m pytest tests/ -v
```

Tested on Python 3.11 to 3.13.

Keys are read from the environment. Nothing under `data/` is committed.

If you contribute notebooks, strip their outputs first, so saved model responses
do not end up in the history:

```bash
pip install nbstripout
nbstripout --install
```

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
tests/       filter semantics, delimiter handling, real-world nulls
```

## Limitations

- MAUDE is mutable. Reports are amended after publication, so a reproduced
  cohort can legitimately differ from the count a paper reports.
- Only cohort extraction has been re-run against the live corpus. The LLM stage
  has not, and its outputs have not been through a scripted manual review, so no
  agreement rates are reported.
- Extract filters apply to `device[0]` by default, preserving the behaviour of
  the original analyses. `match_any_device: true` tests every device block.
- Running this requires a Python environment and a command line. The pipeline it
  is based on was proposed as a method adoptable by teams of varying technical
  background, and a clone-and-configure workflow does not fully meet that.

## Citation

If this pipeline is useful in your work:

> Shi Y, Yu Y, Feng Y, Gong Y. A Data Pipeline for Enhancing Quality of
> MAUDE-Based Studies. Stud Health Technol Inform. 2024;316:1214-1218.
> doi:10.3233/SHTI240629
