"""LLM stage: structured extraction from MDR narratives.

The categorical MAUDE fields do not record which surgical procedure was underway,
when in that procedure the event occurred, or how the device was being used. Those
facts exist only in the free-text narrative. This stage extracts them into a fixed
schema.

Design notes:

* One Pydantic schema, two providers. OpenAI and Gemini differ in how a schema
  is attached to the request, but both constrain generation to the same contract,
  so results are directly comparable across models.
* temperature=0 throughout, for reproducibility.
* Failures are recorded with their report key and error, not silently replaced
  with an empty row. A run that drops reports should say how many and which ones.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

RETRYABLE = ("rate", "quota", "429", "resourceexhausted", "timeout", "overloaded", "503")


def with_retry(fn: Callable, attempts: int = 6, base: float = 2.0) -> Callable:
    """Exponential backoff with jitter on rate-limit and transient errors.

    Both providers throttle aggressively enough that a few hundred sequential
    reports will hit limits without this. Non-retryable errors (a malformed
    schema, a bad key) fail immediately rather than burning six attempts.
    """

    def wrapped(*args, **kwargs):
        last = None
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last = exc
                if not any(t in str(exc).lower() or t in type(exc).__name__.lower()
                           for t in RETRYABLE):
                    raise
                if attempt == attempts - 1:
                    break
                delay = min(base ** attempt, 60) + random.uniform(0, 1)
                log.warning(
                    "Retryable error (attempt %d/%d), sleeping %.1fs: %s",
                    attempt + 1, attempts, delay, str(exc)[:120],
                )
                time.sleep(delay)
        raise last

    return wrapped


# --------------------------------------------------------------------------
# Output schemas
# --------------------------------------------------------------------------

class NarrativeExtraction(BaseModel):
    """Open-ended extraction: what happened, to whom, during what procedure."""

    mdr_report_key: str = Field(description="Unique identifier for the MDR report")
    device_issues: str = Field(description="Device problem and any stated cause")
    patient_outcomes: str = Field(description="Patient involvement and any stated cause")
    procedures: str = Field(description="Medical procedures mentioned in the narrative")
    event_timing: str = Field(description="Event timing relative to the procedure")
    additional_insights: str = Field(description="Other relevant detail from the narrative")


class UseCaseLabel(BaseModel):
    """Closed-set classification, run after the term list has been consolidated."""

    mdr_report_key: str = Field(description="Unique identifier for the MDR report")
    use_case: str = Field(description="Device use case at the time of the event")


USE_CASE_OPTIONS = [
    "clip deployment", "endoscopic marking", "hemostasis",
    "tissue grasping and locking", "wound closure", "N/A",
]

FIELD_GLOSSARY = """\
- mdr_report_key: Unique key for the report
- date_received: Date the initial report was received
- event_type: Type of reportable event (death, injury, or malfunction)
- type_of_report: Initial or follow-up submission
- brand_name: Trade or proprietary name of the suspect device
- generic_name: General descriptive name of the device
- manufacturer_d_name: Full name of the device manufacturer
- device_report_product_code: Three-letter FDA product classification code
- device_name: Common name of the device
- medical_specialty_description: Medical specialty related to the device
- product_problem: Categorical device problems (pipe-delimited)
- patient_problem: Categorical patient adverse events (pipe-delimited)
- mdr_text: Narrative text describing the event"""

SYSTEM_PROMPT = f"""You are an assistant analysing FDA MAUDE medical device reports \
using both their categorical and narrative fields. Report fields are:

{FIELD_GLOSSARY}

Answer only from the report. If the report does not state something, respond \
"not mentioned" rather than inferring it."""

NARRATIVE_PROMPT = """Report to analyse:
{row_data}

1. Device issues: describe the device problem and any cause or contributing factor
   stated in mdr_text. One sentence covering both.
2. Patient outcomes: was a patient involved, and is a cause or contributing factor
   stated in mdr_text? One sentence covering both.
3. Procedures: identify any medical procedures named in mdr_text. Summarise in one
   phrase.
4. Event timing: state whether the event occurred before, during, or after the
   procedure.
5. Additional insights: any other relevant procedural detail.
"""

USE_CASE_PROMPT = f"""Report to analyse:
{{row_data}}

Identify how the device was being used when the event occurred. Choose exactly one
label from this list, excluding the procedure name:
{chr(10).join('* ' + o for o in USE_CASE_OPTIONS)}
"""


@dataclass
class TaskSpec:
    name: str
    schema: type[BaseModel]
    user_prompt: str


TASKS: dict[str, TaskSpec] = {
    "narrative_extraction": TaskSpec(
        "narrative_extraction", NarrativeExtraction, NARRATIVE_PROMPT
    ),
    "use_case": TaskSpec("use_case", UseCaseLabel, USE_CASE_PROMPT),
}


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------

def _openai_caller(model: str, temperature: float, task: TaskSpec) -> Callable[[str], dict]:
    """OpenAI structured output.

    The schema is passed to the API directly rather than described in the prompt,
    so the model is constrained at decode time instead of being asked politely to
    emit valid JSON. Same contract as the Gemini path below.
    """
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Copy .env.example to .env.")

    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    def call(row_data: str) -> dict:
        completion = client.beta.chat.completions.parse(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": task.user_prompt.replace('{row_data}', row_data).replace('{terms}', row_data)},
            ],
            response_format=task.schema,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Model returned no parseable structured output")
        return parsed.model_dump()

    return with_retry(call)


def _gemini_caller(model: str, temperature: float, task: TaskSpec) -> Callable[[str], dict]:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env.")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=task.schema,
    )

    def call(row_data: str) -> dict:
        resp = client.models.generate_content(
            model=model,
            contents=SYSTEM_PROMPT + "\n\n" + task.user_prompt.replace('{row_data}', row_data).replace('{terms}', row_data),
            config=config,
        )
        return task.schema.model_validate_json(resp.text).model_dump()

    return with_retry(call)


PROVIDERS = {"openai": _openai_caller, "gemini": _gemini_caller}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def _row_payload(row: pd.Series) -> str:
    payload = {k: v for k, v in row.to_dict().items() if pd.notna(v) and v != ""}
    return json.dumps(payload, indent=2, default=str)


def run_task(
    df: pd.DataFrame,
    task_name: str,
    provider: str = "openai",
    model: str = "gpt-4-turbo",
    temperature: float = 0.0,
    max_workers: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one extraction task over a cohort.

    Returns (results, failures). Reports are processed concurrently; ordering is
    restored afterwards so output is deterministic regardless of worker count.
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task {task_name!r}; expected one of {list(TASKS)}")
    task = TASKS[task_name]
    call = PROVIDERS[provider](model, temperature, task)

    results: dict[int, dict] = {}
    failures: list[dict] = []

    def work(idx: int, row: pd.Series) -> tuple[int, dict | None, dict | None]:
        try:
            return idx, call(_row_payload(row)), None
        except Exception as exc:
            return idx, None, {
                "mdr_report_key": row.get("mdr_report_key"),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(work, i, pd.Series(record))
            for i, record in enumerate(df.to_dict("records"))
        ]
        for fut in as_completed(futures):
            idx, ok, err = fut.result()
            if ok is not None:
                results[idx] = ok
            else:
                failures.append(err)

    out = pd.DataFrame([results[i] for i in sorted(results)])
    fail_df = pd.DataFrame(failures)

    log.info(
        "%s via %s/%s: %d succeeded, %d failed",
        task_name, provider, model, len(out), len(fail_df),
    )
    if len(fail_df):
        log.warning("Failed report keys: %s", fail_df["mdr_report_key"].tolist())
    return out, fail_df


# --------------------------------------------------------------------------
# Pass 2: term consolidation
# --------------------------------------------------------------------------

class TermCluster(BaseModel):
    standard_term: str = Field(description="Standardised term representing the group")
    original_descriptions: list[str] = Field(description="Raw descriptions in this group")


class TermConsolidation(BaseModel):
    clusters: list[TermCluster] = Field(description="Semantic groupings of the input terms")


CONSOLIDATE_PROMPT = """The following free-text descriptions were extracted from the \
narrative sections of medical device reports:

{terms}

Group them by semantic similarity and give each group a single standardised term \
that best represents it. Use established clinical terminology where one exists. \
Anything that fits no group belongs in a cluster named "Miscellaneous".
"""


def consolidate_terms(
    terms: list[str],
    provider: str = "openai",
    model: str = "gpt-4-turbo",
    temperature: float = 0.0,
) -> pd.DataFrame:
    """Second pass: turn free-text extractions into a controlled vocabulary.

    This sits between the open-ended extraction and the closed-set labelling.
    Running it in this order matters: the label set for pass 3 is derived from
    what the reports actually say, rather than assumed before reading them.

    Returns a long-form mapping of original description to standard term, which
    is what makes the frequency counts in the published analyses reproducible.
    """
    unique = sorted({t.strip() for t in terms if isinstance(t, str) and t.strip()})
    if not unique:
        return pd.DataFrame(columns=["original_description", "standard_term"])

    task = TaskSpec("consolidate", TermConsolidation, CONSOLIDATE_PROMPT)
    call = PROVIDERS[provider](model, temperature, task)
    result = call(json.dumps(unique, indent=2))

    rows = [
        {"original_description": original, "standard_term": cluster["standard_term"]}
        for cluster in result["clusters"]
        for original in cluster["original_descriptions"]
    ]
    mapping = pd.DataFrame(rows)

    unmapped = set(unique) - set(mapping["original_description"])
    if unmapped:
        log.warning("%d terms were not assigned a cluster: %s", len(unmapped), sorted(unmapped)[:5])
        mapping = pd.concat([mapping, pd.DataFrame(
            [{"original_description": u, "standard_term": "Unassigned"} for u in unmapped]
        )], ignore_index=True)

    log.info("Consolidated %d descriptions into %d standard terms",
             len(unique), mapping["standard_term"].nunique())
    return mapping
