"""Study configuration.

Every published analysis in this repo is expressed as a YAML config rather than
an edited copy of the pipeline. See configs/ for the three studies reproduced here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class FieldFilter:
    """A regex applied to one field of the MAUDE device sub-record.

    MAUDE stores the device block as a list; by convention these filters are
    applied to device[0], matching the behaviour of the original analyses.
    """

    field: str
    pattern: str
    ignore_case: bool = True

    def compiled(self) -> re.Pattern:
        flags = re.IGNORECASE if self.ignore_case else 0
        return re.compile(self.pattern, flags)


@dataclass
class ExtractConfig:
    # Which quarters to download. Accepts a year ("2012") or a quarter
    # ("2021 Q1"); None downloads the whole corpus. This is what defines the
    # study period -- there is deliberately no record-level date filter here.
    # Bounding a cohort by date is a downstream concern, and mixing it into the
    # download stage makes a filtered-out quarter indistinguishable from an
    # empty one in the logs.
    partitions: list[str] | None = None
    product_codes: list[str] = field(default_factory=list)
    device_filters: list[FieldFilter] = field(default_factory=list)
    openfda_filters: list[FieldFilter] = field(default_factory=list)
    # Filters normally test the first device block only, which is what the
    # original analyses did. A report can list several devices, so a match on
    # the second would be missed; set this to test every block instead.
    match_any_device: bool = False


@dataclass
class CleanConfig:
    # Inclusive YYYYMMDD bounds on date_received, applied after extraction.
    # Partitions are quarterly, so a study that stops mid-quarter needs this --
    # but it belongs here rather than in the download stage, where a filtered-out
    # quarter would be indistinguishable from an empty one.
    date_start: str | None = None
    date_end: str | None = None
    min_narrative_words: int = 0
    exclude_narrative_patterns: list[str] = field(default_factory=list)


@dataclass
class LLMConfig:
    enabled: bool = False
    provider: str = "openai"  # "openai" | "gemini"
    model: str = "gpt-4-turbo"
    temperature: float = 0.0
    tasks: list[str] = field(default_factory=list)  # "narrative_extraction", "use_case"
    max_workers: int = 4


@dataclass
class StudyConfig:
    name: str
    citation: str
    description: str
    expected_reports: int | None
    extract: ExtractConfig
    clean: CleanConfig
    llm: LLMConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StudyConfig":
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())

        def _filters(items) -> list[FieldFilter]:
            return [FieldFilter(**i) for i in (items or [])]

        ex = raw.get("extract", {})
        stale = {"date_start", "date_end"} & set(ex)
        if stale:
            raise ValueError(
                f"{sorted(stale)} is no longer supported in extract: the download "
                "stage is bounded by `partitions` only. Express the period as "
                'years or quarters, e.g. partitions: ["2012", "2021 Q1"].'
            )
        extract = ExtractConfig(
            partitions=ex.get("partitions"),
            product_codes=ex.get("product_codes", []),
            device_filters=_filters(ex.get("device_filters")),
            openfda_filters=_filters(ex.get("openfda_filters")),
            match_any_device=ex.get("match_any_device", False),
        )
        cl = raw.get("clean", {})

        def _date(v):
            if v is None:
                return None
            s = str(v).replace("-", "").strip()
            if not (s.isdigit() and len(s) == 8):
                raise ValueError(f"clean date must be YYYYMMDD, got {v!r}")
            return s

        clean = CleanConfig(
            date_start=_date(cl.get("date_start")),
            date_end=_date(cl.get("date_end")),
            min_narrative_words=cl.get("min_narrative_words", 0),
            exclude_narrative_patterns=cl.get("exclude_narrative_patterns", []),
        )
        lm = raw.get("llm", {})
        llm = LLMConfig(
            enabled=lm.get("enabled", False),
            provider=lm.get("provider", "openai"),
            model=lm.get("model", "gpt-4-turbo"),
            temperature=lm.get("temperature", 0.0),
            tasks=lm.get("tasks", []),
            max_workers=lm.get("max_workers", 4),
        )
        return cls(
            name=raw["name"],
            citation=raw.get("citation", ""),
            description=raw.get("description", ""),
            expected_reports=raw.get("expected_reports"),
            extract=extract,
            clean=clean,
            llm=llm,
        )
