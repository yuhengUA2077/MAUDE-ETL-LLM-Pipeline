"""Clean stage: drop reports whose narratives cannot support text analysis.

The 100-word threshold reproduces Shi & Gong (2025), which notes that the cutoff
may itself exclude concise but informative reports. It is configurable per study
for that reason, and the rows it removes are written out rather than discarded so
the exclusion can be inspected.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import CleanConfig

log = logging.getLogger(__name__)


def apply_clean(
    flat: pd.DataFrame, cfg: CleanConfig, text_col: str = "mdr_text"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (kept, excluded). Excluded rows carry the reason they were dropped."""
    df = flat.copy()
    df[text_col] = df[text_col].fillna("")
    reasons = pd.Series("", index=df.index)

    if cfg.date_start or cfg.date_end:
        received = df["date_received"].astype(str).str.replace("-", "", regex=False)
        if cfg.date_start:
            early = received < cfg.date_start
            reasons[early & (reasons == "")] = f"before_{cfg.date_start}"
        if cfg.date_end:
            late = received > cfg.date_end
            reasons[late & (reasons == "")] = f"after_{cfg.date_end}"

    if cfg.min_narrative_words:
        too_short = df[text_col].str.split().str.len().fillna(0) < cfg.min_narrative_words
        reasons[too_short & (reasons == "")] = f"under_{cfg.min_narrative_words}_words"

    for pattern in cfg.exclude_narrative_patterns:
        hit = df[text_col].str.contains(pattern, case=False, regex=True, na=False)
        reasons[hit & (reasons == "")] = f"matched:{pattern}"

    kept = df[reasons == ""].copy()
    excluded = df[reasons != ""].copy()
    excluded["exclusion_reason"] = reasons[reasons != ""]

    log.info(
        "Clean: kept %d, excluded %d (%s)",
        len(kept), len(excluded),
        excluded["exclusion_reason"].value_counts().to_dict() if len(excluded) else {},
    )
    return kept, excluded
