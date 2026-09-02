"""Load & visualise stage: Sankey of device-to-patient problem flows, treemaps of
procedural context.

Both figures read the pipeline's own output tables. The original Sankey script
carried its 41 rows pasted inline, which meant the figure could not be regenerated
when the cohort changed and did not match the 42 reports reported in the paper.

Because the long-form problem tables are one row per category, no delimiter
splitting happens here at all. The hand-maintained list of comma-containing
category names the original script needed is gone.
"""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

log = logging.getLogger(__name__)

UNINFORMATIVE = {
    "no information", "insufficient information", "not mentioned", "n/a", "",
}


def problem_flows(
    product_problems: pd.DataFrame,
    patient_problems: pd.DataFrame,
    drop_uninformative: bool = True,
) -> pd.DataFrame:
    """Count device-problem to patient-problem pairs, one row per co-occurrence.

    A report with two device problems and three patient problems contributes six
    pairs, matching the original cross-product behaviour.
    """
    left = product_problems.rename(columns={"product_problem": "source"})
    right = patient_problems.rename(columns={"patient_problem": "target"})

    if drop_uninformative:
        left = left[~left["source"].str.strip().str.lower().isin(UNINFORMATIVE)]
        right = right[~right["target"].str.strip().str.lower().isin(UNINFORMATIVE)]

    pairs = left.merge(right, on="mdr_report_key", how="inner")
    counts = (
        pairs.groupby(["source", "target"]).size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    log.info(
        "Sankey: %d distinct flows from %d reports",
        len(counts), pairs["mdr_report_key"].nunique(),
    )
    return counts


def sankey(counts: pd.DataFrame, title: str = "") -> go.Figure:
    """Device problems (left) to patient adverse events (right)."""
    sources = counts["source"].unique().tolist()
    targets = counts["target"].unique().tolist()
    labels = sources + targets  # kept disjoint so a shared name cannot self-link
    index = {("s", n): i for i, n in enumerate(sources)}
    index.update({("t", n): len(sources) + i for i, n in enumerate(targets)})

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(pad=15, thickness=20, line=dict(width=0.5), label=labels),
        link=dict(
            source=[index[("s", s)] for s in counts["source"]],
            target=[index[("t", t)] for t in counts["target"]],
            value=counts["count"].tolist(),
        ),
    ))
    fig.update_layout(title_text=title, font_size=10)
    return fig


def treemap(
    df: pd.DataFrame,
    path: list[str],
    title: str = "",
    drop_uninformative: bool = True,
) -> go.Figure:
    """Hierarchical view of LLM-extracted procedural context.

    Typical paths: ["event_timing", "use_case"] or ["manufacturer_d_name",
    "brand_name"].
    """
    data = df.copy()
    for col in path:
        data[col] = data[col].fillna("N/A").astype(str).str.strip()
        if drop_uninformative:
            data = data[~data[col].str.lower().isin(UNINFORMATIVE)]

    counts = data.groupby(path).size().reset_index(name="count")
    fig = px.treemap(counts, path=path, values="count", title=title)
    fig.update_traces(textinfo="label+percent root")
    return fig


def save(fig: go.Figure, path, width: int = 1000, height: int = 600) -> None:
    """Write interactive HTML always; a static image only if kaleido is present."""
    fig.write_html(str(path))
    log.info("Wrote %s", path)
    try:
        fig.write_image(str(path).replace(".html", ".png"), width=width, height=height)
    except Exception as exc:
        log.debug("Static export skipped (install kaleido to enable): %s", exc)
