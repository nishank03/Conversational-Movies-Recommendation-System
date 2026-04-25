"""Render evaluation reports as readable markdown tables.

Used to build the comparison artifacts that go into ``docs/`` — one table
per engine, and one overall table comparing engines × prompt versions.
This is what makes the "2 prompt changes" deliverable concrete.
"""
from __future__ import annotations

from collections.abc import Iterable

from crs.evaluation.runner import EvalReport


def render_single(report: EvalReport) -> str:
    """Render one report as a markdown section."""
    lines = [
        f"## Engine: `{report.engine}` · Prompt: `{report.prompt_version}`",
        f"Samples evaluated: **{report.n_samples}**",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, value in sorted(report.aggregate_metrics.items()):
        lines.append(f"| {key} | {value:.4f} |")
    return "\n".join(lines)


def render_comparison(reports: Iterable[EvalReport]) -> str:
    """Render a wide comparison table across all reports.

    Rows: engine+prompt; columns: every metric present.
    """
    reports = list(reports)
    if not reports:
        return "_No evaluation reports provided._"

    metric_keys = sorted(
        {k for r in reports for k in r.aggregate_metrics.keys()}
    )

    header = "| Engine | Prompt | N | " + " | ".join(metric_keys) + " |"
    sep = "|---|---|---|" + "|".join(["---"] * len(metric_keys)) + "|"

    rows = [header, sep]
    for r in reports:
        row = [
            f"`{r.engine}`",
            f"`{r.prompt_version}`",
            str(r.n_samples),
        ]
        for k in metric_keys:
            v = r.aggregate_metrics.get(k, 0.0)
            row.append(f"{v:.4f}")
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join(rows)


def render_full_report(reports: Iterable[EvalReport]) -> str:
    """Markdown document combining comparison + per-engine details."""
    reports = list(reports)
    sections = [
        "# CRS Evaluation Report",
        "",
        "## Comparison",
        "",
        render_comparison(reports),
        "",
        "## Details",
        "",
    ]
    for r in reports:
        sections.append(render_single(r))
        sections.append("")
    return "\n".join(sections)
