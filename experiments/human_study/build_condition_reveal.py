#!/usr/bin/env python3
"""Build an internal HTML view that reveals the blinded A/B condition key."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


LABELS = {
    "full_field_retrieval": "Full-field retrieval",
    "no_retrieval": "No retrieval",
}


def badge(condition: str) -> str:
    if condition not in LABELS:
        raise ValueError(f"unknown condition: {condition}")
    css_class = "retrieval" if condition == "full_field_retrieval" else "baseline"
    return f'<span class="condition {css_class}">{LABELS[condition]}</span>'


def render_page(payload: dict) -> str:
    evaluators = payload.get("evaluators")
    if not isinstance(evaluators, list) or len(evaluators) != 4:
        raise ValueError("condition key must contain exactly four evaluators")

    tabs = []
    panels = []
    for evaluator in evaluators:
        number = int(evaluator["evaluator"])
        items = evaluator.get("items", [])
        if len(items) != 20:
            raise ValueError(f"evaluator {number}: expected 20 items, found {len(items)}")
        a_retrieval = sum(row["A"] == "full_field_retrieval" for row in items)
        if a_retrieval != 10:
            raise ValueError(f"evaluator {number}: A/B positions are not 10/10 balanced")
        tabs.append(
            f'<button type="button" role="tab" data-target="evaluator-{number}" '
            f'aria-controls="evaluator-{number}">Evaluator {number}</button>'
        )
        rows = []
        for position, row in enumerate(items, 1):
            rows.append(
                "<tr>"
                f"<td>{position}</td>"
                f"<td><code>{html.escape(row['review_id'])}</code></td>"
                f"<td><code>{html.escape(row['case_id'])}</code></td>"
                f"<td>{badge(row['A'])}</td>"
                f"<td>{badge(row['B'])}</td>"
                "</tr>"
            )
        panels.append(
            f'<section id="evaluator-{number}" class="panel" role="tabpanel">'
            '<div class="section-head">'
            f'<div><h2>Evaluator {number}</h2><p>20 cases; retrieval appears as A in 10 and B in 10.</p></div>'
            f'<a class="open-link" href="../agent_blind_review_for_humans/evaluator_{number}.html">Open blinded page</a>'
            '</div>'
            '<div class="table-wrap"><table><thead><tr>'
            '<th>Position</th><th>Review ID</th><th>Source case</th><th>A</th><th>B</th>'
            '</tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table></div></section>"
        )

    build_id = html.escape(str(payload.get("build_id", "")))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E4 Condition Reveal Key</title>
<style>
:root{{--ink:#18212b;--muted:#5d6875;--line:#cbd2d9;--bg:#f4f6f8;--paper:#fff;--accent:#1769aa;--warn:#8a3c05;--warn-bg:#fff4e8;--ret:#176b45;--ret-bg:#e9f6ef;--base:#354052;--base-bg:#eef1f4}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif}}
header{{background:var(--paper);border-bottom:1px solid var(--line)}} .inner{{width:min(1120px,100%);margin:auto;padding:20px 24px}}
h1{{font-size:1.45rem;margin:0 0 6px;letter-spacing:0}} h2{{font-size:1.12rem;margin:0 0 3px;letter-spacing:0}} p{{margin:0;color:var(--muted)}}
.warning{{background:var(--warn-bg);border-bottom:1px solid #e3b98f;color:var(--warn)}} .warning .inner{{padding-top:12px;padding-bottom:12px;font-weight:700}}
.definitions{{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin:22px 0}} .definition{{border-left:4px solid var(--line);padding:5px 0 5px 12px}} .definition.retrieval-def{{border-color:var(--ret)}}
.definition strong{{display:block;margin-bottom:3px}} code{{font:0.9em ui-monospace,SFMono-Regular,Menlo,monospace}}
.build{{font-size:.85rem;overflow-wrap:anywhere;margin-top:10px}} .tabs{{display:flex;gap:6px;flex-wrap:wrap;border-bottom:1px solid var(--line);padding-bottom:10px}}
.tabs button{{border:1px solid #8d99a5;background:#fff;color:var(--ink);border-radius:5px;padding:8px 12px;font:inherit;font-weight:700;cursor:pointer}}
.tabs button[aria-selected="true"]{{background:var(--accent);border-color:var(--accent);color:#fff}}
.panel{{display:none;padding-top:20px}} .panel.active{{display:block}} .section-head{{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:12px}}
.open-link{{display:inline-block;color:#fff;background:var(--accent);border-radius:5px;padding:8px 11px;text-decoration:none;font-weight:700;white-space:nowrap}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);background:var(--paper);border-radius:7px}} table{{width:100%;border-collapse:collapse;min-width:680px}}
th,td{{padding:10px 12px;border-bottom:1px solid #e1e5e9;text-align:left;vertical-align:middle}} th{{background:#f7f8fa;color:#4f5b67;font-size:.78rem;text-transform:uppercase}} tbody tr:last-child td{{border-bottom:0}} tbody tr:hover{{background:#f8fafb}}
.condition{{display:inline-block;border-radius:4px;padding:4px 8px;font-weight:750;white-space:nowrap}} .condition.retrieval{{color:var(--ret);background:var(--ret-bg);border:1px solid #9bcbb0}} .condition.baseline{{color:var(--base);background:var(--base-bg);border:1px solid #b8c0c8}}
@media(max-width:640px){{.inner{{padding:16px 12px}}.definitions{{grid-template-columns:1fr;gap:12px}}.section-head{{align-items:flex-start;flex-direction:column}}.open-link{{white-space:normal}}}}
</style></head><body>
<header><div class="inner"><h1>E4 Condition Reveal Key</h1><p>Internal mapping for the retrieval-augmented generation experiment.</p><p class="build">Review build: <code>{build_id}</code></p></div></header>
<div class="warning"><div class="inner">Do not share or open this page until an evaluator has exported all 20 independent judgments.</div></div>
<main class="inner">
<div class="definitions"><div class="definition retrieval-def"><strong>Full-field retrieval</strong><p>Identical text brief plus three retrieved DiagramBank reference images.</p></div><div class="definition"><strong>No retrieval</strong><p>Identical text brief only; no image references.</p></div></div>
<nav class="tabs" role="tablist">{"".join(tabs)}</nav>
{"".join(panels)}
</main>
<script>
const tabs=[...document.querySelectorAll('[role="tab"]')];
const panels=[...document.querySelectorAll('[role="tabpanel"]')];
function selectPanel(id){{tabs.forEach(tab=>tab.setAttribute('aria-selected',String(tab.dataset.target===id)));panels.forEach(panel=>panel.classList.toggle('active',panel.id===id));history.replaceState(null,'','#'+id)}}
tabs.forEach(tab=>tab.addEventListener('click',()=>selectPanel(tab.dataset.target)));
const requested=location.hash.slice(1);selectPanel(document.getElementById(requested)?requested:'evaluator-1');
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.condition_key.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_page(payload), encoding="utf-8")
    print(f"Wrote internal reveal page to {args.output}")


if __name__ == "__main__":
    main()
