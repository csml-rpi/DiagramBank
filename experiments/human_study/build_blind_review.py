#!/usr/bin/env python3
"""Build four self-contained, blinded pairwise-review HTML pages."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import random
from pathlib import Path
from typing import Any


EXPECTED_CASES = 20
EVALUATOR_SEEDS = (41021, 41039, 41047, 41057)
CONDITIONS = ("no_retrieval", "full_field_retrieval")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def index_unique(rows: list[dict[str, Any]], path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{path}: row {row_number} has no non-empty case_id")
        if case_id in indexed:
            raise ValueError(f"{path}: duplicate case_id {case_id!r}")
        indexed[case_id] = row
    return indexed


def require_text(row: dict[str, Any], field: str, case_id: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case {case_id!r} is missing non-empty {field!r}")
    return value.strip()


def resolve_image(manifest_path: Path, value: Any, case_id: str, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"case {case_id!r} is missing {field!r}")
    image_path = Path(value).expanduser()
    if not image_path.is_absolute():
        image_path = manifest_path.parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"case {case_id!r}: image for {field!r} not found: {image_path}")
    return image_path


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not mime.startswith("image/"):
        raise ValueError(f"not a recognized image file: {path}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def validate_inputs(cases_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    cases = index_unique(load_jsonl(cases_path), cases_path)
    generations = index_unique(load_jsonl(manifest_path), manifest_path)
    if len(cases) != EXPECTED_CASES:
        raise ValueError(f"expected exactly {EXPECTED_CASES} cases, found {len(cases)}")
    if len(generations) != EXPECTED_CASES:
        raise ValueError(
            f"expected exactly {EXPECTED_CASES} generation pairs, found {len(generations)}"
        )
    missing = sorted(cases.keys() - generations.keys())
    extra = sorted(generations.keys() - cases.keys())
    if missing or extra:
        raise ValueError(f"case/generation mismatch; missing={missing}, extra={extra}")

    complete: list[dict[str, Any]] = []
    for case_id, case in cases.items():
        generated = generations[case_id]
        images = {
            condition: resolve_image(
                manifest_path, generated.get(f"{condition}_path"), case_id, f"{condition}_path"
            )
            for condition in CONDITIONS
        }
        complete.append(
            {
                "case_id": case_id,
                "brief": {
                    "title": require_text(case, "title", case_id),
                    "abstract": require_text(case, "abstract", case_id),
                    "caption": require_text(case, "figure_caption", case_id),
                    "context": require_text(case, "figure_context", case_id),
                },
                "images": {name: image_data_url(path) for name, path in images.items()},
            }
        )
    return complete


def compute_build_id(cases: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for case in sorted(cases, key=lambda row: row["case_id"]):
        digest.update(case["case_id"].encode("utf-8"))
        for field in ("title", "abstract", "caption", "context"):
            value = case["brief"][field].encode("utf-8")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
        for condition in CONDITIONS:
            value = case["images"][condition].encode("ascii")
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def make_evaluator(
    cases: list[dict[str, Any]], evaluator_number: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    ordered = list(cases)
    rng.shuffle(ordered)
    retrieval_is_a = [True] * (len(ordered) // 2) + [False] * (len(ordered) // 2)
    rng.shuffle(retrieval_is_a)
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for position, (case, retrieval_first) in enumerate(zip(ordered, retrieval_is_a), 1):
        labels = (
            ["full_field_retrieval", "no_retrieval"]
            if retrieval_first
            else ["no_retrieval", "full_field_retrieval"]
        )
        review_id = f"R{evaluator_number}-{position:02d}"
        public_rows.append(
            {
                "id": review_id,
                "brief": case["brief"],
                "a": case["images"][labels[0]],
                "b": case["images"][labels[1]],
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "case_id": case["case_id"],
                "A": labels[0],
                "B": labels[1],
            }
        )
    return public_rows, {"evaluator": evaluator_number, "seed": seed, "items": key_rows}


def render_html(
    evaluator_number: int,
    seed: int,
    build_id: str,
    items: list[dict[str, Any]],
) -> str:
    payload = json.dumps(items, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    title = f"Scientific Diagram Review - Evaluator {evaluator_number}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root{{--ink:#18212b;--muted:#5d6875;--line:#cbd2d9;--paper:#fff;--bg:#f3f5f7;--accent:#1769aa;--ok:#176b45}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
header{{position:sticky;top:0;z-index:3;background:#fff;border-bottom:1px solid var(--line)}}
.bar,main{{width:min(1440px,100%);margin:auto;padding:14px 22px}} .bar{{display:flex;gap:16px;align-items:center;justify-content:space-between}}
h1{{font-size:1.15rem;margin:0;letter-spacing:0}} .meta{{color:var(--muted);font-size:.88rem}} .actions{{display:flex;gap:8px;flex-wrap:wrap}}
button{{border:1px solid #8793a0;background:#fff;color:var(--ink);padding:9px 12px;border-radius:5px;cursor:pointer;font-weight:650}}
button.primary{{background:var(--accent);color:#fff;border-color:var(--accent)}} button:focus-visible,input:focus-visible,textarea:focus-visible{{outline:3px solid #8bc4ef;outline-offset:2px}}
.progress{{height:5px;background:#dce1e6}} .progress>span{{display:block;height:100%;background:var(--ok);transition:width .2s}}
main{{padding-top:24px;padding-bottom:48px}} article{{background:var(--paper);border:1px solid var(--line);border-radius:7px;margin:0 0 22px;padding:20px}}
h2{{font-size:1.05rem;margin:0 0 14px;letter-spacing:0}} .brief{{display:grid;grid-template-columns:1fr 1fr;gap:12px 24px;margin-bottom:18px}}
.brief section{{min-width:0}} .brief h3{{font-size:.8rem;text-transform:uppercase;color:var(--muted);margin:0 0 4px;letter-spacing:0}}
.brief p{{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}} .images{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
figure{{margin:0;min-width:0}} figcaption{{font-size:1rem;font-weight:750;margin-bottom:7px}} img{{display:block;width:100%;height:auto;background:#fff;border:1px solid var(--line)}}
fieldset{{border:0;padding:0;margin:18px 0 0}} legend{{font-weight:700;margin-bottom:8px}} .choices{{display:flex;gap:10px;flex-wrap:wrap}}
.choice{{display:flex;align-items:center;gap:7px;border:1px solid #aeb7c0;border-radius:5px;padding:9px 13px;background:#fafbfc;cursor:pointer}}
.choice:has(input:checked){{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent);background:#eef7ff}}
.reason{{display:block;margin-top:13px;font-weight:650}} textarea{{display:block;width:100%;min-height:62px;margin-top:5px;padding:9px;border:1px solid #aeb7c0;border-radius:5px;font:inherit;resize:vertical}}
.status{{min-height:1.5em;color:var(--ok);font-weight:650}}
@media(max-width:760px){{.bar{{align-items:flex-start;flex-direction:column}}.brief,.images{{grid-template-columns:1fr}}article{{padding:15px}}main{{padding:14px 10px 36px}}.bar{{padding:12px}}}}
@media print{{header{{position:static}}.actions,.progress{{display:none}}body{{background:#fff}}article{{break-inside:avoid}}}}
</style></head><body>
<header><div class="bar"><div><h1>{html.escape(title)}</h1><div class="meta">Randomization seed: {seed} &middot; choose exactly one result per case</div></div>
<div class="actions"><button type="button" onclick="download('csv')">Export CSV</button><button type="button" class="primary" onclick="download('json')">Export JSON</button></div><div id="status" class="status" aria-live="polite"></div></div><div class="progress"><span id="progress"></span></div></header>
<main id="reviews"></main>
<script>
const ITEMS={payload}; const EVALUATOR={evaluator_number}; const SEED={seed}; const BUILD_ID={json.dumps(build_id)}; const STORE=`diagram-review-v1-${{BUILD_ID}}-${{EVALUATOR}}-${{SEED}}`;
let answers={{}}; try{{answers=JSON.parse(localStorage.getItem(STORE)||'{{}}')}}catch(_e){{answers={{}}}}
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const reviews=document.getElementById('reviews');
reviews.innerHTML=ITEMS.map((x,i)=>`<article data-id="${{x.id}}"><h2>Case ${{i+1}} of ${{ITEMS.length}}</h2><div class="brief"><section><h3>Title</h3><p>${{esc(x.brief.title)}}</p></section><section><h3>Intended caption</h3><p>${{esc(x.brief.caption)}}</p></section><section><h3>Abstract</h3><p>${{esc(x.brief.abstract)}}</p></section><section><h3>Figure context</h3><p>${{esc(x.brief.context)}}</p></section></div><div class="images"><figure><figcaption>A</figcaption><img src="${{x.a}}" alt="Candidate diagram A" loading="lazy"></figure><figure><figcaption>B</figcaption><img src="${{x.b}}" alt="Candidate diagram B" loading="lazy"></figure></div><fieldset><legend>Which diagram is more useful for this scientific content?</legend><div class="choices">${{['A wins','B wins','Tie'].map(v=>`<label class="choice"><input required type="radio" name="choice-${{x.id}}" value="${{v}}">${{v}}</label>`).join('')}}</div></fieldset><label class="reason">Optional one-sentence reason<textarea maxlength="300" rows="2" placeholder="One sentence at most"></textarea></label></article>`).join('');
function save(){{localStorage.setItem(STORE,JSON.stringify(answers)); updateProgress()}}
function updateProgress(){{const n=ITEMS.filter(x=>answers[x.id]?.choice).length;document.getElementById('progress').style.width=`${{100*n/ITEMS.length}}%`;document.getElementById('status').textContent=`${{n}} / ${{ITEMS.length}} scored`}}
reviews.querySelectorAll('article').forEach(card=>{{const id=card.dataset.id,a=answers[id]||{{}};if(a.choice){{const radio=[...card.querySelectorAll('input')].find(x=>x.value===a.choice);if(radio)radio.checked=true}}card.querySelector('textarea').value=a.reason||'';card.addEventListener('change',e=>{{if(e.target.matches('input')){{answers[id]={{...(answers[id]||{{}}),choice:e.target.value}};save()}}}});card.querySelector('textarea').addEventListener('input',e=>{{answers[id]={{...(answers[id]||{{}}),reason:e.target.value.replace(/\\s+/g,' ').trimStart()}};save()}})}}); updateProgress();
const csvCell=v=>'"'+String(v??'').replace(/"/g,'""')+'"';
function download(format){{const missing=ITEMS.filter(x=>!answers[x.id]?.choice);if(missing.length&&!confirm(`${{missing.length}} cases are unscored. Export partial results?`))return;const rows=ITEMS.filter(x=>answers[x.id]?.choice).map(x=>({{review_id:x.id,choice:answers[x.id].choice,reason:(answers[x.id].reason||'').trim()}}));let body,type,name;if(format==='csv'){{body=['review_id,choice,reason',...rows.map(r=>[r.review_id,r.choice,r.reason].map(csvCell).join(','))].join('\\r\\n');type='text/csv;charset=utf-8';name=`evaluator_${{EVALUATOR}}_review.csv`}}else{{body=JSON.stringify({{build_id:BUILD_ID,evaluator:EVALUATOR,seed:SEED,responses:rows}},null,2)+'\\n';type='application/json';name=`evaluator_${{EVALUATOR}}_review.json`}}const url=URL.createObjectURL(new Blob([body],{{type}}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}}
</script></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--generation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = validate_inputs(args.cases.resolve(), args.generation_manifest.resolve())
    build_id = compute_build_id(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator_keys = []
    for evaluator_number, seed in enumerate(EVALUATOR_SEEDS, 1):
        public_items, key = make_evaluator(cases, evaluator_number, seed)
        output_path = args.output_dir / f"evaluator_{evaluator_number}.html"
        output_path.write_text(
            render_html(evaluator_number, seed, build_id, public_items),
            encoding="utf-8",
        )
        evaluator_keys.append(key)

    key_path = args.output_dir / "condition_key.json"
    key_path.write_text(
        json.dumps(
            {"schema_version": 2, "build_id": build_id, "evaluators": evaluator_keys},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote 4 evaluator pages and condition key to {args.output_dir}")


if __name__ == "__main__":
    main()
