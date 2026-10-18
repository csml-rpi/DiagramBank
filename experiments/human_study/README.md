# Human study: generation with and without DiagramBank references

Twenty scientific briefs, five per venue with fixed caption-length quotas, selected before generation (`select_cases.py` → `data/case_manifest.jsonl`). Each brief contains the released title, abstract, caption, and figure context.

## References

`retrieve_references.py` ranks candidates with the full-field score of the retrieval ablation, excluding the target paper and normalized title/abstract aliases (`results/reference_rankings.jsonl`, `results/reference_exclusions.jsonl`). `finalize_references.py` keeps the top three per case after image decoding, SHA-256 comparison, and a 64-bit DCT perceptual-hash near-duplicate check (Hamming threshold 5). `results/finalized_references.jsonl` lists all 60 references with their release image paths and hashes; the images are part of the release.

## Generation

The text prompt is byte-identical across the two conditions: no retrieval (brief only) and full-field retrieval (brief plus exactly three ordered reference images); see `agent_generation_protocol.md`. `prepare_agent_tasks.py` writes one task per output (`agent_generations/agent_tasks.jsonl`, `no_retrieval_tasks.jsonl`), and each of the 40 outputs comes from one isolated call to Codex's built-in `image_gen` tool in a fresh agent session. `assemble_agent_generations.py` writes the frozen manifest `agent_generations/generation_manifest.jsonl`, which binds the prompt hash, condition, ordered reference hashes, and output path and hash, and `generation_validation_report.json`. Each `agent_generations/<case>/` folder holds `prompt.txt` and both outputs with their records.

## Evaluation

`build_blind_review.py` builds one self-contained page per evaluator with randomized case order and counterbalanced A/B positions (retrieval is A in 10 cases and B in 10 for every evaluator); condition names, retrieval scores, filenames, and source identifiers are hidden. `annotation_protocol.md` gives the instructions. Four evaluators chose A wins, B wins, or Tie per pair (`results/human_evaluation/reviews/`, hashes in `review_sha256.txt`). `aggregate_human_votes.py` unblinds the votes with `condition_key.json` and applies the predeclared rule: a condition wins a case only with at least three of four votes.

| Preferred condition | Cases (20) | Votes (80) |
|---|---:|---:|
| DiagramBank references | 14 | 59 |
| No references | 1 | 14 |
| Tie | 5 | 7 |

Per-evaluator votes (retrieval / no retrieval / tie) are 13/5/2, 17/2/1, 16/4/0, and 13/3/4; Fleiss' κ = 0.222 (`results/human_evaluation/human_vote_results.json`, `unblinded_votes.jsonl`).

```bash
python aggregate_human_votes.py --condition-key results/human_evaluation/condition_key.json \
  --reviews results/human_evaluation/reviews/evaluator_{1,2,3,4}_review.json --output-dir results-rerun
```
