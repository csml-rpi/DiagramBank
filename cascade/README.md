# Filtering cascade

Selects the 57,100 released diagrams from 94,176 accepted-paper candidates.

1. **CLIP routing** (`extract_candidates.py --accepted-only`): figures that CLIP labels diagram at any confidence enter Tier 1 (60,717); figures with another label and confidence below 0.70 enter Tier 2 (33,459) → `candidates.jsonl`.
2. **VLM classification** (`judge_haiku.py`, `judge_gpt_mini.py`, then `vote.py`): Claude Haiku 4.5 and GPT-5.4-mini label every candidate with the same four-class prompt (`labels_haiku.jsonl`, `labels_gpt_mini.jsonl`). Tier 1 keeps a figure when at least one judge says diagram; Tier 2 forwards a figure only when both do (`voted.jsonl`, `keep_majority.jsonl`, `disagreements.jsonl`).
3. **Disagreement resolution** (`judge_gpt_tiebreak.py`): GPT-5.4 labels the 19,365 candidates in `disagreements.jsonl` (`labels_gpt_tiebreak.jsonl`). Tier 1 candidates rejected by both judges and Tier 2 consensus candidates are kept only when GPT-5.4 says diagram; its label on the 3,645 Tier 1 majority records does not affect their inclusion.
4. **Finalization** (`finalize.py`): applies the rules and writes the kept records.

| Tier | Votes | Action | Input | Kept |
|---|---|---|---:|---:|
| 1 | 2/2 | Retain | 46,524 | 46,524 |
| 1 | 1/2 | Retain | 3,645 | 3,645 |
| 1 | 0/2 | Recheck with GPT-5.4 | 10,548 | 1,865 |
| 2 | 2/2 | Recheck with GPT-5.4 | 5,172 | 5,066 |
| 2 | 1/2 | Reject | 2,602 | 0 |
| 2 | 0/2 | Reject | 25,685 | 0 |

The `.jsonl` files in this folder are the saved outputs of the release run. Raw model responses were not saved; responses are trimmed and lowercased, the first matching class name is stored, and only the stored label `diagram` counts as a positive vote.

## Replaying the release

```bash
python finalize.py --output-dir /path/to/new-run --verify-release-db $FIG_RAG_DIR/faiss/research.db
```

The replay reads the saved votes, applies the decision rules, and checks that the kept (`platform_id`, `figure_number`) keys equal the 57,100 records of the release database exactly (opened read-only). Venue databases are modified only with `--update-db`.

## Running the judges

```bash
bash run_cascade.sh          # all stages
python judge_haiku.py --resume --workers 8
python judge_gpt_mini.py --resume --workers 8
python vote.py
python judge_gpt_tiebreak.py --resume --workers 4
python finalize.py
```

Haiku runs through Amazon Bedrock (model ID from `BEDROCK_HAIKU_MODEL`, region us-west-2, temperature 0, at most 16 output tokens). The GPT judges call a Responses endpoint at default decoding with `store=False` and streaming; see the scripts for credentials and settings.
