# Release supplement

Files that join to the 57,100-record release on (`platform_id`, `figure_number`) or `platform_id`. `../tools/build_release_supplement.py` rebuilds them from the release database and the saved cascade votes in `../cascade/`.

| File | Contents |
|---|---|
| `cascade_labels.jsonl` | 57,100 rows: `clip_type`, `clip_confidence`, `label_haiku`, `label_gpt_mini`, `label_gpt_tiebreak`, `label_cascade`, `cascade_path`. Path counts are 46,524 (`t1_unanimous_diagram`), 3,645 (`t1_majority_diagram`), 1,865 (`t1_minority_gpt_tiebreak`), and 5,066 (`t2_vlm_consensus_gpt_confirmed`). GPT-5.4 is null on the 46,524 unanimous keeps, where it was not queried. |
| `paper_groups.jsonl` | All 26,751 paper identifiers. Identical normalized titles and abstracts link 312 identifiers into 156 groups (all NeurIPS 2021, 562 figure records); other papers have a null group. Keep each group, and all figures of each remaining paper, together when splitting. |
| `unexpected_labels.json` | The 55 saved Claude Haiku 4.5 outputs outside the four labels (51 `table`, 3 `heatmap`, 1 longer response) and their outcomes; 7 are on retained records. |

The database stores the CLIP label and confidence as `figure_type` and `confidence`.

```bash
python tools/build_release_supplement.py --source-db $FIG_RAG_DIR/faiss/research.db \
  --voted cascade/voted.jsonl --tiebreak cascade/labels_gpt_tiebreak.jsonl --output-dir supplement-rerun
```
