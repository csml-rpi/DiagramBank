# Diagram–text alignment review and release-wide context checks

## 50-record review

`prepare_semantic_sample.py` draws a deterministic 50-record sample (seed 20260709) with 40 records that have figure context and 10 that do not. The sample was drawn from an earlier metadata snapshot of the corpus; all 50 records are in the 57,100-record release. `e1_semantic_sample.jsonl` holds the reviewed fields (title, abstract, caption, figure number, stored context, image path); context is shown truncated to 900 characters and captions to 500.

Labels were first proposed by three LLM agents (`e1_annotations_agent_*.csv` for context, `e1_visual_annotations_agent_*.csv` for caption–image match) and then checked by an author against the displayed excerpts and images. `merge_semantic_annotations.py` and `merge_visual_annotations.py` produce the merged labels:

| Outcome | Records |
|---|---:|
| Citing paragraphs explain the diagram (`2_explains`) | 36 / 40 |
| Citing paragraphs refer to the diagram with limited explanation (`1_references_only`) | 4 / 40 |
| Wrong or uncertain citing paragraphs | 0 / 40 |
| Caption and image broadly aligned (`pass` or `minor_issue`) | 50 / 50 |
| With minor issues (`minor_issue`) | 3 / 50 |
| Metadata consistent within the paper (`same_paper_provenance = pass`) | 50 / 50 |

## Release-wide checks

`audit_final_release_57100.py` reconstructs the cascade keep set from `../../cascade/voted.jsonl` and `../../cascade/labels_gpt_tiebreak.jsonl`, checks that it equals the 57,100 keys of `$FIG_RAG_DIR/faiss/research.db` (table `Data`), and measures field availability and the context extraction rule. Results are in `final_release_57100_coverage.{json,md}`: title, abstract, caption, and URL are present for 57,100 / 57,100 records, figure context for 52,052 (91.16%), and every non-empty context matches the extraction rule.

```bash
export FIG_RAG_DIR=/path/to/downloaded/release
python audit_final_release_57100.py
```
