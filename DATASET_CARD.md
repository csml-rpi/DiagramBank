# DiagramBank dataset card

## Summary

DiagramBank is a diagram–text reference resource of 57,100 schematic diagrams from 26,751 accepted papers at four OpenReview-hosted AI/ML venues: ICLR (2017–2026), NeurIPS (2021–2022 and 2024–2025), ICML (2023–2025), and TMLR (2022–2026). Each record contains the diagram image, its caption, the title and abstract of the source paper, and, for 52,052 records (91.16%), the body paragraphs that cite the figure (`figure_context`). Records keep the source identifier, OpenReview URL, and the filtering labels that admitted them. ICLR, NeurIPS, ICML, and TMLR account for 35.9%, 34.4%, 19.7%, and 9.9% of the diagrams.

## Construction

Figures and captions are extracted from the PDFs with PDFFigures 2.0 (tables excluded), and citing paragraphs with PyMuPDF. Diagrams are selected in three stages:

1. **CLIP routing.** OpenCLIP ViT-B/32 labels each figure as diagram, plot, photo, or other. CLIP diagrams at any confidence enter Tier 1; other classes with confidence below 0.70 enter Tier 2 (94,176 candidates from accepted papers).
2. **VLM classification.** Claude Haiku 4.5 and GPT-5.4-mini classify each candidate with the same four-class prompt. Tier 1 keeps a figure when at least one judge says diagram; Tier 2 forwards only figures both judges label as diagrams.
3. **Disagreement resolution.** GPT-5.4 decides when both judges disagree with CLIP; these candidates are kept only when GPT-5.4 says diagram.

| Kept path (`cascade_path` in `data.jsonl`) | Records |
|---|---:|
| `t1_unanimous` | 46,524 |
| `t1_majority` | 3,645 |
| `t1_minority_gpt_tiebreak` | 1,865 |
| `t2_vlm_consensus_gpt_confirmed` | 5,066 |

`release_supplement/cascade_labels.jsonl` spells the first three paths `t1_unanimous_diagram`, `t1_majority_diagram`, and `t1_minority_diagram_gpt_confirmed`.

Prompts, model identifiers, and execution settings are in `cascade/`. Field definitions are in `docs/schema.md`.

## Quality

A blind human audit of 500 released records, with risk-proportional sampling across kept paths, estimates 93.67% precision (95% CI 90.11–97.22). **DiagramBank-Core**, the 51,590 records that both smaller judges label as diagrams (`t1_unanimous` and `t2_vlm_consensus_gpt_confirmed`), has an estimated precision of 95.2% (91.3–99.1); the remaining 5,510 records have 79.3% (73.1–85.5). The weakest kept path is `t1_minority_gpt_tiebreak` (72.8%); screenshots and photo-like figures dominate the observed errors. We recommend DiagramBank-Core for precision-sensitive use. Users should report the subset, confidence threshold, and cascade paths they use, and apply task-specific checks (for example missing or very short captions, incomplete context spans, or very small images).

## Scope and annotation unit

All records come from ICLR, ICML, NeurIPS, and TMLR papers on OpenReview; the resource does not cover other scientific fields. The annotation unit is the complete figure: image, caption, in-text context, paper metadata, provenance, and quality labels. Bounding boxes, node or edge labels, and question–answer annotations are not provided.

## Intended and non-intended use

DiagramBank is intended for scientific diagram retrieval and as a source of reference examples for diagram generation. It is not intended for identity inference, author profiling, or producing misleading scientific artifacts. It has no fixed train/test split; users can form task-specific subsets using text availability, venue metadata, and filtering paths.

## Split units

After Unicode case folding and whitespace normalization, identical non-empty titles and abstracts link 312 paper identifiers into 156 pairs (all NeurIPS 2021, 562 figure records). `release_supplement/paper_groups.jsonl` assigns these pairs stable group identifiers. Keep each group, and all figures of each remaining paper, together when building train/test splits.

## Files

| Hosted file | Contents |
|---|---|
| `data.jsonl` | 57,100 denormalized records |
| `data/<venue>_<year>_<decision>.tar.gz` | Extracted figure images by venue, year, and decision |
| `data/<venue>_research_db.tar.gz` | Venue DuckDB databases |
| `data/faiss_complete.tar.gz` | Title, abstract, and caption FAISS indices and the release database |
| `reproduce/<venue>_metadata.tar.gz` | Raw OpenReview metadata snapshots |

The default download (accepted-paper images and core files) covers all released records. Every payload is listed with its size and SHA-256 in `huggingface/DiagramBank_remote_manifest.json`.

## Licensing and attribution

The code is under the MIT License. The release files provide no separate metadata license or per-image reuse terms; figure rights remain with the source papers. Every record carries a source URL, and all but the ICML 2023 records carry a BibTeX entry. Please cite both DiagramBank and the original source papers when using specific diagrams; reasonable rights-holder takedown requests will be honored.

## Personal information

Records include public author names, OpenReview identifiers, decisions, and, where OpenReview exposes them, reviewer scores, kept for provenance and corpus analysis. The collection does not target non-public personal data. The corpus reflects OpenReview-era AI/ML publication practices and is not demographically or scientifically representative.
