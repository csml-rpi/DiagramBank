# DiagramBank Schema

## `data.jsonl`

Each line is one cascade-filtered diagram record in the 57,100-record primary release.

```text
Data(
  id,
  platform,
  venue,
  year,
  title,
  abstract,
  keywords,
  areas,
  tldr,
  scores,
  decision,
  authors,
  author_ids,
  cdate,
  url,
  platform_id,
  bibtex,
  figure_path,
  figure_number,
  figure_caption,
  figure_context,
  figure_type,
  confidence,
  clip_type,
  clip_confidence,
  label_cascade,
  cascade_verified,
  cascade_tier,
  cascade_vote_category,
  cascade_path,
  cascade_diagram_votes,
  cascade_judges_disagree,
  label_haiku,
  label_gpt_mini,
  label_gpt_tiebreak,
  final_decision
)
```

## Paper Fields

| Field | Description |
| --- | --- |
| `platform` | Source platform. Current value: `OpenReview`. |
| `venue` | Venue: `ICLR`, `ICML`, `NeurIPS`, or `TMLR`. |
| `year` | Venue year. |
| `title` | Paper title. |
| `abstract` | Paper abstract. |
| `keywords` | Comma-separated keywords when available. |
| `areas` | Subject area metadata when available. |
| `tldr` | OpenReview TLDR field when available. |
| `scores` | Reviewer scores when available. |
| `decision` | Paper decision string. |
| `authors` | Comma-separated author names. |
| `author_ids` | Comma-separated OpenReview author ids. |
| `cdate` | Creation date in `YYYYMMDD` format when available. |
| `url` | Source OpenReview URL. |
| `platform_id` | Unique OpenReview paper id. |
| `bibtex` | Source BibTeX entry. |

## Figure Fields

| Field | Description |
| --- | --- |
| `id` | Zero-based record id in the exported JSONL. |
| `figure_path` | Relative path to the extracted image under `FIG_RAG_DIR`. |
| `figure_number` | Figure number in the source paper. |
| `figure_caption` | Extracted figure caption. |
| `figure_context` | In-text paragraphs that reference the figure. |

## CLIP Fields

| Field | Description |
| --- | --- |
| `clip_type` | First-stage CLIP top label. Values include `diagram`, `plot`, `photo`, and `other`. |
| `clip_confidence` | First-stage CLIP confidence for `clip_type`. |
| `figure_type` | Compatibility alias for the original CLIP top label. |
| `confidence` | Compatibility alias for the original CLIP confidence score. |

## Cascade Fields

| Field | Description |
| --- | --- |
| `label_cascade` | Final cascade label. Current primary release records are `diagram`. |
| `cascade_verified` | Whether the record is included in the final cascade-filtered release. |
| `cascade_tier` | Cascade routing tier: `tier1` or `tier2`. |
| `cascade_vote_category` | Raw cascade vote category, such as `t1_unanimous_diagram` or `t2_vlm_consensus`. |
| `cascade_path` | Release path used for reporting and filtering. |
| `cascade_diagram_votes` | Number of first-stage cascade judges that labeled the figure as a diagram before final tiebreaking. |
| `cascade_judges_disagree` | Whether the first-stage cascade judges disagreed. |
| `label_haiku` | Claude Haiku 4.5 figure label in the cascade. |
| `label_gpt_mini` | GPT-5.4-mini figure label in the cascade. |
| `label_gpt_tiebreak` | GPT-5.4 tiebreak/confirmation label where applicable; otherwise `null`. |
| `final_decision` | Final release decision. Current primary release records are `keep`. |

## Release Views

| View | Count | Description |
| --- | ---: | --- |
| Primary cascade-filtered release | 57,100 | Records included in `data.jsonl`. |
| `t1_unanimous` | 46,524 | Tier 1 records unanimously labeled as diagrams. |
| `t1_majority` | 3,645 | Tier 1 records kept by majority vote. |
| `t1_minority_gpt_tiebreak` | 1,865 | Tier 1 records kept after GPT-5.4 tiebreaking. |
| `t2_vlm_consensus_gpt_confirmed` | 5,066 | Tier 2 records recovered by VLM consensus and GPT-5.4 confirmation. |
