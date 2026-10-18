# E1 Final Release Coverage Audit

## Inputs

- Final metadata table: `$FIG_RAG_DIR/faiss/research.db::Data`
- Cascade vote file: `cascade/voted.jsonl`
- Cascade tiebreak file: `cascade/labels_gpt_tiebreak.jsonl`

## Final-Release Key Verification

| Check | Result |
|---|---:|
| cascade final keys reconstructed from logs | 57,100 |
| rows in `Data` table | 57,100 |
| unique `(platform_id, figure_number)` keys in `Data` | 57,100 |
| cascade-final keys missing from `Data` | 0 |
| `Data` keys not in cascade-final set | 0 |

Cascade keep-reason counts:

| Keep reason | Count |
|---|---:|
| `tier1_majority_vote` | 50,169 |
| `tier1_tiebreak_confirmed` | 1,865 |
| `tier2_rescue_confirmed` | 5,066 |

Venue counts:

| Venue | Cascade final keys | `Data` rows |
|---|---:|---:|
| ICLR | 20,516 | 20,516 |
| ICML | 11,267 | 11,267 |
| NeurIPS | 19,655 | 19,655 |
| TMLR | 5,662 | 5,662 |

## Field Coverage Over 57,100 Final Records

| Field | Count | Share |
|---|---:|---:|
| `title` | 57,100 / 57,100 | 100.00% |
| `abstract` | 57,100 / 57,100 | 100.00% |
| `figure_caption` | 57,100 / 57,100 | 100.00% |
| `url` | 57,100 / 57,100 | 100.00% |
| `figure_context` | 52,052 / 57,100 | 91.16% |
| URL matches `platform_id` shape | 57,100 / 57,100 | 100.00% |

## Figure-Reference Traceability

| Check | Count | Share |
|---|---:|---:|
| non-empty `figure_context` | 52,052 / 57,100 | 91.16% |
| matching stored figure number among all records | 52,052 / 57,100 | 91.16% |
| matching stored figure number among non-empty contexts | 52,052 / 52,052 | 100.00% |

The figure-reference check uses the same conservative regex family as the earlier E1 audit: it strips non-digits from the stored figure number and searches each non-empty context span for a nearby `Figure` or `Fig.` reference to that number.
