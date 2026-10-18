# Final-release audit

A blind human audit of 500 released records, allocated across the four retained cascade paths in proportion to risk and across venues in proportion to path composition. The annotator labeled each figure as schematic or non-schematic without seeing CLIP labels, confidence scores, judge outputs, or the cascade path.

`path_counts.csv` lists, per path, the population size (`population`), the number of audited records (`sample_n`), and the number judged schematic (`tp`). These path totals are the available audit artifacts; the item-level judgments and the path-by-venue allocation cells are not available.

```bash
python experiments/final_release_audit/estimate.py
```

`estimate.py` reproduces the stratified estimates with a normal interval and finite-population correction and writes `estimates.json`:

| Subset | Records | Audited | Precision (95% CI) |
|---|---:|---:|---|
| Full release | 57,100 | 500 | 93.67% [90.11, 97.22] |
| DiagramBank-Core (`t1_unanimous_diagram` + `t2_vlm_consensus_gpt_confirmed`) | 51,590 | 300 | 95.2% [91.3, 99.1] |
| Remaining (`t1_majority_diagram` + `t1_minority_diagram_gpt_confirmed`) | 5,510 | 200 | 79.3% [73.1, 85.5] |

The weakest retained path is `t1_minority_diagram_gpt_confirmed` (72.8%); screenshots and photo-like figures dominate the observed errors.
