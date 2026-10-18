# Retrieval-Augmented Generation Pairwise Annotation

## What the evaluator sees

For each of 20 cases, the evaluator sees the same scientific content brief and two generated diagrams labeled A and B. One diagram was generated from the brief alone. The other used the same brief plus three DiagramBank references. Case order is randomized independently, and each evaluator has an exact 10/10 balance of the retrieval condition in positions A and B. Model condition, retrieval scores, filenames, and source identifiers are hidden.

## Scored decision

Choose exactly one:

- `A wins`
- `B wins`
- `Tie`

Select the output that is more useful as a scientific schematic for the supplied content. A better output should represent the intended scientific content, include the important components and relationships, and remain readable. Use `Tie` when neither output is meaningfully more useful overall.

## Optional reason

Write one short sentence explaining the most important reason for the decision. The reason is qualitative only and will not receive a separate score. Useful reasons identify a concrete strength or failure, such as a missing component, incorrect flow, clearer grouping, unreadable labels, unsupported additions, or a more appropriate visual organization.

## Independence

Evaluate every pair independently. Do not discuss decisions with the other evaluators until all 20 cases have been submitted. Do not infer the condition from visual style or filenames. After all 20 choices are complete, use `Export JSON`; this build-bound JSON file is the input to the aggregation script.
