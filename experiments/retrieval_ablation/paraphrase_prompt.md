# Query Rewrite Prompt

## Role

You are constructing a fixed evaluation query for a scientific-diagram retrieval benchmark.

## Input

Each input record contains a paper title, abstract, figure caption, and the paragraph(s) that refer to the figure in the paper.

## Output

Write one natural user request for a diagram that conveys the same scientific content and intended relationships. Return one JSON object on one line with exactly these keys:

```json
{"query_id":"<unchanged input query_id>","query":"<rewritten request>"}
```

## Requirements

1. Preserve the scientific task, important components, and intended flow or relationships.
2. Use 35-90 English words in one to three sentences.
3. Phrase it as a request from someone searching for a suitable scientific schematic.
4. Do not copy the paper title.
5. Do not use a distinctive coined method name or acronym from the source.
6. Do not mention authors, venue, year, a figure number, database fields, DiagramBank, or the existence of a source figure.
7. Do not copy a source phrase of six or more consecutive words.
8. Do not add components, results, or relationships absent from the source.
9. Do not explain your rewrite and do not emit Markdown.

The output will be validated automatically and released as a frozen benchmark artifact.
