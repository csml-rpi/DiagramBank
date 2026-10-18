# Isolated Agent Image-Generation Protocol

The supplementary generation study contains 20 fixed scientific briefs and two conditions per brief. Every output is generated in a fresh Codex agent with exactly one built-in `image_gen` call. Agents do not inherit conversation history from another case or condition.

The two conditions receive the identical text prompt. The no-retrieval condition receives no image input and is prohibited from opening the target figure, retrieval rankings, reference images, or the other condition's output. The retrieval condition additionally receives exactly three full-field-retrieved DiagramBank figures. Same-paper candidates and normalized title/abstract aliases are excluded before image selection; exact and perceptual duplicates of the target or another selected reference are also excluded.

The fixed prompt asks for a polished landscape scientific schematic that faithfully represents the supplied title, abstract, intended caption, and figure-reference context. It prohibits unsupported modules, relationships, results, numerical claims, paper-identifying text, and copying scientific content or identifying labels from references.

Each task records its prompt SHA-256, condition, ordered reference paths and hashes, output path and SHA-256, and decoded image dimensions. `prepare_agent_tasks.py` freezes these inputs before generation. `assemble_agent_generations.py` rejects missing or mismatched metadata, a changed prompt within a pair, an incorrect reference count, an undecodable image, or an output-hash/dimension mismatch before producing the blind-review manifest.

Generated images are not judged by the generating agents. Four people independently receive randomized, condition-blinded A/B pages and choose only `A wins`, `B wins`, or `Tie`; an optional one-sentence reason is collected for qualitative error analysis.
