"""Parse explicit figure-reference expressions without scanning unrelated numbers.

This handles reference syntax in supplied text. It does not validate PDF reading
order or determine whether a PDF text block contains a complete paragraph.
"""
import re

FIGURE_WORD = re.compile(r"\b(?:Figures?|Figs?)\.?\s*", re.IGNORECASE)
LABEL = r"(?:[A-Z][.-]?\d+(?:\.\d+)*|\d+(?:\.\d+)*|[IVXLCDM]+)"
IDENTIFIER = re.compile(r"(?P<label>" + LABEL + r")(?P<panel>\s*\([a-z]\)|[a-z])?(?![A-Za-z0-9]|\.\d)", re.IGNORECASE)
SEPARATOR = re.compile(r"\s*(?P<sep>,\s*(?:(?:and|or)\s+)?|(?:and|or)\b|&|[-–—]|to\b)\s*", re.IGNORECASE)
PANEL = re.compile(r"(?:\([a-z]\)|[a-z])(?![A-Za-z0-9])")


def normalize_key(value):
    text = FIGURE_WORD.sub("", str(value).strip(), count=1).strip()
    match = IDENTIFIER.fullmatch(text)
    return match["label"].upper() if match else text.upper()


def is_caption(paragraph):
    mention = FIGURE_WORD.match(paragraph.lstrip())
    if not mention:
        return False
    text = paragraph.lstrip()
    identifier = IDENTIFIER.match(text, mention.end())
    return bool(identifier and re.match(r"\s*[:.]", text[identifier.end():]))


def range_labels(first, last):
    start = re.fullmatch(r"(.*?)(\d+)", first)
    end = re.fullmatch(r"(.*?)(\d+)", last)
    if start and end and start[1] == end[1]:
        a, b = int(start[2]), int(end[2])
        if 0 <= b - a <= 100:
            return {start[1] + str(n) for n in range(a, b + 1)}
    return {first, last}


def referenced_figures(paragraph):
    found = set()
    for mention in FIGURE_WORD.finditer(paragraph):
        position = mention.end()
        opening = re.match(r"[\[(]\s*", paragraph[position:])
        if opening:
            position += opening.end()
        identifier = IDENTIFIER.match(paragraph, position)
        if not identifier:
            continue
        previous = identifier["label"].upper()
        panel_sequence = bool(identifier["panel"])
        found.add(previous)
        position = identifier.end()
        while True:
            separator = SEPARATOR.match(paragraph, position)
            if not separator:
                break
            panel = PANEL.match(paragraph, separator.end()) if panel_sequence else None
            if panel:
                position = panel.end()
                continue
            identifier = IDENTIFIER.match(paragraph, separator.end())
            if not identifier:
                panel = PANEL.match(paragraph, separator.end())
                if panel:
                    position = panel.end()
                    continue
                break
            current = identifier["label"].upper()
            if separator["sep"].lower() in {"-", "–", "—", "to"}:
                found.update(range_labels(previous, current))
            else:
                found.add(current)
            previous, position = current, identifier.end()
            panel_sequence = bool(identifier["panel"])
    return found


def extract_figure_contexts(paper_text, target_fig_numbers):
    """Return paragraphs under the original figure keys used by the database."""
    targets = {str(number): normalize_key(number) for number in target_fig_numbers}
    results = {number: [] for number in targets}
    for raw in re.split(r"\n\s*\n", paper_text):
        paragraph = raw.strip()
        if not paragraph or is_caption(paragraph):
            continue
        references = referenced_figures(paragraph)
        for number, normalized in targets.items():
            if normalized in references:
                results[number].append(paragraph)
    return results
