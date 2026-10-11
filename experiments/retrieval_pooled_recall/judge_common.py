"""
Shared utilities for the 3-judge graded relevance pipeline.

All three judges (Haiku 4.5, GPT-5.4-mini, GPT-5.4) use the identical prompt
and output schema defined here, ensuring no prompt drift across judges.
"""

import base64
import json
import re
import time
from pathlib import Path

# ── Prompt ─────────────────────────────────────────────────────────────

INSTRUCTIONS = "You are an objective academic evaluator for scientific diagram retrieval."

PROMPT_TEMPLATE = (
    "You are evaluating whether a retrieved scientific diagram is relevant "
    "as a visual reference for creating a new diagram.\n\n"
    "QUERY PAPER:\n"
    "  Title: {query_title}\n"
    "  Abstract: {query_abstract}\n"
    "  Target caption: {query_caption}\n\n"
    "RETRIEVED DIAGRAM:\n"
    "  Source paper: {retrieved_title}\n"
    "  Caption: {retrieved_caption}\n\n"
    "Score the visual relevance of the retrieved diagram as a reference for "
    "creating the query paper's diagram. Use this strict 3-point scale:\n\n"
    "  2 (Highly Relevant): The diagram explicitly depicts a closely related\n"
    "    architecture, pipeline, or concept. Its visual structure could directly\n"
    "    serve as a layout template or reference for the query diagram.\n\n"
    "  1 (Marginally Relevant): The concept is present but secondary, generic,\n"
    "    or from a loosely related domain. A researcher might get partial\n"
    "    inspiration but would need substantial adaptation.\n\n"
    "  0 (Not Relevant): The diagram does not depict a related concept, or\n"
    "    the relationship is superficial (shared keywords but different visual\n"
    "    content).\n\n"
    'Reply with ONLY a JSON object: {{"score": <integer>}}'
)

# ── Parsing ────────────────────────────────────────────────────────────

_SCORE_RE = re.compile(r'"score"\s*:\s*(\d)')
_BARE_INT_RE = re.compile(r'^[012]$')


def parse_score(text: str) -> int:
    """Extract integer score (0, 1, or 2) from VLM response text.

    Handles:
      - JSON: {"score": 2}
      - Bare integer: 2
      - Text containing the digit
    Returns -1 if unparseable.
    """
    text = text.strip()
    # Try JSON pattern first
    m = _SCORE_RE.search(text)
    if m:
        return int(m.group(1))
    # Try bare integer
    if _BARE_INT_RE.match(text):
        return int(text)
    # Last resort: look for any 0/1/2 in the text
    for ch in ('2', '1', '0'):
        if ch in text:
            return int(ch)
    return -1


def format_prompt(item: dict) -> str:
    """Format the graded relevance prompt for a pool item."""
    return PROMPT_TEMPLATE.format(
        query_title=item["query_title"],
        query_abstract=item["query_abstract"][:400],
        query_caption=item["query_caption"],
        retrieved_title=item["retrieved_title"],
        retrieved_caption=item["retrieved_caption"],
    )


# ── Image encoding ─────────────────────────────────────────────────────

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ── Codex OAuth auth ──────────────────────────────────────────────────

CODEX_CONFIG_PATHS = [
    Path.home() / ".clawdbot" / "agents" / "main" / "agent" / "auth-profiles.json",
]


def get_codex_api_key() -> str:
    """Get OpenAI access token from Codex OAuth auth config."""
    for p in CODEX_CONFIG_PATHS:
        if p.exists():
            try:
                with open(p) as f:
                    data = json.load(f)
                profiles = data.get("profiles", {})
                last_good = data.get("lastGood", {}).get("openai-codex")
                if last_good and last_good in profiles:
                    profile = profiles[last_good]
                else:
                    for name, prof in profiles.items():
                        if prof.get("provider") == "openai-codex":
                            profile = prof
                            break
                    else:
                        continue
                token = profile.get("access")
                if token:
                    expires = profile.get("expires", 0)
                    if expires and expires < time.time() * 1000:
                        print("  WARNING: Codex token expired. Run `codex auth login` to refresh.")
                    print(f"  Using Codex OAuth token from {p}")
                    return token
            except Exception:
                continue
    raise RuntimeError("No Codex auth found. Run `codex auth login`.")
