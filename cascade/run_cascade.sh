#!/bin/bash
# ============================================================
# Multi-Stage Filtering Cascade — Full Pipeline
# ============================================================
#
# Replaces the single CLIP confidence threshold (τ=0.85) with
# a multi-judge cascade that achieves higher recall while
# maintaining precision.
#
# Estimated total cost: ~$150
# Estimated total time: ~4-6 hours (with concurrent workers)
#
# Prerequisites:
#   - conda activate DiagramBank
#   - AWS credentials configured (for Bedrock/Haiku)
#   - OPENAI_API_KEY set (for GPT models)
#   - FIG_RAG_DIR set
#
# Usage:
#   bash run_cascade.sh           # Full run
#   bash run_cascade.sh --dry-run # Cost estimates only
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN MODE ==="
fi

echo ""
echo "============================================================"
echo " Stage 1: Extract CLIP=diagram candidates from all venues"
echo "============================================================"
python extract_candidates.py

TOTAL=$(wc -l < candidates.jsonl | tr -d ' ')
echo "Candidates extracted: $TOTAL"
echo ""

echo "============================================================"
echo " Stage 2a+2b: Run Haiku 4.5 and GPT-5.4-mini (parallel)"
echo "============================================================"
echo "Starting both judges in parallel..."
echo ""

if [[ -n "$DRY_RUN" ]]; then
    python judge_haiku.py --dry-run
    python judge_gpt_mini.py --dry-run
else
    # Run both judges in parallel
    python judge_haiku.py --resume --workers 8 &
    PID_HAIKU=$!
    python judge_gpt_mini.py --resume --workers 8 &
    PID_GPT=$!

    echo "  Haiku PID: $PID_HAIKU"
    echo "  GPT-mini PID: $PID_GPT"
    echo "  Waiting for both to complete..."

    wait $PID_HAIKU
    HAIKU_EXIT=$?
    wait $PID_GPT
    GPT_EXIT=$?

    echo ""
    echo "  Haiku exit code: $HAIKU_EXIT"
    echo "  GPT-mini exit code: $GPT_EXIT"

    if [[ $HAIKU_EXIT -ne 0 || $GPT_EXIT -ne 0 ]]; then
        echo "WARNING: One or both judges failed. Check logs and re-run with --resume."
    fi
fi
echo ""

echo "============================================================"
echo " Stage 2 merge: Apply majority vote"
echo "============================================================"
python vote.py
echo ""

if [[ -z "$DRY_RUN" ]]; then
    DISAGREE_COUNT=$(wc -l < disagreements.jsonl | tr -d ' ')
    echo "Disagreements to tiebreak: $DISAGREE_COUNT"
    echo ""

    echo "============================================================"
    echo " Stage 3: GPT-5.4 tiebreaker on disagreements"
    echo "============================================================"
    if [[ "$DISAGREE_COUNT" -gt 0 ]]; then
        python judge_gpt_tiebreak.py --resume --workers 4
    else
        echo "No disagreements — skipping tiebreak."
    fi
    echo ""

    echo "============================================================"
    echo " Stage 4: Finalize dataset"
    echo "============================================================"
    python finalize.py
    echo ""
else
    echo "============================================================"
    echo " Stage 3: GPT-5.4 tiebreaker (skipped in dry-run)"
    echo "============================================================"
    echo "(Run vote.py first to get disagreement count)"
    echo ""
fi

echo "============================================================"
echo " DONE"
echo "============================================================"
echo ""
echo "Output files:"
echo "  candidates.jsonl          — All CLIP=diagram figures"
echo "  labels_haiku.jsonl        — Haiku 4.5 labels"
echo "  labels_gpt_mini.jsonl     — GPT-5.4-mini labels"
echo "  voted.jsonl               — Merged votes with decisions"
echo "  keep_majority.jsonl       — Majority-vote keepers"
echo "  disagreements.jsonl       — Figures sent to tiebreak"
echo "  labels_gpt_tiebreak.jsonl — GPT-5.4 tiebreak labels"
echo "  final_diagrams.jsonl      — Final verified diagrams"
echo "  final_stats.json          — Statistics"
echo "  comparison.txt            — Old vs. new comparison"
