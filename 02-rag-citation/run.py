"""Runnable demo for the RAG Agent — citation grounding.

Three questions against the in-repo Nimbus Cloud corpus, one per behaviour:
  1. well-supported  -> a GROUNDED answer with inline [n] citations + sources
  2. weakly supported -> a LOW-CONFIDENCE flag (thin retrieval + the model can't
                         support it) instead of a hallucinated answer
  3. outside the corpus -> the FALLBACK (simulated web-search) path

Run:  python 02-rag-citation/run.py
"""

from __future__ import annotations

import logging
import os
import sys

# Make `common` (repo root) and this project's modules importable regardless of cwd.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _REPO_ROOT)

# Windows consoles default to cp1252 and choke on the ✓/⚠/↪/… glyphs below.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from agent import FALLBACK, GROUNDED, LOW_CONFIDENCE, Answer, RagAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

CORPUS_DIR = os.path.join(_PROJECT_DIR, "corpus")

CASES = [
    ("CASE 1 — well-supported (expect GROUNDED with citations)",
     "What uptime SLA does the Nimbus Enterprise plan guarantee and what is its monthly price per seat?"),
    ("CASE 2 — weakly supported (expect LOW-CONFIDENCE flag)",
     "Can I pay for my Nimbus subscription using cryptocurrency such as Bitcoin or Ethereum?"),
    ("CASE 3 — outside the corpus (expect FALLBACK to search)",
     "What are the common symptoms of a vitamin D deficiency in adults?"),
]

_BADGE = {GROUNDED: "GROUNDED ✓", LOW_CONFIDENCE: "LOW-CONFIDENCE ⚠", FALLBACK: "FALLBACK ↪"}


def _rule(title: str) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def _show(res: Answer) -> None:
    print(f"\nQ: {res.query}")
    print("\nRetrieved passages (cosine similarity, highest first):")
    for h in res.retrieved:
        print(f"  [{h.rank}] {h.score:.4f}  {h.passage.source:12s} :: {h.passage.text[:64]}…")
    print(f"\ntop_score      : {res.top_score:.4f}")
    print(f"CONFIDENCE     : {_BADGE.get(res.confidence, res.confidence)}")
    print(f"reason         : {res.reason}")
    if res.model_supported is not None:
        print(f"model grounded : {res.model_supported}")
    print(f"fallback used  : {res.fallback_used}")

    print("\nANSWER:")
    print("  " + res.answer.replace("\n", "\n  "))

    print("\nSOURCES:")
    if not res.sources:
        print("  (none — no trusted corpus passage supports this answer)")
    for s in res.sources:
        print(f"  [{s.marker}] {s.source} — {s.title}  (score {s.score})")
        print(f"      {s.snippet}")
    print(f"\nelapsed        : {res.elapsed_s}s")


def main() -> int:
    _rule("RAG AGENT — citation grounding — demo run")
    agent = RagAgent(CORPUS_DIR, k=3)
    print(f"model             : {agent.model}")
    print(f"corpus            : {len(agent.retriever.passages)} passages "
          f"from {CORPUS_DIR}")
    print(f"fallback threshold: {agent.fallback_threshold}  (top<this => web search)")
    print(f"confident thresh. : {agent.confident_threshold}  (top>=this + grounded => GROUNDED)")

    for title, query in CASES:
        _rule(title)
        _show(agent.answer(query))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
