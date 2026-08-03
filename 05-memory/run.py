"""Runnable two-phase demo for the Memory-Enabled Conversational Agent (NVIDIA NIM).

Cross-session memory can only be PROVEN across two separate OS processes, so the demo
is split into two phases that share one on-disk store:

  python 05-memory/run.py session1
      A fresh store. The user states a few durable facts (name, a peanut allergy, a
      project), then chats enough that the short-term buffer OVERFLOWS — each overflow
      COMPRESSES the oldest turns into a long-term memory note. On exit the store is
      persisted to memory_store.json and the process ends.

  python 05-memory/run.py session2
      A brand-new process. It LOADS the store from disk and asks questions that require
      recalling session-1 facts. For each, it shows the relevance SCORES, which memory
      was INJECTED, and that the answer is correct — recall from long-term memory that
      out-lived the first process.

Run both in order:
  python 05-memory/run.py session1
  python 05-memory/run.py session2
"""

from __future__ import annotations

import logging
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _REPO_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from agent import MemoryAgent, TurnResult  # noqa: E402
from memory import MemoryStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

STORE_PATH = os.path.join(_PROJECT_DIR, "memory_store.json")


def _rule(title: str) -> None:
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def _show_turn(res: TurnResult, show_recall: bool = True) -> None:
    print(f"\n👤 USER : {res.user}")
    if show_recall and res.recalled:
        print("   🔎 relevance scoring over long-term memory (top-k cosine):")
        for r in res.recalled:
            tag = "INJECT" if r.injected else " skip "
            print(f"        [{tag}] score={r.score:+.3f}  #{r.memory.id} "
                  f"({r.memory.kind}, {r.memory.session}): {r.memory.text}")
    elif show_recall:
        print("   🔎 long-term memory is empty — nothing to recall yet.")
    print(f"🤖 AGENT: {res.answer}")
    if res.compression:
        c = res.compression
        print(f"   🗜  COMPRESSION FIRED — buffer overflowed; summarised "
              f"{c.compressed_exchanges} oldest exchange(s) into long-term memory "
              f"#{c.summary_id} (freed ~{c.freed_tokens_est} tokens):")
        print(f"        “{c.summary_text}”")
    print(f"   📦 buffer: {res.buffer_size} exchanges (~{res.buffer_tokens} tok) · "
          f"long-term: {res.long_term_size} memories")


def session1() -> int:
    _rule("SESSION 1 — teach the agent facts; let the buffer overflow → COMPRESSION")
    # Start clean so the recorded run is reproducible.
    if os.path.exists(STORE_PATH):
        os.remove(STORE_PATH)
    store = MemoryStore(STORE_PATH, max_turns=4, compress_chunk=2, top_k=3, min_score=0.05)
    store.load()
    agent = MemoryAgent(store, session="session-1")

    # Facts first (they will be compressed into long-term as the buffer overflows),
    # then ordinary chatter that pushes them out of the verbatim window.
    turns = [
        "Hi! My name is Devanshu.",
        "Important: I'm allergic to peanuts — please remember that.",
        "I'm building a travel-planner app called Safar, in Flutter with a Supabase backend.",
        "Give me a catchy one-line tagline for a travel app.",
        "Suggest a calm colour for the app's main theme.",
        "List three trip-planning features worth adding first.",
        "What's a good font pairing for a travel brand?",
        "Recommend one app-store screenshot idea that sells it.",
    ]
    for msg in turns:
        _show_turn(agent.chat(msg))

    _rule("SESSION 1 — end state (persisted to memory_store.json)")
    print(f"short-term buffer now holds {len(store.buffer)} recent exchanges (verbatim):")
    for e in store.buffer:
        print(f"   • {e.user}")
    turns = [m for m in store.long_term if m.kind == "turn"]
    summaries = [m for m in store.long_term if m.kind == "summary"]
    print(f"\nlong-term vector store now holds {len(store.long_term)} memories "
          f"({len(turns)} archived turns + {len(summaries)} compression summaries):")
    for m in store.long_term:
        print(f"   #{m.id} [{m.kind}] {m.text}")
    print("\n💾 saved. The next command starts a FRESH process that only sees this file.")
    return 0


def session2() -> int:
    _rule("SESSION 2 — a FRESH process; recall session-1 facts from long-term memory")
    store = MemoryStore(STORE_PATH, max_turns=4, compress_chunk=2, top_k=3, min_score=0.05)
    n = store.load()
    if n == 0:
        print("No memory_store.json found — run `python 05-memory/run.py session1` first.")
        return 1
    print(f"loaded {n} long-term memories + {len(store.buffer)} residual buffer exchanges "
          f"from a PRIOR process:")
    for m in store.long_term:
        print(f"   #{m.id} [{m.kind}, {m.session}] {m.text}")

    agent = MemoryAgent(store, session="session-2")
    questions = [
        "Remind me — what am I allergic to?",
        "What's my travel app called, and what's it built with?",
        "And what's my name again?",
    ]
    for q in questions:
        _show_turn(agent.chat(q))

    _rule("SESSION 2 — done")
    print("Every answer above came from long-term memory created in session 1, in a "
          "DIFFERENT process — proving cross-session recall + relevance scoring.")
    return 0


def main(argv: list[str]) -> int:
    phase = argv[1] if len(argv) > 1 else ""
    if phase == "session1":
        return session1()
    if phase == "session2":
        return session2()
    print("usage: python 05-memory/run.py [session1|session2]")
    print("  run session1 first (teaches facts + compresses), then session2 (recalls).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
