"""Runnable demo for the Multi-Tool Orchestrator — shows all five sub-points
against NVIDIA NIM.

  DEMO 1 — dynamic tool registry:
      dump the registry the six tools self-registered into + the capability menu.

  DEMO 2 — capability routing + parallel execution + conflict resolution:
      a FULL-permission run asks for AAPL's price. The LLM routes by CAPABILITY
      (price_quote) to the three feeds; they run CONCURRENTLY (timed vs serial);
      beta disagrees, so the deterministic policy resolves the conflict; the model
      synthesises the final answer.

  DEMO 3 — permission scoping (deny-by-default):
      a RESTRICTED run (granted read+network, NOT write) is asked to record a note
      to the ledger. Routing lands on the write tool, which is DENIED and logged.
      The same task under a full grant then SUCCEEDS — proving the scope gates.

Run:  python 04-orchestrator/run.py
"""

from __future__ import annotations

import logging
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _REPO_ROOT)

# Windows consoles default to cp1252 and choke on the box-drawing/✓ glyphs.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from agent import ExecResult, Orchestrator  # noqa: E402
import tools as _tools  # noqa: E402  (import registers the six tools)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)


def _rule(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _show(res: ExecResult) -> None:
    print(f"\nTask     : {res.task}")
    print(f"Granted  : {res.granted}")
    print(f"Route    : via={res.route.via}  capabilities={res.route.capabilities}")
    print(f"           reason: {res.route.reason}")
    print(f"           selected tools: {res.route.tools}")
    if res.denials:
        print("Denials  :")
        for d in res.denials:
            print(f"           DENIED {d.tool} (needs '{d.permission}') — {d.reason}")
    if res.invocations:
        print("Ran      :")
        for i in res.invocations:
            tag = "ok " if i.ok else "ERR"
            print(f"           [{tag}] {i.tool:<14} {i.elapsed_s:>5.2f}s  {i.output or i.error}")
    if res.serial_wall_s is not None:
        saved = res.serial_wall_s - res.parallel_wall_s
        speedup = res.serial_wall_s / res.parallel_wall_s if res.parallel_wall_s else 0
        print("Timing   :")
        print(f"           serial   wall-clock : {res.serial_wall_s:.2f}s")
        print(f"           parallel wall-clock : {res.parallel_wall_s:.2f}s")
        print(f"           saved {saved:.2f}s  ({speedup:.2f}x faster running concurrently)")
    else:
        print(f"Timing   : {res.mode} wall-clock {res.wall_clock_s:.2f}s")
    if res.conflict:
        c = res.conflict
        print("Conflict :")
        print(f"           subject      : {c.subject}")
        print(f"           feeds        : "
              + ", ".join(f"{q['source']}=${q['price']:.2f}(trust={q['trust_priority']})"
                          for q in c.quotes))
        print(f"           policy       : {c.policy}")
        print(f"           winner       : ${c.winner_price:.2f}  from {c.winner_sources}")
        print(f"           disagreeing  : {c.disagreeing_sources}")
        print(f"           recorded     : {c.note}")
    print(f"ANSWER   : {res.final_answer}")


def main() -> int:
    _rule("MULTI-TOOL ORCHESTRATOR — demo run (NVIDIA NIM)")

    # ---------------- DEMO 1 — dynamic tool registry ---------------- #
    _rule("DEMO 1 — dynamic tool registry (tools self-registered via @tool)")
    print(Orchestrator.describe_registry())

    # -------- DEMO 2 — routing + parallel + conflict (full grant) ---- #
    _rule("DEMO 2 — capability routing + PARALLEL execution + CONFLICT resolution")
    full = Orchestrator(granted_permissions={"read", "write", "network"})
    price_task = "What is the current reference price of AAPL? Consult the price feeds."
    _show(full.execute(price_task, args={"symbol": "AAPL"}, compare=True))

    # -------- DEMO 3 — permission scoping (deny-by-default) ---------- #
    _rule("DEMO 3 — permission scoping: a RESTRICTED run is DENIED the write tool")
    restricted = Orchestrator(granted_permissions={"read", "network"})  # NO 'write'
    persist_task = "Record an audit note in the ledger that AAPL was reviewed today."
    note = {"note": "AAPL reviewed 2026-08-03; reference price $150.25 (majority of 3 feeds)."}
    _show(restricted.execute(persist_task, args=note))

    print("\n--- same task, now WITH the write scope granted (proves the gate) ---")
    writer = Orchestrator(granted_permissions={"read", "write", "network"})
    _show(writer.execute(persist_task, args=note))
    print(f"\nledger contents now: {_tools.LEDGER}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
