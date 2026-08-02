"""Multi-Tool Orchestrator — route across many tools safely and in parallel.

Five ideas, hand-rolled (no framework). Each maps to one sub-point:

  1. dynamic tool registry      -> registry.py (the @tool decorator); the router
                                   below reads it only through capability queries,
                                   so new tools need no router change.
  2. capability-based routing   -> `Orchestrator.route`: the model is shown the
                                   registry's ADVERTISED CAPABILITIES (not tool
                                   names) and picks the capabilities a task needs;
                                   the orchestrator resolves those to tools.
  3. permission scoping         -> `Orchestrator._enforce`: deny-by-default. A tool
                                   runs only if its declared permission is in the
                                   run's GRANTED set; otherwise it is refused and
                                   the denial is logged.
  4. parallel execution         -> `Orchestrator.invoke_batch(mode="parallel")`:
                                   independent tools run CONCURRENTLY in a thread
                                   pool; `execute(compare=True)` times serial vs
                                   parallel to show the wall-clock saving.
  5. conflict resolution        -> `Orchestrator.resolve_price_conflict`: when the
                                   price feeds disagree, a documented deterministic
                                   policy (majority -> trust-priority -> freshness)
                                   picks a winner and RECORDS the conflict.

The router is LLM-driven (NVIDIA NIM) with a keyword fallback so a small model can
never wedge the pipeline; every run states which path it used.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from common.client import DEFAULT_MODEL, get_client
import registry
from registry import Tool

logger = logging.getLogger("orchestrator")


# --------------------------------------------------------------------------- #
# Result records
# --------------------------------------------------------------------------- #
@dataclass
class Invocation:
    tool: str
    permission: str
    args: dict
    ok: bool
    output: str
    data: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str = ""


@dataclass
class Denial:
    tool: str
    permission: str
    reason: str


@dataclass
class RouteDecision:
    capabilities: list[str]
    tools: list[str]
    reason: str
    via: str  # "llm" or "fallback"


@dataclass
class ConflictResolution:
    subject: str
    quotes: list[dict]              # every feed's raw {source, price, trust, as_of}
    disagreeing_sources: list[str]  # sources that differ from the winner
    policy: str                     # which policy step decided it
    winner_price: float
    winner_sources: list[str]
    note: str


@dataclass
class ExecResult:
    task: str
    granted: list[str]
    route: RouteDecision
    invocations: list[Invocation]
    denials: list[Denial]
    conflict: ConflictResolution | None
    final_answer: str
    mode: str = "parallel"
    wall_clock_s: float = 0.0
    serial_wall_s: float | None = None
    parallel_wall_s: float | None = None


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else text.strip()


# Keyword -> capability, used only if the LLM returns nothing usable. Keeps the
# demo robust on an 8B model without hiding when the fallback fired.
_FALLBACK_KEYWORDS = {
    "price_quote": ("price", "quote", "stock", "ticker", "share"),
    "persist": ("record", "ledger", "persist", "save", "log ", "audit", "write"),
    "weather": ("weather", "forecast", "temperature"),
    "currency": ("convert", "currency", "exchange", "fx", "usd", "eur", "gbp"),
}


class Orchestrator:
    def __init__(
        self,
        granted_permissions: set[str],
        model: str = DEFAULT_MODEL,
    ) -> None:
        # Import tools so their @tool decorators populate the shared registry.
        # NOTE: we import the module for its side effect only — the router never
        # names a tool, proving new tools need no router change.
        import tools  # noqa: F401
        self.granted = set(granted_permissions)
        self.model = model
        self.client = get_client()

    # ------------------------------------------------------------------ #
    # #1 dynamic registry — a human-readable dump of what self-registered.
    # ------------------------------------------------------------------ #
    @staticmethod
    def describe_registry() -> str:
        rows = ["name            | permission | capabilities            | description",
                "-" * 92]
        for t in sorted(registry.all_tools(), key=lambda x: x.name):
            rows.append(
                f"{t.name:<15} | {t.permission:<10} | "
                f"{','.join(sorted(t.capabilities)):<23} | {t.description}"
            )
        rows.append("")
        rows.append("advertised capabilities (the routing menu):")
        for cap, providers in registry.advertised_capabilities().items():
            rows.append(f"  - {cap:<12} -> {', '.join(providers)}")
        return "\n".join(rows)

    # ------------------------------------------------------------------ #
    # #2 capability-based routing (LLM picks CAPABILITIES, not tool names).
    # ------------------------------------------------------------------ #
    def route(self, task: str) -> RouteDecision:
        menu = registry.advertised_capabilities()
        cap_list = "\n".join(
            f"- {cap}: provided by {', '.join(names)}" for cap, names in menu.items()
        )
        system = (
            "You are the router for a multi-tool orchestrator. You do NOT pick tools "
            "by name. You pick the CAPABILITIES a task needs from the advertised menu; "
            "the orchestrator then resolves those capabilities to concrete tools.\n\n"
            f"Advertised capabilities:\n{cap_list}\n\n"
            "Choose the SMALLEST set of capabilities that satisfies the task. "
            "Reply with ONLY a JSON object, no prose:\n"
            '{"capabilities": ["<cap>", ...], "reason": "<one sentence>"}'
        )
        chosen: list[str] = []
        reason = ""
        via = "llm"
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Task: {task}"},
                ],
                temperature=0,
            )
            data = json.loads(_extract_json(resp.choices[0].message.content or ""))
            chosen = [str(c).strip() for c in data.get("capabilities", [])]
            reason = str(data.get("reason", "")).strip()
        except Exception as exc:  # noqa: BLE001 — router must not crash the run
            logger.warning("router LLM call failed (%s); using keyword fallback", exc)

        # Keep only capabilities the registry actually advertises (drop hallucinations).
        valid = set(menu)
        chosen = [c for c in chosen if c in valid]

        if not chosen:  # LLM gave nothing usable -> deterministic keyword fallback
            via = "fallback"
            low = task.lower()
            for cap, words in _FALLBACK_KEYWORDS.items():
                if cap in valid and any(w in low for w in words):
                    chosen.append(cap)
            reason = reason or "keyword fallback (LLM returned no valid capability)"

        tools = registry.tools_for_capabilities(set(chosen))
        decision = RouteDecision(
            capabilities=sorted(set(chosen)),
            tools=[t.name for t in tools],
            reason=reason,
            via=via,
        )
        logger.info("route via=%s caps=%s -> tools=%s",
                    decision.via, decision.capabilities, decision.tools)
        return decision

    # ------------------------------------------------------------------ #
    # #3 permission scoping — deny-by-default.
    # ------------------------------------------------------------------ #
    def _enforce(self, tools: list[Tool]) -> tuple[list[Tool], list[Denial]]:
        allowed, denials = [], []
        for t in tools:
            if t.permission in self.granted:
                allowed.append(t)
            else:
                d = Denial(
                    tool=t.name,
                    permission=t.permission,
                    reason=(f"permission '{t.permission}' not in granted scope "
                            f"{sorted(self.granted)}"),
                )
                denials.append(d)
                logger.warning("DENIED %s — %s", t.name, d.reason)
        return allowed, denials

    # ------------------------------------------------------------------ #
    # #4 parallel execution — run independent tools concurrently.
    # ------------------------------------------------------------------ #
    def _invoke_one(self, tool: Tool, args: dict) -> Invocation:
        started = time.perf_counter()
        res = tool.run(args)
        return Invocation(
            tool=tool.name,
            permission=tool.permission,
            args=args,
            ok=res.ok,
            output=res.output,
            data=res.data,
            elapsed_s=round(time.perf_counter() - started, 3),
            error=res.error,
        )

    def invoke_batch(
        self, tools: list[Tool], args: dict, mode: str = "parallel"
    ) -> tuple[list[Invocation], float]:
        started = time.perf_counter()
        if mode == "parallel" and len(tools) > 1:
            with ThreadPoolExecutor(max_workers=len(tools)) as ex:
                invocations = list(ex.map(lambda t: self._invoke_one(t, args), tools))
        else:
            invocations = [self._invoke_one(t, args) for t in tools]
        wall = round(time.perf_counter() - started, 3)
        return invocations, wall

    # ------------------------------------------------------------------ #
    # #5 conflict resolution — documented, deterministic policy.
    #   1. MAJORITY: the modal price wins.
    #   2. tie on count  -> TRUST-PRIORITY: lowest trust_priority number wins.
    #   3. still tied     -> FRESHNESS: latest as_of wins.
    # ------------------------------------------------------------------ #
    @staticmethod
    def resolve_price_conflict(invocations: list[Invocation]) -> ConflictResolution | None:
        quotes = [
            inv.data for inv in invocations
            if inv.ok and "price" in inv.data
        ]
        if len(quotes) < 2:
            return None
        subject = quotes[0].get("symbol", "?")
        prices = [q["price"] for q in quotes]
        if len(set(prices)) == 1:
            return None  # unanimous — no conflict to resolve

        counts = Counter(prices)
        top = max(counts.values())
        leaders = [p for p, c in counts.items() if c == top]

        if len(leaders) == 1:
            winner, policy = leaders[0], "majority"
        else:
            # tie on vote count -> most trusted (lowest number) among the leaders
            pool = [q for q in quotes if q["price"] in leaders]
            best_trust = min(q["trust_priority"] for q in pool)
            pool2 = [q for q in pool if q["trust_priority"] == best_trust]
            if len({q["price"] for q in pool2}) == 1:
                winner, policy = pool2[0]["price"], "trust-priority"
            else:
                # still tied -> freshest as_of
                freshest = max(pool2, key=lambda q: q["as_of"])
                winner, policy = freshest["price"], "freshness"

        winner_sources = sorted(q["source"] for q in quotes if q["price"] == winner)
        disagreeing = sorted(q["source"] for q in quotes if q["price"] != winner)
        price_str = ", ".join("{}=${:.2f}".format(q["source"], q["price"]) for q in quotes)
        note = (
            f"{len(quotes)} feeds queried; prices {{{price_str}}}. "
            f"Policy '{policy}' selected ${winner:.2f} "
            f"(sources: {', '.join(winner_sources)}); "
            f"disagreeing: {', '.join(disagreeing)}."
        )
        resolution = ConflictResolution(
            subject=subject,
            quotes=quotes,
            disagreeing_sources=disagreeing,
            policy=policy,
            winner_price=winner,
            winner_sources=winner_sources,
            note=note,
        )
        logger.warning("CONFLICT on %s resolved by %s -> $%.2f (%s)",
                       subject, policy, winner, note)
        return resolution

    # ------------------------------------------------------------------ #
    # Final-answer synthesis (LLM turns tool outputs into a sentence).
    # ------------------------------------------------------------------ #
    def synthesize(
        self,
        task: str,
        invocations: list[Invocation],
        denials: list[Denial],
        conflict: ConflictResolution | None,
    ) -> str:
        lines = [f"- {i.tool}: {i.output if i.ok else 'ERROR: ' + i.error}"
                 for i in invocations]
        deny_lines = [f"- DENIED {d.tool} ({d.permission}): {d.reason}" for d in denials]
        conflict_line = conflict.note if conflict else "(no conflict)"
        system = (
            "You write the final answer for a tool orchestrator. Use ONLY the tool "
            "outputs, denials, and conflict resolution below. If a needed tool was "
            "DENIED, say plainly that the action could not be performed and why. If a "
            "conflict was resolved, state the chosen value and that a conflict was "
            "recorded. Be concise (1-3 sentences). Do NOT invent numbers."
        )
        user = (
            f"Task: {task}\n\n"
            f"Tool outputs:\n{chr(10).join(lines) or '(none ran)'}\n\n"
            f"Denials:\n{chr(10).join(deny_lines) or '(none)'}\n\n"
            f"Conflict resolution: {conflict_line}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("synthesis failed (%s); returning a mechanical summary", exc)
            if denials and not invocations:
                return f"Could not complete '{task}': {denials[0].reason}."
            if conflict:
                return f"{conflict.subject} = ${conflict.winner_price:.2f} (conflict recorded)."
            return "; ".join(i.output for i in invocations if i.ok) or "No result."

    # ------------------------------------------------------------------ #
    # The full pipeline.
    # ------------------------------------------------------------------ #
    def execute(
        self,
        task: str,
        args: dict | None = None,
        mode: str = "parallel",
        compare: bool = False,
    ) -> ExecResult:
        args = args or {}
        decision = self.route(task)
        selected = registry.tools_for_capabilities(set(decision.capabilities))
        allowed, denials = self._enforce(selected)

        serial_wall = parallel_wall = None
        if compare and len(allowed) > 1:
            # Time both ways over the SAME resolved tool set for an honest A/B.
            _, serial_wall = self.invoke_batch(allowed, args, mode="serial")
            invocations, parallel_wall = self.invoke_batch(allowed, args, mode="parallel")
            wall = parallel_wall
            mode = "parallel"
        else:
            invocations, wall = self.invoke_batch(allowed, args, mode=mode)

        conflict = self.resolve_price_conflict(invocations)
        final = self.synthesize(task, invocations, denials, conflict)

        return ExecResult(
            task=task,
            granted=sorted(self.granted),
            route=decision,
            invocations=invocations,
            denials=denials,
            conflict=conflict,
            final_answer=final,
            mode=mode,
            wall_clock_s=wall,
            serial_wall_s=serial_wall,
            parallel_wall_s=parallel_wall,
        )
