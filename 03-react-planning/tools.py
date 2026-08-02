"""Three pure-Python tools for the ReAct agent — no external APIs, no network.

Each tool takes a single string argument (the model's "Action Input") and returns
a `ToolResult`. Tools never crash the agent: a bad argument or a simulated backend
failure comes back as `ToolResult(ok=False, error=...)`, which the agent surfaces
as an ERROR observation and can then reason around (graceful degradation).

  calculator        — a SAFE arithmetic evaluator (AST-walk, never eval())
  knowledge_lookup  — a tiny key/value knowledge base about a fictional company
  web_search        — a MOCK web search over a canned index, with a switchable
                      "backend outage" so the degradation path is demonstrable
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolResult:
    """Uniform tool return. `ok=False` carries a human-readable `error`."""

    ok: bool
    output: str = ""
    error: str = ""

    def as_observation(self) -> str:
        return self.output if self.ok else f"ERROR: {self.error}"


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], ToolResult]

    def run(self, arg: str) -> ToolResult:
        """Invoke the tool, turning ANY unexpected exception into a clean error.

        This is the first line of graceful degradation: a tool that raises never
        propagates up and crashes the loop — it becomes an error observation.
        """
        try:
            return self.func(arg)
        except Exception as exc:  # noqa: BLE001 — deliberately catch-all: tools must not crash the agent
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Tool 1 — calculator (safe arithmetic, no eval)
# --------------------------------------------------------------------------- #
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only +, -, *, /, //, %, ** on numbers are allowed")


def calculator(expr: str) -> ToolResult:
    """Evaluate an arithmetic expression safely (e.g. '47 * 12', '(3+4)**2')."""
    expr = (expr or "").strip()
    if not expr:
        return ToolResult(ok=False, error="empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return ToolResult(ok=False, error=f"could not parse '{expr}': {exc.msg}")
    value = _eval_node(tree)
    # Present whole numbers without a trailing '.0'.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return ToolResult(ok=True, output=f"{expr} = {value}")


# --------------------------------------------------------------------------- #
# Tool 2 — knowledge_lookup (fictional key/value KB)
# --------------------------------------------------------------------------- #
# Deliberately about a FICTIONAL company so the model cannot answer from
# parametric memory — a correct answer must come through this tool.
_KB: dict[str, str] = {
    "founded_year": "2011",
    "headquarters": "Reykjavik, Iceland",
    "ceo": "Dr. Ingrid Solveig",
    "employees": "320",
    "satellites_launched": "47",
    "cost_per_satellite_musd": "12",  # millions of USD, per satellite
    "primary_product": "the Aurora-class weather microsatellite",
}


def knowledge_lookup(key: str) -> ToolResult:
    """Look up a fact about the fictional company 'Zephyr Labs' by key."""
    key = (key or "").strip().lower().replace(" ", "_")
    if key in _KB:
        return ToolResult(ok=True, output=f"{key} = {_KB[key]}")
    return ToolResult(
        ok=False,
        error=(
            f"no fact stored under key '{key}'. "
            f"available keys: {', '.join(sorted(_KB))}"
        ),
    )


# --------------------------------------------------------------------------- #
# Tool 3 — web_search (MOCK, with a switchable outage for the degradation demo)
# --------------------------------------------------------------------------- #
class SearchBackendError(RuntimeError):
    """Raised to simulate an upstream search-backend failure (e.g. HTTP 503)."""


# A tiny canned index so the stub can return *something* for a couple of queries.
_SEARCH_INDEX: dict[str, str] = {
    "speed of light": "The speed of light in vacuum is about 299,792 km/s.",
    "days in a year": "A common year has 365 days; a leap year has 366.",
}

# Flipped on by run.py for the degradation demo to force real error observations.
_OUTAGE = False


def set_search_outage(enabled: bool) -> None:
    """Toggle a simulated search-backend outage (used to demo graceful degradation)."""
    global _OUTAGE
    _OUTAGE = enabled


def web_search(query: str) -> ToolResult:
    """Mock web search. Raises SearchBackendError when the backend is 'down'."""
    query = (query or "").strip()
    if not query:
        return ToolResult(ok=False, error="empty query")
    if _OUTAGE:
        # A genuine exception — Tool.run() catches it and degrades to an error obs.
        raise SearchBackendError("upstream search backend returned 503 (simulated outage)")
    for key, snippet in _SEARCH_INDEX.items():
        if key in query.lower():
            return ToolResult(ok=True, output=f"[mock-search] {snippet}")
    return ToolResult(ok=False, error=f"no results found for '{query}'")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def default_tools() -> dict[str, Tool]:
    return {
        "calculator": Tool(
            name="calculator",
            description="Evaluate an arithmetic expression. Input: an expression like '47 * 12' or '(365 - 12) / 7'.",
            func=calculator,
        ),
        "knowledge_lookup": Tool(
            name="knowledge_lookup",
            description=(
                "Look up a stored fact about the fictional company 'Zephyr Labs' by key. "
                "Input: one key, e.g. 'satellites_launched', 'cost_per_satellite_musd', "
                "'founded_year', 'employees', 'headquarters', 'ceo'."
            ),
            func=knowledge_lookup,
        ),
        "web_search": Tool(
            name="web_search",
            description="Search the (mock) public web for general facts. Input: a search query string.",
            func=web_search,
        ),
    }
