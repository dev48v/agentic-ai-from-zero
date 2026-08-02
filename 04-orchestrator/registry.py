"""Dynamic tool registry — tools register THEMSELVES at runtime via a decorator.

This is sub-point #1 ("dynamic tool registry"). A tool is a plain Python function
tagged with `@tool(...)`; importing the module that defines it registers it into
the global `REGISTRY` with:

    name         — unique id the router never hard-codes
    description   — one line the LLM reads when routing
    capabilities  — a SET of capability tags (what the tool CAN do); routing
                    matches on these, never on the name
    permission    — the single scope the tool needs (read / write / network);
                    the orchestrator refuses to run it unless that scope is granted
    args_schema   — a tiny JSON-schema-style dict ({"properties": ..., "required": ...})
                    used for light argument validation

Because the router (agent.py) only ever asks this registry "which tools advertise
capability X?", a brand-new tool becomes routable the moment its module is
imported — WITHOUT editing the router. tools.py proves that: it adds six tools and
agent.py imports none of them by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# The three permission scopes this series uses. Deny-by-default: a run must be
# GRANTED a scope before a tool needing it may execute (enforced in agent.py).
KNOWN_PERMISSIONS = frozenset({"read", "write", "network"})


@dataclass
class ToolResult:
    """Uniform tool return.

    `output`  — a human-readable string (what a person would read).
    `data`    — structured payload the orchestrator can reason over (e.g. a price
                feed puts {"symbol","price","trust_priority","as_of"} here so the
                conflict resolver can compare feeds numerically).
    `error`   — set with ok=False when the tool could not do its job.
    """

    ok: bool
    output: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""

    def as_line(self) -> str:
        return self.output if self.ok else f"ERROR: {self.error}"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    capabilities: frozenset[str]
    permission: str
    args_schema: dict
    func: Callable[[dict], ToolResult]

    def run(self, args: dict) -> ToolResult:
        """Invoke the tool, turning ANY unexpected exception into a clean error
        and enforcing required args from the schema. A tool must never crash the
        orchestrator — a failure comes back as ToolResult(ok=False)."""
        required = self.args_schema.get("required", [])
        missing = [k for k in required if k not in (args or {})]
        if missing:
            return ToolResult(
                ok=False,
                error=f"missing required arg(s) {missing} for tool '{self.name}'",
            )
        try:
            return self.func(args or {})
        except Exception as exc:  # noqa: BLE001 — tools must not crash the router
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# The registry itself + the self-registration decorator.
# --------------------------------------------------------------------------- #
REGISTRY: dict[str, Tool] = {}


def tool(
    *,
    name: str,
    description: str,
    capabilities: set[str] | frozenset[str],
    permission: str,
    args_schema: dict | None = None,
) -> Callable[[Callable[[dict], ToolResult]], Callable[[dict], ToolResult]]:
    """Decorator: register the decorated function as a Tool at import time.

    New tools are added by writing a function + this decorator — the router is
    never touched.
    """
    if permission not in KNOWN_PERMISSIONS:
        raise ValueError(
            f"tool '{name}' declares unknown permission '{permission}'; "
            f"must be one of {sorted(KNOWN_PERMISSIONS)}"
        )

    def deco(fn: Callable[[dict], ToolResult]) -> Callable[[dict], ToolResult]:
        if name in REGISTRY:
            raise ValueError(f"tool '{name}' is already registered")
        REGISTRY[name] = Tool(
            name=name,
            description=description,
            capabilities=frozenset(capabilities),
            permission=permission,
            args_schema=args_schema or {},
            func=fn,
        )
        return fn

    return deco


# --------------------------------------------------------------------------- #
# Read-only views the router uses. The router asks by CAPABILITY, never by name.
# --------------------------------------------------------------------------- #
def all_tools() -> list[Tool]:
    return list(REGISTRY.values())


def advertised_capabilities() -> dict[str, list[str]]:
    """Map every advertised capability -> the tool names that provide it.

    This is exactly the menu the orchestrator shows the model when routing: the
    model picks capabilities from these keys, not tool names.
    """
    caps: dict[str, list[str]] = {}
    for t in REGISTRY.values():
        for c in sorted(t.capabilities):
            caps.setdefault(c, []).append(t.name)
    return {c: caps[c] for c in sorted(caps)}


def tools_for_capabilities(capabilities: set[str] | frozenset[str]) -> list[Tool]:
    """Every tool whose capability set intersects the requested capabilities."""
    want = frozenset(capabilities)
    selected = [t for t in REGISTRY.values() if t.capabilities & want]
    # Stable, deterministic order.
    return sorted(selected, key=lambda t: t.name)


def reset_registry() -> None:
    """Test hook — clear the registry (not used by the demo)."""
    REGISTRY.clear()
