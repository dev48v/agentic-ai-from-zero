"""agentfuse — deterministic runtime fuses for LLM agents. No dependencies.

Five guards, each extracted from a project in the *Agentic AI from Zero* series after it
had already failed for real at least once:

    PermissionFuse   deny-by-default tool scoping                (Project 4)
    LoopFuse         repeated-state + cycle detection, inline    (Project 11)
    BudgetFuse       pre-flight $ / token ceiling, real usage    (Project 7)
    JudgeGate        LLM self-score co-gated by hard checks      (Project 10)
    CanaryFuse       hashed traffic split + gated rollback       (Project 11)

The doctrine behind all five: **the model reasons, plain code enforces.** Anything a
guard decides is a pure function of recorded facts, so it can be replayed, unit-tested,
and argued with. Anything ambiguous resolves to "stop".

    from agentfuse import FuseBox, LoopFuse, PermissionFuse, ToolSpec, BudgetFuse, Price

    box = FuseBox(
        loop=LoopFuse(repeat_threshold=3),
        permission=PermissionFuse(granted={"read"},
                                  specs=[ToolSpec.of("search", "read"),
                                         ToolSpec.of("refund", "write", "spend_money")]),
        budget=BudgetFuse(max_usd=0.01, price=Price(0.0002, 0.0002)),
    )

    verdict = box.check_tool_call(ToolCall("refund", {"order": 42}))
    assert verdict.blocked
"""

from .budget import BudgetFuse, Price, estimate_prompt_tokens
from .canary import CanaryFuse, CanaryVerdict, Gate, VersionStats, percentile
from .core import ALLOW, BLOCK, FuseError, ToolCall, Verdict, canonical_signature
from .judge import CheckResult, GateVerdict, HardCheck, JudgeGate
from .loops import LoopFuse, LoopState
from .permissions import (EXTERNAL_WRITE, NETWORK, READ, SPEND_MONEY, WRITE,
                          PermissionFuse, ToolSpec)
from .runtime import FuseBox

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "ALLOW", "BLOCK", "ToolCall", "Verdict", "FuseError", "canonical_signature",
    "LoopFuse", "LoopState",
    "PermissionFuse", "ToolSpec", "READ", "WRITE", "NETWORK", "SPEND_MONEY",
    "EXTERNAL_WRITE",
    "BudgetFuse", "Price", "estimate_prompt_tokens",
    "JudgeGate", "HardCheck", "CheckResult", "GateVerdict",
    "CanaryFuse", "CanaryVerdict", "VersionStats", "Gate", "percentile",
    "FuseBox",
]
