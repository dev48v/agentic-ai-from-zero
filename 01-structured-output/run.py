"""Runnable demo for the Structured Output Agent.

Extracts a strict, typed Invoice from a messy purchase note. The source text is
engineered so the model's first attempt violates the schema (the invoice number
isn't zero-padded), which forces a real parse-error -> retry -> success loop.

Run:  python 01-structured-output/run.py
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

from agent import StructuredOutputAgent  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

# The source text a real user might paste. Note "Invoice 4521" — not zero-padded —
# and the total left for the agent to compute. This is what trips attempt #1.
SOURCE_TEXT = (
    "Invoice 4521 for customer billing@acme-corp.com (California, US-CA).\n"
    "They ordered 3 units of the Widget Pro at $12.50 each, billed in US dollars.\n"
    "Apply the standard California sales tax."
)
REGION = "US-CA"


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> int:
    _rule("STRUCTURED OUTPUT AGENT — demo run")
    agent = StructuredOutputAgent(max_retries=3)
    print(f"model      : {agent.model}")
    print(f"max_retries: {agent.max_retries}")
    print("\nSOURCE TEXT (the prompt):\n" + SOURCE_TEXT)

    _rule("STEP 1 — resolve tax via tool (response validated against TaxRateResult)")
    tax = agent.resolve_tax(REGION)
    print("validated tool response:", tax.model_dump())

    _rule("STEP 2 — extract Invoice (enforce schema, retry on parse errors)")
    result = agent.extract_invoice(SOURCE_TEXT, tax)

    _rule("FAILURE LOG (each rejected attempt)")
    if not result.attempts:
        print("(none — model satisfied the schema on the first attempt)")
    for a in result.attempts:
        print(f"\n--- attempt {a.n}: REJECTED ---")
        print(f"validation error: {a.error}")
        print(f"raw model output: {a.raw_output}")

    _rule("FINAL — validated, typed Invoice")
    print(result.invoice.model_dump_json(indent=2))
    print(f"\nattempts used : {len(result.attempts) + 1}")
    print(f"elapsed       : {result.elapsed_s}s")
    print(f"type          : {type(result.invoice).__name__} (Pydantic-validated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
