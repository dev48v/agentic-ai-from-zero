"""Structured Output Agent.

Every output is a validated, typed object — never free text you have to parse.

Four ideas, hand-rolled:
  1. enforce a Pydantic JSON schema  -> Invoice.model_validate_json(...)
  2. validate tool responses         -> TaxRateResult.model_validate(tool_output)
  3. retry on parse errors           -> the parse->reprompt loop below
  4. log validation failures         -> structured logging on every failed attempt
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from common.client import DEFAULT_MODEL, get_client
from schemas import Invoice, TaxRateResult

logger = logging.getLogger("structured_output_agent")


# --------------------------------------------------------------------------- #
# Tool: a plain Python function. Its output is UNTRUSTED and gets validated.
# --------------------------------------------------------------------------- #
_TAX_TABLE = {
    "US-CA": 0.0825,
    "US-NY": 0.08,
    "US": 0.08,
    "GB": 0.20,
    "DE": 0.19,
}


def get_tax_rate(region: str) -> dict[str, Any]:
    """Look up the sales/VAT tax rate (as a fraction) for a region code."""
    return {
        "region": region.upper(),
        "rate": _TAX_TABLE.get(region.upper(), 0.0),
        "source": "static-tax-table-v1",
    }


# OpenAI-compatible tool schema so the model can request the tool itself.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_tax_rate",
            "description": "Get the sales/VAT tax rate (a fraction like 0.08) for a region code such as US-CA, US-NY, GB, or DE.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Region code, e.g. US-CA, US-NY, GB, DE.",
                    }
                },
                "required": ["region"],
            },
        },
    }
]


@dataclass
class Attempt:
    """One record in the structured failure log."""

    n: int
    error: str
    raw_output: str


@dataclass
class AgentResult:
    invoice: Invoice
    tax: TaxRateResult
    attempts: list[Attempt] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    elapsed_s: float = 0.0


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply, tolerating ```json fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else text.strip()


class StructuredOutputAgent:
    def __init__(self, model: str = DEFAULT_MODEL, max_retries: int = 3) -> None:
        self.client = get_client()
        self.model = model
        self.max_retries = max_retries

    # ------------------------------------------------------------------ #
    # Sub-point 2: validate tool responses.
    # Real LLM tool-call round-trip, with a direct-execution fallback so the
    # demo is robust even if the model skips the tool call.
    # ------------------------------------------------------------------ #
    def resolve_tax(self, region: str) -> TaxRateResult:
        raw: dict[str, Any] | None = None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a tax lookup assistant. Use the get_tax_rate tool.",
                    },
                    {"role": "user", "content": f"What is the tax rate for region {region}?"},
                ],
                tools=TOOLS,
                tool_choice="auto",
                temperature=0,
            )
            tool_calls = resp.choices[0].message.tool_calls or []
            for call in tool_calls:
                if call.function.name == "get_tax_rate":
                    args = json.loads(call.function.arguments or "{}")
                    raw = get_tax_rate(args.get("region", region))
                    logger.info("tool get_tax_rate(%s) -> %s (via model tool-call)", args, raw)
                    break
        except Exception as exc:  # network / model / tool-calling hiccup
            logger.warning("tool-call round failed (%s); executing tool directly", exc)

        if raw is None:
            raw = get_tax_rate(region)
            logger.info("tool get_tax_rate(%s) -> %s (direct)", region, raw)

        # Validate the tool response against its schema BEFORE trusting it.
        try:
            return TaxRateResult.model_validate(raw)
        except ValidationError as exc:
            logger.error("tool response failed validation: %s | raw=%s", exc, raw)
            raise

    # ------------------------------------------------------------------ #
    # Sub-points 1, 3, 4: enforce schema, retry on parse errors, log failures.
    # ------------------------------------------------------------------ #
    def extract_invoice(self, source_text: str, tax: TaxRateResult) -> AgentResult:
        started = time.time()
        attempts: list[Attempt] = []

        # Deliberately terse first prompt: it lists field names but NOT the strict
        # formatting rules (zero-padded id, tax as a fraction, math consistency).
        # Those live only in the schema, so attempt #1 almost always violates one,
        # and the validation error teaches the model to self-correct on retry.
        system = (
            "You extract invoices from text and reply with ONLY a JSON object "
            "(no prose, no markdown). Fields: invoice_id, customer_email, currency, "
            "line_items (list of {description, quantity, unit_price}), subtotal, "
            "tax_rate, total."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"{source_text}\n\n"
                    f"Use tax_rate = {tax.rate} for region {tax.region}. "
                    f"Return the invoice as JSON."
                ),
            },
        ]

        for n in range(1, self.max_retries + 1):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            candidate = _extract_json(raw)

            try:
                invoice = Invoice.model_validate_json(candidate)
            except ValidationError as exc:
                errors = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                    for e in exc.errors()
                )
                # Sub-point 4: structured log of the validation failure.
                logger.warning(
                    "validation failed",
                    extra={"attempt": n, "errors": errors},
                )
                logger.warning("attempt %d/%d failed: %s", n, self.max_retries, errors)
                logger.debug("attempt %d raw output: %s", n, raw)
                attempts.append(Attempt(n=n, error=errors, raw_output=raw))

                if n == self.max_retries:
                    raise RuntimeError(
                        f"gave up after {self.max_retries} attempts; last errors: {errors}"
                    ) from exc

                # Sub-point 3: re-prompt with the exact error appended so it self-corrects.
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "That JSON failed schema validation with these errors:\n"
                            f"{errors}\n"
                            "Fix ONLY what the errors mention and return the corrected "
                            "JSON object. Remember: invoice_id must match ^INV-\\d{6}$ "
                            "-- keep EVERY original digit of the invoice number and pad "
                            "with leading zeros on the left to reach 6 digits (e.g. 4521 "
                            "-> INV-004521). tax_rate is a fraction (0.08 not 8). "
                            "subtotal must equal sum(quantity*unit_price). "
                            "total must equal subtotal*(1+tax_rate) (compute it exactly)."
                        ),
                    }
                )
                continue

            logger.info("valid invoice on attempt %d", n)
            return AgentResult(
                invoice=invoice,
                tax=tax,
                attempts=attempts,
                model=self.model,
                elapsed_s=round(time.time() - started, 2),
            )

        # Unreachable (loop either returns or raises), but keeps type-checkers happy.
        raise RuntimeError("extract_invoice exited loop unexpectedly")

    def run(self, source_text: str, region: str) -> AgentResult:
        tax = self.resolve_tax(region)
        return self.extract_invoice(source_text, tax)
