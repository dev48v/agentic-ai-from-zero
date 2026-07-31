"""Pydantic schemas — the single source of truth for every typed I/O in this agent.

The whole point of a structured-output agent: the schema is strict, and the model
must satisfy it. Constraints here (patterns, enums, ranges, cross-field math) are
what the parse->retry loop enforces. Nothing downstream ever touches free text.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Currency(str, Enum):
    """Closed set of currencies we accept. Anything else must be rejected."""

    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class TaxRateResult(BaseModel):
    """Schema a *tool* response must satisfy before the agent trusts it.

    A tool is just another untrusted input. We validate its output exactly the
    way we validate the model's output.
    """

    region: str = Field(min_length=2, max_length=10)
    # rate is a FRACTION (0.08 == 8%), never a percentage. This bound catches a
    # tool that mistakenly returns 8 instead of 0.08.
    rate: float = Field(ge=0.0, le=1.0)
    source: str = Field(min_length=1)


class LineItem(BaseModel):
    description: str = Field(min_length=1, max_length=200)
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)


class Invoice(BaseModel):
    """The typed object the agent must produce. Deliberately strict."""

    # Must be zero-padded to exactly 6 digits: "INV-004521", not "INV-4521".
    invoice_id: str = Field(pattern=r"^INV-\d{6}$")
    customer_email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    currency: Currency
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: float = Field(ge=0)
    # A fraction in [0, 1]. "8" or "8.0" (a percentage) is invalid on purpose.
    tax_rate: float = Field(ge=0.0, le=1.0)
    total: float = Field(ge=0)

    @model_validator(mode="after")
    def _check_math(self) -> "Invoice":
        """Cross-field arithmetic must hold, within a cent of rounding."""
        computed_subtotal = round(
            sum(li.quantity * li.unit_price for li in self.line_items), 2
        )
        if abs(computed_subtotal - self.subtotal) > 0.01:
            raise ValueError(
                f"subtotal {self.subtotal} must equal sum(quantity*unit_price) "
                f"= {computed_subtotal}"
            )
        computed_total = round(self.subtotal * (1 + self.tax_rate), 2)
        if abs(computed_total - self.total) > 0.01:
            raise ValueError(
                f"total {self.total} must equal subtotal*(1+tax_rate) "
                f"= {computed_total}"
            )
        return self
