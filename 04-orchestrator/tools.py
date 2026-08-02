"""Six pure-Python tools with VARIED capabilities + permissions.

Importing this module runs every `@tool(...)` decorator, which self-registers the
tool into `registry.REGISTRY`. agent.py (the router) imports this module ONLY to
trigger that registration — it never references a tool by name. That is the whole
point of the dynamic registry: adding a seventh tool here would make it routable
without changing a line of the router.

The line-up (name — capabilities — permission):

  price_alpha   — {price_quote} — network   ┐ three independent price feeds. They
  price_beta    — {price_quote} — network   │ DISAGREE on AAPL on purpose, so the
  price_gamma   — {price_quote} — network   ┘ orchestrator must resolve a conflict.
  weather_lookup— {weather}     — read      a read-only lookup (no conflict).
  fx_convert    — {currency,math}— read      a read-only currency conversion.
  ledger_write  — {persist}     — WRITE      the one write tool; a restricted run
                                             (no 'write' scope) is DENIED it.

Each price feed also carries a `trust_priority` (1 = most trusted) and an `as_of`
freshness timestamp so the conflict resolver has a fully-specified, deterministic
policy to apply (majority -> trust-priority -> freshness).
"""

from __future__ import annotations

import time

from registry import ToolResult, tool

# --------------------------------------------------------------------------- #
# Simulated network latency. Each price feed "call" costs ~this long, which is
# what makes running the three feeds CONCURRENTLY visibly faster than serially.
# --------------------------------------------------------------------------- #
FEED_LATENCY_S = 0.6

# Canned quotes per feed. AAPL is where alpha & gamma AGREE (150.25) and beta
# DISAGREES (172.40) — the conflict the resolver must settle. MSFT is unanimous.
_ALPHA_QUOTES = {"AAPL": 150.25, "MSFT": 410.00}
_BETA_QUOTES = {"AAPL": 172.40, "MSFT": 410.00}   # beta is the outlier on AAPL
_GAMMA_QUOTES = {"AAPL": 150.25, "MSFT": 410.00}


def _quote(feed: dict[str, float], symbol: str, source: str,
           trust_priority: int, as_of: str) -> ToolResult:
    time.sleep(FEED_LATENCY_S)  # simulate a network round-trip
    symbol = (symbol or "").strip().upper()
    if symbol not in feed:
        return ToolResult(ok=False, error=f"{source}: no quote for symbol '{symbol}'")
    price = feed[symbol]
    return ToolResult(
        ok=True,
        output=f"{source}: {symbol} = ${price:.2f} (trust={trust_priority}, as_of={as_of})",
        data={
            "source": source,
            "symbol": symbol,
            "price": price,
            "trust_priority": trust_priority,  # 1 = most trusted
            "as_of": as_of,                    # ISO instant; later = fresher
        },
    )


@tool(
    name="price_alpha",
    description="Price feed ALPHA. Returns the latest quote for a stock symbol.",
    capabilities={"price_quote"},
    permission="network",
    args_schema={"properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
)
def price_alpha(args: dict) -> ToolResult:
    return _quote(_ALPHA_QUOTES, args.get("symbol", ""), "alpha",
                  trust_priority=2, as_of="2026-08-03T09:30:00Z")


@tool(
    name="price_beta",
    description="Price feed BETA. Returns the latest quote for a stock symbol.",
    capabilities={"price_quote"},
    permission="network",
    args_schema={"properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
)
def price_beta(args: dict) -> ToolResult:
    # Most trusted source (trust=1) but STALEST timestamp — deliberately so the
    # majority-first policy can outvote it, which the recorded run highlights.
    return _quote(_BETA_QUOTES, args.get("symbol", ""), "beta",
                  trust_priority=1, as_of="2026-08-03T09:12:00Z")


@tool(
    name="price_gamma",
    description="Price feed GAMMA. Returns the latest quote for a stock symbol.",
    capabilities={"price_quote"},
    permission="network",
    args_schema={"properties": {"symbol": {"type": "string"}}, "required": ["symbol"]},
)
def price_gamma(args: dict) -> ToolResult:
    return _quote(_GAMMA_QUOTES, args.get("symbol", ""), "gamma",
                  trust_priority=3, as_of="2026-08-03T09:28:00Z")


# --------------------------------------------------------------------------- #
# weather_lookup — a read-only local lookup (capability {weather}, perm read).
# --------------------------------------------------------------------------- #
_WEATHER = {
    "london": "16C, overcast",
    "reykjavik": "9C, windy",
    "mumbai": "31C, humid",
}


@tool(
    name="weather_lookup",
    description="Look up today's weather for a city from a local table. Read-only.",
    capabilities={"weather"},
    permission="read",
    args_schema={"properties": {"city": {"type": "string"}}, "required": ["city"]},
)
def weather_lookup(args: dict) -> ToolResult:
    city = (args.get("city", "") or "").strip().lower()
    if city in _WEATHER:
        return ToolResult(ok=True, output=f"{city.title()}: {_WEATHER[city]}",
                          data={"city": city, "weather": _WEATHER[city]})
    return ToolResult(ok=False, error=f"no weather on file for '{city}'")


# --------------------------------------------------------------------------- #
# fx_convert — a read-only currency conversion (capability {currency, math}).
# --------------------------------------------------------------------------- #
_USD_RATES = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.4}


@tool(
    name="fx_convert",
    description=(
        "Convert an amount between currencies (USD/EUR/GBP/INR) at fixed table "
        "rates. Read-only. Args: amount (number), from (code), to (code)."
    ),
    capabilities={"currency", "math"},
    permission="read",
    args_schema={
        "properties": {
            "amount": {"type": "number"},
            "from": {"type": "string"},
            "to": {"type": "string"},
        },
        "required": ["amount", "from", "to"],
    },
)
def fx_convert(args: dict) -> ToolResult:
    src = (args.get("from", "") or "").upper()
    dst = (args.get("to", "") or "").upper()
    if src not in _USD_RATES or dst not in _USD_RATES:
        return ToolResult(ok=False, error=f"unsupported currency pair {src}->{dst}")
    amount = float(args.get("amount", 0))
    usd = amount / _USD_RATES[src]
    out = usd * _USD_RATES[dst]
    return ToolResult(
        ok=True,
        output=f"{amount:.2f} {src} = {out:.2f} {dst}",
        data={"amount": amount, "from": src, "to": dst, "result": round(out, 2)},
    )


# --------------------------------------------------------------------------- #
# ledger_write — the WRITE tool. Appends to an in-memory audit ledger. A run that
# was not granted the 'write' scope is refused this tool (deny-by-default).
# --------------------------------------------------------------------------- #
LEDGER: list[str] = []


@tool(
    name="ledger_write",
    description="Append a one-line note to the audit ledger. Requires WRITE scope.",
    capabilities={"persist"},
    permission="write",
    args_schema={"properties": {"note": {"type": "string"}}, "required": ["note"]},
)
def ledger_write(args: dict) -> ToolResult:
    note = (args.get("note", "") or "").strip()
    if not note:
        return ToolResult(ok=False, error="empty note")
    entry = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} | {note}"
    LEDGER.append(entry)
    return ToolResult(ok=True, output=f"wrote ledger entry #{len(LEDGER)}: {entry}",
                      data={"entry": entry, "ledger_size": len(LEDGER)})
