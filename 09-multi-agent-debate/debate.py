"""The deterministic core of the Multi-Agent Debate System — the round orchestration, the
ballot tally, the consensus ratio, the mind-change detection, and the confidence derivation.
NO model calls live here on purpose.

A debate is only worth trusting if the thing that decides "how sure are we?" is legible and
reproducible. So the MODEL (in `agents.py`) does the arguing — propose, critique, rebut,
synthesize — and everything that turns those arguments into a VERDICT is plain Python here:

  normalize_stance  — squash a free-text stance label to a canonical key so two agents that
                      mean the same thing ("$0.05", "0.05 dollars", "5 cents") tally together.
  tally             — count the canonical stances into positions; the winner is the position
                      with the most ballots. Each agent's FINAL stance IS its ballot.
  consensus_ratio   — winner_ballots / total_ballots: how much the panel actually agrees.
  derive_confidence — maps that ratio to HIGH / MEDIUM / LOW confidence. Unanimous → HIGH;
                      a split with no majority → LOW, flagged. The judge does NOT get to set
                      this — it falls out of the agreement math.
  detect_mind_change— compares each agent's stance BEFORE vs AFTER the rebuttal (by canonical
                      key), so "the critique changed at least one mind" is measured, not
                      claimed.

Same stances in → same tally, same consensus, same confidence out, every run — regardless of
how the 8B model happens to phrase its prose.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from agents import (
    Critique, Judge, Proposal, Proposer, Synthesis, build_roles,
)

# small map so common number-words tally with their digits
_NUM_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_stance(stance: str) -> str:
    """Canonicalize a free-text stance into a comparable key. Deterministic and pure.

    Lowercases, strips currency/punctuation, collapses whitespace, normalizes a leading
    number (so "$0.05", "0.05", "0.050" all key to "0.05"), and maps number-words to digits.
    Two agents that mean the same position land on the same key and tally together; genuinely
    different positions stay distinct. It is intentionally conservative — when unsure it keeps
    the cleaned text rather than guessing two things are equal.
    """
    s = (stance or "").strip().lower()
    s = s.replace("$", " ").replace("%", " ")
    s = re.sub(r"[\"'`.,;:!?()]+$", "", s)              # trailing punctuation
    s = re.sub(r"^[\"'`(]+", "", s)                      # leading quotes/brackets
    s = re.sub(r"\s+", " ", s).strip()
    if s in _NUM_WORDS:
        return _NUM_WORDS[s]
    # if the whole label is a number, normalize its numeric value (0.05 == 0.050 == .05)
    m = re.fullmatch(r"-?\d*\.?\d+", s)
    if m:
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else ("%g" % f)
        except ValueError:
            pass
    return s


@dataclass
class Tally:
    ballots: dict[str, str]            # persona_id -> canonical stance key (its ballot)
    counts: dict[str, int]             # canonical stance key -> number of ballots
    winner: str                        # the plurality/majority position key
    winner_votes: int
    total: int
    tied: bool                         # is the top position tied with another?

    @property
    def consensus_ratio(self) -> float:
        return (self.winner_votes / self.total) if self.total else 0.0

    def summary(self) -> str:
        parts = ", ".join(f'"{k}"×{v}' for k, v in
                          sorted(self.counts.items(), key=lambda kv: (-kv[1], kv[0])))
        tie = " (TIE)" if self.tied else ""
        return f"{parts}  → winner \"{self.winner}\" {self.winner_votes}/{self.total}{tie}"


def tally(final_proposals: list[Proposal]) -> Tally:
    """Count each agent's FINAL stance as a ballot for a position (deterministic)."""
    ballots = {p.persona_id: normalize_stance(p.stance) for p in final_proposals}
    counts = dict(Counter(ballots.values()))
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, winner_votes = ordered[0]
    tied = len(ordered) > 1 and ordered[1][1] == winner_votes
    return Tally(ballots=ballots, counts=counts, winner=winner,
                 winner_votes=winner_votes, total=len(final_proposals), tied=tied)


@dataclass
class Confidence:
    label: str                         # "high" | "medium" | "low"
    ratio: float                       # consensus ratio it was derived from
    reason: str

    @property
    def pct(self) -> int:
        return round(self.ratio * 100)


def derive_confidence(t: Tally) -> Confidence:
    """Confidence falls OUT of agreement — it is not something the judge asserts.

    unanimous (all agree, no tie)      → HIGH
    strict majority (> half, no tie)   → MEDIUM
    plurality / a tie / no majority    → LOW (flagged — the panel is genuinely split)
    """
    r = t.consensus_ratio
    if t.total and t.winner_votes == t.total and not t.tied:
        return Confidence("high", r, "unanimous — every agent landed on the same position")
    if t.winner_votes > t.total / 2 and not t.tied:
        return Confidence("medium", r,
                          f"a majority ({t.winner_votes}/{t.total}) agreed but not everyone")
    return Confidence("low", r,
                      f"split — no majority (top position only {t.winner_votes}/{t.total}"
                      + (", and tied" if t.tied else "") + "); flagged as low-confidence")


def detect_mind_changes(before: list[Proposal], after: list[Proposal]) -> list[str]:
    """Which agents' canonical stance actually MOVED between round 1 and the rebuttal.
    Measured from the stances themselves, not the model's self-report."""
    pre = {p.persona_id: normalize_stance(p.stance) for p in before}
    moved: list[str] = []
    for p in after:
        if pre.get(p.persona_id) is not None and pre[p.persona_id] != normalize_stance(p.stance):
            moved.append(p.persona_id)
    return moved


# --------------------------------------------------------------------------- #
# The full result of one debate — every intermediate artifact, for the transcript.
# --------------------------------------------------------------------------- #
@dataclass
class DebateResult:
    question: str
    proposals: list[Proposal]              # round 1 (cold)
    critique: Critique
    rebuttals: list[Proposal]              # round 2 (post-critique)
    tally: Tally
    confidence: Confidence
    synthesis: Synthesis
    mind_changed: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The orchestrator — runs the rounds in order. The ONLY non-determinism is the
# model's prose inside each role; the tally + consensus + confidence are pure.
# --------------------------------------------------------------------------- #
class Debate:
    def __init__(self, model: str | None = None, log=None) -> None:
        self._log = log or (lambda *a, **k: None)
        proposers, critic, judge = build_roles(
            model=model or _default_model(), log=self._log)
        self.proposers: list[Proposer] = proposers
        self.critic = critic
        self.judge = judge

    def run(self, question: str) -> DebateResult:
        log = self._log

        # ---- Round 1: PROPOSE — each persona answers cold, independently. ----
        log("phase", title="1 · PROPOSE — each agent answers the same question independently")
        proposals: list[Proposal] = []
        for pr in self.proposers:
            p = pr.propose(question)
            proposals.append(p)
            log("proposal", proposal=p)

        # ---- CRITIC — pressure-tests every proposal at once. ----
        log("phase", title="2 · CRITIQUE — one critic reviews every proposal for flaws")
        critique = self.critic.evaluate(question, proposals)
        log("critique", critique=critique)

        # ---- Round 2: REBUT — each persona revises or defends after the critique. ----
        log("phase", title="3 · REBUTTAL — each agent revises or defends after the critique")
        by_id = {p.persona_id: p for p in proposals}
        rebuttals: list[Proposal] = []
        for pr in self.proposers:
            own = by_id[pr.persona.id]
            others = [p for p in proposals if p.persona_id != pr.persona.id]
            r = pr.rebut(question, own, others, critique)
            rebuttals.append(r)
            log("rebuttal", before=own, after=r)

        # ---- VOTE / CONSENSUS — deterministic tally of the FINAL stances. ----
        log("phase", title="4 · VOTE & CONSENSUS — tally final stances (deterministic Python)")
        t = tally(rebuttals)
        mind_changed = detect_mind_changes(proposals, rebuttals)
        confidence = derive_confidence(t)
        log("tally", tally=t, confidence=confidence, mind_changed=mind_changed)

        # ---- JUDGE — synthesize the final answer; confidence is attached from the math. ----
        log("phase", title="5 · SYNTHESIS — the judge combines the best points (confidence from the math)")
        synthesis = self.judge.synthesize(question, rebuttals, critique, t.summary())
        log("synthesis", synthesis=synthesis, confidence=confidence)

        return DebateResult(
            question=question, proposals=proposals, critique=critique,
            rebuttals=rebuttals, tally=t, confidence=confidence,
            synthesis=synthesis, mind_changed=mind_changed)


def _default_model() -> str:
    from common.client import DEFAULT_MODEL
    return DEFAULT_MODEL
