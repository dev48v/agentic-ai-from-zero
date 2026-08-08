"""The deterministic scoring core of the Self-Reflective Agent — the RUBRIC, the
aggregation math, the ground-truth HARD CHECKS, and the pass/fail GATE. NO model
calls live here on purpose.

A self-reflective agent is only trustworthy if the thing that decides "is this good
enough yet?" is legible and reproducible. So the MODEL (in `agent.py`) does the soft,
judgement-heavy work — draft an answer, score it against this rubric with reasoning,
refine it — and everything that turns those judgements into a VERDICT is plain Python
here:

  Criterion / Rubric  — an EXPLICIT rubric: named criteria, each with guidance and a
                        weight, scored 1-5. The judge scores against THIS, not a vague
                        "is it good?" — which is the first defence against a model
                        rubber-stamping its own work.
  aggregate           — normalize each 1-5 score to 0-1 ((s-1)/4), take the weighted
                        mean → one overall score in [0, 1]. Same per-criterion scores
                        in → same overall out, every time.
  HardCheck           — a DETERMINISTIC, non-LLM check on the answer text (does the
                        docstring actually name `Raises`? does the answer contain the
                        correct number `28`?). These anchor the soft LLM score against
                        an objective floor the model cannot flatter its way past.
  gate                — the pass decision: overall >= threshold AND every hard check
                        passes. This is the quality gate the loop stops on — quality,
                        not vibes.

The split is the lesson of the series: the MODEL judges and rewrites; the aggregation,
the hard checks, the gate, and (in `agent.py`) the loop control + improvement metrics
are deterministic Python you can read and test. Same scores in → same verdict out.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


# --------------------------------------------------------------------------- #
# The rubric — criteria, weights, and the pass threshold.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Criterion:
    """One scored dimension. The judge gives it an integer 1-5 against `guidance`."""

    id: str
    name: str
    guidance: str            # exactly what a 5 vs a 1 looks like — the judge scores THIS
    weight: float = 1.0


@dataclass(frozen=True)
class HardCheck:
    """A deterministic, non-LLM check on the answer text.

    This is the anti-self-flattery anchor: no matter how generously the model grades its
    own output, a hard check either finds the required substance in the text or it does
    not. `check` is a pure predicate over the answer string.
    """

    id: str
    description: str
    check: Callable[[str], bool]


@dataclass(frozen=True)
class Rubric:
    name: str
    criteria: list[Criterion]
    hard_checks: list[HardCheck] = field(default_factory=list)
    threshold: float = 0.80          # overall (0-1) must reach this to pass the gate

    def criterion(self, cid: str) -> Criterion | None:
        for c in self.criteria:
            if c.id == cid:
                return c
        return None


@dataclass(frozen=True)
class Task:
    """A task = a prompt for the drafter + the rubric its answer is graded against."""

    id: str
    title: str
    prompt: str
    rubric: Rubric
    note: str = ""                   # what we EXPECT to happen (for the transcript)


# --------------------------------------------------------------------------- #
# What the judge returns per criterion (built from the model reply in agent.py).
# --------------------------------------------------------------------------- #
@dataclass
class CriterionScore:
    criterion_id: str
    score: int                       # 1-5, from the LLM judge
    evidence: str = ""               # a quote/observation justifying the score (anti-bias)
    critique: str = ""               # what is wrong / missing + how to fix it (actionable)


# --------------------------------------------------------------------------- #
# The pure scoring functions — deterministic, reproducible from the scores alone.
# --------------------------------------------------------------------------- #
def normalize_score(score_1_5: int) -> float:
    """Map an integer 1-5 onto [0, 1]:  1→0.0, 3→0.5, 5→1.0. Clamps out-of-range."""
    s = max(1, min(5, int(score_1_5)))
    return (s - 1) / 4.0


def aggregate(scores: list[CriterionScore], rubric: Rubric) -> float:
    """Weighted mean of the normalized per-criterion scores → one overall in [0, 1].

    Only criteria that actually belong to the rubric count, so a stray/hallucinated
    criterion id from the model can never move the score.
    """
    total_w = 0.0
    acc = 0.0
    for cs in scores:
        crit = rubric.criterion(cs.criterion_id)
        if crit is None:
            continue
        acc += normalize_score(cs.score) * crit.weight
        total_w += crit.weight
    return round(acc / total_w, 4) if total_w else 0.0


def run_hard_checks(answer: str, rubric: Rubric) -> list[tuple[str, bool]]:
    """Run every deterministic hard check against the answer text. Pure and non-LLM."""
    results: list[tuple[str, bool]] = []
    for hc in rubric.hard_checks:
        try:
            results.append((hc.id, bool(hc.check(answer or ""))))
        except Exception:
            results.append((hc.id, False))
    return results


def gate(overall: float, hard_results: list[tuple[str, bool]],
         threshold: float) -> tuple[bool, str]:
    """The quality gate: pass ONLY if the score clears the threshold AND every hard
    check passes. Returns (passed, human-readable reason).

    Requiring the hard checks too is the point — it stops an over-generous LLM judge
    from waving through an answer that is objectively missing something required.
    """
    failed = [cid for cid, ok in hard_results if not ok]
    if overall < threshold:
        return False, f"score {overall:.2f} < threshold {threshold:.2f}"
    if failed:
        return False, f"score OK ({overall:.2f}) but hard checks failed: {', '.join(failed)}"
    return True, f"score {overall:.2f} >= {threshold:.2f} and all hard checks passed"


# --------------------------------------------------------------------------- #
# Concrete tasks — chosen so a fast first draft is genuinely mediocre and there is
# real room for reflection to improve it. Each ships an explicit rubric + hard checks.
# --------------------------------------------------------------------------- #

# ---- Task A: a precise function docstring with edge cases --------------------
_FUNCTION_UNDER_TEST = '''\
def split_payment(total_cents, num_people):
    base, remainder = divmod(total_cents, num_people)
    shares = [base] * num_people
    for i in range(remainder):
        shares[i] += 1
    return shares
'''

_re_args = re.compile(r"total_cents", re.I), re.compile(r"num_people", re.I)
_re_returns = re.compile(r"\breturn(s|ed)?\b", re.I)
_re_raises = re.compile(r"\b(raise[sd]?|zerodivision|valueerror|exception|error)\b", re.I)
_re_edge = re.compile(r"\b(zero|negative|remainder|leftover|empty|0\b|<=\s*0|even(ly)?)\b", re.I)


def _has_both_args(text: str) -> bool:
    return all(rx.search(text) for rx in _re_args)


DOCSTRING_TASK = Task(
    id="docstring",
    title="Write a precise docstring (with edge cases) for split_payment()",
    prompt=(
        "Write a Python docstring for this function:\n\n"
        f"{_FUNCTION_UNDER_TEST}\n"
        "Return ONLY the docstring text (the content that goes between the triple quotes)."
    ),
    rubric=Rubric(
        name="docstring-quality",
        threshold=0.80,
        criteria=[
            Criterion("summary", "Accurate summary",
                      "5 = a correct one-line summary that says it splits an integer "
                      "amount into per-person shares AND distributes the remainder; "
                      "1 = missing, vague, or wrong.", weight=1.0),
            Criterion("args_returns", "Args & Returns documented",
                      "5 = documents BOTH parameters (total_cents, num_people) with "
                      "meaning/type AND the return value (a list of per-person cents); "
                      "1 = neither documented.", weight=1.2),
            Criterion("edge_cases", "Edge cases covered",
                      "5 = explicitly covers the tricky cases: the remainder is spread "
                      "over the first few people (not lost), and behaviour when "
                      "num_people is 0 or negative; 1 = no edge cases at all.", weight=1.5),
            Criterion("raises", "Exceptions documented",
                      "5 = notes that num_people=0 raises ZeroDivisionError (and/or that "
                      "callers should guard non-positive counts); 1 = silent on errors.",
                      weight=1.0),
            Criterion("precision", "Precise & unambiguous",
                      "5 = precise, concrete, no hand-waving, matches the code exactly; "
                      "1 = generic boilerplate that could describe any function.",
                      weight=1.0),
        ],
        hard_checks=[
            HardCheck("names_both_params", "mentions both total_cents and num_people",
                      _has_both_args),
            HardCheck("documents_return", "mentions the return value",
                      lambda t: bool(_re_returns.search(t))),
            HardCheck("documents_raises", "mentions an exception / error condition",
                      lambda t: bool(_re_raises.search(t))),
            HardCheck("mentions_edge", "mentions a remainder / zero / negative edge case",
                      lambda t: bool(_re_edge.search(t))),
        ],
    ),
    note=("a fast first draft is usually a one-liner missing Args/Returns/Raises and the "
          "remainder + zero-people edge cases — reflection has real room to improve it"),
)


# ---- Task B: answer a tricky question completely -----------------------------
_re_28 = re.compile(r"\b28\b")
_re_072 = re.compile(r"\b(0\.72|72\s*%|72\b)")
_re_not_additive = re.compile(
    r"(not\s+30|isn'?t\s+30|rather than 30|do(es)? ?n'?t\s+add|"
    r"can(no|')t\s+(just\s+)?add|multiplicativ|multiply|compound)", re.I)


def _addresses_trap(text: str) -> bool:
    return bool(_re_not_additive.search(text)) or ("30" in text and bool(_re_28.search(text)))


QUESTION_TASK = Task(
    id="discount",
    title="Answer a tricky question completely (successive discounts)",
    prompt=(
        "A store takes 20% off an item, then takes a further 10% off the already-"
        "discounted price. What is the single equivalent percentage discount off the "
        "ORIGINAL price? Give the final answer and explain."
    ),
    rubric=Rubric(
        name="complete-answer-quality",
        threshold=0.80,
        criteria=[
            Criterion("correctness", "Correct final answer",
                      "5 = states the equivalent discount is 28% off (price is 72% of "
                      "original); 1 = says 30% (the additive trap) or another wrong "
                      "number.", weight=1.6),
            Criterion("shows_work", "Shows the calculation",
                      "5 = shows 0.80 x 0.90 = 0.72 → 28% off, step by step; "
                      "1 = just asserts a number with no working.", weight=1.2),
            Criterion("addresses_trap", "Addresses the 30% trap",
                      "5 = explicitly explains why it is NOT 30% — percentage discounts "
                      "compound, they don't add; 1 = ignores the intuitive trap.",
                      weight=1.3),
            Criterion("completeness", "Complete",
                      "5 = also notes the order of the two discounts doesn't change the "
                      "result (multiplication commutes); 1 = bare minimum.", weight=0.8),
            Criterion("clarity", "Clear & well organized",
                      "5 = clear, correct, easy to follow; 1 = confusing or rambling.",
                      weight=1.0),
        ],
        hard_checks=[
            HardCheck("has_28", "states the correct answer 28(%)",
                      lambda t: bool(_re_28.search(t))),
            HardCheck("shows_072", "shows the 0.72 / 72% intermediate",
                      lambda t: bool(_re_072.search(t))),
            HardCheck("addresses_trap", "explains why it is not the additive 30%",
                      _addresses_trap),
        ],
    ),
    note=("a fast first answer often says '30%' (the additive trap) or gives 28% with no "
          "explanation — reflection must fix the number and/or complete the reasoning"),
)


TASKS: list[Task] = [DOCSTRING_TASK, QUESTION_TASK]
TASK_BY_ID = {t.id: t for t in TASKS}
