"""The Self-Reflective Agent — the MODEL-BACKED ROLES plus the deterministic
GENERATE → REFLECT → REFINE loop.

The model does three jobs, and only three:

  Drafter  — produces a FAST first-pass answer to the task. The prompt deliberately asks
             for a quick draft, so attempt #1 is usually mediocre (a one-line docstring
             with no edge cases; a snap "30%" to the discount question). That is the point:
             self-reflection only means something if there is something to improve.
  Judge    — LLM-AS-JUDGE on the agent's OWN output. It scores the answer against the
             EXPLICIT rubric (each criterion 1-5) and must give EVIDENCE (a quote/observation)
             plus SPECIFIC, ACTIONABLE critique (what's wrong, what's missing, how to fix)
             for each — not just a number. Scoring against a fixed rubric + demanding
             evidence is the first mitigation of self-judge bias.
  Refiner  — REGENERATES the answer UNDER CONSTRAINTS: keep what already scored well, and
             fix exactly the gaps the judge named (plus any failed deterministic hard check).

Everything that turns those judgements into a verdict is deterministic Python: the
aggregation + hard checks + gate live in `rubric.py`; the LOOP CONTROL (bounded iterations),
the THRESHOLD CHECK, the "what changed" diff, the improvement DELTA, and the STOP REASON
live in `ReflectionLoop` below. Same per-criterion scores in → same gate, same metrics out,
no matter how the 8B model phrases its prose.

Self-judge bias — the honest caveat: the judge is the same model family grading its own
work, so it can share the drafter's blind spots and drift optimistic. We reduce (not
eliminate) that with: (1) a fixed rubric with per-criterion definitions, (2) required
evidence per score, (3) a fresh, stateless judge call framed as an independent grader, and
(4) deterministic hard checks that co-gate the pass decision so the model cannot flatter a
missing-something answer past the bar.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from common.client import DEFAULT_MODEL, get_client
from rubric import (
    CriterionScore, Rubric, Task, aggregate, gate, run_hard_checks,
)


# --------------------------------------------------------------------------- #
# What each stage produces (plain dataclasses — no model magic here).
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    """A full evaluation of one answer: the model's per-criterion scores + critique,
    plus the DETERMINISTIC overall, hard-check results, and gate verdict."""

    scores: list[CriterionScore]
    overall: float                          # deterministic weighted mean (rubric.aggregate)
    hard_results: list[tuple[str, bool]]    # deterministic hard checks (rubric.run_hard_checks)
    passed: bool                            # deterministic gate
    gate_reason: str
    judge_comment: str = ""

    def score_of(self, cid: str) -> int:
        for s in self.scores:
            if s.criterion_id == cid:
                return s.score
        return 0

    @property
    def hard_passed(self) -> int:
        return sum(1 for _, ok in self.hard_results if ok)

    @property
    def hard_total(self) -> int:
        return len(self.hard_results)


@dataclass
class Iteration:
    """One turn of the loop: the answer, its evaluation, and what changed vs the last one."""

    n: int                                  # 0 = first draft, 1.. = refinements
    kind: str                               # "draft" | "refine"
    answer: str
    evaluation: Evaluation
    change_note: str = ""                   # the refiner's self-report of what it changed
    improved: list[str] = field(default_factory=list)   # criteria whose score rose (measured)
    regressed: list[str] = field(default_factory=list)  # criteria whose score fell (measured)


@dataclass
class ReflectionResult:
    task_id: str
    iterations: list[Iteration]
    stop_reason: str                        # "passed" | "max-iters"
    model: str = DEFAULT_MODEL
    elapsed_s: float = 0.0

    @property
    def first(self) -> Iteration:
        return self.iterations[0]

    @property
    def final(self) -> Iteration:
        return self.iterations[-1]

    @property
    def delta(self) -> float:
        return round(self.final.evaluation.overall - self.first.evaluation.overall, 4)

    @property
    def score_trail(self) -> list[float]:
        return [round(it.evaluation.overall, 2) for it in self.iterations]

    @property
    def hard_trail(self) -> list[str]:
        """Per-iteration hard-checks-passed, e.g. ["2/4", "4/4"]. Captures a real
        improvement even when the (sometimes over-generous) LLM score is flat/saturated."""
        return [f"{it.evaluation.hard_passed}/{it.evaluation.hard_total}"
                for it in self.iterations]


# --------------------------------------------------------------------------- #
# The shared model call — strict-JSON, tolerant parse + one retry (same approach as
# the rest of the series). NEVER crashes the loop on a malformed reply.
# --------------------------------------------------------------------------- #
class _ModelBackedRole:
    def __init__(self, model: str, client, log=None) -> None:
        self.model = model
        self.client = client
        self._log = log

    def _chat(self, system: str, user: str, temperature: float) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature)
        return (resp.choices[0].message.content or "").strip()

    def _chat_json(self, system: str, user: str, temperature: float) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        raw = ""
        for _ in (1, 2):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature)
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _parse_json(raw)
            if parsed is not None:
                return parsed
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "That was not valid JSON. Reply ONLY the JSON object."})
        return {"_unparsed": raw}


# --------------------------------------------------------------------------- #
# Drafter — a fast first pass. Deliberately terse so there is room to reflect.
# --------------------------------------------------------------------------- #
class Drafter(_ModelBackedRole):
    def draft(self, task: Task) -> str:
        system = ("You are answering quickly. Give a FAST FIRST DRAFT — a short, quick "
                  "answer, do not over-think it or polish it. Reply with ONLY the answer "
                  "itself, no preamble.")
        return self._chat(system, task.prompt, temperature=0.5)


# --------------------------------------------------------------------------- #
# Judge — LLM-as-judge on the agent's OWN output, against the explicit rubric.
# --------------------------------------------------------------------------- #
class Judge(_ModelBackedRole):
    def evaluate(self, task: Task, answer: str) -> Evaluation:
        rubric = task.rubric
        crit_lines = "\n".join(
            f'  - id="{c.id}"  ({c.name}): {c.guidance}' for c in rubric.criteria)
        ids = ", ".join(f'"{c.id}"' for c in rubric.criteria)
        system = (
            "You are a STRICT, INDEPENDENT grader. You did not write the answer; judge it on "
            "its merits, not charitably. Score it against the rubric below. For EACH criterion "
            "give an integer score 1-5, a short piece of EVIDENCE (a quote or concrete "
            "observation from the answer that justifies the score), and specific ACTIONABLE "
            "critique — what is wrong or missing and exactly how to fix it. Do not inflate: "
            "reserve 5 for an answer that fully meets the criterion, and be willing to give "
            "low scores. Return STRICT JSON only, no markdown:\n"
            '{"scores": [{"criterion": <one of ' + ids + '>, '
            '"score": <integer 1-5>, '
            '"evidence": <quote/observation>, '
            '"critique": <what is wrong/missing + how to fix, one or two sentences>}], '
            '"comment": <one sentence overall>}\n'
            "Score EVERY criterion exactly once.")
        user = (f"TASK:\n{task.prompt}\n\n"
                f"RUBRIC (score each 1-5):\n{crit_lines}\n\n"
                f"ANSWER TO GRADE:\n{answer}")
        d = self._chat_json(system, user, temperature=0.2)

        scores = _to_scores(d, rubric)
        overall = aggregate(scores, rubric)
        hard = run_hard_checks(answer, rubric)
        passed, reason = gate(overall, hard, rubric.threshold)
        return Evaluation(
            scores=scores, overall=overall, hard_results=hard,
            passed=passed, gate_reason=reason,
            judge_comment=str(d.get("comment", "")).strip())


# --------------------------------------------------------------------------- #
# Refiner — regenerate UNDER CONSTRAINTS: keep the good, fix exactly what was named.
# --------------------------------------------------------------------------- #
class Refiner(_ModelBackedRole):
    def refine(self, task: Task, answer: str, ev: Evaluation) -> tuple[str, str]:
        rubric = task.rubric
        # Feed back the LOW-scoring criteria with their critique, plus failed hard checks —
        # this is the "constraint" the regeneration must satisfy.
        weak = sorted(ev.scores, key=lambda s: s.score)
        fix_lines = []
        for s in weak:
            crit = rubric.criterion(s.criterion_id)
            if crit is None:
                continue
            fix_lines.append(
                f'  - {crit.name} (scored {s.score}/5): {s.critique or "improve this"}')
        failed = [cid for cid, ok in ev.hard_results if not ok]
        hard_line = ("\nMANDATORY — these objective checks currently FAIL and MUST pass: "
                     + ", ".join(_HARD_HINTS.get(cid, cid) for cid in failed)) if failed else ""
        system = (
            "You are revising your own earlier answer after an evaluation. Produce an IMPROVED "
            "answer that KEEPS everything that already scored well and FIXES exactly the gaps "
            "listed. Do not pad, do not drop correct content, do not start over from scratch. "
            "Return STRICT JSON only, no markdown:\n"
            '{"answer": <the full improved answer>, '
            '"changes": <one sentence: what you changed and why>}')
        user = (f"TASK:\n{task.prompt}\n\n"
                f"YOUR CURRENT ANSWER:\n{answer}\n\n"
                f"WHAT TO FIX (lowest-scoring first):\n" + "\n".join(fix_lines) + hard_line +
                "\n\nReturn the improved answer.")
        d = self._chat_json(system, user, temperature=0.3)
        new_answer = str(d.get("answer", "")).strip()
        changes = str(d.get("changes", "")).strip()
        if not new_answer:                       # malformed reply → keep the old answer, don't crash
            new_answer, changes = answer, "(refiner returned no answer; kept previous)"
        return new_answer, changes


# short human hints for the hard-check ids, used when telling the refiner what MUST pass
_HARD_HINTS = {
    "names_both_params": "document BOTH parameters total_cents and num_people",
    "documents_return": "document the return value",
    "documents_raises": "document the error/exception (num_people=0 raises ZeroDivisionError)",
    "mentions_edge": "cover the remainder-distribution and zero/negative-people edge cases",
    "has_28": "state the correct answer, 28%",
    "shows_072": "show the 0.80 x 0.90 = 0.72 step",
    "addresses_trap": "explain why it is NOT 30% (discounts compound, they don't add)",
}


# --------------------------------------------------------------------------- #
# The deterministic loop — generate → (reflect → refine)*. NO model logic here:
# the model calls happen inside the roles; the loop only decides when to stop and
# records the improvement metrics.
# --------------------------------------------------------------------------- #
class ReflectionLoop:
    def __init__(self, model: str | None = None, max_iters: int = 3, log=None) -> None:
        self.model = model or DEFAULT_MODEL
        self.max_iters = max_iters           # max REFINEMENTS after the first draft
        self._log = log or (lambda *a, **k: None)
        client = get_client()
        self.drafter = Drafter(self.model, client, self._log)
        self.judge = Judge(self.model, client, self._log)
        self.refiner = Refiner(self.model, client, self._log)

    def solve(self, task: Task) -> ReflectionResult:
        log = self._log
        started = time.time()
        iterations: list[Iteration] = []

        # ---- GENERATE — the fast first draft. ----
        log("phase", title="1 · GENERATE — a fast first-pass draft")
        answer = self.drafter.draft(task)
        ev = self.judge.evaluate(task, answer)
        it = Iteration(n=0, kind="draft", answer=answer, evaluation=ev)
        iterations.append(it)
        log("iteration", it=it, task=task)

        # ---- REFLECT → REFINE, bounded, until the gate passes or we hit the cap. ----
        # The gate (deterministic, in rubric.py) is the only thing that stops the loop
        # early; otherwise it runs to max_iters. Same scores in → same stop decision out.
        for n in range(1, self.max_iters + 1):
            if ev.passed:
                break
            log("phase", title=f"{n + 1} · REFLECT → REFINE (iteration {n}) — fix what the judge named")
            new_answer, change_note = self.refiner.refine(task, answer, ev)
            new_ev = self.judge.evaluate(task, new_answer)
            improved, regressed = _score_diff(ev, new_ev, task.rubric)
            it = Iteration(n=n, kind="refine", answer=new_answer, evaluation=new_ev,
                           change_note=change_note, improved=improved, regressed=regressed)
            iterations.append(it)
            log("iteration", it=it, task=task, prev=ev)
            answer, ev = new_answer, new_ev
        stop_reason = "passed" if ev.passed else "max-iters"

        result = ReflectionResult(
            task_id=task.id, iterations=iterations, stop_reason=stop_reason,
            model=self.model, elapsed_s=round(time.time() - started, 2))
        log("done", result=result, task=task)
        return result


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _score_diff(before: Evaluation, after: Evaluation, rubric: Rubric):
    """Which criteria actually MOVED between two evaluations — measured from the scores,
    not the model's self-report. Deterministic."""
    improved, regressed = [], []
    for c in rubric.criteria:
        b, a = before.score_of(c.id), after.score_of(c.id)
        if a > b:
            improved.append(c.id)
        elif a < b:
            regressed.append(c.id)
    return improved, regressed


def _to_scores(d: dict, rubric: Rubric) -> list[CriterionScore]:
    """Turn the judge's JSON into validated CriterionScores. A missing criterion defaults
    to the worst score (1), so the model cannot lift the overall by simply omitting a hard
    criterion — an omission counts against it."""
    by_id: dict[str, CriterionScore] = {}
    for row in (d.get("scores", []) or []):
        cid = str(row.get("criterion", "")).strip()
        if rubric.criterion(cid) is None:
            continue
        try:
            sc = int(round(float(row.get("score", 1))))
        except (TypeError, ValueError):
            sc = 1
        by_id[cid] = CriterionScore(
            criterion_id=cid, score=max(1, min(5, sc)),
            evidence=str(row.get("evidence", "")).strip(),
            critique=str(row.get("critique", "")).strip())
    scores: list[CriterionScore] = []
    for c in rubric.criteria:
        scores.append(by_id.get(c.id, CriterionScore(
            criterion_id=c.id, score=1, evidence="(not scored by judge)",
            critique="the judge did not score this criterion")))
    return scores


# --------------------------------------------------------------------------- #
# Tolerant JSON extraction (same approach the rest of the series uses).
# --------------------------------------------------------------------------- #
def _parse_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.split("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
