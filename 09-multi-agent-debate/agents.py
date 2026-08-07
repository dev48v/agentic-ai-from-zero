"""The Multi-Agent Debate System — the MODEL-BACKED ROLES.

This is the only place a model call happens. Three kinds of agent share one small model
(`meta/llama-3.1-8b-instruct`) but are given DIFFERENT jobs, and — for the proposers —
different PERSONAS, so their answers genuinely diverge instead of echoing each other:

  Proposer — answers the question from ONE persona's framing (cautious / creative / literal).
             The whole point of a debate is perspective diversity, so each proposer gets a
             distinct system prompt that pushes it toward a different kind of answer. A
             proposer is asked twice: once to PROPOSE (cold, before seeing anyone else), and
             once to REBUT (after reading the critique + the other proposals), where it may
             revise — or defend — its stance.
  Critic   — reads ALL the proposals at once and points out each one's main flaw, hidden
             assumption, or unsupported leap, plus an overall read. It does not answer the
             question; its job is to pressure-test the proposals so the rebuttal round has
             something to push against.
  Judge    — reads the proposals, the critique, and the final (post-rebuttal) stances and
             SYNTHESIZES a single answer that combines the best points. Crucially the judge
             does NOT get to invent the confidence: the confidence is derived in
             `debate.py` from how much the agents actually AGREE (deterministic Python).

The split is the lesson of the series: the MODEL proposes, critiques, and synthesizes; the
round orchestration, the ballot tally, the consensus ratio, and the confidence are
deterministic Python in `debate.py`. Same stances in → same tally + confidence out, no
matter how the model phrases things.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from common.client import DEFAULT_MODEL, get_client


# --------------------------------------------------------------------------- #
# The personas — three deliberately different framings, so proposals diverge.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    emoji: str
    system: str


PERSONAS: list[Persona] = [
    Persona(
        id="cautious", name="Cautious", emoji="🛡️",
        system=(
            "You are the CAUTIOUS debater. You are risk-averse and conservative: you prefer "
            "proven, safe, low-downside answers, you distrust hype, and you call out what could "
            "go wrong. When a question has a correct answer you double-check the arithmetic and "
            "the edge cases before committing.")),
    Persona(
        id="creative", name="Creative", emoji="💡",
        system=(
            "You are the CREATIVE debater. You think laterally and are willing to back a bold, "
            "unconventional answer if the upside is high. You look for the option others dismiss "
            "too quickly. You still have to justify your answer — flair is not an excuse for being "
            "wrong on a question that has a definite answer.")),
    Persona(
        id="literal", name="Literal", emoji="📏",
        system=(
            "You are the LITERAL debater. You are precise and methodical: you answer EXACTLY what "
            "was asked, show the steps, and refuse to read anything into the question that is not "
            "there. On a math or logic question you work it out step by step and state the result "
            "plainly.")),
]

PERSONA_BY_ID = {p.id: p for p in PERSONAS}


# --------------------------------------------------------------------------- #
# What each role returns (plain dataclasses — no model magic here).
# --------------------------------------------------------------------------- #
@dataclass
class Proposal:
    persona_id: str
    persona_name: str
    answer: str                     # the answer, in prose
    stance: str                     # a SHORT canonical label — this is the ballot Python tallies
    reasoning: str                  # why, briefly
    self_confidence: str = ""       # the model's OWN confidence (color only — not the system's)
    changed_mind: bool = False      # rebuttal only: did the critique move this agent?
    change_note: str = ""           # rebuttal only: what the critique made it reconsider


@dataclass
class Assessment:
    persona_id: str
    verdict: str                    # "sound" | "flawed" | "unsupported"
    flaw: str                       # the main flaw / hidden assumption the critic found


@dataclass
class Critique:
    assessments: list[Assessment] = field(default_factory=list)
    overall: str = ""

    def for_persona(self, pid: str) -> Assessment | None:
        for a in self.assessments:
            if a.persona_id == pid:
                return a
        return None


@dataclass
class Synthesis:
    final_answer: str
    rationale: str
    key_points: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The shared model call — strict-JSON, tolerant parse + one retry (same approach
# as the rest of the series). NEVER crashes the debate on a malformed reply.
# --------------------------------------------------------------------------- #
class _ModelBackedRole:
    def __init__(self, model: str, client, log=None) -> None:
        self.model = model
        self.client = client
        self._log = log

    def _reason(self, system: str, user: str, temperature: float) -> dict:
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
# Proposer — answers from ONE persona; proposes cold, then rebuts after critique.
# --------------------------------------------------------------------------- #
class Proposer(_ModelBackedRole):
    def __init__(self, persona: Persona, model: str, client, log=None) -> None:
        super().__init__(model, client, log)
        self.persona = persona

    def propose(self, question: str) -> Proposal:
        """Round 1 — answer the question cold, from this persona's framing, having seen no
        one else's answer. Higher temperature so the personas genuinely diverge."""
        system = (
            self.persona.system + "\n\n"
            "Answer the user's question. Return STRICT JSON only, no markdown:\n"
            '{"answer": <your answer in 1-3 sentences>, '
            '"stance": <a SHORT canonical label for the POSITION you are taking — the ANSWER '
            'ITSELF, never a description of your method or persona. For a numeric question use '
            'just the number (e.g. "0.05"). For a choice, use the option you pick in one or two '
            'words (e.g. "bootstrap"). Lowercase, no units, no punctuation, no explanation. Two '
            'agents who agree MUST be able to produce the identical label.>, '
            '"reasoning": <one or two sentences of why>, '
            '"confidence": "low"|"medium"|"high"}')
        d = self._reason(system, f"Question: {question}", temperature=0.7)
        return _to_proposal(self.persona, d)

    def rebut(self, question: str, own: Proposal, others: list[Proposal],
              critique: Critique) -> Proposal:
        """Round 2 — after reading the critique of your answer and the OTHER proposals, either
        REVISE your stance or defend it. This is where a good critique can change a mind."""
        my_flaw = critique.for_persona(self.persona.id)
        others_txt = "\n".join(
            f'  - {o.persona_name}: answer={o.answer!r} stance={o.stance!r}' for o in others)
        system = (
            self.persona.system + "\n\n"
            "This is a debate. You already gave an answer; now you have read a critic's review "
            "of it and the other debaters' answers. Reconsider honestly: change your answer if "
            "the critique or another answer is genuinely more convincing, or defend it if not. "
            "Do NOT change just to agree — only if it is actually better. Return STRICT JSON only:\n"
            '{"answer": <your possibly-revised answer, 1-3 sentences>, '
            '"stance": <SHORT canonical label = the POSITION/ANSWER itself, NOT your method — same '
            'rules as before: just the number for a numeric answer, or the option you pick in one '
            'or two words; lowercase, no units, no punctuation. If two agents now agree, their '
            'labels MUST be identical.>, '
            '"changed_mind": true|false, '
            '"why": <one sentence: what, if anything, the critique made you reconsider>}')
        user = (
            f"Question: {question}\n\n"
            f"YOUR earlier answer: {own.answer!r} (stance: {own.stance!r})\n"
            f"The critic said about YOUR answer: "
            f"{(my_flaw.flaw if my_flaw else 'no specific flaw noted')} "
            f"[verdict: {my_flaw.verdict if my_flaw else 'n/a'}]\n"
            f"The critic's overall read: {critique.overall}\n\n"
            f"The OTHER debaters' answers:\n{others_txt}")
        d = self._reason(system, user, temperature=0.5)
        p = _to_proposal(self.persona, d)
        p.changed_mind = bool(d.get("changed_mind", False))
        p.change_note = str(d.get("why", "")).strip()
        return p


# --------------------------------------------------------------------------- #
# Critic — reviews ALL proposals; finds each one's main flaw. Does not answer.
# --------------------------------------------------------------------------- #
class Critic(_ModelBackedRole):
    def evaluate(self, question: str, proposals: list[Proposal]) -> Critique:
        listing = "\n".join(
            f'  [{p.persona_id}] {p.persona_name}: answer={p.answer!r} '
            f'stance={p.stance!r} reasoning={p.reasoning!r}' for p in proposals)
        ids = ", ".join(f'"{p.persona_id}"' for p in proposals)
        system = (
            "You are the CRITIC in a multi-agent debate. You do NOT answer the question. You "
            "pressure-test every proposal: name its single most important flaw, hidden "
            "assumption, or unsupported leap — and if a proposal is actually sound, say so. "
            "Check arithmetic and logic on questions that have a definite answer. Be specific "
            "and brief. Return STRICT JSON only, no markdown:\n"
            '{"assessments": [{"persona": <one of ' + ids + '>, '
            '"verdict": "sound"|"flawed"|"unsupported", '
            '"flaw": <one sentence naming the main problem, or why it is sound>}], '
            '"overall": <one sentence on where the debate actually stands>}')
        user = f"Question: {question}\n\nProposals to critique:\n{listing}"
        d = self._reason(system, user, temperature=0.3)
        assessments: list[Assessment] = []
        for a in d.get("assessments", []) or []:
            pid = str(a.get("persona", "")).strip()
            if pid in PERSONA_BY_ID or pid in {p.persona_id for p in proposals}:
                assessments.append(Assessment(
                    persona_id=pid,
                    verdict=str(a.get("verdict", "flawed")).lower().strip() or "flawed",
                    flaw=str(a.get("flaw", "")).strip()))
        return Critique(assessments=assessments, overall=str(d.get("overall", "")).strip())


# --------------------------------------------------------------------------- #
# Judge — synthesizes a final answer from everything. Does NOT set confidence.
# --------------------------------------------------------------------------- #
class Judge(_ModelBackedRole):
    def synthesize(self, question: str, proposals: list[Proposal],
                   critique: Critique, tally_summary: str) -> Synthesis:
        listing = "\n".join(
            f'  {p.persona_name}: answer={p.answer!r} stance={p.stance!r}'
            + (f' (revised — {p.change_note!r})' if p.changed_mind else '')
            for p in proposals)
        system = (
            "You are the JUDGE of a multi-agent debate. You have read every debater's final "
            "answer, the critic's review, and the tally of where they landed. Synthesize ONE "
            "final answer that takes the best-supported points and discards the flawed ones. "
            "If the debaters split with no clear winner, say the honest, balanced answer and "
            "name the genuine trade-off — do not fake a consensus that is not there. Do NOT "
            "state a confidence level; that is computed separately from how much they agreed. "
            "Return STRICT JSON only, no markdown:\n"
            '{"final_answer": <2-4 sentences>, '
            '"rationale": <one or two sentences on why this is the synthesis>, '
            '"key_points": [<a few short bullet strings you kept>]}')
        user = (f"Question: {question}\n\nFinal stances:\n{listing}\n\n"
                f"Critic's overall read: {critique.overall}\n\n"
                f"Ballot tally (computed in Python): {tally_summary}")
        d = self._reason(system, user, temperature=0.3)
        kps = d.get("key_points", []) or []
        return Synthesis(
            final_answer=str(d.get("final_answer", "")).strip(),
            rationale=str(d.get("rationale", "")).strip(),
            key_points=[str(k).strip() for k in kps if str(k).strip()])


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _to_proposal(persona: Persona, d: dict) -> Proposal:
    return Proposal(
        persona_id=persona.id,
        persona_name=persona.name,
        answer=str(d.get("answer", "")).strip() or "(no answer parsed)",
        stance=str(d.get("stance", "")).strip() or "(no stance)",
        reasoning=str(d.get("reasoning", "")).strip(),
        self_confidence=str(d.get("confidence", "")).strip().lower())


def build_roles(model: str = DEFAULT_MODEL, client=None, log=None):
    """Make the three proposers (one per persona), a critic, and a judge — all sharing one
    NIM client. The model does the talking; debate.py does the deterministic orchestration."""
    client = client or get_client()
    proposers = [Proposer(p, model, client, log) for p in PERSONAS]
    critic = Critic(model, client, log)
    judge = Judge(model, client, log)
    return proposers, critic, judge


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
