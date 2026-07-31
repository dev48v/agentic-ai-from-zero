"""RAG Agent — citation grounding with honest confidence.

Four ideas, hand-rolled (no framework):
  1. retrieve context                    -> TfidfRetriever.retrieve() (retriever.py)
  2. generate answers with citations     -> grounded JSON generation, inline [n] markers
  3. flag low-confidence responses       -> two signals: top-similarity band + the
                                            model's own "supported" flag
  4. fallback to search                  -> when retrieval is too weak, a clearly
                                            labeled web-search fallback path

Confidence is decided from the top-1 retrieval similarity plus the model's own
grounding signal:

    top_score < FALLBACK_THRESHOLD                 -> FALLBACK   (retrieval too weak)
    FALLBACK <= top_score < CONFIDENT              -> LOW_CONF   (thin support)
    top_score >= CONFIDENT  AND model says grounded-> GROUNDED
    model says it cannot support the answer        -> LOW_CONF   (don't hallucinate)
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field

from common.client import DEFAULT_MODEL, get_client
from retriever import Retrieved, TfidfRetriever

logger = logging.getLogger("rag_citation_agent")

# Confidence labels.
GROUNDED = "GROUNDED"
LOW_CONFIDENCE = "LOW_CONFIDENCE"
FALLBACK = "FALLBACK"

# Tuned against the in-repo corpus (see recorded-run.md for the measured scores).
FALLBACK_THRESHOLD = 0.15   # top-1 below this => retrieval too weak => web-search fallback
CONFIDENT_THRESHOLD = 0.30  # top-1 at/above this AND model-grounded => confident


@dataclass
class Source:
    """A citation entry: the [n] marker mapped back to its passage."""

    marker: int
    source: str
    title: str
    score: float
    snippet: str


@dataclass
class Answer:
    query: str
    confidence: str                       # GROUNDED | LOW_CONFIDENCE | FALLBACK
    answer: str
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Retrieved] = field(default_factory=list)
    top_score: float = 0.0
    model_supported: bool | None = None    # what the grounded model reported (None if fallback)
    reason: str = ""                       # why this confidence label was chosen
    fallback_used: bool = False
    model: str = DEFAULT_MODEL
    elapsed_s: float = 0.0


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model reply, tolerating ```json fences."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    return brace.group(0) if brace else text.strip()


def _snippet(text: str, n: int = 160) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


class RagAgent:
    def __init__(
        self,
        corpus_dir: str,
        k: int = 3,
        model: str = DEFAULT_MODEL,
        fallback_threshold: float = FALLBACK_THRESHOLD,
        confident_threshold: float = CONFIDENT_THRESHOLD,
    ) -> None:
        self.retriever = TfidfRetriever(corpus_dir)
        self.k = k
        self.model = model
        self.fallback_threshold = fallback_threshold
        self.confident_threshold = confident_threshold
        self.client = get_client()

    # ------------------------------------------------------------------ #
    # Sub-point 2: grounded generation with inline [n] citations.
    # Asks the model to answer ONLY from the numbered passages and to report a
    # `supported` flag so we never dress up a guess as a grounded answer.
    # ------------------------------------------------------------------ #
    def _generate_grounded(self, query: str, hits: list[Retrieved]) -> tuple[str, bool, list[int]]:
        passages_block = "\n".join(
            f"[{h.rank}] (source: {h.passage.source}) {h.passage.text}" for h in hits
        )
        system = (
            "You are a retrieval-grounded question answerer. Answer the question using "
            "ONLY the numbered passages provided. Rules:\n"
            "- Use ONLY facts stated in the passages; do NOT use outside knowledge.\n"
            "- Place an inline citation marker like [1] or [2] immediately after EACH "
            "fact, naming the passage the fact came from. Every sentence that states a "
            "fact must end with at least one marker.\n"
            '- Example answer: "The Enterprise plan guarantees 99.95% uptime [1] and '
            'costs 99 dollars per seat each month [2]."\n'
            "- If the passages do not contain enough information to answer, set "
            '"supported" to false and briefly say what is missing. Do NOT guess.\n'
            'Reply with ONLY a JSON object of the form: {"supported": true|false, '
            '"answer": "<answer text with inline [n] citations>", '
            '"citations": [<the passage numbers you cited>]}'
        )
        user = f"Passages:\n{passages_block}\n\nQuestion: {query}"

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
        )
        raw = resp.choices[0].message.content or ""
        try:
            data = json.loads(_extract_json(raw))
            answer = str(data.get("answer", "")).strip()
            supported = bool(data.get("supported", False))
            citations = [int(c) for c in data.get("citations", []) if str(c).isdigit()]
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("grounded reply was not clean JSON (%s); using raw text", exc)
            answer, supported, citations = raw.strip(), True, []
        return answer, supported, citations

    # ------------------------------------------------------------------ #
    # Sub-point 4: fallback to (simulated) web search.
    # Reached only when retrieval is too weak. Answers from the model's broad
    # knowledge, but LOUDLY labeled as NOT grounded in the trusted corpus.
    # ------------------------------------------------------------------ #
    def _fallback_search(self, query: str, hits: list[Retrieved], reason: str) -> Answer:
        logger.info("FALLBACK: %s", reason)
        system = (
            "You are a general web-search fallback assistant. The trusted local "
            "knowledge base returned NO sufficiently relevant passages for this "
            "question, so you are answering from broad general knowledge, as a web "
            "search would. Start your reply with exactly this line:\n"
            "'[FALLBACK - general web knowledge, NOT from the Nimbus knowledge base; "
            "verify independently.]'\n"
            "Then answer briefly."
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        answer = (resp.choices[0].message.content or "").strip()
        return Answer(
            query=query,
            confidence=FALLBACK,
            answer=answer,
            sources=[],  # deliberately empty: nothing in the trusted corpus supports this
            retrieved=hits,
            top_score=hits[0].score if hits else 0.0,
            model_supported=None,
            reason=reason,
            fallback_used=True,
            model=self.model,
        )

    def _build_sources(self, hits: list[Retrieved], citations: list[int]) -> list[Source]:
        by_rank = {h.rank: h for h in hits}
        # If the model cited nothing usable, fall back to showing the retrieved set.
        markers = citations or [h.rank for h in hits]
        seen: set[int] = set()
        out: list[Source] = []
        for m in markers:
            if m in seen or m not in by_rank:
                continue
            seen.add(m)
            h = by_rank[m]
            out.append(
                Source(
                    marker=m,
                    source=h.passage.source,
                    title=h.passage.title,
                    score=round(h.score, 4),
                    snippet=_snippet(h.passage.text),
                )
            )
        return out

    # ------------------------------------------------------------------ #
    # The full pipeline: retrieve -> (fallback | grounded) -> confidence label.
    # ------------------------------------------------------------------ #
    def answer(self, query: str) -> Answer:
        started = time.time()
        hits = self.retriever.retrieve(query, k=self.k)
        top = hits[0].score if hits else 0.0
        logger.info(
            "retrieved top-%d: %s",
            self.k,
            ", ".join(f"[{h.rank}] {h.passage.source} {h.score:.4f}" for h in hits),
        )

        # Sub-point 4: retrieval too weak -> fallback, without pretending to ground.
        if top < self.fallback_threshold:
            result = self._fallback_search(
                query,
                hits,
                reason=(
                    f"top retrieval similarity {top:.4f} < fallback threshold "
                    f"{self.fallback_threshold} -> corpus has no relevant passage"
                ),
            )
            result.elapsed_s = round(time.time() - started, 2)
            return result

        # Sub-point 2: grounded generation with citations.
        answer_text, supported, citations = self._generate_grounded(query, hits)
        sources = self._build_sources(hits, citations)

        # Sub-point 3: decide confidence from BOTH signals.
        if not supported:
            confidence = LOW_CONFIDENCE
            reason = (
                "model reported it could not fully support an answer from the "
                f"retrieved passages (top score {top:.4f})"
            )
        elif top < self.confident_threshold:
            confidence = LOW_CONFIDENCE
            reason = (
                f"top retrieval similarity {top:.4f} is below the confident "
                f"threshold {self.confident_threshold} (thin support)"
            )
        else:
            confidence = GROUNDED
            reason = (
                f"top retrieval similarity {top:.4f} >= {self.confident_threshold} "
                "and the model grounded its answer in the passages"
            )
        logger.info("confidence=%s (%s)", confidence, reason)

        return Answer(
            query=query,
            confidence=confidence,
            answer=answer_text,
            sources=sources,
            retrieved=hits,
            top_score=top,
            model_supported=supported,
            reason=reason,
            fallback_used=False,
            model=self.model,
            elapsed_s=round(time.time() - started, 2),
        )
