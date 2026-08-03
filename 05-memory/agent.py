"""Memory-Enabled Conversational Agent — remembers across turns AND sessions.

The agent owns the MODEL; the store (`memory.py`) owns the memory POLICY. On each
user turn the agent:

  1. RECALL   — scores every long-term memory against the user message (hand-rolled
                cosine) and injects only the top, above-threshold ones (relevance
                scoring) — see `MemoryStore.recall`.
  2. PROMPT   — builds: system + [injected long-term memories] + the verbatim
                short-term buffer + the new user message.
  3. ANSWER   — one NVIDIA NIM chat call produces the reply.
  4. REMEMBER — appends the exchange to the short-term buffer; if the buffer
                overflows, the oldest turns are COMPRESSED (a second NIM call) into a
                compact note stored long-term, and the raw turns are dropped.
  5. PERSIST  — the whole store is written to a local JSON file, so a fresh process
                run recalls these facts (cross-session sync).

Only two things use the model: compression (summarise old turns) and the final
answer. Recall + relevance scoring + the buffer window are deterministic Python.
"""

from __future__ import annotations

import logging

from common.client import DEFAULT_MODEL, get_client
from memory import CompressionEvent, MemoryStore, Recalled
from dataclasses import dataclass

logger = logging.getLogger("memory-agent")


@dataclass
class TurnResult:
    user: str
    recalled: list[Recalled]
    injected: list[Recalled]
    answer: str
    compression: CompressionEvent | None
    buffer_size: int
    buffer_tokens: int
    long_term_size: int


_SYSTEM = (
    "You are a helpful assistant with memory. You are given (a) MEMORY — durable facts "
    "recalled from earlier in this or a PAST session, and (b) the recent conversation. "
    "Treat MEMORY as true things the user told you before; use it to answer questions "
    "about the user even if it was said in an earlier session. If the memory does not "
    "contain the answer, say you don't have it — do not invent facts. Be concise."
)

_SUMMARISE_SYSTEM = (
    "You compress old conversation turns into ONE compact memory note for long-term "
    "storage. Preserve EVERY durable fact the user revealed, keeping their exact key "
    "words: personal names, preferences, allergies (keep the word 'allergic' and the "
    "allergen), project names, technology/tool names, numbers, and decisions. Write 1-2 "
    "terse third-person sentences. No preamble, no 'the user asked' filler — just the "
    "facts worth remembering."
)


class MemoryAgent:
    def __init__(self, store: MemoryStore, session: str, model: str = DEFAULT_MODEL) -> None:
        self.store = store
        self.session = session
        self.model = model
        self.client = get_client()

    # -- the model-backed compressor handed to the store -------------------- #
    def _summarize(self, raw_turns: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SUMMARISE_SYSTEM},
                {"role": "user", "content": raw_turns},
            ],
            temperature=0,
        )
        note = (resp.choices[0].message.content or "").strip()
        return note or raw_turns  # never lose the memory if the model returns nothing

    # -- build the prompt from injected memories + the verbatim buffer ------ #
    def _build_messages(self, user_msg: str, injected: list[Recalled]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": _SYSTEM}]
        if injected:
            mem_block = "\n".join(
                f"- (memory #{r.memory.id}, relevance {r.score:.2f}, from {r.memory.session}) "
                f"{r.memory.text}"
                for r in injected
            )
            messages.append({
                "role": "system",
                "content": f"MEMORY (recalled by relevance to the user's message):\n{mem_block}",
            })
        for ex in self.store.buffer:          # short-term buffer, verbatim
            messages.append({"role": "user", "content": ex.user})
            messages.append({"role": "assistant", "content": ex.assistant})
        messages.append({"role": "user", "content": user_msg})
        return messages

    # -- one full conversational turn --------------------------------------- #
    def chat(self, user_msg: str) -> TurnResult:
        # 1. RECALL + relevance scoring (deterministic Python)
        recalled = self.store.recall(user_msg)
        injected = [r for r in recalled if r.injected]
        logger.info(
            "recall q=%r -> %s",
            user_msg,
            [(r.memory.id, r.score, "inject" if r.injected else "skip") for r in recalled],
        )

        # 2 + 3. PROMPT + ANSWER (NVIDIA NIM)
        messages = self._build_messages(user_msg, injected)
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2,
        )
        answer = (resp.choices[0].message.content or "").strip()

        # 4. REMEMBER — append to buffer; compress on overflow (a 2nd NIM call)
        compression = self.store.append(user_msg, answer, self.session, self._summarize)
        if compression:
            logger.info(
                "COMPRESSED %d old exchange(s) -> long-term memory #%d (freed ~%d tok): %s",
                compression.compressed_exchanges, compression.summary_id,
                compression.freed_tokens_est, compression.summary_text,
            )

        # 5. PERSIST (cross-session sync)
        self.store.save()

        return TurnResult(
            user=user_msg,
            recalled=recalled,
            injected=injected,
            answer=answer,
            compression=compression,
            buffer_size=len(self.store.buffer),
            buffer_tokens=self.store.buffer_tokens(),
            long_term_size=len(self.store.long_term),
        )
