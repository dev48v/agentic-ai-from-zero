"""Two-tier memory for a conversational agent — store + embed + compress, hand-rolled.

Four ideas, no framework, no paid embedding API. Each maps to one sub-point:

  1. short-term buffer        -> `MemoryStore.buffer`: the most recent exchanges kept
                                 VERBATIM in a turn-bounded window (also token-estimated).
  2. long-term vector recall  -> `HashingEmbedder` + `MemoryStore.recall`: a hand-rolled
                                 hashing bag-of-words embedding + cosine similarity over
                                 stored memories (same spirit as Project 2's TF-IDF
                                 retriever, but incremental so a growing store needs no
                                 refit). NO network, NO paid embeddings.
  3. context compression      -> `MemoryStore.append` overflow path: when the buffer
                                 overflows, the oldest exchanges are SUMMARISED (via an
                                 injected model callback) into one compact memory note
                                 and moved to long-term; the raw turns are dropped.
  4. cross-session sync       -> `MemoryStore.save`/`load`: the long-term store (and the
                                 residual buffer) persist to a local JSON file, so a NEW
                                 process run remembers facts from a PRIOR run.

Design choice: long-term memories store their TEXT (human-readable JSON) and are
re-embedded deterministically on load. The embedding uses a STABLE hash (hashlib,
not Python's per-process-salted `hash()`), so a memory embeds to the same vector in
session 2 as it did in session 1 — which is what makes cross-session recall work.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Hand-rolled embedding: hashing bag-of-words -> cosine. No API, deterministic.
# --------------------------------------------------------------------------- #
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "with", "you", "your", "me", "my", "am", "was", "were", "do",
    "does", "did", "can", "could", "would", "should", "how", "what", "which",
    "when", "who", "whom", "this", "these", "those", "all", "so", "if", "im",
    "s", "re", "ll", "ve", "about", "just", "keep", "mind", "please", "hey",
    "hi", "hello", "there", "some", "any", "get", "give", "tell", "want",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if t not in _STOPWORDS and len(t) > 1]


def _stable_bucket(token: str, dim: int) -> tuple[int, float]:
    """Map a token to a (bucket, sign) with a STABLE hash.

    Python's built-in `hash("str")` is salted per process (PYTHONHASHSEED), which
    would give a memory a DIFFERENT vector in a fresh process and break
    cross-session recall. hashlib is stable across processes and machines.
    Signed hashing (the second hash bit) reduces bucket-collision bias.
    """
    digest = hashlib.md5(token.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % dim
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


class HashingEmbedder:
    """Deterministic hashing bag-of-words embedder; cosine == dot of L2-norm vectors."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        toks = _tokenize(text)
        if not toks:
            return vec
        for t in toks:
            bucket, sign = _stable_bucket(t, self.dim)
            vec[bucket] += sign
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        # both are L2-normalized already, so the dot product IS the cosine.
        return float(np.dot(a, b))


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Exchange:
    """One verbatim user->assistant turn kept in the short-term buffer."""
    user: str
    assistant: str
    ts: str = ""


@dataclass
class Memory:
    """One long-term memory item in the vector store."""
    id: int
    kind: str          # "turn" (an archived past user turn) | "summary" (from compression)
    text: str
    session: str       # which session created it
    ts: str = ""


@dataclass
class Recalled:
    """A long-term memory plus the relevance score it earned for a query."""
    rank: int          # 1-based rank for THIS query
    score: float       # cosine similarity in [-1, 1]
    injected: bool     # did it clear the relevance threshold and enter the prompt?
    memory: Memory


@dataclass
class CompressionEvent:
    """Recorded whenever the buffer overflowed and older turns were summarised."""
    compressed_exchanges: int
    summary_id: int
    summary_text: str
    freed_tokens_est: int


# --------------------------------------------------------------------------- #
# The two-tier store
# --------------------------------------------------------------------------- #
def _est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) so the window can be token-aware."""
    return max(1, len(text) // 4)


class MemoryStore:
    """Short-term buffer (verbatim, bounded) + long-term vector store (recalled).

    Compression is delegated: the caller injects a `summarizer(text) -> note` so the
    STORE owns the *policy* (when to compress, which turns, where the note goes) while
    the AGENT owns the *model call*. That keeps this module testable offline.
    """

    def __init__(
        self,
        path: str,
        max_turns: int = 4,          # verbatim exchanges kept in the short-term window
        compress_chunk: int = 2,     # how many oldest exchanges fold into one note
        embed_dim: int = 512,
        top_k: int = 3,              # long-term memories retrieved per user turn
        min_score: float = 0.05,     # relevance threshold to actually INJECT a memory
    ) -> None:
        self.path = path
        self.max_turns = max_turns
        self.compress_chunk = compress_chunk
        self.top_k = top_k
        self.min_score = min_score
        self.embedder = HashingEmbedder(embed_dim)

        self.buffer: list[Exchange] = []
        self.long_term: list[Memory] = []
        self._next_id = 1
        self._vectors: list[np.ndarray] = []   # parallel to long_term, recomputed on load

    # ---------------------------- persistence --------------------------- #
    def load(self) -> int:
        """Load long-term memories + residual buffer from disk. Returns #memories."""
        if not os.path.exists(self.path):
            return 0
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.long_term = [Memory(**m) for m in data.get("long_term", [])]
        self.buffer = [Exchange(**e) for e in data.get("buffer", [])]
        self._next_id = data.get("next_id", len(self.long_term) + 1)
        # Re-embed every stored memory deterministically (stable hash -> same vectors).
        self._vectors = [self.embedder.embed(m.text) for m in self.long_term]
        return len(self.long_term)

    def save(self) -> None:
        data = {
            "version": 1,
            "next_id": self._next_id,
            "long_term": [asdict(m) for m in self.long_term],
            "buffer": [asdict(e) for e in self.buffer],
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # ---------------------- long-term vector recall --------------------- #
    def add_memory(self, text: str, kind: str, session: str) -> Memory:
        mem = Memory(id=self._next_id, kind=kind, text=text.strip(),
                     session=session, ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._next_id += 1
        self.long_term.append(mem)
        self._vectors.append(self.embedder.embed(mem.text))
        return mem

    def recall(self, query: str) -> list[Recalled]:
        """Top-k long-term memories by cosine score; flag which clear the threshold."""
        if not self.long_term:
            return []
        q = self.embedder.embed(query)
        sims = [self.embedder.cosine(q, v) for v in self._vectors]
        order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[: self.top_k]
        out: list[Recalled] = []
        for rank, i in enumerate(order, start=1):
            out.append(Recalled(
                rank=rank,
                score=round(sims[i], 4),
                injected=sims[i] >= self.min_score,
                memory=self.long_term[i],
            ))
        return out

    # ------------------- short-term buffer + compression ---------------- #
    def buffer_tokens(self) -> int:
        return sum(_est_tokens(e.user) + _est_tokens(e.assistant) for e in self.buffer)

    def append(
        self,
        user: str,
        assistant: str,
        session: str,
        summarizer: Callable[[str], str],
    ) -> CompressionEvent | None:
        """Add an exchange to the buffer; if it overflows, COMPRESS the oldest chunk.

        The user's turn is ALSO archived to the long-term vector store (kind="turn"),
        because the user's own words are the best keys for later recall — this is the
        "vector recall over past turns" tier. Returns a CompressionEvent when
        compression fired (kind="summary"), else None.
        """
        self.add_memory(user, kind="turn", session=session)
        self.buffer.append(Exchange(user=user, assistant=assistant,
                                    ts=time.strftime("%Y-%m-%dT%H:%M:%S")))
        if len(self.buffer) <= self.max_turns:
            return None

        # OVERFLOW -> summarise the oldest `compress_chunk` exchanges into one note.
        chunk = self.buffer[: self.compress_chunk]
        self.buffer = self.buffer[self.compress_chunk :]
        freed = sum(_est_tokens(e.user) + _est_tokens(e.assistant) for e in chunk)
        raw = "\n".join(f"User: {e.user}\nAssistant: {e.assistant}" for e in chunk)
        note = summarizer(raw).strip()
        mem = self.add_memory(note, kind="summary", session=session)
        return CompressionEvent(
            compressed_exchanges=len(chunk),
            summary_id=mem.id,
            summary_text=note,
            freed_tokens_est=freed,
        )
