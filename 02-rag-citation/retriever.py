"""Deterministic, offline retriever: TF-IDF bag-of-words + cosine top-k.

No embedding API, no network — so retrieval is reproducible and the whole demo
runs offline-friendly. (NVIDIA NIM does expose an embedding endpoint; this
module is written so you could swap `embed()` for a NIM `embeddings.create`
call without touching the agent. TF-IDF is used here precisely because it makes
the recorded run byte-for-byte reproducible.)

Pipeline:
  load corpus  ->  split into passages  ->  fit TF-IDF vocabulary/IDF over
  passages  ->  embed query with the SAME vocabulary  ->  cosine similarity
  ->  return the top-k passages with their scores.
"""

from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass

import numpy as np

# A tiny stopword list — enough to stop "the/of/a" from dominating the vectors.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "to", "with", "you", "your", "per", "each", "does", "do", "can", "how",
    "what", "which", "when", "who", "whom", "this", "these", "those", "all",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


@dataclass
class Passage:
    """One retrievable unit of the corpus."""

    idx: int          # stable 0-based position in the corpus
    source: str       # source doc filename, e.g. "pricing.md"
    title: str        # the doc's first heading, for nicer citations
    text: str         # the passage text


@dataclass
class Retrieved:
    """A passage plus the similarity score it earned for a given query."""

    rank: int         # 1-based rank in THIS query's result (drives the [n] marker)
    score: float      # cosine similarity in [0, 1]
    passage: Passage


def _split_passages(doc_text: str) -> list[str]:
    """Split a doc into passages on blank lines; drop the leading heading line."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", doc_text) if b.strip()]
    out: list[str] = []
    for b in blocks:
        # Skip a pure markdown heading block ("# Title") — it is captured as title.
        if b.startswith("#") and "\n" not in b:
            continue
        out.append(" ".join(line.strip() for line in b.splitlines()))
    return out


class TfidfRetriever:
    """Fit TF-IDF over a folder of markdown docs and retrieve top-k by cosine."""

    def __init__(self, corpus_dir: str) -> None:
        self.corpus_dir = corpus_dir
        self.passages: list[Passage] = []
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)
        self._matrix: np.ndarray = np.zeros((0, 0))  # (n_passages, vocab) L2-normalized
        self._load()
        self._fit()

    # -- corpus loading ---------------------------------------------------- #
    def _load(self) -> None:
        paths = sorted(glob.glob(os.path.join(self.corpus_dir, "*.md")))
        if not paths:
            raise FileNotFoundError(f"no .md docs found in {self.corpus_dir}")
        idx = 0
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read()
            source = os.path.basename(path)
            first = raw.strip().splitlines()[0] if raw.strip() else source
            title = first.lstrip("#").strip() or source
            for chunk in _split_passages(raw):
                self.passages.append(Passage(idx=idx, source=source, title=title, text=chunk))
                idx += 1

    # -- TF-IDF fit -------------------------------------------------------- #
    def _fit(self) -> None:
        docs_tokens = [_tokenize(p.text) for p in self.passages]
        vocab: dict[str, int] = {}
        for toks in docs_tokens:
            for t in set(toks):
                vocab.setdefault(t, len(vocab))
        self.vocab = vocab

        n_docs = len(self.passages)
        df = np.zeros(len(vocab))
        for toks in docs_tokens:
            for t in set(toks):
                df[vocab[t]] += 1
        # Smoothed IDF, always positive.
        self.idf = np.log((1 + n_docs) / (1 + df)) + 1.0

        matrix = np.zeros((n_docs, len(vocab)))
        for i, toks in enumerate(docs_tokens):
            matrix[i] = self._vectorize(toks)
        self._matrix = matrix

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(len(self.vocab))
        if not tokens:
            return vec
        for t in tokens:
            j = self.vocab.get(t)
            if j is not None:
                vec[j] += 1.0
        vec = (vec / len(tokens)) * self.idf   # tf * idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    # -- retrieval --------------------------------------------------------- #
    def embed(self, text: str) -> np.ndarray:
        """Embed arbitrary text into the fitted TF-IDF space (L2-normalized)."""
        return self._vectorize(_tokenize(text))

    def retrieve(self, query: str, k: int = 3) -> list[Retrieved]:
        """Return the top-k passages by cosine similarity, highest first."""
        q = self.embed(query)
        # Both sides are L2-normalized, so the dot product IS the cosine similarity.
        sims = self._matrix @ q
        order = np.argsort(-sims)[:k]
        return [
            Retrieved(rank=r + 1, score=float(sims[i]), passage=self.passages[i])
            for r, i in enumerate(order)
        ]
