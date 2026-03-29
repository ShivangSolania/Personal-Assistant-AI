"""
memory/memory_store.py
──────────────────────
Conversation + tool-output memory with optional FAISS vector search.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from utils.logger import get_logger

log = get_logger(__name__)


# ── Data Containers ───────────────────────────────────────

@dataclass
class MemoryEntry:
    role: str                 # "user" | "assistant" | "tool"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


# ── In-Memory Store ───────────────────────────────────────
#create custom class which self decides whether to use vector store or not based on the size of the memory (size<1MB)
class InMemoryStore:
    """Simple list-backed conversation memory."""

    def __init__(self, max_entries: int = 200) -> None:
        self._entries: list[MemoryEntry] = []
        self._max = max_entries

    def add(self, role: str, content: str, **metadata: Any) -> None:
        entry = MemoryEntry(role=role, content=content, metadata=metadata)
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max :]
        log.debug("Memory +1 (%s) — total %d", role, len(self._entries))

    def get_history(self, last_n: int | None = None) -> list[dict[str, Any]]:
        entries = self._entries if last_n is None else self._entries[-last_n:]
        return [
            {"role": e.role, "content": e.content, "metadata": e.metadata}
            for e in entries
        ]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Keyword-based search (fallback when no vector store)."""
        q_lower = query.lower()
        scored = []
        for e in self._entries:
            score = sum(1 for w in q_lower.split() if w in e.content.lower())
            if score > 0:
                scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"role": e.role, "content": e.content, "metadata": e.metadata, "score": s}
            for s, e in scored[:top_k]
        ]

    def clear(self) -> None:
        self._entries.clear()


# ── FAISS Vector Store ────────────────────────────────────

class VectorMemoryStore:
    """
    FAISS-backed semantic memory store.
    Falls back to InMemoryStore if FAISS/numpy aren't available.
    """

    def __init__(self, embedding_dim: int = 384, max_entries: int = 500) -> None:
        self._fallback = InMemoryStore(max_entries)
        self._texts: list[str] = []
        self._roles: list[str] = []
        self._meta: list[dict] = []
        self._dim = embedding_dim
        self._index = None

        try:
            import faiss
            import numpy as np
            self._index = faiss.IndexFlatL2(embedding_dim)
            self._np = np
            log.info("FAISS vector memory initialised (dim=%d)", embedding_dim)
        except ImportError:
            log.warning("faiss-cpu not installed — falling back to keyword search")

    # very simple bag-of-chars embedding (replace with a real model in prod) REPLACEE
    def _embed(self, text: str):
        vec = self._np.zeros(self._dim, dtype="float32")
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self._dim] += ch
        norm = self._np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.reshape(1, -1)

    def add(self, role: str, content: str, **metadata: Any) -> None:
        self._fallback.add(role, content, **metadata)
        if self._index is not None:
            self._index.add(self._embed(content))
            self._texts.append(content)
            self._roles.append(role)
            self._meta.append(metadata)

    def get_history(self, last_n: int | None = None) -> list[dict[str, Any]]:
        return self._fallback.get_history(last_n)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._index is None or self._index.ntotal == 0:
            return self._fallback.search(query, top_k)
        q_vec = self._embed(query)
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(q_vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "role": self._roles[idx],
                "content": self._texts[idx],
                "metadata": self._meta[idx],
                "distance": float(dist),
            })
        return results

    def clear(self) -> None:
        self._fallback.clear()
        self._texts.clear()
        self._roles.clear()
        self._meta.clear()
        if self._index is not None:
            self._index.reset()


# ── Factory ───────────────────────────────────────────────

class MemoryManager:
    """Facade — picks the right backend based on config."""

    def __init__(self, backend: str | None = None) -> None:
        backend = (backend or os.getenv("MEMORY_BACKEND", "in_memory")).lower()
        if backend == "faiss":
            self.store = VectorMemoryStore()
        else:
            self.store = InMemoryStore()
        log.info("Memory backend: %s", type(self.store).__name__)

    def add(self, role: str, content: str, **meta: Any) -> None:
        self.store.add(role, content, **meta)

    def get_history(self, last_n: int | None = None) -> list[dict[str, Any]]:
        return self.store.get_history(last_n)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.store.search(query, top_k)

    def clear(self) -> None:
        self.store.clear()
