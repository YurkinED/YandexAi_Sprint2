"""Чанкинг + e5 + метрики retrieval по doc_id."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np


def fixed_window(text: str, size: int = 400, overlap: int = 0) -> list[str]:
    if not text:
        return []
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def sentence_window(text: str, size: int = 400, overlap_sentences: int = 1) -> list[str]:
    # как в Lesson11 task6
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]
    parts, buf = [], []
    for sentence in sentences:
        candidate = " ".join([*buf, sentence])
        if not buf or len(candidate) <= size:
            buf.append(sentence)
        else:
            parts.append(" ".join(buf))
            buf = (buf[-overlap_sentences:] + [sentence]
                   if overlap_sentences else [sentence])
            while len(" ".join(buf)) > size and len(buf) > 1:
                buf.pop(0)
    if buf:
        parts.append(" ".join(buf))
    return [p for p in parts if p.strip()]


def build_chunks(
    docs: list[dict],
    splitter: Callable[[str], list[str]],
) -> tuple[list[str], list[int], list[dict]]:
    texts, doc_ids, metas = [], [], []
    for d in docs:
        pieces = splitter(d["text"])
        for i, piece in enumerate(pieces):
            # title в чанк — иначе часто непонятно, о чём кусок
            body = f"{d['title']}. {piece}" if d.get("title") else piece
            texts.append(body)
            doc_ids.append(d["doc_id"])
            metas.append({
                "doc_id": d["doc_id"],
                "title": d.get("title"),
                "category": d.get("category"),
                "chunk_i": i,
            })
    return texts, doc_ids, metas


def embed_passages(embedder, texts: list[str], batch_size: int = 64) -> np.ndarray:
    inputs = ["passage: " + t for t in texts]
    vecs = embedder.encode(
        inputs, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=True
    )
    return np.asarray(vecs, dtype=np.float32)


def dense_search(
    embedder,
    doc_vecs: np.ndarray,
    query: str,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (indices, scores) top-k по косинусу."""
    q = embedder.encode("query: " + query, normalize_embeddings=True)
    scores = q @ doc_vecs.T
    top = np.argsort(scores)[::-1][:k]
    return top, scores[top]


def section_hits(got_doc_ids: list[int], relevant: list[int], k: int | None = None) -> set[int]:
    ids = got_doc_ids[:k] if k is not None else got_doc_ids
    return set(ids) & set(relevant)


def precision_at_k(got_doc_ids: list[int], relevant: list[int], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    return len(section_hits(got_doc_ids, relevant, k)) / k


def recall_at_k(got_doc_ids: list[int], relevant: list[int], k: int = 5) -> float:
    if not relevant:
        return 0.0
    return len(section_hits(got_doc_ids, relevant, k)) / len(relevant)


def mrr(got_doc_ids: list[int], relevant: list[int]) -> float:
    rel = set(relevant)
    for rank, d in enumerate(got_doc_ids, start=1):
        if d in rel:
            return 1.0 / rank
    return 0.0


def eval_retrieval(
    embedder,
    doc_vecs: np.ndarray,
    texts: list[str],
    doc_ids: list[int],
    queries: list[dict],
    k: int = 5,
) -> dict:
    rows = []
    for q in queries:
        if q["kind"] == "gap" or not q["relevant_docs"]:
            continue
        top, _ = dense_search(embedder, doc_vecs, q["query"], k=max(k, 20))
        got = [doc_ids[i] for i in top]
        rows.append({
            "kind": q["kind"],
            "p": precision_at_k(got, q["relevant_docs"], k),
            "r": recall_at_k(got, q["relevant_docs"], k),
            "mrr": mrr(got, q["relevant_docs"]),
        })
    return {
        "n": len(rows),
        "precision@5": float(np.mean([r["p"] for r in rows])),
        "recall@5": float(np.mean([r["r"] for r in rows])),
        "MRR": float(np.mean([r["mrr"] for r in rows])),
        "by_kind": {
            kind: {
                "precision@5": float(np.mean([r["p"] for r in rows if r["kind"] == kind])),
                "recall@5": float(np.mean([r["r"] for r in rows if r["kind"] == kind])),
                "MRR": float(np.mean([r["mrr"] for r in rows if r["kind"] == kind])),
            }
            for kind in sorted({r["kind"] for r in rows})
        },
    }
