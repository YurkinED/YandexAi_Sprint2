"""Chroma + baseline / hybrid RRF / rerank."""

from __future__ import annotations

import re
from typing import Any

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi


def build_chroma(
    texts: list[str],
    metas: list[dict],
    vecs: np.ndarray,
    path: str | None = None,
    collection: str = "filmoskop",
):
    # path=None → в памяти
    client = (
        chromadb.PersistentClient(path=path)
        if path
        else chromadb.EphemeralClient()
    )
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    col = client.create_collection(
        name=collection,
        configuration={"hnsw": {"space": "cosine"}},
    )
    clean_metas = []
    for m in metas:
        clean_metas.append({
            "doc_id": int(m["doc_id"]),
            "category": str(m.get("category") or ""),
            "title": str(m.get("title") or "")[:200],
            "chunk_i": int(m.get("chunk_i") or 0),
        })
    col.add(
        ids=[str(i) for i in range(len(texts))],
        embeddings=vecs.tolist(),
        documents=texts,
        metadatas=clean_metas,
    )
    return client, col


def chroma_search(
    col,
    q_vec: np.ndarray,
    k: int = 5,
    where: dict | None = None,
) -> list[dict[str, Any]]:
    kwargs = {
        "query_embeddings": [q_vec.tolist()],
        "n_results": k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    res = col.query(**kwargs)
    out = []
    for i, doc in enumerate(res["documents"][0]):
        dist = float(res["distances"][0][i])  # Chroma cosine dist = 1 - sim
        meta = res["metadatas"][0][i]
        out.append({
            "id": int(res["ids"][0][i]),
            "text": doc,
            "doc_id": int(meta["doc_id"]),
            "category": meta.get("category"),
            "title": meta.get("title"),
            "dist": dist,
            "score": 1.0 - dist,
        })
    return out


def apply_threshold(hits: list[dict], threshold: float) -> list[dict]:
    return [h for h in hits if h["score"] >= threshold]


class BM25Index:
    def __init__(self, texts: list[str]):
        self.texts = texts
        self.corpus = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]


def rrf_fuse(
    rankings: list[list[int]],
    k: int = 5,
    rrf_k: int = 60,
) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]


def hybrid_search(
    embedder,
    col,
    bm25: BM25Index,
    texts: list[str],
    doc_ids: list[int],
    query: str,
    k: int = 5,
    n_cand: int = 20,
) -> list[dict]:
    q_vec = embedder.encode("query: " + query, normalize_embeddings=True)
    dense = chroma_search(col, q_vec, k=n_cand)
    dense_ids = [h["id"] for h in dense]
    sparse_ids = [i for i, _ in bm25.search(query, k=n_cand)]
    fused = rrf_fuse([dense_ids, sparse_ids], k=k)
    out = []
    for cid in fused:
        out.append({
            "id": cid,
            "text": texts[cid],
            "doc_id": doc_ids[cid],
            "score": None,
        })
    return out


def _archive_penalty(text: str) -> float:
    t = (text or "").lower()
    if "устарел" in t or "в архиве" in t or "не ориентируйтесь" in t:
        return 1.5
    return 0.0


def _query_variants(query: str) -> list[str]:
    """Для составных вопросов — доп. подзапросы (вторая половина часто теряется)."""
    variants = [query]
    parts = [p.strip() for p in re.split(r"\s+и\s+|,\s*|/\s*", query) if p.strip()]
    if len(parts) >= 2:
        variants.extend(parts[:3])
    # хвост после тире/вопроса часто содержит вторую сущность
    m = re.split(r"\s+[—–-]\s+", query)
    if len(m) >= 2 and len(m[-1]) > 12:
        variants.append(m[-1].strip(" ?"))
    # уникальные, без слишком коротких
    seen, out = set(), []
    for v in variants:
        key = v.lower()
        if key in seen or len(v) < 8:
            continue
        seen.add(key)
        out.append(v)
    return out


def rerank_search(
    embedder,
    col,
    reranker,
    query: str,
    k: int = 5,
    n_cand: int = 20,
    diversify: bool = True,
) -> list[dict]:
    # собираем кандидатов по основному запросу + подзапросам
    by_id: dict[Any, dict] = {}
    for q in _query_variants(query):
        q_vec = embedder.encode("query: " + q, normalize_embeddings=True)
        for h in chroma_search(col, q_vec, k=n_cand):
            prev = by_id.get(h["id"])
            if prev is None or h.get("score", 0) > prev.get("score", 0):
                by_id[h["id"]] = h
    cand = list(by_id.values())
    if not cand:
        return []

    pairs = [(query, h["text"]) for h in cand]
    scores = np.asarray(reranker.predict(pairs), dtype=float)
    for i, h in enumerate(cand):
        scores[i] -= _archive_penalty(h.get("text", ""))

    order = list(np.argsort(scores)[::-1])
    out: list[dict] = []
    seen_docs: set[int] = set()

    if diversify:
        for j in order:
            doc_id = int(cand[int(j)]["doc_id"])
            if doc_id in seen_docs:
                continue
            h = dict(cand[int(j)])
            h["rerank_score"] = float(scores[int(j)])
            out.append(h)
            seen_docs.add(doc_id)
            if len(out) >= k:
                return out
        for j in order:
            if any(x.get("id") == cand[int(j)].get("id") for x in out):
                continue
            h = dict(cand[int(j)])
            h["rerank_score"] = float(scores[int(j)])
            out.append(h)
            if len(out) >= k:
                break
        return out

    for j in order[:k]:
        h = dict(cand[int(j)])
        h["rerank_score"] = float(scores[int(j)])
        out.append(h)
    return out


def recall_curve(
    embedder,
    doc_vecs: np.ndarray,
    doc_ids: list[int],
    queries: list[dict],
    ks: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[int, float]:
    from .chunking import recall_at_k, dense_search

    usable = [q for q in queries if q["kind"] != "gap" and q["relevant_docs"]]
    out = {}
    max_k = max(ks)
    for k in ks:
        vals = []
        for q in usable:
            top, _ = dense_search(embedder, doc_vecs, q["query"], k=max_k)
            got = [doc_ids[i] for i in top]
            vals.append(recall_at_k(got, q["relevant_docs"], k))
        out[k] = float(np.mean(vals))
    return out
