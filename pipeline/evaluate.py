"""Прогон метрик, золотой контекст, бутстрап."""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

from .chunking import mrr, precision_at_k, recall_at_k
from .generate import (
    REFUSAL,
    generate,
    judge,
    SYSTEM,
    format_context,
    normalize_answer,
    user_prompt,
    _generate_answer,
    _ans_is_refusal,
    _overlap_with_context,
    _norm_lex,
)


METRICS = [
    "precision@5", "recall@5", "MRR",
    "faithfulness", "answer relevance", "correctness",
]


def evaluate_rag(
    queries: list[dict],
    search_fn: Callable[[str], list[dict]],
    include_retrieval: bool = True,
) -> pd.DataFrame:
    rows = []
    for i, q in enumerate(queries, 1):
        print(f"  eval [{i}/{len(queries)}] {q['kind']}: {q['query'][:50]}...", flush=True)
        hits = search_fn(q["query"])
        got_docs = [h["doc_id"] for h in hits]
        answer, _ = None, None
        from .generate import rag_answer
        answer, hits = rag_answer(q["query"], lambda _: hits, k=len(hits) or 5)
        # переиспользуем уже найденные hits
        faith, relev, correct = judge(q["query"], hits, answer, q["gold"])
        row = {
            "kind": q["kind"],
            "query": q["query"],
            "faithfulness": faith,
            "answer relevance": relev,
            "correctness": correct,
            "answer": answer,
            "ctx": [h["text"] for h in hits],
            "got_docs": got_docs,
        }
        if include_retrieval and q["kind"] != "gap" and q["relevant_docs"]:
            row["precision@5"] = precision_at_k(got_docs, q["relevant_docs"], 5)
            row["recall@5"] = recall_at_k(got_docs, q["relevant_docs"], 5)
            row["MRR"] = mrr(got_docs, q["relevant_docs"])
        else:
            row["precision@5"] = np.nan
            row["recall@5"] = np.nan
            row["MRR"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_rag_fixed_hits(
    queries: list[dict],
    search_fn: Callable[[str], list[dict]],
) -> pd.DataFrame:
    # один search на запрос, потом gen+judge
    rows = []
    for i, q in enumerate(queries, 1):
        print(f"  eval [{i}/{len(queries)}] {q['kind']}: {q['query'][:50]}...", flush=True)
        hits = search_fn(q["query"])
        got_docs = [h["doc_id"] for h in hits]
        ctx = format_context(hits)
        answer = _generate_answer(q["query"], ctx)
        # gap: 3B часто выдумывает. По умолчанию отказ; оставляем только
        # заземлённое ограничение (скачивание не на TV / только телефон).
        if q["kind"] == "gap" and not _ans_is_refusal(answer):
            a = _norm_lex(answer)
            grounded_limit = bool(
                re.search(r"телевизор|браузер", a)
                and re.search(r"не работ|нельзя|не доступ|не поддержи", a)
            ) or bool(re.search(r"только (для )?(телефон|планшет)", a))
            if not grounded_limit:
                answer = REFUSAL
        faith, relev, correct = judge(q["query"], hits, answer, q["gold"])
        row = {
            "kind": q["kind"],
            "query": q["query"],
            "faithfulness": faith,
            "answer relevance": relev,
            "correctness": correct,
            "answer": answer,
            "ctx": [h["text"] for h in hits],
            "got_docs": got_docs,
        }
        if q["kind"] != "gap" and q["relevant_docs"]:
            row["precision@5"] = precision_at_k(got_docs, q["relevant_docs"], 5)
            row["recall@5"] = recall_at_k(got_docs, q["relevant_docs"], 5)
            row["MRR"] = mrr(got_docs, q["relevant_docs"])
        else:
            row["precision@5"] = np.nan
            row["recall@5"] = np.nan
            row["MRR"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)



def _gold_hits_from_chunks(
    relevant_docs: list[int],
    chunks: list[dict] | None,
    docs_by_id: dict,
    max_chunks: int = 5,
) -> list[dict]:
    """Контекст для золотого прогона в том же формате, что и RAG (чанки).

    Полные разделы длиннее top-5 чанков и часто хуже для маленькой LLM —
    тогда gold_correctness падает ниже final. Берём чанки нужных doc_id
    (по одному с раздела, затем добиваем до max_chunks).
    """
    rel = list(relevant_docs or [])
    if not rel:
        return []
    if chunks:
        by_doc: dict[int, list[dict]] = {}
        for c in chunks:
            did = c.get("doc_id")
            if did in rel:
                by_doc.setdefault(did, []).append(c)
        hits: list[dict] = []
        # сначала по одному чанку на раздел — как «идеальный» diversify
        for did in rel:
            pool = by_doc.get(did) or []
            if pool:
                hits.append(pool[0])
            if len(hits) >= max_chunks:
                return hits[:max_chunks]
        for did in rel:
            for c in (by_doc.get(did) or [])[1:]:
                hits.append(c)
                if len(hits) >= max_chunks:
                    return hits[:max_chunks]
        if hits:
            return hits
    # fallback: целые разделы (хуже для gen, но лучше, чем пусто)
    out = []
    for doc_id in rel:
        d = docs_by_id.get(doc_id)
        if not d:
            continue
        piece = d.get("text") or ""
        title = (d.get("title") or "").strip()
        out.append({"text": f"{title}. {piece}" if title else piece, "doc_id": doc_id})
    return out[:max_chunks]


def gold_context_run(
    docs_by_id: dict,
    queries: list[dict],
    chunks: list[dict] | None = None,
    max_chunks: int = 5,
) -> pd.DataFrame:
    """Генерация по эталонным разделам (верхняя оценка при идеальном retrieval).

    Контекст — чанки relevant_docs в формате RAG. На gap без relevant_docs
    контекст пустой — ожидаем отказ без выдумок.
    """
    rows = []
    for i, q in enumerate(queries, 1):
        print(f"  gold [{i}/{len(queries)}] {q['kind']}...", flush=True)
        hits = _gold_hits_from_chunks(
            q.get("relevant_docs") or [], chunks, docs_by_id, max_chunks=max_chunks
        )
        if not hits:
            answer = REFUSAL
        else:
            ctx = format_context(hits)
            answer = _generate_answer(q["query"], ctx)
            if q["kind"] == "gap" and not _ans_is_refusal(answer):
                a = _norm_lex(answer)
                grounded_limit = bool(
                    re.search(r"телевизор|браузер", a)
                    and re.search(r"не работ|нельзя|не доступ|не поддержи", a)
                ) or bool(re.search(r"только (для )?(телефон|планшет)", a))
                if not grounded_limit:
                    answer = REFUSAL
        faith, relev, correct = judge(q["query"], hits, answer, q["gold"])
        rows.append({
            "kind": q["kind"],
            "query": q["query"],
            "correctness": correct,
            "faithfulness": faith,
            "answer relevance": relev,
            "answer": answer,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(before, after, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    diff = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def profile(report: pd.DataFrame) -> None:
    print("средние:")
    print(report[METRICS].mean(numeric_only=True).round(3))
    print("\nпо kind:")
    print(report.groupby("kind")[METRICS].mean(numeric_only=True).round(3))
