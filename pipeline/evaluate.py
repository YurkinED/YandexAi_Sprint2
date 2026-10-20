"""Прогон метрик, золотой контекст, бутстрап."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .chunking import mrr, precision_at_k, recall_at_k
from .generate import (
    generate,
    judge,
    SYSTEM,
    format_context,
    normalize_answer,
    user_prompt,
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
        answer = normalize_answer(generate([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(q["query"], ctx)},
        ]))
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


def gold_context_run(docs_by_id: dict, queries: list[dict]) -> pd.DataFrame:
    # gen по целым relevant_docs; на gap — пустой ctx
    rows = []
    for i, q in enumerate(queries, 1):
        print(f"  gold [{i}/{len(queries)}] {q['kind']}...", flush=True)
        if q["relevant_docs"]:
            ctx_docs = [docs_by_id[d]["text"] for d in q["relevant_docs"] if d in docs_by_id]
        else:
            ctx_docs = []
        numbered = "\n".join(f"[{j}] {c}" for j, c in enumerate(ctx_docs, 1)) or "[нет фрагментов]"
        answer = normalize_answer(generate([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_prompt(q["query"], numbered)},
        ]))
        faith, relev, correct = judge(q["query"], ctx_docs, answer, q["gold"])
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
