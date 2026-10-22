"""Этапы 1–4 без LLM. Запуск: python run_stages_1_4.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.clean import clean_corpus, save_cleaned, stats_to_dict
from pipeline.chunking import (
    build_chunks,
    embed_passages,
    eval_retrieval,
    fixed_window,
    sentence_window,
)
from pipeline.retrieve import (
    BM25Index,
    build_chroma,
    chroma_search,
    hybrid_search,
    recall_curve,
    rerank_search,
    apply_threshold,
)
from pipeline.chunking import precision_at_k, recall_at_k, mrr

ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads((ROOT / "filmoskop_kb.json").read_text(encoding="utf-8"))
    docs_raw, queries = data["docs"], data["queries"]
    print(f"Сырых разделов: {len(docs_raw)} | Запросов: {len(queries)}")

    cleaned, stats = clean_corpus(docs_raw)
    save_cleaned(cleaned, ART / "corpus_clean.json")
    (ART / "clean_stats.json").write_text(
        json.dumps(stats_to_dict(stats), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n[1] очистка")
    print(stats_to_dict(stats))

    import os
    device = os.environ.get(
        "FILMOSKOP_RETRIEVE_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"\n[2] чанкинг + эмбеды ({device})")
    embedder = SentenceTransformer("intfloat/multilingual-e5-base", device=device)

    variants = {
        "fixed_400": lambda t: fixed_window(t, size=400, overlap=80),
        "sentence_500": lambda t: sentence_window(t, size=500, overlap_sentences=1),
    }
    chunk_reports = {}
    stores = {}
    for name, splitter in variants.items():
        texts, doc_ids, metas = build_chunks(cleaned, splitter)
        print(f"  {name}: чанков {len(texts)}")
        vecs = embed_passages(embedder, texts)
        np.save(ART / f"vecs_{name}.npy", vecs)
        (ART / f"chunks_{name}.json").write_text(
            json.dumps(
                [{"text": t, "doc_id": d, **m} for t, d, m in zip(texts, doc_ids, metas)],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        metrics = eval_retrieval(embedder, vecs, texts, doc_ids, queries, k=5)
        chunk_reports[name] = {"n_chunks": len(texts), **metrics}
        stores[name] = (texts, doc_ids, metas, vecs)
        print(f"    R@5={metrics['recall@5']:.3f} P@5={metrics['precision@5']:.3f} MRR={metrics['MRR']:.3f}")

    (ART / "chunk_compare.json").write_text(
        json.dumps(chunk_reports, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # по recall@5; ничья — sentence (как в Lesson11)
    chosen = max(chunk_reports, key=lambda n: (chunk_reports[n]["recall@5"], "sentence" in n))
    print(f"чанкинг: {chosen}")
    texts, doc_ids, metas, vecs = stores[chosen]

    print("\n[3] chroma + baseline")
    _, col = build_chroma(texts, metas, vecs, path=str(ART / "chroma_filmoskop"))
    print("в коллекции:", col.count())

    # порог: 15-й перцентиль top-1 на non-gap
    top1 = []
    for q in queries:
        if q["kind"] == "gap":
            continue
        qv = embedder.encode("query: " + q["query"], normalize_embeddings=True)
        hits = chroma_search(col, qv, k=1)
        if hits:
            top1.append(hits[0]["score"])
    thr = float(np.percentile(top1, 15)) if top1 else 0.35
    print(f"порог: {thr:.3f}")

    demo_q = "как отменить подписку"
    qv = embedder.encode("query: " + demo_q, normalize_embeddings=True)
    filt = chroma_search(col, qv, k=3, where={"category": "Подписка и оплата"})
    print("фильтр category:", [h["category"] for h in filt])

    curve = recall_curve(embedder, vecs, doc_ids, queries)
    print("recall@K:", {str(k): round(v, 3) for k, v in curve.items()})

    def search_baseline(query: str, k: int = 5):
        qv = embedder.encode("query: " + query, normalize_embeddings=True)
        hits = chroma_search(col, qv, k=k)
        return apply_threshold(hits, thr) or hits  # пустой список не хочу

    rows = []
    for q in queries:
        if q["kind"] == "gap" or not q["relevant_docs"]:
            continue
        hits = search_baseline(q["query"], k=5)
        got = [h["doc_id"] for h in hits]
        rows.append({
            "kind": q["kind"],
            "precision@5": precision_at_k(got, q["relevant_docs"], 5),
            "recall@5": recall_at_k(got, q["relevant_docs"], 5),
            "MRR": mrr(got, q["relevant_docs"]),
        })
    base_df = pd.DataFrame(rows)
    print("Baseline:", base_df[["precision@5", "recall@5", "MRR"]].mean().round(3).to_dict())
    base_df.to_csv(ART / "baseline_retrieval.csv", index=False)

    print("\n[4] hybrid + rerank")
    bm25 = BM25Index(texts)

    def search_hybrid(query: str, k: int = 5):
        return hybrid_search(embedder, col, bm25, texts, doc_ids, query, k=k)

    def metrics_for(search_fn):
        rows = []
        for q in queries:
            if q["kind"] == "gap" or not q["relevant_docs"]:
                continue
            hits = search_fn(q["query"])
            got = [h["doc_id"] for h in hits]
            rows.append({
                "precision@5": precision_at_k(got, q["relevant_docs"], 5),
                "recall@5": recall_at_k(got, q["relevant_docs"], 5),
                "MRR": mrr(got, q["relevant_docs"]),
            })
        return pd.DataFrame(rows).mean().to_dict()

    m_base = metrics_for(search_baseline)
    m_hyb = metrics_for(search_hybrid)
    print("baseline", {k: round(v, 3) for k, v in m_base.items()})
    print("hybrid  ", {k: round(v, 3) for k, v in m_hyb.items()})

    print("гружу reranker...")
    reranker = CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        device=device,
    )

    def search_rerank(query: str, k: int = 5):
        return rerank_search(embedder, col, reranker, query, k=k, n_cand=20)

    m_rr = metrics_for(search_rerank)
    print("rerank  ", {k: round(v, 3) for k, v in m_rr.items()})

    # hybrid тут чуть хуже baseline; CE поднимает R@5 выше 0.85 — его и беру
    search_final = search_rerank
    m_final = m_rr
    print("final (=rerank)", {k: round(v, 3) for k, v in m_final.items()})

    summary = {
        "chosen_chunking": chosen,
        "final_retriever": "dense_top20 + bge-reranker-v2-m3 → top5",
        "threshold": thr,
        "baseline": m_base,
        "hybrid": m_hyb,
        "rerank": m_rr,
        "final": m_final,
        "recall_curve_baseline_vecs": {str(k): v for k, v in curve.items()},
        "n_chunks": len(texts),
        "n_docs_clean": len(cleaned),
    }
    (ART / "retrieval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(ART / "vecs_final.npy", vecs)
    (ART / "chunks_final.json").write_text(
        json.dumps(
            [{"text": t, "doc_id": d, **m} for t, d, m in zip(texts, doc_ids, metas)],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\nСохранено в", ART)


if __name__ == "__main__":
    main()
