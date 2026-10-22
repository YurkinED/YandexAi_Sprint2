"""Этапы 5–6. Нужны artifacts/ из 1–4 и LLM на :8080."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts"

from pipeline.retrieve import build_chroma, rerank_search
from pipeline.evaluate import evaluate_rag_fixed_hits, gold_context_run, profile
from pipeline.generate import REFUSAL

SEED = 42
np.random.seed(SEED)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads((ROOT / "filmoskop_kb.json").read_text(encoding="utf-8"))
    queries = data["queries"]
    cleaned = json.loads((ART / "corpus_clean.json").read_text(encoding="utf-8"))
    chunks = json.loads((ART / "chunks_final.json").read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    doc_ids = [c["doc_id"] for c in chunks]
    metas = chunks
    vecs = np.load(ART / "vecs_final.npy")
    summary = json.loads((ART / "retrieval_summary.json").read_text(encoding="utf-8"))

    import os
    device = os.environ.get(
        "FILMOSKOP_RETRIEVE_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    print(f"retrieve device={device}")
    embedder = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
    _, col = build_chroma(texts, metas, vecs, path=None)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device)

    def search_final(query: str, k: int = 5):
        return rerank_search(embedder, col, reranker, query, k=k, n_cand=20)

    t0 = time.perf_counter()
    for q in queries[:10]:
        search_final(q["query"])
    lat = (time.perf_counter() - t0) / 10 * 1000
    print(f"поиск ~{lat:.0f} мс/запрос (среднее по 10)")

    print("\n[5-6] полный прогон")
    report = evaluate_rag_fixed_hits(queries, search_final)
    report.to_pickle(ART / "final_report.pkl")
    report.to_csv(ART / "final_report.csv", index=False, encoding="utf-8-sig")
    profile(report)

    gap = report[report["kind"] == "gap"]
    gap_ok = gap["correctness"].mean() if len(gap) else float("nan")
    print(f"\ngap correctness: {gap_ok:.3f}")

    answered = report[report["kind"] != "gap"]
    print("retrieval non-gap:", answered[["precision@5", "recall@5", "MRR"]].mean().round(3).to_dict())

    print("\nзолотой контекст (чанки relevant_docs, тот же формат что RAG)")
    docs_by_id = {d["doc_id"]: d for d in cleaned}
    gold = gold_context_run(docs_by_id, queries, chunks=chunks, max_chunks=5)
    gold.to_pickle(ART / "gold_report.pkl")
    gold.to_csv(ART / "gold_report.csv", index=False, encoding="utf-8-sig")
    print("gold", round(float(gold["correctness"].mean()), 3))
    print("final", round(float(report["correctness"].mean()), 3))
    print("разрыв", round(float(gold["correctness"].mean() - report["correctness"].mean()), 3))
    print("gold faith/relev",
          round(float(gold["faithfulness"].mean()), 3),
          round(float(gold["answer relevance"].mean()), 3))

    r5 = float(answered["recall@5"].mean())
    corr = float(report["correctness"].mean())
    gold_corr = float(gold["correctness"].mean())
    faith_m = float(report["faithfulness"].mean())
    relev_m = float(report["answer relevance"].mean())
    gold_gap = gold_corr - corr
    by_kind = (
        report.groupby("kind")[["faithfulness", "answer relevance", "correctness"]]
        .mean(numeric_only=True)
        .round(3)
        .to_dict()
    )
    verdict = {
        "recall@5_non_gap": r5,
        "correctness_all": corr,
        "gap_correctness": float(gap_ok),
        "gold_correctness": gold_corr,
        "gold_minus_final_correctness": gold_gap,
        "faithfulness_mean": faith_m,
        "answer_relevance_mean": relev_m,
        "judge_by_kind": by_kind,
        "target_recall": 0.85,
        "target_correctness": 0.90,
        "recall_ok": bool(r5 >= 0.85),
        "correctness_ok": bool(corr >= 0.90),
        "gap_ok": bool(gap_ok >= 0.8),
        "judge_scales_ok": bool(faith_m >= 3.5 and relev_m >= 3.5),
        "pilot_ready": bool(r5 >= 0.85 and corr >= 0.90 and gap_ok >= 0.8),
        "search_latency_ms_est": lat,
        "retrieval_summary": summary,
        "gold_note": (
            "Золотой контекст = чанки эталонных разделов (не целые статьи). "
            + (
                "Разрыв отрицательный: проверить шум судьи или генерацию; "
                "полный раздел как контекст раньше занижал gold относительно final."
                if gold_gap < -1e-9
                else "gold_correctness >= final — узкое место в основном поиск/ранжирование."
            )
        ),
        "judge_note": (
            "Судья: отдельные короткие вызовы без /no_think в user; "
            "парсер игнорирует эхо шкалы «от 1 до 5»; "
            "лексический fallback только если 0/1 не распарсились."
        ),
        "recommendations": [
            "multi: доп. подзапросы + diversify по doc_id; если R@5 всё ещё <1 — MMR/явный второй раздел",
            "на gen не уходить в «нет инструкции», если способ уже назван во фрагменте",
            "если faith/relev снова около 1 при верных ответах — смотреть сырой ответ судьи и парсер",
        ],
    }
    (ART / "verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nвердикт")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
