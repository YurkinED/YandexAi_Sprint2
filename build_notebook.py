"""Сборка ноутбука для сдачи (filmoskop_rag.ipynb)."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import nbformat as nbf
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nbformat", "-q"])
    import nbformat as nbf

ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Спринт 2 — RAG для поддержки «Фильмоскоп»

Делал без LangChain: проще следить за префиксами e5 и за тем, что уходит в промпт.

Ориентиры из задания: recall@5 >= 0.85 на non-gap, correctness >= 0.90, на gap — отказ без фантазий.
""")

md("""
## Окружение
""")

code("""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path('.').resolve()
sys.path.insert(0, str(ROOT))
ART = ROOT / 'artifacts'
ART.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print('cuda:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
""")

md("""
## 1. Корпус

В сырых разделах каша: html, markdown, плюс куски где utf-8 прочитали как latin-1 (ÐÑ…).
Сначала чиню кодировку и снимаю разметку. Выкидываю только совсем короткие заглушки —
разделы с кракозябрами не мусор, их можно восстановить.
""")

code("""
from pipeline.clean import clean_corpus, save_cleaned, stats_to_dict

data = json.loads((ROOT / 'filmoskop_kb.json').read_text(encoding='utf-8'))
docs_raw, queries = data['docs'], data['queries']
print('разделов', len(docs_raw), 'запросов', len(queries))

cleaned, stats = clean_corpus(docs_raw)
save_cleaned(cleaned, ART / 'corpus_clean.json')
print(stats_to_dict(stats))
print('было → стало (пара примеров):')
for ex in stats.examples[:2]:
    print('---', ex['flag'], ex['title'])
    print('BEFORE:', ex['before'][:160])
    print('AFTER:', ex['after'][:160])
""")

md("""
Осталось ~179 из 181. Два раздела без нормального текста (схема/таблица) — дропнул.
MIN_CHARS=60 взял на глаз по гистограмме длин.
""")

md("""
## 2. Чанкинг

Сравнил fixed+overlap и sentence+overlap при похожем размере окна.
Эмбеддер — multilingual-e5-base. Считаю метрики по doc_id раздела, не по чанкам
(как в уроках спринта).
""")

code("""
# тяжёлый прогон (эмбеды + chroma + rerank) вынес в скрипт:
# !python run_stages_1_4.py
summary = json.loads((ART / 'retrieval_summary.json').read_text(encoding='utf-8'))
chunk = json.loads((ART / 'chunk_compare.json').read_text(encoding='utf-8'))
print(json.dumps(chunk, ensure_ascii=False, indent=2))
print('беру', summary['chosen_chunking'])
""")

md("""
sentence_500 выиграл по recall@5 — границы предложений целее, плюс title в начале чанка.
fixed режет фразы посередине, это заметно на multi.
""")

md("""
## 3. Индекс

Chroma с cosine/HNSW. Корпус маленький, нужны фильтры по category, отдельный сервис не хотел поднимать.
Если бы чанков было сильно больше — смотрел бы FAISS/Qdrant, тут оверкилл.
""")

code("""
print('baseline', summary['baseline'])
print('recall@K', summary['recall_curve_baseline_vecs'])
print('порог', round(summary['threshold'], 3))
""")

md("""
Baseline dense около 0.66 по recall@5 (в курсе ориентир ~0.67, ок).
По кривой видно: релевантное часто есть в top-20, но не в top-5 — значит реранкер логичнее, чем просто крутить K.
""")

md("""
## 4. Улучшение поиска

Пробовал hybrid BM25+RRF — на этом корпусе почти не дал. Итоговый вариант:
dense top-20 → bge-reranker-v2-m3 → top-5.
""")

code("""
print('baseline', {k: round(v,3) for k,v in summary['baseline'].items()})
print('hybrid  ', {k: round(v,3) for k,v in summary['hybrid'].items()})
print('rerank  ', {k: round(v,3) for k,v in summary['rerank'].items()})
print('final   ', summary.get('final_retriever'), {k: round(v,3) for k,v in summary['final'].items()})
""")

md("""
С реранкером recall@5 перешагнул 0.85. Цена — тяжёлый CE и лишняя latency, но прирост измеримый, оставляю.
Hybrid оставил в сравнении, в final не пошёл.
""")

md("""
## 5–6. Генерация и судья

Промпт: отвечать только по фрагментам, иначе «В базе знаний ответа нет».
Судья — faithfulness / relevance / correctness (шкалы как в уроках).
Парсинг оценки: берём цифру из «Оценка: N» или последнюю изолированную 1..5,
а не первую из текста вроде «от 1 до 5» — иначе relevance залипала на 1.
Полный прогон: `python run_stages_5_6.py` (нужен LLM на :8080).
""")

code("""
verdict_path = ART / 'verdict.json'
report_path = ART / 'final_report.pkl'
if verdict_path.exists():
    verdict = json.loads(verdict_path.read_text(encoding='utf-8'))
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    if report_path.exists():
        report = pd.read_pickle(report_path)
        print()
        cols = ['precision@5','recall@5','MRR','faithfulness','answer relevance','correctness']
        print(report[cols].mean(numeric_only=True).round(3))
        print(report.groupby('kind')[cols].mean(numeric_only=True).round(3))
else:
    print('сначала прогони: python run_stages_5_6.py')
""")

md("""
### Вердикт

Ориентиры пилота и шкалы судьи — в `artifacts/verdict.json`.

По замечанию ревью: раньше faithfulness/relevance залипали около 1–2 из‑за
парсинга эха шкалы «от 1 до 5» у маленькой LLM. Сейчас отдельный вызов судьи,
парсер чистит эхо шкалы, на exact relevance ≈ 4.7. Золотой контекст берём
чанками relevant_docs (тот же формат, что RAG) — иначе целые статьи занижали gold
относительно final.

Если ещё крутить: добить multi (полнота ответа) и gap-галлюцинации 3B.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}
out = ROOT / "filmoskop_rag.ipynb"
out.write_text(nbf.writes(nb), encoding="utf-8")
print("Wrote", out)
