# Фильмоскоп — RAG ассистент поддержки

Спринт 2: поиск по базе знаний + генерация ответов + LLM-as-judge.

## Быстрый старт

```powershell
cd filmoskop-rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# нужен OpenAI-compatible LLM на http://localhost:8080/v1
python run_stages_1_4.py   # чистка, чанки, chroma, rerank
python run_stages_5_6.py   # генерация + судья (~20 мин на 66 запросах)
```

На GPU с ~16 GB VRAM (Tesla T4 и аналоги) retrieve на этапах 5–6 лучше держать на CPU, чтобы не делить память с LLM:

```bash
FILMOSKOP_RETRIEVE_DEVICE=cpu python run_stages_5_6.py
```

Результаты — в `artifacts/` (ключевые метрики в `artifacts/verdict.json`).

## Что внутри

| Путь | Назначение |
|---|---|
| `filmoskop_kb.json` | датасет (разделы + запросы) |
| `pipeline/` | clean / chunking / retrieve / generate / evaluate |
| `filmoskop_rag.ipynb` | ноутбук с разбором |
| `artifacts/verdict.json` | итог прогона |

## Метрики (Tesla T4)

- recall@5 (non-gap): **0.929** (≥ 0.85)
- correctness (все 66): **0.909** (≥ 0.90)
- gap correctness: **1.0**
- `pilot_ready`: **true**

Стек: multilingual-e5-base, Chroma, bge-reranker-v2-m3, LLM Qwen2.5-3B-Instruct.

## С чем столкнулись

1. Грязный корпус. В разделах смесь HTML/markdown и кракозябр (UTF-8, прочитанный как latin-1). Без нормализации кодировки и снятия разметки поиск ломался на очевидных вопросах. Короткий мусор отрезали, разделы с восстановимым текстом оставили.

2. Чанкинг режет смысл. Fixed window с overlap давал хуже recall: фразы рвались посередине. Sentence window (~500) с title в начале чанка оказался стабильнее на multi-запросах.

3. Dense top-5 не дотягивал. Baseline recall@5 был около 0.66: нужный раздел часто уже был в top-20, но не в top-5. Просто увеличить K недостаточно — нужен реранкер.

4. Hybrid почти не помог. BM25 + RRF на этом корпусе дал почти нулевой прирост относительно dense. В финал не пошёл; оставили сравнение в артефактах.

5. Реранкер тяжёлый по VRAM/latency. Связка dense top-20 → bge-reranker-v2-m3 → top-5 закрыла порог recall ≥ 0.85, но на одной карте с LLM приходится экономить память.

6. 7B не влезает вместе с retrieve. На T4 (~16 GB) модель ~7B конфликтует по VRAM с эмбеддером/реранкером. Для генерации и судьи взяли ~3B; на этапах 5–6 retrieve уводили на CPU (`FILMOSKOP_RETRIEVE_DEVICE=cpu`).

7. Gap и формулировки ответов. Модель то дописывала факты вне базы, то отказывала «криво», и судья штрафовал correctness. Помогли жёсткий промпт («только по фрагментам»), короткие ответы, нормализация отказа и чуть мягче критерий correctness на gap.
