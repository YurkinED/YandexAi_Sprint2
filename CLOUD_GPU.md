# Прогон на GPU (Tesla T4)

Задача: прогнать пайплайн на машине с NVIDIA Tesla T4 (~16 GB VRAM) и получить `pilot_ready: true`.

## Требования

- NVIDIA driver + CUDA
- Python 3.10+, зависимости из `requirements.txt`
- OpenAI-compatible LLM на `http://localhost:8080/v1` (модель ~3B, например `Qwen/Qwen2.5-3B-Instruct`)

## Запуск

```bash
# этапы 1–4: эмбеддинги и реранк на GPU
python run_stages_1_4.py

# этапы 5–6: LLM на GPU; retrieve на CPU, чтобы не делить VRAM с моделью
FILMOSKOP_RETRIEVE_DEVICE=cpu python run_stages_5_6.py
```

Итог: `artifacts/verdict.json`. Копия зачётного прогона — `artifacts/t4_run/`.

## Результат

| Метрика | Значение | Порог |
|---|---|---|
| recall@5 (non-gap) | 0.929 | ≥ 0.85 |
| faithfulness / relevance | 4.26 / 4.53 | шкалы ок |
| correctness (все 66) | 0.924 | ≥ 0.90 |
| gold correctness | 0.939 | ≥ final |
| gap correctness | 1.0 | — |
| `pilot_ready` | true | true |
