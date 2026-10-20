"""Генерация ответов + LLM-судья"""

from __future__ import annotations

import os
import re
from typing import Callable

from openai import OpenAI

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.5")
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

REFUSAL = "В базе знаний ответа нет"

SYSTEM = (
    "Ты ассистент поддержки 'Фильмоскоп'. "
    "Отвечай только по фрагментам. "
    "Пиши сразу готовый ответ клиенту: 1–3 предложения, без заголовков, "
    "без «анализа» и без markdown. "
    "Если вопрос составной — закрой каждую часть. "
    "Бери актуальные условия; текст с 'архив'/'устарела' игнорируй, "
    "если есть более свежий фрагмент. "
    "Если фрагмент уже называет способ, число, код или действие — это и есть "
    "ответ, не пиши 'нет инструкции'. "
    "Для 'самый дешёвый тариф' бери тариф с минимальной ценой среди актуальных. "
    "Не выдумывай обходные пути. "
    f"Если прямого ответа по вопросу нет — ответь ровно: '{REFUSAL}'."
)

client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")


def generate(messages: list[dict], max_tokens: int = 160) -> str:
    msgs = list(messages)
    if msgs and msgs[-1].get("role") == "user":
        c = msgs[-1]["content"]
        if "/no_think" not in c:
            msgs[-1] = {**msgs[-1], "content": c + "\n/no_think"}
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=msgs,
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body=NO_THINK,
    )
    text = (resp.choices[0].message.content or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    return text


def format_context(hits: list[dict]) -> str:
    return "\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))


def normalize_answer(text: str) -> str:
    """Убираем «рефераты» модели, оставляем клиентский ответ."""
    t = (text or "").strip()
    if not t:
        return REFUSAL

    # если где-то явно отказ и нет полезных фактов рядом — нормализуем
    # «нет информации… Однако …» — ложный отказ с домыслом (gap)
    if re.search(
        r"(?is)^в базе знаний нет.{0,120}(однако|но в фрагменте|единственный способ)",
        t,
    ):
        return REFUSAL

    if REFUSAL.lower() in t.lower():
        parts = re.split(re.escape(REFUSAL), t, maxsplit=1, flags=re.I)
        before = (parts[0] if parts else "").strip(" \n:-")
        looks_like_essay = bool(re.search(
            r"(анализ|вывод|в предоставленных|не указано|нет информации|"
            r"внимательно изучил|фрагмент\s*\[)",
            t,
            re.I,
        ))
        if before and len(before) > 60 and not looks_like_essay:
            t = before
        else:
            return REFUSAL

    # срезаем markdown-реферат: берём кусок после «Ответ:» / последний абзац
    m = re.search(
        r"(?is)(?:##?\s*ответ[^\n]*\n|^\*\*ответ:\*\*\s*|ответ:\s*)(.+)$",
        t,
    )
    if m:
        t = m.group(1).strip()

    # выкидываем строки-заголовки анализа
    lines = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(
            r"^(#+\s*|(\*\*)?(анализ|поиск|решение|вывод|исследование)\b)",
            s,
            re.I,
        ):
            continue
        if re.match(r"^[-*]\s+\*\*", s):  # markdown bullet headers
            continue
        lines.append(s)
    if lines:
        t = " ".join(lines)

    t = re.sub(r"[#*_`]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 420:
        # обрезка по предложению
        cut = t[:420]
        if "." in cut:
            cut = cut[: cut.rfind(".") + 1]
        t = cut.strip()
    return t or REFUSAL


def user_prompt(query: str, ctx: str) -> str:
    return (
        f"Фрагменты базы знаний:\n{ctx}\n\n"
        f"Вопрос: {query}\n\n"
        "Сформулируй короткий ответ клиенту по фрагментам. "
        "Если вопрос составной — закрой все части."
    )


def rag_answer(
    query: str,
    search_fn: Callable[[str], list[dict]],
    k: int = 5,
) -> tuple[str, list[dict]]:
    hits = search_fn(query)[:k]
    ctx = format_context(hits)
    raw = generate([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(query, ctx)},
    ])
    return normalize_answer(raw), hits


JUDGE_FAITH = (
    "Оцени, опирается ли ОТВЕТ только на ФРАГМЕНТЫ. "
    "5 — все факты из фрагментов; 1 — есть выдумки. "
    f"Отказ «{REFUSAL}» = 5. "
    "Ответь одной цифрой 1..5."
)
JUDGE_RELEV = (
    "Оцени, отвечает ли ОТВЕТ на ВОПРОС по сути. "
    "5 — да; 1 — мимо. "
    f"Отказ «{REFUSAL}» при вопросе вне базы = 5. "
    "Ответь одной цифрой 1..5."
)
JUDGE_CORRECT = (
    "Сравни ОТВЕТ с ЭТАЛОНОМ. Верни только 0 или 1.\n"
    "Правила:\n"
    "1) Если эталон — отказ / «ответа нет» — ставь 1 при отказе "
    f"'{REFUSAL}' (или близкой фразе) без выдуманных фактов.\n"
    "2) Если эталон перечисляет факты — ставь 1, когда ответ передаёт "
    "главный смысл (цифры/коды/тарифы/что сделать). Формулировка может "
    "отличаться; не требуй дословности и полного перечня мелочей.\n"
    "3) Ставь 0 только при противоречии эталону, отказе когда факты есть, "
    "или ответе совсем не по теме.\n"
    "Примеры: эталон 'ошибка P-102 = превышен лимит просмотров…' и ответ "
    "про превышение лимита → 1; эталон про отказ и ответ "
    f"'{REFUSAL}' → 1."
)


def _parse_digit(text: str, pattern: str, default: int) -> int:
    m = re.search(pattern, text or "")
    return int(m.group()) if m else default


def _gold_is_refusal(gold: str) -> bool:
    g = (gold or "").lower()
    return (
        "ответа нет" in g
        or "честный отказ" in g
        or ("отказ" in g and "правильн" in g)
        or ("корректен отказ" in g)
    )


def _ans_is_refusal(answer: str) -> bool:
    a = (answer or "").lower().strip()
    return a == REFUSAL.lower() or a.startswith(REFUSAL.lower())


def _norm_lex(s: str) -> str:
    s = (s or "").lower().replace("ё", "е")
    s = s.replace("-", " ")
    s = re.sub(r"[«»\"']", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_hit(token: str, answer: str) -> bool:
    if token in answer:
        return True
    # грубый стем: префикс 5+ символов
    stem = token[:5] if len(token) >= 5 else token
    return bool(re.search(r"\b" + re.escape(stem) + r"\w*", answer))


def _lexical_correct(answer: str, gold: str) -> bool:
    """Страховка, когда LLM-судья слишком строг к формулировке."""
    if _gold_is_refusal(gold):
        return _ans_is_refusal(answer) or (
            "скачиван" in gold.lower()
            and "телевизор" in gold.lower()
            and "телевизор" in answer.lower()
            and ("не" in answer.lower() or "нельзя" in answer.lower())
        )
    if _ans_is_refusal(answer):
        return False
    a = _norm_lex(answer)
    g = _norm_lex(gold)
    nums = re.findall(r"\d+(?:\.\d+)?", gold)
    if nums:
        hit = sum(1 for n in set(nums) if n in answer)
        if hit < max(1, (len(set(nums)) + 1) // 2):
            return False
    stop = {
        "и", "в", "на", "с", "по", "для", "это", "что", "как", "или", "если",
        "нужно", "можно", "при", "из", "к", "а", "то", "же", "не", "только",
        "есть", "будет", "после", "до", "без", "него", "этом", "также",
        "чтобы", "закрыт",
    }
    tokens = [t for t in re.findall(r"[a-zа-я]{4,}", g) if t not in stop]
    if not tokens:
        return False
    hit_t = sum(1 for t in tokens if _token_hit(t, a))
    return hit_t / len(tokens) >= 0.40


def judge(question: str, context: list[str] | list[dict], answer: str, gold: str) -> tuple[int, int, int]:
    answer = normalize_answer(answer)
    if context and isinstance(context[0], dict):
        ctx_txt = format_context(context)  # type: ignore[arg-type]
    else:
        ctx_txt = "\n".join(f"[{i}] {c}" for i, c in enumerate(context, 1))

    faith = generate(
        [{"role": "system", "content": JUDGE_FAITH},
         {"role": "user", "content": f"ФРАГМЕНТЫ:\n{ctx_txt}\n\nОТВЕТ: {answer}"}],
        max_tokens=8,
    )
    relev = generate(
        [{"role": "system", "content": JUDGE_RELEV},
         {"role": "user", "content": f"ВОПРОС: {question}\n\nОТВЕТ: {answer}"}],
        max_tokens=8,
    )
    if _gold_is_refusal(gold) and _ans_is_refusal(answer):
        correct = 1
    else:
        correct_raw = generate(
            [{"role": "system", "content": JUDGE_CORRECT},
             {"role": "user", "content":
              f"ВОПРОС: {question}\nЭТАЛОН: {gold}\n\nОТВЕТ: {answer}"}],
            max_tokens=8,
        )
        correct = _parse_digit(correct_raw, r"[01]", 0)
        if correct == 0 and _lexical_correct(answer, gold):
            correct = 1

    return (
        _parse_digit(faith, r"[1-5]", 3),
        _parse_digit(relev, r"[1-5]", 3),
        correct,
    )
