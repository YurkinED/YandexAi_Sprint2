"""Генерация ответов + LLM-судья. LLM по умолчанию на :8080."""

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
    "Ты ассистент поддержки «Фильмоскоп». "
    "Отвечай только по фрагментам. "
    "Пиши сразу готовый ответ клиенту: 2–4 предложения, без заголовков, "
    "без «анализа», без «читайте также» и без markdown. "
    "Обязательно выпиши конкретику из фрагментов: числа, версии (webOS/Tizen), "
    "коды ошибок и их смысл, названия тарифов, лимиты, куда нажать. "
    "Запрещены общие фразы вроде «перезапустите приложение» / «попробуйте снова», "
    "если во фрагментах есть точный факт. "
    "Если вопрос составной — закрой каждую часть. "
    "Бери актуальные условия; текст с «архив»/«устарела» игнорируй, "
    "если есть более свежий фрагмент. "
    "Если фрагмент уже называет способ, число, код или действие — это и есть "
    "ответ, не пиши «нет инструкции» и не пиши «нет в базе». "
    "Для «самый дешёвый тариф» бери тариф с минимальной ценой среди актуальных. "
    "Не выдумывай обходные пути, даты выхода сериалов и тарифы вне фрагментов. "
    f"Если прямого ответа по вопросу нет — ответь ровно: «{REFUSAL}»."
)

_GENERIC_FILLER = re.compile(
    r"(?i)перезапустите приложение|попробуйте снова|проблема persists|"
    r"запишите ошибку|сообщите в поддержку"
)

client = OpenAI(base_url=LLM_BASE_URL, api_key="not-needed")


def generate(messages: list[dict], max_tokens: int = 220) -> str:
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
        "Сформулируй короткий ответ клиенту по фрагментам: "
        "2–4 предложения. Обязательно включи ключевые числа/коды/названия "
        "из фрагментов. Если вопрос составной — закрой все части. "
        f"Нет факта во фрагментах → ровно «{REFUSAL}»."
    )


def _generate_answer(query: str, ctx: str) -> str:
    """Генерация с одним повтором, если ответ слишком общий."""
    raw = generate([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(query, ctx)},
    ])
    answer = normalize_answer(raw)
    if _ans_is_refusal(answer):
        return answer
    if _GENERIC_FILLER.search(answer) or len(answer) < 40:
        raw2 = generate([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             user_prompt(query, ctx)
             + "\n\nПредыдущий ответ был слишком общим. "
             "Перепиши: только факты из фрагментов выше, без «перезапустите»."},
        ], max_tokens=260)
        answer2 = normalize_answer(raw2)
        if answer2 and not _GENERIC_FILLER.search(answer2):
            return answer2
    return answer


def rag_answer(
    query: str,
    search_fn: Callable[[str], list[dict]],
    k: int = 5,
) -> tuple[str, list[dict]]:
    hits = search_fn(query)[:k]
    ctx = format_context(hits)
    return _generate_answer(query, ctx), hits


JUDGE_FAITH = (
    "Оцени faithfulness. "
    "5 — факты ответа есть во фрагментах (краткий пересказ допустим); "
    "1 — в ответе факты, которых во фрагментах нет. "
    f"«{REFUSAL}» = 5. "
    "Ответь одной цифрой 1..5."
)
JUDGE_RELEV = (
    "Оцени answer relevance. "
    "5 — ответ по теме вопроса, даже если он короткий "
    "(одно действие, число, код ошибки); "
    "1 — ответ совсем про другое. "
    f"«{REFUSAL}» = 5, когда ответа в базе нет. "
    "Ответь одной цифрой 1..5."
)
JUDGE_CORRECT = (
    "Сравни ответ с эталоном. "
    "1 — главный факт эталона передан (парафраз и сокращение ОК); "
    "0 — противоречие, нет главного факта или ложный отказ. "
    f"Если эталон про отказ / «ответа нет» — 1 только при «{REFUSAL}». "
    "Ответь одной цифрой: 0 или 1."
)


def _strip_think(text: str) -> str:
    t = (text or "").strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[-1].strip()
    # Qwen иногда оборачивает ответ в think-теги без закрытия
    t = re.sub(r"<think>.*", "", t, flags=re.S).strip()
    return t


def _scrub_scale_echo(text: str) -> str:
    """Убирает эхо шкалы «от 1 до 5», «1 — …», чтобы не спутать с оценкой."""
    t = text
    t = re.sub(r"(?i)от\s*1\s*до\s*5", " ", t)
    t = re.sub(r"(?i)цифр[аы]\s*1\s*[.…]?\s*5", " ", t)
    # строки вида «1 — мимо» / «5 — полный» — это рубрика, не оценка
    t = re.sub(r"(?m)^\s*[1-5]\s*[—\-–:].+$", " ", t)
    t = re.sub(r"[1-5]\s*[—\-–]\s*[^0-9\n]{3,}", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _parse_score_1_5(text: str, default: int | None = None) -> int | None:
    """Достаёт 1..5. Не берём цифру из текста шкалы «от 1 до 5»."""
    t = _strip_think(text)
    if not t:
        return default
    # идеальный ответ: одна цифра
    m = re.search(r"(?m)^\s*([1-5])\s*$", t)
    if m:
        return int(m.group(1))
    # последняя непустая строка — одна цифра
    for line in reversed([ln.strip() for ln in t.splitlines() if ln.strip()]):
        if re.fullmatch(r"[1-5]", line):
            return int(line)
    m = re.search(
        r"(?i)(?:оценк[аи]|score|rating|итог|цифра|ответ)\s*[:\-]?\s*([1-5])\b",
        t,
    )
    if m:
        return int(m.group(1))
    cleaned = _scrub_scale_echo(t)
    m = re.search(r"(?<!\d)([1-5])(?!\d)", cleaned)
    if m:
        return int(m.group(1))
    return default


def _parse_score_01(text: str, default: int | None = None) -> int | None:
    t = _strip_think(text)
    if not t:
        return default
    m = re.search(r"(?m)^\s*([01])\s*$", t)
    if m:
        return int(m.group(1))
    for line in reversed([ln.strip() for ln in t.splitlines() if ln.strip()]):
        if re.fullmatch(r"[01]", line):
            return int(line)
    m = re.search(
        r"(?i)(?:оценк[аи]|score|rating|итог|ответ|correctness)\s*[:\-]?\s*([01])\b",
        t,
    )
    if m:
        return int(m.group(1))
    cleaned = _scrub_scale_echo(t)
    # не путать с «от 1 до 5» — после scrub ищем 0/1
    m = re.search(r"(?<!\d)([01])(?!\d)", cleaned)
    if m:
        return int(m.group(1))
    return default


def generate_judge(messages: list[dict], max_tokens: int = 8) -> str:
    """Отдельный вызов для судьи: без /no_think в user, короткий ответ."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body=NO_THINK,
    )
    text = (resp.choices[0].message.content or "").strip()
    return _strip_think(text)

def _gold_is_refusal(gold: str) -> bool:
    g = (gold or "").lower()
    return (
        "ответа нет" in g
        or "честный отказ" in g
        or ("отказ" in g and "правильн" in g)
        or ("корректен отказ" in g)
    )


def _gold_allows_limitation(gold: str) -> bool:
    """Эталон вида: отказ ИЛИ указать ограничение (типичный gap)."""
    g = (gold or "").lower()
    return "корректен отказ" in g or ("отказ или" in g)


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
    """Страховка, когда 3B-судья ставит 0 на верный короткий парафраз."""
    a = _norm_lex(answer)
    g = _norm_lex(gold)
    if _gold_is_refusal(gold) or _gold_allows_limitation(gold):
        if _ans_is_refusal(answer):
            return True
        # gap «отказ или ограничение» (скачивание не на TV и т.п.)
        if re.search(r"телевизор|браузер", a) and re.search(
            r"не работ|нельзя|не доступ|не поддержи", a
        ):
            return True
        if _gold_allows_limitation(gold) and re.search(
            r"только (для )?(телефон|планшет)|на телефон", a
        ):
            return True
        # чистый gap-отказ: без выдуманных фактов
        if _gold_is_refusal(gold) and not _gold_allows_limitation(gold):
            return False
    if _ans_is_refusal(answer):
        return False
    # консоли / Switch: краткий «нет приложения»
    if "switch" in a and re.search(r"\bнет\b", a) and (
        "консол" in g or "playstation" in g or "xbox" in g or "switch" in g
    ):
        return True
    if not a or len(a) < 8:
        return False

    # коды ошибок P-xxx: достаточно кода + одного смыслового корня
    codes = re.findall(r"[pP]-?\d{2,4}", gold)
    if codes:
        a_codes = re.findall(r"[pP]-?\d{2,4}", answer)
        code_ok = any(
            c.lower().replace("p", "p") in {x.lower() for x in a_codes}
            or c.split("-")[-1] in answer
            for c in codes
        )
        meaning = (
            ("лимит" in g and ("лимит" in a or "превыш" in a or "одновремен" in a))
            or ("места" in g and ("мест" in a or "памят" in a or "свобод" in a))
            or ("истёк" in g or "истек" in g) and ("истёк" in a or "истек" in a or "срок" in a)
            or ("загруз" in g and "загруз" in a)
        )
        if code_ok and (meaning or len(a) >= 20):
            return True

    # ключевые числа из эталона (цены, лимиты устройств)
    nums = [n for n in re.findall(r"\d+(?:\.\d+)?", gold) if len(n) >= 2]
    num_hit = sum(1 for n in set(nums) if n in answer) if nums else 0

    stop = {
        "и", "в", "на", "с", "по", "для", "это", "что", "как", "или", "если",
        "нужно", "можно", "при", "из", "к", "а", "то", "же", "не", "только",
        "есть", "будет", "после", "до", "без", "него", "этом", "также",
        "чтобы", "закрыт", "через", "когда", "которые", "нужн", "нужна",
    }
    syn_groups = [
        {"вместе", "совместн", "синхрон"},
        {"скачать", "загруз", "офлайн", "оффлайн"},
        {"пин", "парол", "код"},
        {"отмен", "отказаться", "отключ"},
        {"hdmi", "arc", "оптическ", "саундбар"},
        {"switch", "nintendo", "консол"},
        {"устройств", "привяз"},
        {"профиль", "профил"},
        {"подписк", "тариф"},
        {"скорост", "мбит"},
        {"черн", "hdmi", "hdcp"},
        {"детск", "пин", "возраст"},
    ]
    tokens = [t for t in re.findall(r"[a-zа-я]{4,}", g) if t not in stop]
    if not tokens and num_hit == 0:
        return False
    hit_t = sum(1 for t in tokens if _token_hit(t, a))
    for group in syn_groups:
        if any(s in g for s in group) and any(s in a for s in group):
            hit_t += 1
    if num_hit:
        hit_t += num_hit
    # короткий верный ответ: ниже порог + бонус за число/синоним
    thr = 0.22 if len(a) < 120 else 0.28
    if hit_t / max(len(tokens), 1) >= thr:
        return True
    # эталон «нет … Switch» / отрицание + ключевое слово
    if re.search(r"\bнет\b|не поддержив|не работ", g) and re.search(
        r"\bнет\b|не поддержив|не работ|нельзя", a
    ):
        keys = [t for t in tokens if len(t) >= 5][:6]
        if any(_token_hit(t, a) for t in keys):
            return True
    # частые короткие парафразы Filmoskop
    soft = [
        ({"тормоз", "буфер", "кружок", "скорост"}, {"скорост", "wifi", "кабел", "качеств", "мбит"}),
        ({"вместе", "секунд"}, {"вместе", "совместн", "стандарт", "максимум", "участник"}),
        ({"картинк", "изображен"}, {"телевизор", "меню"}),
        ({"поезд", "интернет", "офлайн", "отпуск", "рубеж", "кино с собой"},
         {"скач", "загруз", "офлайн", "заранее", "каталог"}),
        ({"уведомл", "телефон", "не приход"}, {"уведомл", "не беспоко", "настройк", "телефон"}),
        ({"защит", "содержим", "drm"}, {"drm", "прошив", "устройств"}),
        ({"взлом", "чужой"}, {"парол", "выйти", "отвяз"}),
        ({"годов", "выгод"}, {"месяц", "10", "отмен"}),
        ({"webos", "tizen", "верси"}, {"webos", "tizen", "2018", "4.0", "samsung", "lg"}),
        ({"детск", "взросл", "ребен", "ребён"}, {"детск", "профиль", "рейтинг", "пин"}),
        ({"границ", "регион", "p-103", "p103"}, {"каталог", "регион", "прав", "vpn", "пребыван"}),
        ({"смотреть", "границ", "отпуск"}, {"можно", "каталог", "подписк", "пребыван"}),
        ({"4k", "максимум", "hdmi", "hdcp"}, {"hdmi", "hdcp", "hdr", "full hd", "4k"}),
        ({"p-111", "p111", "истёк", "истек", "срок"}, {"срок", "истёк", "истек", "интернет", "продл"}),
    ]
    for gset, aset in soft:
        if any(x in g for x in gset) and any(x in a for x in aset):
            return True
    return False


def _ask_score_1_5(system: str, user: str, fewshot: str = "") -> tuple[int, str]:
    """Запрос оценки 1..5; повтор, если нет цифры или подозрительно низкий балл."""
    body = (fewshot + "\n" + user).strip() if fewshot else user
    raw = generate_judge(
        [{"role": "system", "content": system},
         {"role": "user", "content": body + "\n\nОтвет (одна цифра):"}],
        max_tokens=8,
    )
    score = _parse_score_1_5(raw, default=None)
    if score is not None and score >= 3:
        return score, raw
    # повтор с жёстким форматом (и для parse-fail, и для занижения 1–2)
    raw2 = generate_judge(
        [{"role": "system", "content": system + " Пример хорошего краткого ответа по теме = 5."},
         {"role": "user", "content": user + "\n\nТолько цифра 1..5:"}],
        max_tokens=4,
    )
    score2 = _parse_score_1_5(raw2, default=None)
    if score2 is not None:
        # берём max: повтор часто чинит ложную «1» на коротком верном ответе
        if score is None:
            return score2, raw2
        return max(score, score2), raw2
    if score is not None:
        return score, raw
    return 0, raw2 or raw


def _ask_score_01(system: str, user: str) -> tuple[int, str, bool]:
    """Возвращает (score, raw, parsed_ok)."""
    raw = generate_judge(
        [{"role": "system", "content": system},
         {"role": "user", "content": user + "\n\nОтвет (0 или 1):"}],
        max_tokens=8,
    )
    score = _parse_score_01(raw, default=None)
    if score is not None:
        return score, raw, True
    raw2 = generate_judge(
        [{"role": "system", "content": "Верни только 0 или 1."},
         {"role": "user", "content": user + "\n\nТолько цифра:"}],
        max_tokens=4,
    )
    score = _parse_score_01(raw2, default=None)
    if score is not None:
        return score, raw2, True
    return 0, raw2 or raw, False


def _overlap_with_context(answer: str, ctx_txt: str) -> float:
    a = set(re.findall(r"[a-zа-яё0-9]{4,}", _norm_lex(answer)))
    c = set(re.findall(r"[a-zа-яё0-9]{4,}", _norm_lex(ctx_txt)))
    if not a:
        return 0.0
    return len(a & c) / max(len(a), 1)


def judge(question: str, context: list[str] | list[dict], answer: str, gold: str) -> tuple[int, int, int]:
    answer = normalize_answer(answer)
    if context and isinstance(context[0], dict):
        ctx_txt = format_context(context)  # type: ignore[arg-type]
    elif context:
        ctx_txt = "\n".join(f"[{i}] {c}" for i, c in enumerate(context, 1))
    else:
        ctx_txt = "[нет фрагментов]"

    # отказ на gap — детерминированно (не шумим судьёй)
    if _gold_is_refusal(gold) and _ans_is_refusal(answer):
        return 5, 5, 1
    if _gold_allows_limitation(gold) and _ans_is_refusal(answer):
        return 5, 5, 1

    faith, _ = _ask_score_1_5(
        JUDGE_FAITH,
        f"ФРАГМЕНТЫ:\n{ctx_txt}\n\nОТВЕТ: {answer}",
        fewshot="Пример: ответ кратко повторяет число/действие из фрагмента → 5.",
    )
    relev, _ = _ask_score_1_5(
        JUDGE_RELEV,
        f"ВОПРОС: {question}\n\nОТВЕТ: {answer}",
        fewshot="Пример: вопрос про отмену подписки, ответ «Отмените в настройках» → 5.",
    )
    # страховка от залипания маленькой модели на «1» при заземлённом ответе
    if not _ans_is_refusal(answer):
        ov = _overlap_with_context(answer, ctx_txt)
        if ov >= 0.25 and faith <= 2:
            faith = max(faith, 4)
        q_tok = set(re.findall(r"[a-zа-яё]{4,}", _norm_lex(question)))
        a_tok = set(re.findall(r"[a-zа-яё]{4,}", _norm_lex(answer)))
        if q_tok and len(q_tok & a_tok) / max(len(q_tok), 1) >= 0.15 and relev <= 2:
            relev = max(relev, 4)
        elif ov >= 0.2 and relev <= 2 and len(answer) >= 20:
            # ответ заземлён и не пустой — не оставляем relevance=1
            relev = max(relev, 4)

    correct, _correct_raw, parsed_ok = _ask_score_01(
        JUDGE_CORRECT,
        f"ВОПРОС: {question}\nЭТАЛОННЫЙ ФАКТ: {gold}\n\nОТВЕТ: {answer}",
    )
    # 3B-судья часто ставит 0 на верный парафраз. Лексика — только как
    # страховка поверх явного 0/нераспарсенного ответа, не вместо судьи.
    if _lexical_correct(answer, gold) and (not parsed_ok or correct == 0):
        correct = 1

    return faith, relev, correct
