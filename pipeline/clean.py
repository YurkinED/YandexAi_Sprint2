"""Очистка корпуса: html/md, mojibake, короткие заглушки."""

from __future__ import annotations

import hashlib
import html as html_lib
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

MIN_CHARS = 60  # короче обычно пустышка
MAX_GARBAGE = 0.18
DROP_EMPTY = True


@dataclass
class CleanStats:
    before_n: int
    after_n: int
    html_stripped: int
    md_normalized: int
    mojibake_fixed: int
    dropped_short: int
    dropped_garbage: int
    examples: list[dict]


def garbage_ratio(text: str) -> float:
    """Сколько «не буквы/цифры/пробел/пунктуация»."""
    if not text:
        return 1.0
    ok = set(".,;:!?-—–«»\"'()/%+@№№…")
    good = sum(ch.isalnum() or ch.isspace() or ch in ok for ch in text)
    return 1.0 - good / len(text)


def strip_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return html_lib.unescape(text)


def strip_markdown(raw: str) -> str:
    t = raw
    t = re.sub(r"(?m)^#{1,6}\s*", "", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"(?m)^\s*[-*+]\s+", "• ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    return t


def fix_mojibake(text: str) -> tuple[str, bool]:
    """UTF-8, который кто-то прочитал как latin-1/cp1252."""
    if not text:
        return text, False
    markers = text.count("Ð") + text.count("Ñ") + text.count("Â")
    if markers < 3 and "ï»¿" not in text[:10]:
        return text, False
    for enc in ("latin-1", "cp1252"):
        try:
            fixed = text.encode(enc).decode("utf-8")
            if garbage_ratio(fixed) < garbage_ratio(text):
                return fixed, True
        except (UnicodeDecodeError, UnicodeEncodeError):
            continue
    return text, False


def normalize_text(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    t = t.replace("“", "«").replace("”", "»").replace("„", "«")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def clean_doc(doc: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    # (doc|None, reason) — reason пустой если ок
    raw = doc.get("raw") or ""
    fmt = doc.get("format", "text")
    flag = ""

    # кодировку чиним до парсеров, иначе BeautifulSoup видит мусор
    raw_fixed, was_mojibake = fix_mojibake(raw)
    title_fixed, _ = fix_mojibake(doc.get("title") or "")

    if fmt == "html" or ("<" in raw_fixed and re.search(r"</?\w+[^>]*>", raw_fixed)):
        body = strip_html(raw_fixed)
        flag = "html"
    elif fmt == "markdown":
        body = strip_markdown(raw_fixed)
        flag = "markdown"
    else:
        body = raw_fixed

    body = normalize_text(body)
    if was_mojibake:
        flag = (flag + "+mojibake") if flag else "mojibake"
    if len(body) < MIN_CHARS:
        return None, "short"
    if garbage_ratio(body) > MAX_GARBAGE:
        return None, "garbage"

    out = {
        "doc_id": doc["doc_id"],
        "title": normalize_text(title_fixed),
        "category": doc.get("category"),
        "format": fmt,
        "raw": raw,
        "text": body,
        "clean_flag": flag or "text",
        "n_chars": len(body),
    }
    return out, ""


def clean_corpus(docs: list[dict]) -> tuple[list[dict], CleanStats]:
    cleaned: list[dict] = []
    html_n = md_n = moj_n = short_n = garb_n = 0
    examples: list[dict] = []

    for d in docs:
        out, reason = clean_doc(d)
        if reason == "short":
            short_n += 1
            continue
        if reason == "garbage":
            garb_n += 1
            continue
        assert out is not None
        flag = out["clean_flag"]
        if "html" in flag:
            html_n += 1
        if "markdown" in flag:
            md_n += 1
        if "mojibake" in flag:
            moj_n += 1
        if len(examples) < 4 and any(x in flag for x in ("html", "markdown", "mojibake")):
            examples.append({
                "doc_id": out["doc_id"],
                "title": out["title"],
                "before": (d.get("raw") or "")[:220],
                "after": out["text"][:220],
                "flag": flag,
            })
        cleaned.append(out)

    stats = CleanStats(
        before_n=len(docs),
        after_n=len(cleaned),
        html_stripped=html_n,
        md_normalized=md_n,
        mojibake_fixed=moj_n,
        dropped_short=short_n,
        dropped_garbage=garb_n,
        examples=examples,
    )
    return cleaned, stats


def content_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()


def save_cleaned(docs: list[dict], path: Path) -> None:
    path.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")


def stats_to_dict(stats: CleanStats) -> dict:
    return asdict(stats)
