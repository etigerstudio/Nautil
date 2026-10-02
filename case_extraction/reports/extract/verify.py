from __future__ import annotations

import re
import unicodedata
from typing import Any

STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "for", "with", "was", "were", "is",
        "are", "be", "been", "had", "has", "have", "that", "this", "it", "its", "as", "by", "from", "not",
        "he", "she", "they", "his", "her", "their", "which", "when", "no", "there", "but", "about", "after"}


CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_control(text: str) -> str:
    return CONTROL.sub(" ", text or "")


ARTICLE_FIX = ((re.compile(r"\b([Aa]n?)\s+(the|a|an)\b"), r"\2"),
               (re.compile(r"\b(the)\s+the\b", re.I), r"\1"),
               (re.compile(r"\s+([,.;:])"), r"\1"))


def tidy(text: str) -> str:
    for pattern, replacement in ARTICLE_FIX:
        text = pattern.sub(replacement, text or "")
    return re.sub(r"\s{2,}", " ", text).strip()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―−]", "-", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'\-\.]*", norm(text)) if w not in STOP and len(w) > 1]


def contains(haystack: str, needle: str, *, min_words: int = 5) -> bool:
    h, n = norm(haystack), norm(needle)
    if not n:
        return False
    if n in h:
        return True
    tokens = n.split()
    if len(tokens) < min_words:
        return False
    for start, end in ((1, None), (None, -1), (1, -1)):
        trimmed = " ".join(tokens[start:end])
        if trimmed and trimmed in h:
            return True
    return False


def overlap(text: str, source: str) -> float:
    item_words = words(text)
    if not item_words:
        return 0.0
    pool = set(words(source))
    return sum(w in pool for w in item_words) / len(item_words)


def coverage(items: list[str], source: str) -> float:
    pool = set()
    for item in items:
        pool.update(words(item))
    source_words = words(source)
    if not source_words:
        return 1.0
    return sum(w in pool for w in source_words) / len(source_words)


CONCLUSION_WORDS = re.compile(
    r"probable cause|reason for closing|root cause was|was the cause of the|final diagnosis|"
    r"diagnosis was confirmed|contributing factor|"
    r"\b(the (investigation|investigators|board|bureau|branch|office|analysis)|ODI|we)\b[^.]{0,40}"
    r"\b(concluded|determined|found that|attributed)\b", re.I)


def conclusion_hits(text: str) -> list[str]:
    return [m.group(0) for m in CONCLUSION_WORDS.finditer(text or "")]


DATE = re.compile(
    r"\b(?:\d{1,2}\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}?,?\s*(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b|\b(?:19|20)\d{2}-\d{2}-\d{2}\b", re.I)
REGO = re.compile(r"\b(?:[A-Z]{1,2}-?\d{2,5}[A-Z]{0,3}|N\d{2,5}[A-Z]{0,2})\b")


def residual_identifiers(text: str) -> dict[str, list[str]]:
    return {"dates": sorted({m.group(0) for m in DATE.finditer(text)})[:20],
            "registration_like": sorted({m.group(0) for m in REGO.finditer(text)})[:20]}


def leakage(package_text: str, withheld: list[str]) -> list[str]:
    low = norm(package_text)
    hits = []
    for value in withheld:
        if not value or not norm(value):
            continue
        pattern = flexible_pattern(value)
        if norm(value) in low or (pattern and pattern.search(package_text or "")):
            hits.append(value)
    return hits


def model_visible_text(package: dict) -> str:
    items = package.get("evidence_items") or []
    return " ".join([str(package.get("initial_context", "")), str(package.get("task_question", ""))]
                    + [f"{i.get('neutral_title','')} {i.get('text','')}" for i in items])


PROPER = re.compile(r"\b(?:[A-Z][a-z'’\-]{2,}|[A-Z]{2,6})(?:\s+(?:of|the|de|van|von)?\s*[A-Z][a-z'’\-]{2,}){0,3}\b")
COMMON_CAPS = {
    "The", "A", "An", "This", "That", "These", "Those", "It", "He", "She", "They", "There", "When", "After",
    "Before", "During", "Both", "One", "Two", "Three", "Four", "Five", "No", "Not", "All", "Each", "Its",
    "His", "Her", "Their", "Sources", "Source", "Evidence", "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday", "Title", "Code", "Federal", "Regulations", "Part", "Class", "Type",
    "Model", "Serial", "Number", "Left", "Right", "Front", "Rear", "North", "South", "East", "West",
}


def proper_nouns(text: str, *, min_count: int = 1) -> list[tuple[str, int]]:
    from collections import Counter
    counts: Counter[str] = Counter()
    for match in PROPER.finditer(text or ""):
        token = match.group(0).strip()
        head = token.split()[0]
        if head in COMMON_CAPS or len(token) < 3:
            continue
        counts[token] += 1
    def priority(entry: tuple[str, int]) -> tuple[int, int]:
        name, count = entry
        acronym = name.isupper() and 2 <= len(name) <= 6
        multiword = " " in name
        return (0 if acronym else (1 if multiword else 2), -count)

    return sorted([(n, c) for n, c in counts.items() if c >= min_count], key=priority)[:60]


IDENTIFIER_LIKE = re.compile(r"^(?:[A-Z0-9][\w'’\-\./ ]{1,48})$")


ROLE_HEADS = {"pilot", "passenger", "crew", "driver", "master", "patient", "occupant", "witness",
              "investigator", "operator", "manufacturer", "owner", "captain", "engineer"}


def identifier_hygiene(strings: list[str]) -> list[str]:
    keep = []
    for raw in strings:
        value = (raw or "").strip().strip(".,;")
        if not value or len(value) > 50 or len(value.split()) > 6:
            continue
        if not IDENTIFIER_LIKE.match(value):
            continue
        lowered = value.lower()
        if lowered.startswith(("at the ", "an ", "a ", "the ", "on the ", "in the ", "of the ")):
            continue
        if len(value) < 4 and not any(c.isdigit() for c in value):
            continue
        tokens = value.split()
        head = re.split(r"[^A-Za-z]", value)[0].lower()
        if head in ROLE_HEADS and len(tokens) > 1:
            continue
        connectors = {"of", "the", "and", "de", "van", "von", "for", "at", "on", "in", "&"}
        if any(t.islower() and t not in connectors and t.isalpha() for t in tokens[1:]):
            continue
        keep.append(value)
    return sorted(set(keep))


def near_duplicates(texts: list[str], threshold: float = 0.88) -> list[int]:
    seen: list[set[str]] = []
    drop: list[int] = []
    for index, text in enumerate(texts):
        bag = set(words(text))
        if len(bag) < 3:
            seen.append(bag)
            continue
        hit = False
        for other in seen:
            if not other:
                continue
            union = len(bag | other)
            if union and len(bag & other) / union >= threshold:
                hit = True
                break
        if hit:
            drop.append(index)
        else:
            seen.append(bag)
    return drop


def residual_withheld_tokens(package_text: str, withheld: list[str]) -> list[str]:
    out: set[str] = set()
    for entry in withheld:
        for token in re.findall(r"[A-Z]{2,6}", entry or ""):
            if re.search(r"(?<![A-Za-z])" + re.escape(token) + r"(?![A-Za-z])", package_text or ""):
                out.add(token)
    return sorted(out)


NEUTRAL_FOR_KEY = {
    "date": "the event day", "location": "the location", "operator": "the operator",
    "registration": "the vehicle", "native_id": "the investigation", "aircraft": "the aircraft type",
    "manufacturer": "the manufacturer", "products": "the subject vehicles", "title": "the report",
}


def flexible_pattern(value: str) -> re.Pattern[str] | None:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’]*", value or "")
    if not tokens:
        return None
    body = r"[^A-Za-z0-9]{0,3}".join(re.escape(t) for t in tokens)
    tail = r"[A-Za-z]*" if len(tokens[-1]) >= 4 and tokens[-1].isalpha() else ""
    return re.compile(r"(?<![A-Za-z0-9])" + body + tail + r"(?![A-Za-z0-9])", re.I)


def force_remove(text: str, strings: list[str], replacement: str = "the subject") -> tuple[str, list[str]]:
    removed = []
    for value in sorted({s for s in strings if s and s.strip()}, key=len, reverse=True):
        pattern = flexible_pattern(value)
        if pattern is None:
            continue
        text, n = pattern.subn(replacement, text)
        if n:
            removed.append(value)
    return text, removed
