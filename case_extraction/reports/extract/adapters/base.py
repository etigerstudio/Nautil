from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any



@dataclass
class Block:
    label: str
    text: str
    role: str = "unknown"
    source_hint: str = ""


@dataclass
class RawCase:
    native_id: str
    source_dataset: str
    domain: str
    record_kind: str
    blocks: list[Block]
    analysis_text: str
    conclusion_text: str
    identity_seed: dict[str, str] = field(default_factory=dict)
    license: str = ""
    document: str = ""
    url: str = ""
    anonymize: bool = True
    keep_extra: str = ""
    task_question: str = ("Based only on the supplied materials, what caused this? "
                          "Cite the evidence_id for each key claim.")
    needs_block_labelling: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


WS = re.compile(r"\s+")


def squash(text: str) -> str:
    return WS.sub(" ", text or "").strip()


def dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n\s*(\w)", r"\1\2", text)


def clean_pdf_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x0c", "\n")
    text = dehyphenate(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def paragraphs(text: str, min_chars: int = 40) -> list[str]:
    out = []
    for part in re.split(r"\n\s*\n", text):
        part = part.strip()
        if len(part) >= min_chars:
            out.append(part)
        elif out and part:
            out[-1] = out[-1] + "\n" + part
    return out


def looks_like_page_furniture(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    lowered = stripped.lower()
    return bool(
        re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", lowered)
        or re.search(r"\bpage \d+ of \d+\b", lowered)
        or re.fullmatch(r"[-–—_.•\s]+", stripped)
    )


HEADING_PATTERNS = (
    re.compile(r"^\s{0,12}((?:\d+|[A-Z])(?:\.\d+){0,3})\.?\s+([A-Z][\w][^\n]{2,80})$"),
    re.compile(r"^\s{0,20}([A-Z][A-Z0-9 &/,'()\-\.]{7,70})$"),
    re.compile(r"^\s{0,20}((?:[A-Z][a-z'\-]+)(?: (?:of|the|and|for|to|in|on|a)| [A-Z][a-z'\-]+){1,7})$"),
)


def detect_headings(text: str, *, min_body: int = 200) -> list[tuple[int, str]]:
    lines = text.split("\n")
    found: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if looks_like_page_furniture(line) or len(line.strip()) < 4:
            continue
        if line.strip().endswith((".", ";", ",")) and not re.match(r"^\s*\d+(\.\d+)*\.?\s", line):
            continue
        for pattern in HEADING_PATTERNS:
            match = pattern.match(line.rstrip())
            if match:
                heading = squash(" ".join(match.groups()))
                if 4 <= len(heading) <= 90 and not heading[0].islower():
                    found.append((index, heading))
                break
    return found


def split_blocks(text: str, headings: list[tuple[int, str]], *, min_chars: int = 200,
                 max_chars: int = 9000) -> list[Block]:
    lines = text.split("\n")
    if not headings:
        return [Block(label="body", text=chunk) for chunk in _chunk(text, max_chars)]
    blocks: list[Block] = []
    bounds = [(index, name) for index, name in headings]
    if bounds[0][0] > 0:
        head = "\n".join(lines[: bounds[0][0]]).strip()
        if len(head) >= min_chars:
            blocks.append(Block(label="(front)", text=head))
    for position, (index, name) in enumerate(bounds):
        end = bounds[position + 1][0] if position + 1 < len(bounds) else len(lines)
        body = "\n".join(lines[index + 1: end]).strip()
        if len(body) < min_chars:
            if blocks:
                blocks[-1].text += f"\n\n{name}\n{body}"
                continue
            if not body:
                continue
        for part_index, chunk in enumerate(_chunk(body, max_chars)):
            label = name if part_index == 0 else f"{name} (continued {part_index + 1})"
            blocks.append(Block(label=label, text=chunk))
    return blocks


def _chunk(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    out: list[str] = []
    current = ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) + 2 > max_chars:
            out.append(current.strip())
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        out.append(current.strip())
    return out


def garbled_score(text: str) -> float:
    if not text.strip():
        return 1.0
    letters = sum(char.isalpha() for char in text)
    words = re.findall(r"[A-Za-z]{2,}", text)
    long_words = [w for w in words if len(w) >= 4]
    weird = len(re.findall(r"[^\x09\x0a\x0d\x20-\x7e£§°±–—’‘“”é]", text))
    if letters == 0:
        return 1.0
    single_char_lines = sum(1 for line in text.split("\n") if 0 < len(line.strip()) <= 2)
    lines = max(1, len(text.split("\n")))
    return min(1.0, 0.5 * (weird / max(1, len(text)) * 20)
               + 0.3 * (1 - min(1.0, len(long_words) / max(1, len(words) or 1)))
               + 0.2 * min(1.0, single_char_lines / lines * 3))
