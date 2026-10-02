from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from .base import Block, RawCase, clean_pdf_text, detect_headings, garbled_score, split_blocks, squash
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from nautil_common import paths

EXTRACT = paths.CASES / "boards" / "_extract"
INDEX = EXTRACT / "report_index.jsonl"
MODE_DOMAIN = {"rail": "rail accident investigation", "marine": "marine accident investigation",
               "aviation": "aviation accident investigation", "chemical": "chemical process safety investigation"}
BOARD_KIND = {
    "RAIB": "an independent national rail accident investigation body",
    "MAIB": "an independent national marine accident investigation body",
    "ATSB": "an independent national transport safety board",
    "CSB": "an independent national chemical process safety investigation board",
}
_INDEX: list[dict[str, Any]] = []


def index() -> list[dict[str, Any]]:
    global _INDEX
    if not _INDEX:
        _INDEX = [json.loads(line) for line in INDEX.read_text(encoding="utf-8").splitlines() if line.strip()]
    return _INDEX


def text_path(board: str, rid: str) -> Path:
    return EXTRACT / board.lower() / f"{rid}.txt"


def candidates(limit: int, *, seed: int = 11, min_chars: int = 25000, max_chars: int = 400000,
               boards: tuple[str, ...] = ("ATSB", "RAIB", "MAIB", "CSB")) -> list[dict[str, Any]]:
    import random
    rows: list[dict[str, Any]] = []
    for item in index():
        board = item["board"]
        if board not in boards:
            continue
        path = text_path(board, item["rid"])
        if not path.exists():
            continue
        size = path.stat().st_size
        if not (min_chars <= size <= max_chars):
            continue
        rows.append({"native_id": item["rid"], "board": board, "bytes": size,
                     "title": item.get("title", ""), "date": item.get("date", ""),
                     "mode": item.get("mode", ""), "url": item.get("url", "")})
    rng = random.Random(seed)
    by_board: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_board.setdefault(row["board"], []).append(row)
    chosen: list[dict[str, Any]] = []
    board_names = [b for b in boards if by_board.get(b)]
    per = {b: limit // len(board_names) for b in board_names} if board_names else {}
    for extra_index in range(limit - sum(per.values())):
        per[board_names[extra_index % len(board_names)]] += 1
    for board, want in per.items():
        pool = by_board[board]
        rng.shuffle(pool)
        chosen.extend(pool[:want])
    return sorted(chosen, key=lambda r: (r["board"], r["native_id"]))


def load(native_id: str, board: str | None = None) -> RawCase:
    item = next((x for x in index() if x["rid"] == native_id and (board is None or x["board"] == board)), None)
    if item is None:
        raise KeyError(f"{native_id} not in the board index")
    board = item["board"]
    path = text_path(board, native_id)
    text = clean_pdf_text(path.read_text(encoding="utf-8", errors="replace"))
    headings = detect_headings(text)
    blocks = split_blocks(text, headings)
    domain = MODE_DOMAIN.get(item.get("mode", ""), "accident investigation")
    return RawCase(
        native_id=native_id, source_dataset=board, domain=domain,
        record_kind=f"the final investigation report of {BOARD_KIND.get(board, 'an investigation board')}",
        blocks=blocks, analysis_text="", conclusion_text="",
        identity_seed={"native_id": native_id, "title": squash(item.get("title", "")),
                       "date": item.get("date", "") or ""},
        license={"ATSB": "CC BY 4.0", "RAIB": "Open Government Licence v3.0",
                 "MAIB": "Open Government Licence v3.0", "CSB": "US government work"}.get(board, "UNRESOLVED"),
        document=squash(item.get("title", "")), url=item.get("url", ""),
        keep_extra="- the make, model and type of the vehicle, vessel, locomotive or plant item, and part numbers",
        needs_block_labelling=True,
        extra={"reference_note": ('   A possibility counts as set aside when the report states the evidence does not support it, that it was examined and excluded, that a component was tested and found sound, or that it was considered unlikely.'), "board": board, "pdf_files": item.get("files", []), "text_path": str(path),
               "garbled": round(garbled_score(text), 3)})


def iter_cases(limit: int, **kw) -> Iterator[RawCase]:
    for item in candidates(limit, **kw):
        yield load(item["native_id"], item["board"])
