from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

from .base import Block, RawCase, _chunk, clean_pdf_text, paragraphs, squash
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from nautil_common import paths

EXTRACT = paths.CASES / "nhtsa" / "_extract"
HEADING = re.compile(r"^\s{0,60}([A-Z][A-Z0-9 &/,'()\-\.]{7,70})\s*$")
DOC_KIND = {"INOA": "opening resume", "INCLA": "closing resume", "INCR": "closing report",
            "INCV": "index of cited complaint numbers"}
FIELD = re.compile(r"^\s*(Manufacturer|Products?|Population|Problem Description|Subject|Action|Date Opened|"
                   r"Date Closed|Investigator|Reviewer|Approver|Prompted [Bb]y|Investigation)\s*:", re.M)


def _docs(directory: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for path in sorted(directory.glob("*.txt")):
        prefix = path.name.split("-", 1)[0].upper()
        out.setdefault(prefix, []).append(path)
    return out


def candidates(limit: int, *, seed: int = 11, min_chars: int = 4000) -> list[dict[str, Any]]:
    import random
    rows: list[dict[str, Any]] = []
    for directory in sorted(EXTRACT.iterdir()):
        if not directory.is_dir():
            continue
        docs = _docs(directory)
        if "INOA" not in docs or "INCLA" not in docs:
            continue
        size = sum(p.stat().st_size for kind in ("INOA", "INCLA", "INCR") for p in docs.get(kind, []))
        if size < min_chars:
            continue
        rows.append({"native_id": directory.name, "bytes": size, "has_report": "INCR" in docs})
    rows.sort(key=lambda r: (-r["has_report"], -r["bytes"]))
    pool = rows[: max(limit * 3, limit)]
    random.Random(seed).shuffle(pool)
    return sorted(pool[:limit], key=lambda r: r["native_id"])


def load(native_id: str) -> RawCase:
    directory = EXTRACT / native_id
    docs = _docs(directory)
    if "INOA" not in docs and "INCLA" not in docs:
        raise KeyError(f"{native_id}: no resume in the text cache")
    blocks: list[Block] = []
    used: list[str] = []
    header_fields: dict[str, str] = {}
    for prefix in ("INOA", "INCLA", "INCR"):
        for path in docs.get(prefix, []):
            text = clean_pdf_text(path.read_text(encoding="utf-8", errors="replace"))
            used.append(path.name)
            kind = DOC_KIND.get(prefix, prefix)
            lines = text.split("\n")
            marks = [(i, squash(m.group(1))) for i, line in enumerate(lines)
                     if (m := HEADING.match(line.rstrip())) and not FIELD.match(line)]
            if not marks:
                marks = [(0, "body")]
            for position, (index, name) in enumerate(marks):
                end = marks[position + 1][0] if position + 1 < len(marks) else len(lines)
                body = "\n".join(lines[index + 1: end]).strip()
                if len(body) < 60:
                    continue
                label = f"{kind}, {name.title()}"
                upper = name.upper()
                role = "unknown"
                if any(k in upper for k in ("MANUFACTURER", "PRODUCT INFORMATION", "FAILURE REPORT",
                                            "PROBLEM DESCRIPTION", "COMPLAINT", "TESTING", "SIMULATION",
                                            "FIELD", "WARRANTY", "WHAT WE KNOW")):
                    role = "factual"
                if "SUMMARY" in name.upper() and len(body) > 900:
                    for para_index, para in enumerate(paragraphs(body, min_chars=80), 1):
                        blocks.append(Block(label=f"{label} (paragraph {para_index})", text=para, role=role))
                else:
                    for part, chunk in enumerate(_chunk(body, 9000), 1):
                        suffix = "" if part == 1 else f" (continued {part})"
                        blocks.append(Block(label=label + suffix, text=chunk, role=role))
            for match in FIELD.finditer(text):
                line = text[match.start(): text.find("\n", match.start())]
                key, _, value = line.partition(":")
                if squash(value):
                    header_fields.setdefault(squash(key), squash(value))
    manufacturer = header_fields.get("Manufacturer", "")
    products = header_fields.get("Products", header_fields.get("Product", ""))
    return RawCase(
        native_id=native_id, source_dataset="NHTSA ODI", domain="road vehicle defect investigation",
        record_kind=("the opening and closing record of a national road-vehicle safety regulator's "
                     "defect investigation office on an alleged vehicle defect"),
        blocks=blocks, analysis_text="", conclusion_text="",
        identity_seed={"native_id": native_id, "manufacturer": manufacturer, "products": products,
                       "date_opened": header_fields.get("Date Opened", ""),
                       "date_closed": header_fields.get("Date Closed", "")},
        license="US government work", document=" + ".join(used),
        keep_extra=("- the vehicle manufacturer, make, model and model years, and component and part numbers "
                    "(this source keeps them: they are technical information)\n"
                    "- complaint, crash, injury and population counts"),
        needs_block_labelling=True,
        extra={"reference_note": ('   This is a defect investigation. A possibility counts as set aside when the office states that the data do not support it, that the alleged mechanism was not confirmed, that a component was tested and found sound, or that the rate or trend does not bear it out.'), "files": used, "identity_keys_withheld": ("date", "location", "native_id"),
               "factual_headings": ("MANUFACTURER", "PRODUCT INFORMATION", "FAILURE REPORT", "PROBLEM DESCRIPTION",
                                    "COMPLAINT", "TESTING", "SIMULATION", "FIELD", "WARRANTY")})


def iter_cases(limit: int, **kw) -> Iterator[RawCase]:
    for item in candidates(limit, **kw):
        yield load(item["native_id"])
