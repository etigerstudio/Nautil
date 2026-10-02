from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .base import RawCase, clean_pdf_text, detect_headings, garbled_score, split_blocks, squash
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from nautil_common import paths

SOURCES = {
    "ntsb_pipeline": {"dir": paths.RAW / "ntsb_modes/pipeline", "agency": "NTSB",
                      "kind": "an independent national transportation safety board",
                      "domain": "pipeline accident investigation", "licence": "US government work"},
    "ntsb_highway": {"dir": paths.RAW / "ntsb_modes/highway", "agency": "NTSB",
                     "kind": "an independent national transportation safety board",
                     "domain": "highway accident investigation", "licence": "US government work"},
    "tsb_pipeline": {"dir": paths.RAW / "tsb_canada/pipeline", "agency": "TSB",
                     "kind": "an independent national transportation safety board",
                     "domain": "pipeline accident investigation",
                     "licence": "TSB Canada: reproduction for non-commercial purposes permitted with attribution"},
}


def _manifest(source: str) -> list[dict[str, Any]]:
    path = SOURCES[source]["dir"] / "manifest.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    if not rows:
        rows = [{"report_id": p.stem} for p in sorted(SOURCES[source]["dir"].glob("*.pdf"))]
    return [r for r in rows if (SOURCES[source]["dir"] / f"{r['report_id']}.pdf").exists() and not r.get("interim_report")]


def _text(source: str, rid: str) -> str:
    d = SOURCES[source]["dir"]
    cache = d / "_text" / f"{rid}.txt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftotext", str(d / f"{rid}.pdf"), str(cache)], check=True, capture_output=True)
    return cache.read_text(encoding="utf-8", errors="replace")


def make(source: str) -> SimpleNamespace:
    spec = SOURCES[source]

    def candidates(limit: int, *, seed: int = 11, min_chars: int = 8000, max_chars: int = 400000) -> list[dict[str, Any]]:
        rows = []
        for r in _manifest(source):
            size = len(_text(source, r["report_id"]))
            if min_chars <= size <= max_chars:
                rows.append({"native_id": r["report_id"], "bytes": size, "title": r.get("title", ""),
                             "date": r.get("event_date", "")})
        rows.sort(key=lambda r: r["native_id"])
        return rows[:limit]

    def load(native_id: str, board: str | None = None) -> RawCase:
        row = next((r for r in _manifest(source) if r["report_id"] == native_id), {"report_id": native_id})
        text = clean_pdf_text(_text(source, native_id))
        blocks = split_blocks(text, detect_headings(text))
        return RawCase(
            native_id=native_id, source_dataset=spec["agency"], domain=spec["domain"],
            record_kind=f"the final investigation report of {spec['kind']}",
            blocks=blocks, analysis_text="", conclusion_text="",
            identity_seed={"native_id": native_id, "title": squash(row.get("title", "")),
                           "date": row.get("event_date", "") or ""},
            license=spec["licence"], document=squash(row.get("title", "")), url=row.get("url", ""),
            keep_extra="- the make, model and type of the vehicle, pipe, valve or equipment item, pipe grades and part numbers",
            needs_block_labelling=True,
            extra={"reference_note": ("   A possibility counts as set aside when the report states the evidence does not "
                                      "support it, that it was examined and excluded, that a component was tested and found "
                                      "sound, or that it was considered unlikely."),
                   "board": spec["agency"], "text_path": str(spec["dir"] / "_text" / f"{native_id}.txt"),
                   "garbled": round(garbled_score(text), 3)})

    def index() -> list[dict[str, Any]]:
        return [{"rid": r["report_id"], "board": spec["agency"]} for r in _manifest(source)]

    return SimpleNamespace(candidates=candidates, load=load, index=index, __name__=f"reports_pdf.{source}")
