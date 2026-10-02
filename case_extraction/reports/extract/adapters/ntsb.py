from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .base import Block, RawCase, clean_pdf_text, squash
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from nautil_common import paths

PARQUET = paths.CASES / "ntsb" / "_extract" / "workingset_2014_2024_finals.parquet"
SECTION = re.compile(
    r"(HISTORY OF FLIGHT|PERSONNEL INFORMATION|AIRCRAFT INFORMATION|METEOROLOGICAL INFORMATION|"
    r"AIRPORT INFORMATION|WRECKAGE AND IMPACT INFORMATION|MEDICAL AND PATHOLOGICAL INFORMATION|"
    r"TESTS? AND RESEARCH|FLIGHT RECORDERS?|COMMUNICATIONS|SURVIVAL ASPECTS|FIRE|"
    r"ADDITIONAL INFORMATION|ORGANIZATIONAL AND MANAGEMENT INFORMATION|USEFUL OR EFFECTIVE INVESTIGATION TECHNIQUES)")
_CACHE: dict[str, pd.DataFrame] = {}


def frame() -> pd.DataFrame:
    if "df" not in _CACHE:
        _CACHE["df"] = pd.read_parquet(PARQUET)
    return _CACHE["df"]


def candidates(limit: int, *, undetermined_share: float = 0.7, seed: int = 11,
               min_accp: int = 2500, min_accf: int = 700) -> list[dict[str, Any]]:
    df = frame()
    ok = df[(df.narr_accp.str.len() >= min_accp) & (df.narr_accf.str.len() >= min_accf)
            & (df.narr_cause.str.len() >= 60)].copy()
    ok["rank"] = ok["difficulty_score"] if "difficulty_score" in ok else 0
    want_undet = int(round(limit * undetermined_share))
    undet = ok[ok.is_undetermined].sort_values("rank", ascending=False).head(want_undet * 3)
    det = ok[~ok.is_undetermined].sort_values("rank", ascending=False).head((limit - want_undet) * 3)
    undet = undet.sample(n=min(want_undet, len(undet)), random_state=seed)
    det = det.sample(n=min(limit - len(undet), len(det)), random_state=seed)
    rows = pd.concat([undet, det]).sort_values("ntsb_no")
    return [{"native_id": r.ntsb_no, "undetermined": bool(r.is_undetermined)} for r in rows.itertuples()]


def load(native_id: str) -> RawCase:
    df = frame()
    hit = df[df.ntsb_no == native_id]
    if hit.empty:
        raise KeyError(f"NTSB {native_id} not in the working set")
    row = hit.iloc[0]
    factual = clean_pdf_text(str(row.narr_accp))
    parts = SECTION.split(factual)
    blocks: list[Block] = []
    if parts and parts[0].strip():
        blocks.append(Block(label="Narrative", text=parts[0].strip(), role="factual"))
    for index in range(1, len(parts) - 1, 2):
        body = parts[index + 1].strip()
        if body:
            blocks.append(Block(label=squash(parts[index]).title(), text=body, role="factual"))
    if not blocks:
        blocks = [Block(label="Narrative", text=factual, role="factual")]
    aircraft = squash(f"{row.acft_make} {row.acft_model}").title()
    return RawCase(
        native_id=native_id, source_dataset="NTSB", domain="aviation accident investigation",
        record_kind=f"the final report of a national safety board on an accident involving a {aircraft}",
        blocks=blocks, analysis_text=clean_pdf_text(str(row.narr_accf)),
        conclusion_text=squash(str(row.narr_cause)),
        identity_seed={"date": str(row.ev_date)[:10], "location": squash(f"{row.ev_city}, {row.ev_state}, {row.ev_country}"),
                       "operator": squash(str(row.oper_name)), "registration": squash(str(row.regis_no)),
                       "native_id": native_id, "aircraft": aircraft},
        license="US government work",
        document="final report (narr_accp / narr_accf / narr_cause in avall-2026-09-01)",
        keep_extra="- the make and model of the aircraft and of its engine, and part numbers",
        extra={"is_undetermined": bool(row.is_undetermined)})


def iter_cases(limit: int, **kw) -> Iterator[RawCase]:
    for item in candidates(limit, **kw):
        yield load(item["native_id"])
