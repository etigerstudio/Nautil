from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .base import Block, RawCase, paragraphs, squash
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from nautil_common import paths

PARQUET = paths.RAW / "medcasereasoning" / "medcasereasoning_core.pqt"
_CACHE: dict[str, pd.DataFrame] = {}

DIRECTION = re.compile(
    r"\b(initial(ly)? (diagnos|treat|manage)|initial diagnosis|working diagnosis|presumed |presumptive |"
    r"was diagnosed with|misdiagnos|diagnosed as|treated (for|with|empirically)|empiric(al)? (treatment|therapy|antibiotic)|"
    r"started on|commenced on|referred with a diagnosis of|admitted with a diagnosis of)", re.I)
FAILURE = re.compile(
    r"\b(without improvement|no improvement|did not (improve|respond)|failed to (respond|improve)|"
    r"despite (treatment|therapy|antibiotic|steroid)|symptoms persisted|persisted despite|continued to (worsen|deteriorate)|"
    r"deteriorated|relapsed|recurred|refractory|unresponsive to|no clinical response)", re.I)
REVEAL = re.compile(r"\b(final diagnosis|was diagnosed as having|confirmed the diagnosis of|biopsy (confirmed|revealed))", re.I)


def frame() -> pd.DataFrame:
    if "df" not in _CACHE:
        _CACHE["df"] = pd.read_parquet(PARQUET)
    return _CACHE["df"]


def candidates(limit: int, *, seed: int = 11, min_words: int = 140, max_words: int = 700,
               exclude: set[str] | None = None) -> list[dict[str, Any]]:
    df = frame().copy()
    df["words"] = df.case_prompt.str.split().str.len()
    keep = df[(df.words >= min_words) & (df.words <= max_words)
              & df.case_prompt.str.contains(DIRECTION, regex=True)
              & df.case_prompt.str.contains(FAILURE, regex=True)
              & (df.final_diagnosis.str.split().str.len() <= 12)
              & (df.diagnostic_reasoning.str.len() >= 300)].copy()
    if exclude:
        keep = keep[~keep.pmcid.isin(exclude)]
    def leak(row: Any) -> float:
        answer = {w for w in re.findall(r"[a-z]{5,}", str(row.final_diagnosis).lower())}
        if not answer:
            return 1.0
        prompt = str(row.case_prompt).lower()
        return sum(w in prompt for w in answer) / len(answer)
    keep["leak"] = keep.apply(leak, axis=1)
    keep = keep[keep.leak < 0.35].sort_values(["leak", "words"])
    keep = keep.head(max(limit * 3, limit)).sample(n=min(limit, len(keep)), random_state=seed)
    return [{"native_id": r.pmcid} for r in keep.sort_values("pmcid").itertuples()]


def load(pmcid: str) -> RawCase:
    df = frame()
    hit = df[df.pmcid == pmcid]
    if hit.empty:
        raise KeyError(f"{pmcid} not in the parquet")
    row = hit.iloc[0]
    prompt = str(row.case_prompt).strip()
    blocks: list[Block] = []
    chunk: list[str] = []
    for para in paragraphs(prompt, min_chars=1):
        chunk.append(para)
        if sum(len(p) for p in chunk) > 2600:
            blocks.append(Block(label=f"Case narrative {len(blocks) + 1}", text="\n\n".join(chunk), role="factual"))
            chunk = []
    if chunk:
        blocks.append(Block(label=f"Case narrative {len(blocks) + 1}", text="\n\n".join(chunk), role="factual"))
    year = str(row.publication_date)[:4]
    return RawCase(
        native_id=pmcid, source_dataset="MedCaseReasoning (zou-lab), PMC case reports",
        domain="clinical case report",
        record_kind="a published clinical case report, as it stood before the case was resolved",
        blocks=blocks,
        analysis_text=(str(row.diagnostic_reasoning).strip()
                       + "\n\nFULL ARTICLE TEXT (use it to find the sentence that sets a possibility aside):\n"
                       + str(row.text).strip()),
        conclusion_text=squash(str(row.final_diagnosis)),
        identity_seed={"pmcid": pmcid, "title": squash(str(row.title)),
                       "journal": squash(str(row.journal)), "year": year},
        license="UNRESOLVED (PMC article-level license not in parquet)",
        document=f"{squash(str(row.journal))}, {str(row.publication_date)[:10]}",
        url=str(row.article_link), anonymize=False,
        task_question=("Based only on the supplied materials, what is the diagnosis? "
                       "Cite the evidence_id for each key claim."),
        extra={"full_text": str(row.text), "final_diagnosis": squash(str(row.final_diagnosis)),
               "min_items": 6,
               "reference_note": (
                   "   This is a clinical case report. A possibility counts as set aside when the record shows it "
                   "was considered and then dropped: a test result that excludes it, a treatment given for it that "
                   "failed, a working or admitting diagnosis that the later course contradicts, or an explicit "
                   "sentence dismissing it. The first thing to look for is the diagnosis or treatment direction the "
                   "clinicians took at the time and later abandoned. Quote the sentence that shows it was set "
                   "aside, from anywhere in the record below."),
               "granularity": ("- This is a clinical record, where each finding is cited on its own. Give every "
                               "distinct symptom, sign, vital sign, laboratory value, imaging finding, treatment "
                               "and response its own item, even when several appear in one sentence.")})


def iter_cases(limit: int, **kw) -> Iterator[RawCase]:
    for item in candidates(limit, **kw):
        yield load(item["native_id"])
