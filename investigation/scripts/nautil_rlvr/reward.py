from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .common import EVIDENCE_ID, initial_record_ids

SECTION = re.compile(r"^##\s+(.+?)\s*$", re.M)
STATE_SECTIONS = ("Current hypotheses", "Hypothesis ledger")
UPDATE_SECTIONS = ("Current hypotheses", "Evidence and hypothesis update", "Hypothesis ledger")
FETCH_SECTIONS = ("Current hypotheses", "Hypothesis ledger", "Next discriminating check")

DEFAULT_STRUCTURE = {
    "citation_none": -0.25, "citation_invalid": -0.5,
    "turn_free": 6, "turn_step": -0.1, "turn_cap": -1.0,
    "rerequest_free": 10, "rerequest_step": -0.1, "rerequest_cap": -1.0,
    "invalid_fetch_step": -0.2, "invalid_fetch_cap": -1.0,
    "missing_marker": -2.0,
}
DEFAULT_SCALING = {
    "fetch_helpfulness": {"raw": [0, 10], "to": [0.0, 0.5]},
    "hypothesis_update": {"raw": [0, 10], "to": [-0.5, 0.5]},
    "hypothesis_total_clip": [-1.0, 1.0],
    "closure_conclusion_closed": {"raw": [0, 10], "to": [-1.0, 1.0]},
    "closure_conclusion_not_closed": {"raw": [0, 10], "to": [-0.5, 0.5]},
    "closure_gap_not_closed": {"raw": [0, 10], "to": [-0.5, 0.5]},
    "closure_status_match": 0.5, "closure_status_mismatch": -0.5,
    "key_hit_multiplier": 3.0,
}


@dataclass
class RewardConfig:
    phase: int = 1
    structure: dict = field(default_factory=lambda: dict(DEFAULT_STRUCTURE))
    scaling: dict = field(default_factory=lambda: json.loads(json.dumps(DEFAULT_SCALING)))
    citation_scope: str = "final"
    weights: dict = field(default_factory=lambda: {"fetch": 1.0, "hypothesis": 1.0,
                                                   "closure": 1.0, "structure": 1.0})
    max_chars_official: int = 3000

    @classmethod
    def from_dict(cls, raw: dict) -> "RewardConfig":
        cfg = cls()
        cfg.phase = int(raw.get("phase", 1))
        if cfg.phase not in (1, 2):
            raise ValueError("reward.phase must be 1 or 2")
        cfg.structure.update(raw.get("structure", {}))
        for key, value in raw.get("scaling", {}).items():
            cfg.scaling[key] = value
        cfg.citation_scope = raw.get("citation_scope", "final")
        cfg.weights.update(raw.get("weights", {}))
        cfg.max_chars_official = int(raw.get("max_chars_official", 3000))
        return cfg


def scale(raw_score: float, spec: dict) -> float:
    (r0, r1), (t0, t1) = spec["raw"], spec["to"]
    raw_score = min(max(float(raw_score), r0), r1)
    return t0 + (raw_score - r0) * (t1 - t0) / (r1 - r0)


def sections(text: str) -> dict[str, str]:
    text = text or ""
    marks = list(SECTION.finditer(text))
    out = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1).strip()] = text[m.end():end].strip()
    return out


def pick_sections(text: str, names, fallback_chars: int = 4000) -> str:
    found = sections(text)
    parts = [f"## {n}\n{found[n]}" for n in names if n in found]
    if parts:
        return "\n\n".join(parts)
    text = (text or "").strip()
    return text[:fallback_chars] if text else "(empty)"


def valid_fetch(turn: dict) -> bool:
    return (turn.get("call") is not None and not turn.get("call_errors")
            and isinstance(turn.get("tool_return"), dict) and "error" not in turn["tool_return"])


def invalid_fetch(turn: dict) -> bool:
    if turn.get("call_errors"):
        return True
    ret = turn.get("tool_return")
    return isinstance(ret, dict) and "error" in ret


def task_question(user_message: str) -> str:
    found = sections(user_message)
    return found.get("Task question", "").strip() or user_message[:600]


def final_turn(result: dict) -> dict | None:
    if result.get("final_answer") is None or not result.get("turns"):
        return None
    return result["turns"][-1]


def model_closed(result: dict) -> bool | None:
    marker = result.get("conclusion_marker")
    if marker == "CASE CLOSED":
        return True
    if marker == "CASE NOT CLOSED":
        return False
    return None


def key_hit(result: dict, case: dict, ref: dict, multiplier: float = 3.0) -> dict:
    key = set(ref.get("key_fetchable_ids") or [])
    disclosed = set(initial_record_ids(case["user_message"]))
    requests, union = [], []
    for turn in result["turns"]:
        if not valid_fetch(turn):
            continue
        ids = list(turn["call"]["evidence_ids"])
        new = [i for i in ids if i not in disclosed]
        hits = len(set(new) & key)
        requests.append(min(1.0, multiplier * hits / len(ids)) if ids else 0.0)
        disclosed.update(ids)
        union.extend(ids)
    unique = set(union)
    if not key or not unique:
        return {"value": 0.0, "whole": 0.0, "per_request_mean": 0.0, "requests": len(requests),
                "unique_fetched": len(unique), "key_hits": 0, "key_size": len(key),
                "key_set_empty": not key}
    hits = len(unique & key)
    whole = min(1.0, multiplier * hits / len(unique))
    per = sum(requests) / len(requests)
    return {"value": (whole + per) / 2, "whole": whole, "per_request_mean": per,
            "requests": len(requests), "unique_fetched": len(unique), "key_hits": hits,
            "key_size": len(key), "key_set_empty": False}


def structure(result: dict, case: dict, cfg: RewardConfig) -> dict:
    s = cfg.structure
    out = {}
    marker = result.get("conclusion_marker")
    out["missing_marker"] = s["missing_marker"] if marker is None else 0.0
    citation = 0.0
    detail = {"cited": [], "bad": []}
    if marker is not None and result.get("final_answer") is not None:
        disclosed_final = set(result.get("disclosed_evidence_ids") or [])
        cited = sorted(set(EVIDENCE_ID.findall(result["final_answer"])))
        bad = [c for c in cited if c not in disclosed_final]
        if cfg.citation_scope == "all_turns":
            seen = set(initial_record_ids(case["user_message"]))
            for turn in result["turns"][:-1]:
                prose = turn.get("assistant") or ""
                bad += [c for c in set(EVIDENCE_ID.findall(prose)) if c not in seen]
                if valid_fetch(turn):
                    seen.update(turn["call"]["evidence_ids"])
        detail = {"cited": cited, "bad": sorted(set(bad))}
        if not cited:
            citation = s["citation_none"]
        elif bad:
            citation = s["citation_invalid"]
    out["citation"] = citation
    turns = result.get("assistant_turns", len(result["turns"]))
    out["turns"] = max(s["turn_cap"], s["turn_step"] * max(0, turns - s["turn_free"]))
    seen = set(initial_record_ids(case["user_message"]))
    rereq = 0
    for turn in result["turns"]:
        if valid_fetch(turn):
            ids = turn["call"]["evidence_ids"]
            rereq += sum(i in seen for i in ids)
            seen.update(ids)
    out["rerequest"] = max(s["rerequest_cap"], s["rerequest_step"] * max(0, rereq - s["rerequest_free"]))
    n_invalid = sum(invalid_fetch(t) for t in result["turns"])
    out["invalid_fetch"] = max(s["invalid_fetch_cap"], s["invalid_fetch_step"] * n_invalid)
    out["total"] = sum(out[k] for k in ("missing_marker", "citation", "turns", "rerequest", "invalid_fetch"))
    out["counts"] = {"assistant_turns": turns, "rerequested_items": rereq,
                     "invalid_fetches": n_invalid, **detail}
    return out


def closure_status(result: dict, ref: dict, cfg: RewardConfig) -> float:
    closed = model_closed(result)
    if closed is None:
        return 0.0
    ref_closed = ref["closure"] == "closed"
    return cfg.scaling["closure_status_match"] if closed == ref_closed else cfg.scaling["closure_status_mismatch"]


def _title(store: dict, cid: str) -> str:
    item = store.get(cid)
    return item.get("neutral_title", "") if item else "(no such item in the case)"


def _item_text(item: dict) -> str:
    return f"{item['evidence_id']} | {item.get('neutral_title', '')} | {item.get('kind', '')}\n{item.get('text', '')}"


def fetch_steps(result: dict, case: dict) -> list[dict]:
    store = {i["evidence_id"]: i for i in case["evidence_items"]}
    disclosed = set(initial_record_ids(case["user_message"]))
    steps = []
    for turn in result["turns"]:
        if not valid_fetch(turn):
            continue
        ids = turn["call"]["evidence_ids"]
        steps.append({"turn": turn["turn"],
                      "state": pick_sections(turn.get("assistant") or "", FETCH_SECTIONS),
                      "requested": [f"{i} | {_title(store, i)} | "
                                    f"{'ALREADY READ' if i in disclosed else 'NEW'}" for i in ids],
                      "reason": turn["call"].get("reason", ""),
                      "already_read_before": sorted(disclosed)})
        disclosed.update(ids)
    return steps


def hypothesis_updates(result: dict) -> list[dict]:
    turns = result["turns"]
    updates = []
    for i in range(1, len(turns)):
        prev, cur = turns[i - 1], turns[i]
        if prev.get("call") is None and not prev.get("call_errors"):
            continue
        if cur.get("call") is None:
            continue
        ret = prev.get("tool_return")
        if not isinstance(ret, dict):
            returned = "(no tool result)"
        elif "error" in ret:
            returned = "(the fetch failed; no evidence was returned: " + json.dumps(ret, ensure_ascii=False)[:400] + ")"
        elif not ret.get("evidence_items"):
            returned = "(no new evidence: every requested item had already been disclosed)"
        else:
            returned = "\n\n".join(_item_text(it) for it in ret["evidence_items"])
        updates.append({"turn": cur["turn"],
                        "before": pick_sections(prev.get("assistant") or "", STATE_SECTIONS),
                        "returned": returned,
                        "update": pick_sections(cur.get("assistant") or "", UPDATE_SECTIONS)})
    return updates


def last_state(result: dict) -> str:
    for turn in reversed(result["turns"][:-1] if result.get("final_answer") is not None else result["turns"]):
        found = sections(turn.get("assistant") or "")
        if any(n in found for n in STATE_SECTIONS):
            return pick_sections(turn["assistant"], STATE_SECTIONS)
    return "(no explicit hypothesis state was written)"


def render_fetch_request(result: dict, case: dict, ref: dict) -> tuple[str, int] | None:
    steps = fetch_steps(result, case)
    if not steps:
        return None
    store = {i["evidence_id"]: i for i in case["evidence_items"]}
    key = [f"{k} | {_title(store, k)}" for k in ref.get("key_fetchable_ids") or []]
    out = [f"# Task question\n{task_question(case['user_message'])}",
           "# KEY EVIDENCE (reference investigation relied on these)\n" + ("\n".join(key) or "(none)")]
    for n, st in enumerate(steps, 1):
        out.append(f"# Step {n}\n## Hypothesis state at the moment of the request\n{st['state']}\n\n"
                   f"## Already read before this step (IDs)\n{', '.join(st['already_read_before']) or '(none)'}\n\n"
                   f"## Requested items\n" + "\n".join(st["requested"]) +
                   f"\n\n## Stated reason\n{st['reason']}")
    out.append(f"Grade all {len(steps)} steps.")
    return "\n\n".join(out), len(steps)


def render_hypothesis_request(result: dict, case: dict) -> tuple[str, int] | None:
    ups = hypothesis_updates(result)
    if not ups:
        return None
    out = [f"# Task question\n{task_question(case['user_message'])}"]
    for n, up in enumerate(ups, 1):
        out.append(f"# Update {n}\n## Hypothesis state BEFORE\n{up['before']}\n\n"
                   f"## Evidence JUST RETURNED\n{up['returned']}\n\n"
                   f"## The investigator's update\n{up['update']}")
    out.append(f"Grade all {len(ups)} updates.")
    return "\n\n".join(out), len(ups)


def render_closure_request(result: dict, case: dict, ref: dict, cfg: RewardConfig) -> tuple[str, int] | None:
    closed = model_closed(result)
    if closed is None:
        return None
    store = {i["evidence_id"]: i for i in case["evidence_items"]}
    disclosed = set(result.get("disclosed_evidence_ids") or [])
    cited = sorted(set(EVIDENCE_ID.findall(result["final_answer"])),
                   key=lambda x: tuple(int(p) for p in x[1:].split(".")))
    cited_text = []
    for c in cited:
        if c not in store:
            cited_text.append(f"{c} | (DOES NOT EXIST in this case file)")
        else:
            flag = "" if c in disclosed else "  [CITED BUT NEVER READ by the investigator]"
            cited_text.append(_item_text(store[c]) + flag)
    order = sorted(store, key=lambda x: tuple(int(p) for p in x[1:].split(".")))
    read = [f"{c} | {_title(store, c)}" for c in order if c in disclosed]
    unread = [f"{c} | {_title(store, c)}" for c in order if c not in disclosed]
    official = ref.get("official_conclusion")
    if official and len(official) > cfg.max_chars_official:
        official = official[:cfg.max_chars_official] + " [...]"
    alts = ref.get("alternatives_left_open") or []
    out = [f"# Task question\n{task_question(case['user_message'])}",
           f"# Investigator's decision\n{'CASE CLOSED' if closed else 'CASE NOT CLOSED'}",
           f"# Final answer\n{result['final_answer'].strip()}",
           f"# Investigator's last hypothesis state\n{last_state(result)}",
           "# Full text of evidence cited in the final answer\n" + ("\n\n".join(cited_text) or "(no citations)"),
           "# Titles of items the investigator READ\n" + ("\n".join(read) or "(none)"),
           "# Titles of items the investigator did NOT READ\n" + ("\n".join(unread) or "(none)"),
           "# REFERENCE (not shown to the investigator)\n"
           f"## Reference closure decision\n{'CASE CLOSED' if ref['closure'] == 'closed' else 'CASE NOT CLOSED'}"
           f" ({ref.get('closure_provenance', '')})\n\n"
           f"## Reference final answer (expert investigator)\n{ref.get('teacher_final_answer') or '(none)'}\n\n"
           f"## Official conclusion\n{official or '(none: this source has no official conclusion)'}\n\n"
           f"## Alternatives the record left open\n" +
           ("\n".join(f"- {a}" for a in alts) if alts else "(none listed)"),
           "Score the conclusion" + ("; the gap must be null (the case was closed)." if closed
                                     else " and the gap.")]
    return "\n\n".join(out), 1


def judge_requests(result: dict, case: dict, ref: dict, cfg: RewardConfig) -> dict:
    if cfg.phase < 2:
        return {}
    reqs = {}
    fetch = render_fetch_request(result, case, ref)
    if fetch:
        reqs["fetch"] = fetch
    hyp = render_hypothesis_request(result, case)
    if hyp:
        reqs["hypothesis"] = hyp
    clo = render_closure_request(result, case, ref, cfg)
    if clo:
        reqs["closure"] = clo
    return reqs


class JudgeFormatError(ValueError):

    def __init__(self, message: str, kind: str = "schema"):
        super().__init__(message)
        self.kind = kind


def _int_score(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) \
            or not 0 <= value <= 10:
        raise JudgeFormatError(f"score must be an integer 0-10, got {value!r}", "out_of_range")
    return int(value)


def parse_judge(dimension: str, obj: dict, n_items: int, closed: bool | None = None) -> dict:
    if not isinstance(obj, dict):
        raise JudgeFormatError("top level is not an object")
    if dimension in ("fetch", "hypothesis"):
        key, idx = ("steps", "step") if dimension == "fetch" else ("updates", "update")
        items = obj.get(key)
        if not isinstance(items, list) or len(items) != n_items:
            raise JudgeFormatError(f"expected {n_items} {key}, got "
                                   f"{len(items) if isinstance(items, list) else type(items).__name__}",
                                   "missing_items" if isinstance(items, list) else "schema")
        out = []
        for n, item in enumerate(items, 1):
            if not isinstance(item, dict) or item.get(idx) != n:
                raise JudgeFormatError(f"{key}[{n-1}] out of order or malformed")
            just = item.get("justification")
            if not isinstance(just, str) or not just.strip():
                raise JudgeFormatError(f"{key}[{n-1}] missing justification")
            row = {"score": _int_score(item.get("score")), "justification": just.strip()}
            if dimension == "hypothesis":
                if not isinstance(item.get("new_information"), bool):
                    raise JudgeFormatError(f"{key}[{n-1}] new_information must be boolean")
                row["new_information"] = item["new_information"]
            out.append(row)
        return {"items": out}
    if dimension == "closure":
        con = obj.get("conclusion")
        if not isinstance(con, dict) or not isinstance(con.get("justification"), str):
            raise JudgeFormatError("conclusion malformed")
        res = {"conclusion": {"score": _int_score(con.get("score")),
                              "justification": con["justification"].strip()}, "gap": None}
        gap = obj.get("gap")
        if closed is False:
            if not isinstance(gap, dict) or not isinstance(gap.get("justification"), str) or \
                    not isinstance(gap.get("claims_missing_evidence_that_was_available"), bool):
                raise JudgeFormatError("gap required for CASE NOT CLOSED")
            res["gap"] = {"score": _int_score(gap.get("score")),
                          "justification": gap["justification"].strip(),
                          "claims_missing_evidence_that_was_available":
                              gap["claims_missing_evidence_that_was_available"]}
        return res
    raise ValueError(dimension)


def compute(result: dict, case: dict, ref: dict, cfg: RewardConfig,
            judged: dict | None = None, excluded: set | None = None) -> dict:
    excluded = set(excluded or ())
    judged = {k: v for k, v in (judged or {}).items() if k not in excluded}
    sc = cfg.scaling
    kh = key_hit(result, case, ref, sc["key_hit_multiplier"])
    st = structure(result, case, cfg)
    closed = model_closed(result)
    comp = {"key_hit": kh["value"], "closure_status": closure_status(result, ref, cfg),
            "fetch_judge": 0.0, "hypothesis_judge": 0.0,
            "closure_conclusion": 0.0, "closure_gap": 0.0,
            "structure_missing_marker": st["missing_marker"], "structure_citation": st["citation"],
            "structure_turns": st["turns"], "structure_rerequest": st["rerequest"],
            "structure_invalid_fetch": st["invalid_fetch"]}
    raw = {}
    judge_failed = []
    if cfg.phase >= 2:
        f = judged.get("fetch")
        if f and "items" in f:
            vals = [scale(i["score"], sc["fetch_helpfulness"]) for i in f["items"]]
            comp["fetch_judge"] = sum(vals) / len(vals)
            raw["fetch"] = [i["score"] for i in f["items"]]
        elif f:
            judge_failed.append("fetch")
        h = judged.get("hypothesis")
        if h and "items" in h:
            total = sum(scale(i["score"], sc["hypothesis_update"]) for i in h["items"])
            lo, hi = sc["hypothesis_total_clip"]
            comp["hypothesis_judge"] = min(hi, max(lo, total))
            raw["hypothesis"] = [i["score"] for i in h["items"]]
        elif h:
            judge_failed.append("hypothesis")
        c = judged.get("closure")
        if c and "conclusion" in c and closed is not None:
            if closed:
                comp["closure_conclusion"] = scale(c["conclusion"]["score"], sc["closure_conclusion_closed"])
            else:
                comp["closure_conclusion"] = scale(c["conclusion"]["score"], sc["closure_conclusion_not_closed"])
                comp["closure_gap"] = scale(c["gap"]["score"], sc["closure_gap_not_closed"])
            raw["closure"] = {"conclusion": c["conclusion"]["score"],
                              "gap": c["gap"]["score"] if c.get("gap") else None}
        elif c:
            judge_failed.append("closure")
    dims = {"fetch": comp["fetch_judge"] + comp["key_hit"],
            "hypothesis": comp["hypothesis_judge"],
            "closure": comp["closure_status"] + comp["closure_conclusion"] + comp["closure_gap"],
            "structure": st["total"]}
    w = cfg.weights
    total = sum(w[k] * v for k, v in dims.items())
    return {"total": total, "dimensions": dims, "components": comp, "raw_judge_scores": raw,
            "judge_failed": judge_failed, "judge_excluded": sorted(excluded), "phase": cfg.phase,
            "model_closed": closed, "reference_closed": ref["closure"] == "closed",
            "key_hit_detail": kh, "structure_counts": st["counts"]}
