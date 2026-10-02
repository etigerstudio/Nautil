from __future__ import annotations

import re
from dataclasses import dataclass

from . import reward as r1
from .common import initial_record_ids
from .reward import JudgeFormatError
from .reward_v2 import _answer, render_closure as _render_closure_v2

YN = ("yes", "no")
VARIANTS = ("full", "outcome", "decision")
LEDGER = re.compile(r"H(\d+)\s*\[([^\]→>]+?)\s*(?:→|->)\s*([^\]]+?)\]")
HYP_ID = re.compile(r"^\s*(?:[-*]\s*)?\**H(\d+)\b", re.M)
DEFAULT_WEIGHTS = {
    "gate_wrong": 0.3, "kcov": 0.5,
    "C1a": 0.2, "C1b": 0.2, "C1b_partial_credit": 0.5, "C2": 0.2, "C3": 0.2, "C4": 0.2,
    "G1": 0.3, "G2": 0.4, "G3": 0.3, "G1no_G3": 0.5,
    "H2": 0.4, "H3_not": 0.3, "H4_not": 0.15, "H5": 0.15,
    "H1no_H4_not": 0.7, "H1no_H5": 0.3,
}
W = DEFAULT_WEIGHTS


@dataclass
class RewardV2Config:
    variant: str = "full"
    base: r1.RewardConfig = None
    weights: dict = None
    quality_only_if_correct: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "RewardV2Config":
        variant = raw.get("variant", "full")
        if variant not in VARIANTS:
            raise ValueError(f"reward.variant must be one of {VARIANTS}")
        extra = raw.get("weights_v2") or {}
        unknown = set(extra) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"unknown reward.weights_v2 keys: {sorted(unknown)}")
        return cls(variant=variant, base=r1.RewardConfig.from_dict({**raw, "phase": 2}),
                   weights={**DEFAULT_WEIGHTS, **extra},
                   quality_only_if_correct=bool(raw.get("quality_only_if_correct", False)))

    @property
    def dims(self) -> tuple:
        return {"full": ("hypothesis", "closure"), "outcome": ("closure",), "decision": ()}[self.variant]


def status_rank(s: str):
    s = s.strip().lower()
    if s.startswith(("ruled out", "excluded", "eliminated", "rejected", "refuted")):
        return -2
    if s.startswith(("weaken", "weaker", "disfavo", "less likely", "unlikely")):
        return -1
    if s.startswith(("open", "unresolved", "undetermined", "uncertain", "possible")):
        return 0
    if s.startswith(("favo", "support", "stronger", "strengthen", "leading", "likely", "more likely")):
        return 1
    if s.startswith(("confirmed", "established", "strongly")):
        return 2
    return None


def ledger(text: str) -> dict:
    out = {}
    for h, a, b in LEDGER.findall(text or ""):
        out.setdefault(f"H{h}", (a.strip().lower(), b.strip().lower()))
    return out


def hyp_ids(update: dict) -> list[str]:
    led = ledger(update["update"])
    if led:
        return list(led)
    for text in (update["update"], update["before"]):
        ids = list(dict.fromkeys(f"H{h}" for h in HYP_ID.findall(text or "")))
        if ids:
            return ids
    return ["H1"]


def render_hypothesis(result: dict, case: dict):
    ups = r1.hypothesis_updates(result)
    if not ups:
        return None
    out = [f"# Task question\n{r1.task_question(case['user_message'])}"]
    for n, up in enumerate(ups, 1):
        out.append(f"# Update {n}\n## Hypothesis state BEFORE\n{up['before']}\n\n"
                   f"## Evidence JUST RETURNED\n{up['returned']}\n\n"
                   f"## The investigator's update\n{up['update']}\n\n"
                   f"## Hypotheses to assess (in this order)\n{', '.join(hyp_ids(up))}")
    out.append(f"Answer for all {len(ups)} updates.")
    return "\n\n".join(out), len(ups), [hyp_ids(u) for u in ups]


def render_closure(result: dict, case: dict, ref: dict, base: r1.RewardConfig):
    got = _render_closure_v2(result, case, ref, base)
    if got is None:
        return None
    text, n = got
    return text.replace("Answer the CONCLUSION checklist and the GAP checklist.",
                        "Answer the CONCLUSION checklist (C1a, C1b, C2, C3 claims, C4) and the GAP checklist."), n, None


def judge_requests(result: dict, case: dict, ref: dict, cfg: RewardV2Config) -> dict:
    out = {}
    if cfg.variant == "decision":
        return out
    if cfg.variant == "full":
        h = render_hypothesis(result, case)
        if h:
            out["hypothesis"] = h
    c = render_closure(result, case, ref, cfg.base)
    if c:
        out["closure"] = c
    return out


def parse_judge(dimension: str, obj: dict, n_items: int, closed: bool | None = None, context=None) -> dict:
    if not isinstance(obj, dict):
        raise JudgeFormatError("top level is not an object")
    if dimension == "hypothesis":
        items = obj.get("updates")
        if not isinstance(items, list):
            raise JudgeFormatError("updates missing")
        if len(items) != n_items:
            raise JudgeFormatError(f"expected {n_items} updates, got {len(items)}", "missing_items")
        if context is None or len(context) != n_items:
            raise ValueError("hypothesis parsing needs the hypothesis ids of every update")
        out = []
        for n, (it, ids) in enumerate(zip(items, context), 1):
            if not isinstance(it, dict) or it.get("update") != n:
                raise JudgeFormatError(f"updates[{n-1}] out of order")
            hs = it.get("hypotheses")
            if not isinstance(hs, list) or [h.get("id") if isinstance(h, dict) else None for h in hs] != list(ids):
                raise JudgeFormatError(f"updates[{n-1}] hypotheses ids != {ids}", "missing_items")
            rows = []
            for h in hs:
                t, d = str(h.get("touched", "")).lower(), str(h.get("direction", "")).lower()
                if t not in YN or d not in ("supports", "weakens", "unrelated"):
                    raise JudgeFormatError(f"{h.get('id')} touched/direction invalid", "out_of_range")
                if (t == "yes") != (d != "unrelated"):
                    raise JudgeFormatError(f"{h.get('id')} touched={t} but direction={d}", "out_of_range")
                if not isinstance(h.get("why"), str) or not h["why"].strip():
                    raise JudgeFormatError(f"{h.get('id')} missing why")
                rows.append({"id": h["id"], "touched": t == "yes", "direction": d, "why": h["why"].strip()})
            touched = [r["id"] for r in rows if r["touched"]]
            main = it.get("main")
            if main in ("null", "", "none"):
                main = None
            if touched and main not in touched:
                raise JudgeFormatError(f"main {main!r} not a touched hypothesis {touched}", "out_of_range")
            if not touched and main is not None:
                raise JudgeFormatError(f"main {main!r} although nothing touched", "out_of_range")
            out.append({"hyps": rows, "main": main, "H5": _answer(it.get("H5"), "H5", YN)})
        return {"items": out}
    if dimension == "closure":
        con = obj.get("conclusion")
        if not isinstance(con, dict):
            raise JudgeFormatError("conclusion missing")
        res = {"conclusion": {"C1a": _answer(con.get("C1a"), "C1a", YN),
                              "C1b": _answer(con.get("C1b"), "C1b", ("yes", "partial", "no")),
                              "C2": _answer(con.get("C2"), "C2", YN),
                              "C4": _answer(con.get("C4"), "C4", YN)}, "gap": None}
        c3 = con.get("C3")
        if not isinstance(c3, dict) or not isinstance(c3.get("claims"), list):
            raise JudgeFormatError("C3 claims missing")
        claims = []
        for c in c3["claims"]:
            if not isinstance(c, dict) or str(c.get("supported", "")).lower() not in YN:
                raise JudgeFormatError("C3 claim malformed", "out_of_range")
            claims.append({"claim": str(c.get("claim", ""))[:200], "supported": c["supported"].lower() == "yes"})
        res["conclusion"]["C3"] = {"claims": claims, "why": str(c3.get("why", ""))}
        if closed is False:
            gap = obj.get("gap")
            if not isinstance(gap, dict):
                raise JudgeFormatError("gap required for CASE NOT CLOSED")
            g1 = _answer(gap.get("G1"), "G1", YN)
            g2 = _answer(gap.get("G2"), "G2", ("yes", "no", "n/a"))
            if (g1["answer"] == "no") != (g2["answer"] == "n/a"):
                raise JudgeFormatError(f"G1={g1['answer']} but G2={g2['answer']}", "out_of_range")
            res["gap"] = {"G1": g1, "G2": g2, "G3": _answer(gap.get("G3"), "G3", YN)}
        return res
    raise ValueError(f"reward v2.2 has no judge dimension {dimension!r}")


def fetch_program(result: dict, case: dict, ref: dict) -> dict:
    key = set(ref.get("key_fetchable_ids") or [])
    disclosed = set(initial_record_ids(case["user_message"]))
    fetched, first, n = set(), None, 0
    for turn in result["turns"]:
        if not r1.valid_fetch(turn):
            continue
        n += 1
        ids = turn["call"]["evidence_ids"]
        new = [i for i in ids if i not in disclosed]
        if first is None and set(new) & key:
            first = n
        fetched.update(new)
        disclosed.update(ids)
    kcov = len(fetched & key) / len(key) if key else 0.0
    ktime = max(0.0, (5 - first) / 4) if first else 0.0
    return {"Kcov": kcov, "Ktime": ktime, "first_key_request": first, "n_requests": n}


def hyp_program(item: dict, update_text: str) -> dict:
    led = ledger(update_text)

    def moved(hid):
        if hid not in led:
            return 0, False
        a, b = led[hid]
        if a == b:
            return 0, False
        ra, rb = status_rank(a), status_rank(b)
        return (0 if ra is None or rb is None else (rb > ra) - (rb < ra)), True
    touched = [h for h in item["hyps"] if h["touched"]]
    H1 = bool(touched)
    H2 = H3 = None
    if H1:
        m = next(h for h in item["hyps"] if h["id"] == item["main"])
        d, ch = moved(m["id"])
        H2 = ch and d == (1 if m["direction"] == "supports" else -1)
        H3 = any(not moved(h["id"])[1] for h in touched)
    H4 = any(moved(h["id"])[1] for h in item["hyps"] if not h["touched"])
    return {"H1": H1, "H2": H2, "H3": H3, "H4": H4, "H5": item["H5"]["answer"] == "yes",
            "changed": any(a != b for a, b in led.values())}


def hyp_item_score(p: dict, w: dict = W) -> float:
    if p["H1"]:
        return w["H2"] * p["H2"] + w["H3_not"] * (1 - p["H3"]) + w["H4_not"] * (1 - p["H4"]) + w["H5"] * p["H5"]
    return w["H1no_H4_not"] * (1 - p["H4"]) + w["H1no_H5"] * p["H5"]


def c3_score(claims: list) -> float:
    return (1 - sum(not c["supported"] for c in claims) / len(claims)) if claims else 0.0


def q_conc(c: dict, w: dict = W) -> float:
    b = {"yes": 1.0, "partial": w["C1b_partial_credit"], "no": 0.0}[c["C1b"]["answer"]]
    y = lambda q: 1.0 if c[q]["answer"] == "yes" else 0.0
    return (w["C1a"] * y("C1a") + w["C1b"] * b + w["C2"] * y("C2") + w["C3"] * c3_score(c["C3"]["claims"])
            + w["C4"] * y("C4"))


def q_gap(g: dict, w: dict = W) -> float:
    y = lambda q: 1.0 if g[q]["answer"] == "yes" else 0.0
    if g["G1"]["answer"] == "yes":
        return w["G1"] + w["G2"] * y("G2") + w["G3"] * y("G3")
    return w["G1no_G3"] * y("G3")


def answer_counts(judged: dict, hp: list[dict]) -> dict:
    out: dict = {}

    def add(q, is_yes):
        c = out.setdefault(q, [0, 0])
        c[0] += bool(is_yes)
        c[1] += 1
    for it in (judged.get("hypothesis") or {}).get("items") or []:
        for h in it["hyps"]:
            add("hyp_touched", h["touched"])
            if h["touched"]:
                add("hyp_direction_supports", h["direction"] == "supports")
        add("H5", it["H5"]["answer"] == "yes")
    for p in hp:
        add("H1_prog", p["H1"])
        if p["H1"]:
            add("H2_prog", p["H2"])
            add("H3_prog_missed_update", p["H3"])
        add("H4_prog_unfounded_change", p["H4"])
        add("ledger_changed", p["changed"])
    c = judged.get("closure") or {}
    con = c.get("conclusion") or {}
    for q in ("C1a", "C2", "C4"):
        if q in con:
            add(q, con[q]["answer"] == "yes")
    if "C1b" in con:
        add("C1b", con["C1b"]["answer"] == "yes")
        add("C1b_partial", con["C1b"]["answer"] == "partial")
    if "C3" in con:
        for cl in con["C3"]["claims"]:
            add("C3_claim_supported", cl["supported"])
        add("C3_no_claims", not con["C3"]["claims"])
    gap = c.get("gap") or {}
    if gap:
        add("G1", gap["G1"]["answer"] == "yes")
        if gap["G2"]["answer"] != "n/a":
            add("G2", gap["G2"]["answer"] == "yes")
        add("G3", gap["G3"]["answer"] == "yes")
    return out


def compute(result: dict, case: dict, ref: dict, cfg: RewardV2Config,
            judged: dict | None = None, excluded: set | None = None) -> dict:
    excluded = set(excluded or ())
    judged = {k: v for k, v in (judged or {}).items() if k not in excluded}
    ok = {k: v for k, v in judged.items() if isinstance(v, dict) and "error" not in v}
    judge_failed = sorted(k for k, v in judged.items() if isinstance(v, dict) and "error" in v)
    full = cfg.variant == "full"
    w = cfg.weights or W
    closed = r1.model_closed(result)
    ref_closed = ref["closure"] == "closed"
    D = 1.0 if (closed is not None and closed == ref_closed) else -1.0
    g = 1.0 if D > 0 else w["gate_wrong"]
    qc = qg = None
    if closed is None or cfg.variant == "decision" or (cfg.quality_only_if_correct and D < 0):
        outcome = D
    else:
        c = ok.get("closure")
        qc = q_conc(c["conclusion"], w) if c else 0.5
        if closed:
            outcome = D + (2 * qc - 1)
        else:
            qg = q_gap(c["gap"], w) if c else 0.5
            outcome = D + 0.5 * (2 * qc - 1) + 0.5 * (2 * qg - 1)
    kp = fetch_program(result, case, ref)
    fetch = w["kcov"] * kp["Kcov"] if kp["n_requests"] else 0.0
    h = ok.get("hypothesis") if full else None
    hp = []
    if h:
        for it, up in zip(h["items"], r1.hypothesis_updates(result)):
            p = hyp_program(it, up["update"])
            p["s"] = hyp_item_score(p, w)
            hp.append(p)
    hyp = 2 * (sum(p["s"] for p in hp) / len(hp)) - 1 if hp else 0.0
    st = r1.structure(result, case, cfg.base)
    claims = ((ok.get("closure") or {}).get("conclusion") or {}).get("C3", {}).get("claims")
    comp = {"D": D, "gate": g, "q_conc": qc, "q_gap": qg, "conclusion_part": outcome - D,
            "Kcov": kp["Kcov"], "Ktime_unrewarded": kp["Ktime"], "n_requests": kp["n_requests"],
            "c3_claims": len(claims) if claims is not None else None,
            "c3_supported_share": c3_score(claims) if claims else None,
            "structure_missing_marker": st["missing_marker"], "structure_citation": st["citation"],
            "structure_turns": st["turns"], "structure_rerequest": st["rerequest"],
            "structure_invalid_fetch": st["invalid_fetch"]}
    if full:
        total = outcome + g * (fetch + hyp) + st["total"]
        dims = {"outcome": outcome, "fetch": fetch, "hypothesis": hyp, "structure": st["total"]}
        comp.update({"g_fetch": g * fetch, "g_hyp": g * hyp, "judge_part": (outcome - D) + g * hyp})
    else:
        total = outcome + st["total"]
        dims = {"outcome": outcome, "structure": st["total"]}
        comp.update({"fetch_kcov_term_unrewarded": w["kcov"] * kp["Kcov"] if kp["n_requests"] else 0.0,
                     "judge_part": outcome - D})
    return {"total": total, "dimensions": dims, "components": comp,
            "raw_judge_scores": {}, "answers": answer_counts(ok, hp),
            "fetch_items": [], "hyp_items": [p["s"] for p in hp],
            "hyp_program": [{k: p[k] for k in ("H1", "H2", "H3", "H4", "H5", "changed", "s")} for p in hp],
            "judge_failed": judge_failed, "judge_excluded": sorted(excluded), "phase": 2,
            "reward_version": "2.2", "variant": cfg.variant,
            "model_closed": closed, "reference_closed": ref_closed, "structure_counts": st["counts"]}
