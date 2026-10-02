from __future__ import annotations

from dataclasses import dataclass

from . import reward as r1
from .common import initial_record_ids
from .reward import JudgeFormatError

YN = ("yes", "no")
FETCH_Q = ("F1", "F2", "F3")
HYP_Q = ("H1", "H2", "H3", "H4", "H5")
CONC_Q = ("C1", "C2", "C3", "C4")
GAP_Q = ("G1", "G2", "G3")
VARIANTS = ("full", "outcome")
DEFAULT_WEIGHTS = {
    "gate_wrong": 0.3,
    "C1": 0.35, "C2": 0.25, "C3": 0.20, "C4": 0.20, "C1_partial_credit": 0.5,
    "G1": 0.3, "G2": 0.4, "G3": 0.3,
    "H2": 0.4, "H3_not": 0.3, "H4_not": 0.15, "H5": 0.15,
    "H1no_H4_not": 0.7, "H1no_H5": 0.3,
    "K1": 0.3, "K2": 0.2, "F1": 0.2, "F2_not": 0.2, "F3_not": 0.1,
}
W = DEFAULT_WEIGHTS


@dataclass
class RewardV2Config:
    variant: str = "full"
    base: r1.RewardConfig = None
    weights: dict = None

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
                   weights={**DEFAULT_WEIGHTS, **extra})

    @property
    def dims(self) -> tuple:
        return ("fetch", "hypothesis", "closure") if self.variant == "full" else ("closure",)


def fetch_program(result: dict, case: dict, ref: dict) -> list[dict]:
    key = set(ref.get("key_fetchable_ids") or [])
    disclosed = set(initial_record_ids(case["user_message"]))
    out = []
    for turn in result["turns"]:
        if not r1.valid_fetch(turn):
            continue
        ids = list(turn["call"]["evidence_ids"])
        new = [i for i in ids if i not in disclosed]
        hits = len(set(new) & key)
        out.append({"turn": turn["turn"], "n_ids": len(ids), "n_new": len(new), "key_new": hits,
                    "K1": 1.0 if hits > 0 else 0.0, "K2": hits / len(ids) if ids else 0.0,
                    "unread_key_before": sorted(key - disclosed)})
        disclosed.update(ids)
    return out


def render_fetch(result: dict, case: dict, ref: dict):
    steps = r1.fetch_steps(result, case)
    if not steps:
        return None
    prog = fetch_program(result, case, ref)
    assert len(prog) == len(steps)
    store = {i["evidence_id"]: i for i in case["evidence_items"]}
    out = [f"# Task question\n{r1.task_question(case['user_message'])}"]
    for n, (st, pg) in enumerate(zip(steps, prog), 1):
        unread_key = "\n".join(f"{k} | {r1._title(store, k)}" for k in pg["unread_key_before"]) or "(none)"
        out.append(f"# Request {n}\n## Hypothesis state at the moment of the request\n{st['state']}\n\n"
                   f"## Already read before this request (IDs)\n{', '.join(st['already_read_before']) or '(none)'}\n\n"
                   f"## Requested items\n" + "\n".join(st["requested"]) +
                   f"\n\n## Stated reason\n{st['reason']}\n\n"
                   f"## KEY items still UNREAD at this moment (not shown to the investigator)\n{unread_key}")
    out.append(f"Answer F1-F3 for all {len(steps)} requests.")
    return "\n\n".join(out), len(steps)


def render_hypothesis(result: dict, case: dict):
    ups = r1.hypothesis_updates(result)
    if not ups:
        return None
    out = [f"# Task question\n{r1.task_question(case['user_message'])}"]
    for n, up in enumerate(ups, 1):
        out.append(f"# Update {n}\n## Hypothesis state BEFORE\n{up['before']}\n\n"
                   f"## Evidence JUST RETURNED\n{up['returned']}\n\n"
                   f"## The investigator's update\n{up['update']}")
    out.append(f"Answer H1-H5 for all {len(ups)} updates.")
    return "\n\n".join(out), len(ups)


def render_closure(result: dict, case: dict, ref: dict, base: r1.RewardConfig):
    got = r1.render_closure_request(result, case, ref, base)
    if got is None:
        return None
    text, _ = got
    head, _last = text.rsplit("\n\n", 1)
    closed = r1.model_closed(result)
    tail = ("Answer the CONCLUSION checklist; the gap must be null (the case was closed)." if closed
            else "Answer the CONCLUSION checklist and the GAP checklist.")
    return head + "\n\n" + tail, 1


def judge_requests(result: dict, case: dict, ref: dict, cfg: RewardV2Config) -> dict:
    out = {}
    if cfg.variant == "full":
        for dim, got in (("fetch", render_fetch(result, case, ref)),
                         ("hypothesis", render_hypothesis(result, case))):
            if got:
                out[dim] = got
    got = render_closure(result, case, ref, cfg.base)
    if got:
        out["closure"] = got
    return out


def _answer(obj, q, allowed):
    if not isinstance(obj, dict):
        raise JudgeFormatError(f"{q} not an object")
    why, ans = obj.get("why"), obj.get("answer")
    if not isinstance(why, str) or not why.strip():
        raise JudgeFormatError(f"{q} missing why")
    if not isinstance(ans, str) or ans.strip().lower() not in allowed:
        kind = "inconsistent" if isinstance(ans, str) and ans.strip().lower() in ("n/a", "na", "not applicable") \
            else "out_of_range"
        raise JudgeFormatError(f"{q} answer {ans!r} not in {allowed}", kind)
    return {"answer": ans.strip().lower(), "why": why.strip()}


def parse_judge(dimension: str, obj: dict, n_items: int, closed: bool | None = None) -> dict:
    if not isinstance(obj, dict):
        raise JudgeFormatError("top level is not an object")
    if dimension in ("fetch", "hypothesis"):
        key, idx, qs = ("requests", "request", FETCH_Q) if dimension == "fetch" else ("updates", "update", HYP_Q)
        items = obj.get(key)
        if not isinstance(items, list):
            raise JudgeFormatError(f"{key} missing")
        if len(items) != n_items:
            raise JudgeFormatError(f"expected {n_items} {key}, got {len(items)}", "missing_items")
        out = []
        for n, it in enumerate(items, 1):
            if not isinstance(it, dict) or it.get(idx) != n:
                raise JudgeFormatError(f"{key}[{n-1}] out of order")
            row = {}
            for q in qs:
                allowed = YN + ("n/a",) if q == "H2" else YN
                row[q] = _answer(it.get(q), q, allowed)
            if dimension == "hypothesis":
                if row["H1"]["answer"] == "no":
                    row["H2"]["answer"] = "n/a"
                elif row["H2"]["answer"] == "n/a":
                    raise JudgeFormatError("H2 n/a although H1 yes", "inconsistent")
            out.append(row)
        return {"items": out}
    if dimension == "closure":
        con = obj.get("conclusion")
        if not isinstance(con, dict):
            raise JudgeFormatError("conclusion missing")
        res = {"conclusion": {q: _answer(con.get(q), q, ("yes", "partial", "no") if q == "C1" else YN)
                              for q in CONC_Q}, "gap": None}
        if closed is False:
            gap = obj.get("gap")
            if not isinstance(gap, dict):
                raise JudgeFormatError("gap required for CASE NOT CLOSED")
            res["gap"] = {q: _answer(gap.get(q), q, YN) for q in GAP_Q}
        return res
    raise ValueError(dimension)


def yes(item, q) -> float:
    return 1.0 if item[q]["answer"] == "yes" else 0.0


def hyp_item_score(it: dict, w: dict = W) -> float:
    if it["H1"]["answer"] == "yes":
        return (w["H2"] * yes(it, "H2") + w["H3_not"] * (1 - yes(it, "H3")) + w["H4_not"] * (1 - yes(it, "H4"))
                + w["H5"] * yes(it, "H5"))
    return w["H1no_H4_not"] * (1 - yes(it, "H4")) + w["H1no_H5"] * yes(it, "H5")


def fetch_program_score(pg: dict, w: dict = W) -> float:
    return w["K1"] * pg["K1"] + w["K2"] * pg["K2"]


def fetch_item_score(pg: dict, it: dict | None, w: dict = W) -> float:
    prog = fetch_program_score(pg, w)
    if it is None:
        return prog
    return prog + w["F1"] * yes(it, "F1") + w["F2_not"] * (1 - yes(it, "F2")) + w["F3_not"] * (1 - yes(it, "F3"))


def q_conc(c: dict, w: dict = W) -> float:
    c1 = {"yes": 1.0, "partial": w["C1_partial_credit"], "no": 0.0}[c["C1"]["answer"]]
    return w["C1"] * c1 + w["C2"] * yes(c, "C2") + w["C3"] * yes(c, "C3") + w["C4"] * yes(c, "C4")


def q_gap(g: dict, w: dict = W) -> float:
    return w["G1"] * yes(g, "G1") + w["G2"] * yes(g, "G2") + w["G3"] * yes(g, "G3")


def answer_counts(judged: dict) -> dict:
    out: dict = {}

    def add(q, ans):
        if ans == "n/a":
            return
        c = out.setdefault(q, [0, 0])
        c[0] += ans == "yes"
        c[1] += 1
        if q == "C1":
            p = out.setdefault("C1_partial", [0, 0])
            p[0] += ans == "partial"
            p[1] += 1
    for dim in ("fetch", "hypothesis"):
        for it in (judged.get(dim) or {}).get("items") or []:
            for q, a in it.items():
                add(q, a["answer"])
    c = judged.get("closure") or {}
    for part in ("conclusion", "gap"):
        for q, a in (c.get(part) or {}).items():
            add(q, a["answer"])
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
    if closed is None:
        outcome = D
    else:
        c = ok.get("closure")
        qc = q_conc(c["conclusion"], w) if c else 0.5
        if closed:
            outcome = D + (2 * qc - 1)
        else:
            qg = q_gap(c["gap"], w) if c else 0.5
            outcome = D + 0.5 * (2 * qc - 1) + 0.5 * (2 * qg - 1)
    prog = fetch_program(result, case, ref)
    f = ok.get("fetch") if full else None
    items = f["items"] if f else [None] * len(prog)
    fscores = [fetch_item_score(pg, it, w) for pg, it in zip(prog, items)]
    fetch = sum(fscores) / len(fscores) if fscores else 0.0
    fetch_prog = (sum(fetch_program_score(p, w) for p in prog) / len(prog)) if prog else 0.0
    h = ok.get("hypothesis") if full else None
    hscores = [hyp_item_score(it, w) for it in h["items"]] if h else []
    hyp = 2 * (sum(hscores) / len(hscores)) - 1 if hscores else 0.0
    st = r1.structure(result, case, cfg.base)
    comp = {"D": D, "gate": g, "q_conc": qc, "q_gap": qg, "conclusion_part": outcome - D,
            "K1_mean": (sum(p["K1"] for p in prog) / len(prog)) if prog else None,
            "K2_mean": (sum(p["K2"] for p in prog) / len(prog)) if prog else None,
            "structure_missing_marker": st["missing_marker"], "structure_citation": st["citation"],
            "structure_turns": st["turns"], "structure_rerequest": st["rerequest"],
            "structure_invalid_fetch": st["invalid_fetch"]}
    if full:
        total = outcome + g * (fetch + hyp) + st["total"]
        dims = {"outcome": outcome, "fetch": fetch, "hypothesis": hyp, "structure": st["total"]}
        comp.update({"fetch_program": fetch_prog, "fetch_judge_part": fetch - fetch_prog,
                     "g_fetch": g * fetch, "g_hyp": g * hyp,
                     "judge_part": (outcome - D) + g * (fetch - fetch_prog + hyp)})
    else:
        total = outcome + st["total"]
        dims = {"outcome": outcome, "structure": st["total"]}
        comp["fetch_program_unrewarded"] = fetch_prog
        comp["judge_part"] = outcome - D
    return {"total": total, "dimensions": dims, "components": comp,
            "raw_judge_scores": {}, "answers": answer_counts(ok),
            "fetch_items": fscores, "hyp_items": hscores,
            "fetch_program_items": [{k: p[k] for k in ("turn", "n_ids", "n_new", "key_new", "K1", "K2")}
                                    for p in prog],
            "judge_failed": judge_failed, "judge_excluded": sorted(excluded), "phase": 2,
            "reward_version": 2, "variant": cfg.variant,
            "model_closed": closed, "reference_closed": ref_closed, "structure_counts": st["counts"]}
