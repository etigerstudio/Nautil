from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nautil_harness_v2_2 import run_case

from nautil_rlvr import reward as R
from nautil_rlvr.testing import mock_judge, synthetic

CASES, REFS, _ = synthetic.make_dataset(4)
CASE_C, REF_C = CASES[0], REFS[0]
CASE_N, REF_N = CASES[1], REFS[1]
CALL = synthetic.call_text


def run(case: dict, replies: list[str], max_turns: int = 15) -> dict:
    it = iter(replies)

    def gen(model, tok, messages, tools, device, limit):
        return next(it), 100, False
    return run_case(None, None, "system", case, None, 32768, max_turns, generate_fn=gen)


def run_variant(case: dict, variant: str, seed: int = 0) -> dict:
    rng = random.Random(seed)
    returned: list[str] = []
    turn = 0

    def gen(model, tok, messages, tools, device, limit):
        nonlocal turn
        turn += 1
        returned.clear()
        for m in messages:
            if m["role"] == "tool":
                returned.extend(i["text"] for i in json.loads(m["content"]).get("evidence_items", []))
        return synthetic.scripted_reply(variant, turn, list(returned), None, rng), 100, False
    return run_case(None, None, "system", case, None, 32768, 15, generate_fn=gen)


def judged_by_mock(result: dict, case: dict, ref: dict, cfg: R.RewardConfig) -> dict:
    out = {}
    for dim, (text, n) in R.judge_requests(result, case, ref, cfg).items():
        out[dim] = R.parse_judge(dim, mock_judge.score(dim, text), n, R.model_closed(result))
    return out


P1 = R.RewardConfig.from_dict({"phase": 1})
P2 = R.RewardConfig.from_dict({"phase": 2})
FINAL_OK_C = "CASE CLOSED. Bearing seizure (H1) shown by E2.2, E2.5, E2.7."
FINAL_OK_N = "CASE NOT CLOSED - E2.2, E2.5, E2.7 are inconclusive between H1 and H2. Unresolved: teardown."


def fetch_key():
    return CALL("## Current hypotheses\nH1 - a\nH2 - b", ["E2.2", "E2.5", "E2.7"], "Which hypothesis do the logs support?")


def closure_value(case, ref, final, conclusion_raw, gap_raw):
    res = run(case, [fetch_key(), final])
    closed = R.model_closed(res)
    judged = {"closure": {"conclusion": {"score": conclusion_raw, "justification": "x"},
                          "gap": None if closed else {"score": gap_raw, "justification": "x",
                                                      "claims_missing_evidence_that_was_available": False}}}
    return R.compute(res, case, ref, P2, judged)["dimensions"]["closure"]


def test_closure_matrix_ranges():
    cells = {"closed/closed": (CASE_C, REF_C, FINAL_OK_C, (-0.5, 1.5)),
             "closed/not_closed": (CASE_N, REF_N, FINAL_OK_C, (-1.5, 0.5)),
             "not_closed/closed": (CASE_C, REF_C, FINAL_OK_N, (-1.5, 0.5)),
             "not_closed/not_closed": (CASE_N, REF_N, FINAL_OK_N, (-0.5, 1.5))}
    for name, (case, ref, final, (lo, hi)) in cells.items():
        assert abs(closure_value(case, ref, final, 0, 0) - lo) < 1e-9, name
        assert abs(closure_value(case, ref, final, 10, 10) - hi) < 1e-9, name
    assert R.compute(run(CASE_C, [fetch_key(), FINAL_OK_C]), CASE_C, REF_C, P1)["dimensions"]["closure"] == 0.5
    assert R.compute(run(CASE_C, [fetch_key(), FINAL_OK_N]), CASE_C, REF_C, P1)["dimensions"]["closure"] == -0.5


def test_key_hit_formula():
    res = run(CASE_C, [CALL("x", ["E2.2", "E2.1", "E2.3"], "first request here"),
                       CALL("y", ["E2.4", "E2.6", "E2.8", "E2.9", "E2.10", "E2.11"], "second request here"),
                       FINAL_OK_C])
    kh = R.key_hit(res, CASE_C, REF_C)
    assert abs(kh["per_request_mean"] - 0.5) < 1e-9
    assert abs(kh["whole"] - 1 / 3) < 1e-9
    assert abs(kh["value"] - (0.5 + 1 / 3) / 2) < 1e-9
    full = R.key_hit(run(CASE_C, [fetch_key(), FINAL_OK_C]), CASE_C, REF_C)
    assert full["value"] == 1.0
    none = R.key_hit(run(CASE_C, [FINAL_OK_C]), CASE_C, REF_C)
    assert none["value"] == 0.0


def test_turn_penalty_and_cap():
    fetches = [CALL("x", [f"E2.{j}"], "reading one more item") for j in range(1, 8)]
    res = run(CASE_C, fetches + [FINAL_OK_C])
    assert abs(R.structure(res, CASE_C, P1)["turns"] - (-0.2)) < 1e-9
    many = [CALL("x", ["E1.1"], "re-reading the record item") for _ in range(15)]
    res = run(CASE_C, many)
    st = R.structure(res, CASE_C, P1)
    assert abs(st["turns"] - (-0.9)) < 1e-9
    assert st["missing_marker"] == -2.0
    cfg = R.RewardConfig.from_dict({"structure": {"turn_free": 2}})
    assert R.structure(res, CASE_C, cfg)["turns"] == -1.0


def test_rerequest_penalty_and_cap():
    ids = ["E1.1", "E1.2"]
    reqs = [CALL("x", ids, "re-reading the record items") for _ in range(6)]
    st = R.structure(run(CASE_C, reqs + [FINAL_OK_C]), CASE_C, P1)
    assert st["counts"]["rerequested_items"] == 12 and abs(st["rerequest"] - (-0.2)) < 1e-9
    reqs = [CALL("x", [f"E2.{j}" for j in range(1, 13)], "reading everything at once")] + \
           [CALL("x", [f"E2.{j}" for j in range(1, 13)], "reading everything again") for _ in range(2)]
    st = R.structure(run(CASE_C, reqs + [FINAL_OK_C]), CASE_C, P1)
    assert st["counts"]["rerequested_items"] == 24 and st["rerequest"] == -1.0


def test_invalid_fetch_penalty_and_cap():
    bad = CALL("x", [f"E2.{j}" for j in range(1, 13)] + ["E1.1"], "too many ids at once")
    st = R.structure(run(CASE_C, [bad] * 3 + [FINAL_OK_C]), CASE_C, P1)
    assert st["counts"]["invalid_fetches"] == 3 and abs(st["invalid_fetch"] - (-0.6)) < 1e-9
    nonjson = "x\n\n<tool_call>\n<function=request_evidence>\n<parameter=evidence_ids>\nE2.1, E2.2\n</parameter>\n<parameter=reason>\nnon-json id list here\n</parameter>\n</function>\n</tool_call>"
    st = R.structure(run(CASE_C, [nonjson] * 6 + [FINAL_OK_C]), CASE_C, P1)
    assert st["counts"]["invalid_fetches"] == 6 and st["invalid_fetch"] == -1.0


def test_citations_and_marker():
    st = R.structure(run(CASE_C, [fetch_key(), "CASE CLOSED. Bearing seizure."]), CASE_C, P1)
    assert st["citation"] == -0.25
    st = R.structure(run(CASE_C, [fetch_key(), "CASE CLOSED. Bearing seizure E2.2 and E2.9."]), CASE_C, P1)
    assert st["citation"] == -0.5
    st = R.structure(run(CASE_C, [fetch_key(), FINAL_OK_C]), CASE_C, P1)
    assert st["citation"] == 0.0 and st["missing_marker"] == 0.0
    res = run(CASE_C, [fetch_key(), "Bearing seizure per E2.2."])
    r = R.compute(res, CASE_C, REF_C, P1)
    assert r["components"]["structure_missing_marker"] == -2.0 and r["components"]["closure_status"] == 0.0


def test_judge_scaling_and_parsing():
    s = P2.scaling
    assert R.scale(10, s["fetch_helpfulness"]) == 0.5 and R.scale(0, s["fetch_helpfulness"]) == 0.0
    assert R.scale(5, s["hypothesis_update"]) == 0.0 and R.scale(10, s["hypothesis_update"]) == 0.5
    try:
        R.parse_judge("fetch", {"steps": [{"step": 1, "justification": "a", "score": 11}]}, 1)
        raise AssertionError("out of range accepted")
    except R.JudgeFormatError as exc:
        assert exc.kind == "out_of_range"
    try:
        R.parse_judge("fetch", {"steps": []}, 2)
        raise AssertionError("missing items accepted")
    except R.JudgeFormatError as exc:
        assert exc.kind == "missing_items"
    res = run(CASE_C, [fetch_key(), CALL(synthetic.state_block("favored", "weakened", "u", None), ["E2.1"], "one more item"),
                       CALL(synthetic.state_block("favored", "weakened", "u", None), ["E2.3"], "one more item"),
                       CALL(synthetic.state_block("favored", "weakened", "u", None), ["E2.4"], "one more item"),
                       FINAL_OK_C])
    judged = {"hypothesis": {"items": [{"score": 10}, {"score": 10}, {"score": 10}]}}
    assert R.compute(res, CASE_C, REF_C, P2, judged)["components"]["hypothesis_judge"] == 1.0
    judged = {"hypothesis": {"items": [{"score": 0}] * 3}}
    assert R.compute(res, CASE_C, REF_C, P2, judged)["components"]["hypothesis_judge"] == -1.0
    r = R.compute(res, CASE_C, REF_C, P2, {"hypothesis": {"items": [{"score": 0}] * 3}}, excluded={"hypothesis"})
    assert r["components"]["hypothesis_judge"] == 0.0 and r["judge_excluded"] == ["hypothesis"]


def test_judge_contexts_are_focused():
    res = run_variant(CASE_C, "bulk")
    reqs = R.judge_requests(res, CASE_C, REF_C, P2)
    fetch_text, n = reqs["fetch"]
    assert n == 2 and "KEY EVIDENCE" in fetch_text and "REFERENCE" not in fetch_text
    assert "supports H1" not in fetch_text
    hyp_text, n = reqs["hypothesis"]
    assert n == 1 and "Evidence JUST RETURNED" in hyp_text and "KEY EVIDENCE" not in hyp_text
    clo_text, _ = reqs["closure"]
    assert "Reference final answer" in clo_text and "did NOT READ" in clo_text


BEHAVIOURS = [("good investigation", "careful"), ("fetch everything", "bulk"),
              ("never close", "never_close"), ("premature close", "premature"),
              ("random hypothesis flips", "flip")]


def ranking(case, ref, seed: int = 0) -> list[dict]:
    rows = []
    for label, variant in BEHAVIOURS:
        res = run_variant(case, variant, seed)
        r = R.compute(res, case, ref, P2, judged_by_mock(res, case, ref, P2))
        r1 = R.compute(res, case, ref, P1)
        rows.append({"behaviour": label, "total_phase2": round(r["total"], 3), "total_phase1": round(r1["total"], 3),
                     **{k: round(v, 3) for k, v in r["dimensions"].items()},
                     "marker": res["conclusion_marker"], "turns": res["assistant_turns"]})
    return rows


def test_adversarial_ranking():
    for seed in range(3):
        totals = [row["total_phase2"] for row in ranking(CASE_C, REF_C, seed)]
        assert totals == sorted(totals, reverse=True) and len(set(totals)) == len(totals), totals


if __name__ == "__main__":
    out = {"reference_closed": ranking(CASE_C, REF_C), "reference_not_closed": ranking(CASE_N, REF_N)}
    print(json.dumps(out, indent=1))
