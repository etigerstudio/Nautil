from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nautil_harness_v2_2 import run_case

from nautil_rlvr import reward as R1
from nautil_rlvr import reward_v2 as R
from nautil_rlvr.reward import JudgeFormatError
from nautil_rlvr.testing import mock_judge_v2, synthetic

CASES, REFS, _ = synthetic.make_dataset(4)
CASE_C, REF_C = CASES[0], REFS[0]
CASE_N, REF_N = CASES[1], REFS[1]
CALL = synthetic.call_text
FULL = R.RewardV2Config.from_dict({"version": 2, "variant": "full", "phase": 2})
OUTC = R.RewardV2Config.from_dict({"version": 2, "variant": "outcome", "phase": 2})
V1P1 = R1.RewardConfig.from_dict({"phase": 1})
EPS = 1e-9


def run(case: dict, replies: list[str], max_turns: int = 15) -> dict:
    it = iter(replies)

    def gen(model, tok, messages, tools, device, limit):
        return next(it), 100, False
    return run_case(None, None, "system", case, None, 32768, max_turns, generate_fn=gen)


def close(a, b):
    return abs(a - b) < 1e-9


def yn(b):
    return {"answer": "yes" if b else "no", "why": "x"}


def conc(c1="yes", c2=True, c3=True, c4=True):
    return {"C1": {"answer": c1, "why": "x"}, "C2": yn(c2), "C3": yn(c3), "C4": yn(c4)}


def gap(g1=True, g2=True, g3=True):
    return {"G1": yn(g1), "G2": yn(g2), "G3": yn(g3)}


def hyp_item(h1, h2, h3, h4, h5):
    return {"H1": yn(h1), "H2": {"answer": "n/a" if h2 is None else ("yes" if h2 else "no"), "why": "x"},
            "H3": yn(h3), "H4": yn(h4), "H5": yn(h5)}


def fetch_item(f1, f2, f3):
    return {"F1": yn(f1), "F2": yn(f2), "F3": yn(f3)}


FINAL_C = "CASE CLOSED. Bearing seizure (H1) shown by E2.2, E2.5, E2.7. Discounted: H2 electrical fault."
FINAL_N = "CASE NOT CLOSED - E2.2, E2.5, E2.7 are inconclusive between H1 and H2. Unresolved: no teardown report."
KEYS = ["E2.2", "E2.5", "E2.7"]


def one_fetch(final):
    return [CALL("## Current hypotheses\nH1 - a\nH2 - b", KEYS, "Which hypothesis do the logs support?"), final]


def outcome_of(case, ref, final, c, g=None):
    res = run(case, one_fetch(final))
    closed = R1.model_closed(res)
    judged = {"closure": {"conclusion": c, "gap": g if closed is False else None}}
    return R.compute(res, case, ref, OUTC, judged)


def test_outcome_closed_right():
    r = outcome_of(CASE_C, REF_C, FINAL_C, conc())
    assert r["components"]["D"] == 1 and r["components"]["gate"] == 1.0
    assert close(r["dimensions"]["outcome"], 2.0)
    r = outcome_of(CASE_C, REF_C, FINAL_C, conc("partial", True, False, True))
    assert close(r["components"]["q_conc"], 0.625) and close(r["dimensions"]["outcome"], 1.25)
    r = outcome_of(CASE_C, REF_C, FINAL_C, conc("no", False, False, False))
    assert close(r["dimensions"]["outcome"], 0.0) and r["components"]["q_gap"] is None


def test_outcome_closed_wrong():
    r = outcome_of(CASE_N, REF_N, FINAL_C, conc())
    assert r["components"]["D"] == -1 and r["components"]["gate"] == 0.3
    assert close(r["dimensions"]["outcome"], 0.0)
    r = outcome_of(CASE_N, REF_N, FINAL_C, conc("no", False, False, False))
    assert close(r["dimensions"]["outcome"], -2.0)


def test_outcome_not_closed_right():
    r = outcome_of(CASE_N, REF_N, FINAL_N, conc(), gap())
    assert r["components"]["D"] == 1 and close(r["dimensions"]["outcome"], 2.0)
    r = outcome_of(CASE_N, REF_N, FINAL_N, conc("partial", True, False, False), gap(True, False, True))
    assert close(r["components"]["q_conc"], 0.425) and close(r["components"]["q_gap"], 0.6)
    assert close(r["dimensions"]["outcome"], 1.025)
    r = outcome_of(CASE_N, REF_N, FINAL_N, conc("no", False, False, False), gap(False, False, False))
    assert close(r["dimensions"]["outcome"], 0.0)


def test_outcome_not_closed_wrong():
    r = outcome_of(CASE_C, REF_C, FINAL_N, conc(), gap())
    assert r["components"]["D"] == -1 and close(r["dimensions"]["outcome"], 0.0)
    r = outcome_of(CASE_C, REF_C, FINAL_N, conc("no", False, False, False), gap(False, False, False))
    assert close(r["dimensions"]["outcome"], -2.0)


def test_no_marker():
    res = run(CASE_C, one_fetch("Bearing seizure per E2.2."))
    assert "closure" not in R.judge_requests(res, CASE_C, REF_C, FULL)
    r = R.compute(res, CASE_C, REF_C, FULL, {})
    assert r["components"]["D"] == -1 and r["dimensions"]["outcome"] == -1 and r["components"]["gate"] == 0.3
    assert r["components"]["structure_missing_marker"] == -2.0 and r["components"]["q_conc"] is None


def three_fetches(final):
    s = synthetic.state_block
    return [CALL(s("open", "open", "Initial hypotheses.", "logs?"), ["E2.2"], "Does log 2 support H1?"),
            CALL(s("favored", "open", "E2.2 supports H1.", "logs?"), ["E2.5", "E2.1"], "Does log 5 support H1?"),
            CALL(s("favored", "weakened", "E2.5 supports H1.", "logs?"), ["E2.7"], "Does log 7 support H1?"),
            final]


def full_judged(res, case, ref, fetch_items, hyp_items, c, g=None):
    reqs = R.judge_requests(res, case, ref, FULL)
    assert reqs["fetch"][1] == len(fetch_items) and reqs["hypothesis"][1] == len(hyp_items), \
        (reqs["fetch"][1], reqs["hypothesis"][1])
    closed = R1.model_closed(res)
    return {"fetch": {"items": fetch_items}, "hypothesis": {"items": hyp_items},
            "closure": {"conclusion": c, "gap": g if closed is False else None}}


def test_gate_and_total():
    fi = [fetch_item(True, False, False)] * 3
    hi = [hyp_item(True, True, False, False, True)] * 2
    right = run(CASE_C, three_fetches(FINAL_C))
    wrong = run(CASE_N, three_fetches(FINAL_C))
    r1 = R.compute(right, CASE_C, REF_C, FULL, full_judged(right, CASE_C, REF_C, fi, hi, conc()))
    r2 = R.compute(wrong, CASE_N, REF_N, FULL, full_judged(wrong, CASE_N, REF_N, fi, hi, conc()))
    f = ((0.3 + 0.2 + 0.5) + (0.3 + 0.1 + 0.5) + (0.3 + 0.2 + 0.5)) / 3
    for r in (r1, r2):
        assert close(r["dimensions"]["fetch"], f), r["dimensions"]
        assert close(r["dimensions"]["hypothesis"], 1.0)
    assert close(r1["total"], 2.0 + 1.0 * (f + 1.0) + r1["dimensions"]["structure"])
    assert close(r2["total"], 0.0 + 0.3 * (f + 1.0) + r2["dimensions"]["structure"])
    assert close(r1["components"]["g_fetch"], f) and close(r2["components"]["g_fetch"], 0.3 * f)
    assert close(r2["components"]["g_hyp"], 0.3)


def test_hypothesis_branches():
    res = run(CASE_C, three_fetches(FINAL_C))
    fi = [fetch_item(True, False, False)] * 3
    cases = [
        ([hyp_item(True, True, False, False, True)] * 2, 1.0),
        ([hyp_item(True, False, True, False, False)] * 2, 2 * 0.15 - 1),
        ([hyp_item(True, False, False, True, True)] * 2, 2 * 0.45 - 1),
        ([hyp_item(False, None, False, False, True)] * 2, 1.0),
        ([hyp_item(False, None, False, True, False)] * 2, -1.0),
        ([hyp_item(False, None, False, False, False), hyp_item(True, True, False, False, True)],
         2 * ((0.7 + 1.0) / 2) - 1),
    ]
    for items, want in cases:
        r = R.compute(res, CASE_C, REF_C, FULL, full_judged(res, CASE_C, REF_C, fi, items, conc()))
        assert close(r["dimensions"]["hypothesis"], want), (want, r["dimensions"]["hypothesis"])
    missed = R.hyp_item_score(hyp_item(True, False, True, False, True))
    no_news_ok = R.hyp_item_score(hyp_item(False, None, False, False, True))
    assert missed < no_news_ok, (missed, no_news_ok)
    r = R.compute(run(CASE_C, one_fetch(FINAL_C)), CASE_C, REF_C, FULL, {"closure": {"conclusion": conc(), "gap": None}})
    assert r["dimensions"]["hypothesis"] == 0.0
    r = R.compute(res, CASE_C, REF_C, FULL, full_judged(res, CASE_C, REF_C, fi, cases[1][0], conc()),
                  excluded={"hypothesis"})
    assert r["dimensions"]["hypothesis"] == 0.0 and r["judge_excluded"] == ["hypothesis"]


def test_hypothesis_parser_h2():
    ok = {"updates": [{"update": 1, **hyp_item(False, True, False, False, True)}]}
    p = R.parse_judge("hypothesis", json.loads(json.dumps(ok)), 1)
    assert p["items"][0]["H2"]["answer"] == "n/a"
    bad = {"updates": [{"update": 1, **hyp_item(True, None, False, False, True)}]}
    try:
        R.parse_judge("hypothesis", bad, 1)
        raise AssertionError("H2 n/a with H1 yes accepted")
    except JudgeFormatError as exc:
        assert exc.kind == "inconsistent"
    h3na = {"updates": [{"update": 1, **hyp_item(False, None, False, False, True), "H3": {"answer": "n/a", "why": "x"}}]}
    for obj, n, kind in [(h3na, 1, "inconsistent"), ({"updates": []}, 1, "missing_items"),
                         ({"requests": [{"request": 1, **fetch_item(True, False, False), "F2": {"answer": "maybe", "why": "x"}}]}, 1, "out_of_range"),
                         ({"requests": [{"request": 1, "F1": {"answer": "yes"}, "F2": yn(0), "F3": yn(0)}]}, 1, "schema")]:
        dim = "hypothesis" if "updates" in obj else "fetch"
        try:
            R.parse_judge(dim, obj, n)
            raise AssertionError(f"accepted {obj}")
        except JudgeFormatError as exc:
            assert exc.kind == kind, (exc.kind, kind)
    try:
        R.parse_judge("closure", {"conclusion": conc(), "gap": None}, 1, closed=False)
        raise AssertionError("missing gap accepted")
    except JudgeFormatError:
        pass
    try:
        R.parse_judge("closure", {"conclusion": {**conc(), "C2": {"answer": "partial", "why": "x"}}}, 1, closed=True)
        raise AssertionError("partial accepted for C2")
    except JudgeFormatError as exc:
        assert exc.kind == "out_of_range"


def test_fetch_program_and_judge():
    res = run(CASE_C, [CALL("x", ["E2.2", "E2.1", "E2.3"], "first request here"),
                       CALL("y", ["E2.2", "E2.4"], "second request re-reads E2.2"),
                       CALL("z", ["E2.5", "E2.7"], "third request, two key items"),
                       FINAL_C])
    prog = R.fetch_program(res, CASE_C, REF_C)
    assert [p["K1"] for p in prog] == [1.0, 0.0, 1.0]
    assert close(prog[0]["K2"], 1 / 3) and prog[1]["K2"] == 0.0 and prog[2]["K2"] == 1.0
    assert prog[2]["unread_key_before"] == ["E2.5", "E2.7"]
    fi = [fetch_item(True, False, False), fetch_item(False, True, True), fetch_item(True, False, True)]
    judged = {"fetch": {"items": fi}, "closure": {"conclusion": conc(), "gap": None}}
    r = R.compute(res, CASE_C, REF_C, FULL, judged)
    want = [0.3 + 0.2 / 3 + 0.2 + 0.2 + 0.1, 0.0, 0.3 + 0.2 + 0.2 + 0.2 + 0.0]
    assert all(close(a, b) for a, b in zip(r["fetch_items"], want)), r["fetch_items"]
    assert close(r["dimensions"]["fetch"], sum(want) / 3)
    assert close(r["components"]["fetch_program"], (0.3 + 0.2 / 3 + 0 + 0.5) / 3)
    r = R.compute(res, CASE_C, REF_C, FULL, judged, excluded={"fetch"})
    assert close(r["dimensions"]["fetch"], r["components"]["fetch_program"])
    r = R.compute(run(CASE_C, [FINAL_C]), CASE_C, REF_C, FULL, {"closure": {"conclusion": conc(), "gap": None}})
    assert r["dimensions"]["fetch"] == 0.0 and "fetch" not in R.judge_requests(run(CASE_C, [FINAL_C]), CASE_C, REF_C, FULL)
    text, n = R.judge_requests(res, CASE_C, REF_C, FULL)["fetch"]
    assert n == 3 and "KEY items still UNREAD" in text and "supports H1" not in text and "REFERENCE" not in text


def test_structure_same_as_v1():
    trajs = [
        run(CASE_C, [CALL("x", [f"E2.{j}"], "reading one more item") for j in range(1, 8)] + [FINAL_C]),
        run(CASE_C, [CALL("x", ["E1.1"], "re-reading the record item") for _ in range(15)]),
        run(CASE_C, [CALL("x", [f"E2.{j}" for j in range(1, 13)], "reading everything at once")] +
            [CALL("x", [f"E2.{j}" for j in range(1, 13)], "reading everything again") for _ in range(2)] + [FINAL_C]),
        run(CASE_C, [CALL("x", [f"E2.{j}" for j in range(1, 13)] + ["E1.1"], "too many ids at once")] * 6 + [FINAL_C]),
        run(CASE_C, [CALL("x", KEYS, "keys"), "CASE CLOSED. Bearing seizure E2.2 and E2.9."]),
    ]
    want = [(-0.2, "turns"), (-0.9, "turns"), (-1.0, "rerequest"), (-1.0, "invalid_fetch"), (-0.5, "citation")]
    for res, (value, key) in zip(trajs, want):
        v1 = R1.structure(res, CASE_C, V1P1)
        r = R.compute(res, CASE_C, REF_C, FULL, {})
        assert close(r["dimensions"]["structure"], v1["total"]) and close(v1[key], value), (key, v1[key])
        assert close(r["components"][f"structure_{key}"], value)


def test_outcome_variant():
    res = run(CASE_C, three_fetches(FINAL_C))
    reqs_full = R.judge_requests(res, CASE_C, REF_C, FULL)
    reqs_out = R.judge_requests(res, CASE_C, REF_C, OUTC)
    assert set(reqs_full) == {"fetch", "hypothesis", "closure"} and set(reqs_out) == {"closure"}
    assert reqs_full["closure"] == reqs_out["closure"]
    judged = full_judged(res, CASE_C, REF_C, [fetch_item(True, False, False)] * 3,
                         [hyp_item(True, True, False, False, True)] * 2, conc("partial"))
    r = R.compute(res, CASE_C, REF_C, OUTC, judged)
    assert set(r["dimensions"]) == {"outcome", "structure"}
    assert close(r["total"], r["dimensions"]["outcome"] + r["dimensions"]["structure"])
    assert close(r["dimensions"]["outcome"], 1 + (2 * 0.825 - 1))
    assert "fetch_program_unrewarded" in r["components"]


def test_answer_counts_and_weights():
    res = run(CASE_N, three_fetches(FINAL_N))
    judged = full_judged(res, CASE_N, REF_N, [fetch_item(True, False, True)] * 3,
                         [hyp_item(False, None, False, False, True), hyp_item(True, False, True, False, True)],
                         conc("partial", True, False, True), gap(True, False, True))
    r = R.compute(res, CASE_N, REF_N, FULL, judged)
    a = r["answers"]
    assert a["F1"] == [3, 3] and a["F3"] == [3, 3] and a["F2"] == [0, 3]
    assert a["H1"] == [1, 2] and a["H2"] == [0, 1] and a["C1"] == [0, 1] and a["C1_partial"] == [1, 1]
    assert a["G2"] == [0, 1]
    heavy = R.RewardV2Config.from_dict({"variant": "full", "weights_v2": {"gate_wrong": 0.0, "K1": 0.5}})
    assert heavy.weights["K1"] == 0.5 and FULL.weights["K1"] == 0.3
    try:
        R.RewardV2Config.from_dict({"weights_v2": {"K9": 1}})
        raise AssertionError("unknown weight accepted")
    except ValueError:
        pass


def _client(schema, **kw):
    from nautil_rlvr.judge import JudgeClient, JudgeConfig
    cfg = JudgeConfig(model="mock", schema=schema, per_key_concurrency=1, retry_backoff=[], **kw)
    return JudgeClient(cfg, keys={"API_KEY": "k1"})


def test_judge_client_prompts_and_feedback_retry():
    from nautil_rlvr.common import sha256_file
    from nautil_rlvr.judge import PROMPT_DIR
    c1 = _client("v1")
    assert c1.prompt_sha["closure"] == sha256_file(PROMPT_DIR / "closure_judge_v1.txt")
    assert "0-10" in c1.prompts["fetch"] or "score" in c1.prompts["fetch"].lower()
    c2 = _client("v2", format_retry_feedback=True)
    assert c2.prompt_sha["hypothesis"] == sha256_file(PROMPT_DIR / "hypothesis_v2.txt")
    assert c1.cache_key("closure", "u") != c2.cache_key("closure", "u")
    bad = {"updates": [{"update": 1, **hyp_item(True, None, False, False, True)}]}
    good = {"updates": [{"update": 1, **hyp_item(True, True, False, False, True)}]}
    seen = []

    def fake_post(messages):
        seen.append(json.loads(json.dumps(messages)))
        body = bad if len(seen) == 1 else good
        return {"choices": [{"message": {"content": json.dumps(body)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}, "key1"
    c2._post = fake_post
    out = c2.call("hypothesis", "user text", 1)
    assert "items" in out and out["items"][0]["H2"]["answer"] == "yes"
    assert len(seen) == 2 and len(seen[1]) == 4 and "H2 n/a although H1 yes" in seen[1][3]["content"]
    assert c2.cumulative()["parse_failures"] == {"inconsistent": 1}
    import collections
    from nautil_rlvr.judge import _new_counters
    cnt = _new_counters()
    cnt.update({"http_attempts": 10, "parse_failures": collections.Counter({"inconsistent": 2, "json": 1, "truncated": 1})})
    h = c2.summarize(cnt)
    assert close(h["parse_failure_rate"], 0.1) and close(h["inconsistent_rate"], 0.2) and close(h["truncated_rate"], 0.1)
    c3 = _client("v2")
    seen.clear()
    c3._post = fake_post
    c3.call("hypothesis", "user text", 1)
    assert len(seen[1]) == 2
    for c in (c1, c2, c3):
        c.close()


S = synthetic.state_block


def scripted(behaviour: str, ref_closed: bool):
    def good_process(turn, returned, final):
        if turn == 1:
            return CALL(S("open", "open", "Initial hypotheses from E1.1 and E1.2.",
                          "Do the decisive logs support H1 (bearing seizure) or H2?"),
                        ["E2.2", "E2.5"], "Logs 2 and 5 would support H1 (bearing seizure) or weaken it in favour of H2.")
        if turn == 2:
            sup = any("supports H1" in t for t in returned)
            return CALL(S("favored" if sup else "open", "weakened" if sup else "open",
                          "E2.2 and E2.5 support H1." if sup else "E2.2 and E2.5 are inconclusive between H1 and H2.",
                          "Does log 7 confirm H1?"),
                        ["E2.7"], "Log 7 is the last decisive item for H1 versus H2.")
        return final

    right_c = FINAL_C
    right_n = FINAL_N
    wrong_c = ("CASE NOT CLOSED - E2.2, E2.5, E2.7 point to bearing seizure (H1) but no teardown report exists; "
               "H2 electrical fault is not excluded. Tentative: bearing seizure (H1).")
    wrong_n = ("CASE CLOSED. Supported mechanism: bearing seizure (H1) shown by E2.2, E2.5, E2.7. "
               "Discounted alternatives: H2 electrical fault.")

    def reply(turn, returned):
        if behaviour == "good investigation":
            return good_process(turn, returned, right_c if ref_closed else right_n)
        if behaviour == "wrong decision, good process":
            return good_process(turn, returned, wrong_c if ref_closed else wrong_n)
        if behaviour == "fetch everything":
            ids = [f"E2.{j}" for j in range(1, 13)]
            if turn <= 2:
                return CALL(S("open", "open", "Reading the file." if turn == 1 else "Read items E2.1 to E2.6; they support the picture.",
                              "Read the rest."), ids[:6] if turn == 1 else ids[6:], "Read every item to be thorough.")
            return right_c if ref_closed else right_n
        if behaviour == "always close (lazy)":
            return "CASE CLOSED. Supported mechanism: electrical fault (H2), because the operator saw a pressure drop (E1.2)."
        if behaviour == "never close (lazy)":
            return "CASE NOT CLOSED - the record E1.1 does not settle the cause."
        if behaviour == "random hypothesis flips":
            if turn <= 3:
                h1, h2 = ("weakened -> favored", "favored -> weakened") if turn % 2 else ("favored -> weakened", "weakened -> favored")
                return CALL(S(h1, h2, "Re-reading the event summary changes the picture.", "Re-check the event summary."),
                            ["E1.1"], "Re-check the event summary once more.")
            return "CASE CLOSED. Electrical fault (H2) per E1.1."
        if behaviour == "no marker":
            return good_process(turn, returned, "Summary: E2.2, E2.5 and E2.7 were read; bearing seizure (H1) is likely.")
        raise ValueError(behaviour)
    return reply


BEHAVIOURS = ["good investigation", "fetch everything", "wrong decision, good process",
              "never close (lazy)", "always close (lazy)", "random hypothesis flips", "no marker"]


def run_behaviour(case, behaviour, ref_closed):
    policy = scripted(behaviour, ref_closed)
    returned: list[str] = []
    turn = 0

    def gen(model, tok, messages, tools, device, limit):
        nonlocal turn
        turn += 1
        returned.clear()
        for m in messages:
            if m["role"] == "tool":
                returned.extend(i["text"] for i in json.loads(m["content"]).get("evidence_items", []))
        return policy(turn, list(returned)), 100, False
    return run_case(None, None, "system", case, None, 32768, 15, generate_fn=gen)


def judged_by_mock(res, case, ref, cfg):
    out = {}
    for dim, (text, n) in R.judge_requests(res, case, ref, cfg).items():
        out[dim] = R.parse_judge(dim, mock_judge_v2.answer(dim, text), n, R1.model_closed(res))
    return out


def ranking() -> list[dict]:
    rows = []
    for b in BEHAVIOURS:
        row = {"behaviour": b}
        for tag, case, ref in (("refC", CASE_C, REF_C), ("refN", CASE_N, REF_N)):
            res = run_behaviour(case, b, ref["closure"] == "closed")
            for vname, cfg in (("full", FULL), ("outcome", OUTC)):
                r = R.compute(res, case, ref, cfg, judged_by_mock(res, case, ref, cfg))
                row[f"{vname}_{tag}"] = round(r["total"], 3)
                if vname == "full":
                    row[f"parts_{tag}"] = {**{k: round(v, 3) for k, v in r["dimensions"].items()},
                                           "D": r["components"]["D"], "q_conc": r["components"]["q_conc"],
                                           "q_gap": r["components"]["q_gap"], "marker": res["conclusion_marker"]}
        for vname in ("full", "outcome"):
            row[f"{vname}_mean"] = round((row[f"{vname}_refC"] + row[f"{vname}_refN"]) / 2, 3)
        rows.append(row)
    return rows


def test_adversarial_ranking():
    rows = {r["behaviour"]: r for r in ranking()}
    for v in ("full", "outcome"):
        m = {b: rows[b][f"{v}_mean"] for b in rows}
        assert m["good investigation"] == max(m.values()), (v, m)
        assert min(m["never close (lazy)"], m["always close (lazy)"]) > m["no marker"], (v, m)
    m = {b: rows[b]["full_mean"] for b in rows}
    assert m["good investigation"] > m["wrong decision, good process"] > max(m["never close (lazy)"],
                                                                             m["always close (lazy)"]), m
    assert m["good investigation"] > m["fetch everything"] > m["wrong decision, good process"], m
    assert m["random hypothesis flips"] < m["wrong decision, good process"], m
    for tag in ("refC", "refN"):
        assert rows["good investigation"][f"full_{tag}"] > rows["fetch everything"][f"full_{tag}"]


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
    rows = ranking()
    print(json.dumps(rows, indent=1))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps({"tests": len(tests), "failed": failed,
                                                                             "ranking": rows}, indent=1))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    random.seed(0)
    sys.exit(main())
