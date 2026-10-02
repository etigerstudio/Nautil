from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nautil_rlvr import reward as R1
from nautil_rlvr import reward_v22 as R
from nautil_rlvr.reward import JudgeFormatError
from nautil_rlvr.testing import mock_judge_v22
from nautil_rlvr.testing.test_reward_v2 import (BEHAVIOURS, CALL, CASE_C, CASE_N, FINAL_C, FINAL_N, KEYS,
                                                REF_C, REF_N, S, close, run, run_behaviour, yn)

FULL = R.RewardV2Config.from_dict({"version": "2.2", "variant": "full", "phase": 2})
OUTC = R.RewardV2Config.from_dict({"version": "2.2", "variant": "outcome", "phase": 2})
FIX = Path(__file__).resolve().parent / "fixtures_v22_parity.json"


def conc(c1a=True, c1b="yes", c2=True, claims=(True, True), c4=True):
    return {"C1a": yn(c1a), "C1b": {"answer": c1b, "why": "x"}, "C2": yn(c2),
            "C3": {"why": "x", "claims": [{"claim": f"c{i}", "supported": s} for i, s in enumerate(claims)]},
            "C4": yn(c4)}


def gap(g1=True, g2=True, g3=True):
    return {"G1": yn(g1), "G2": {"answer": "n/a", "why": "x"} if not g1 else yn(g2), "G3": yn(g3)}


def upd(hyps, main, h5=True):
    return {"hyps": [{"id": i, "touched": t, "direction": d, "why": "x"} for i, t, d in hyps], "main": main,
            "H5": yn(h5)}


def one_fetch(final):
    return [CALL("## Current hypotheses\nH1 - a\nH2 - b", KEYS, "Which hypothesis do the logs support?"), final]


def three_fetches(final):
    return [CALL(S("open", "open", "Initial hypotheses.", "logs?"), ["E2.2"], "Does log 2 support H1?"),
            CALL(S("favored", "open", "E2.2 supports H1.", "logs?"), ["E2.5", "E2.1"], "Does log 5 support H1?"),
            CALL(S("favored", "weakened", "E2.5 supports H1.", "logs?"), ["E2.7"], "Does log 7 support H1?"),
            final]


def outcome_of(case, ref, final, c, g=None):
    res = run(case, one_fetch(final))
    closed = R1.model_closed(res)
    return R.compute(res, case, ref, OUTC, {"closure": {"conclusion": c, "gap": g if closed is False else None}})


def test_outcome_cells_and_checklist():
    r = outcome_of(CASE_C, REF_C, FINAL_C, conc())
    assert r["components"]["D"] == 1 and close(r["dimensions"]["outcome"], 2.0)
    r = outcome_of(CASE_C, REF_C, FINAL_C, conc(claims=(True, False)))
    assert close(r["components"]["q_conc"], 0.9) and close(r["components"]["c3_supported_share"], 0.5)
    r = outcome_of(CASE_C, REF_C, FINAL_C, conc(claims=()))
    assert close(r["components"]["q_conc"], 0.8) and r["components"]["c3_claims"] == 0
    r = outcome_of(CASE_C, REF_C, FINAL_C, conc(False, "partial", True, (True,), False))
    assert close(r["components"]["q_conc"], 0.5) and close(r["dimensions"]["outcome"], 1.0)
    r = outcome_of(CASE_N, REF_N, FINAL_C, conc())
    assert r["components"]["D"] == -1 and r["components"]["gate"] == 0.3 and close(r["dimensions"]["outcome"], 0.0)
    r = outcome_of(CASE_N, REF_N, FINAL_C, conc(False, "no", False, (), False))
    assert close(r["dimensions"]["outcome"], -2.0)
    r = outcome_of(CASE_N, REF_N, FINAL_N, conc(), gap(True, True, False))
    assert close(r["components"]["q_gap"], 0.7) and close(r["dimensions"]["outcome"], 1 + 0 + 0.5 * 0.4 + 0.5)
    r = outcome_of(CASE_N, REF_N, FINAL_N, conc(), gap(False, None, True))
    assert close(r["components"]["q_gap"], 0.5)
    r = outcome_of(CASE_N, REF_N, FINAL_N, conc(), gap(False, None, False))
    assert close(r["components"]["q_gap"], 0.0)
    r = outcome_of(CASE_C, REF_C, FINAL_N, conc(), gap())
    assert r["components"]["D"] == -1 and close(r["dimensions"]["outcome"], 0.0)
    res = run(CASE_C, one_fetch("Bearing seizure per E2.2."))
    assert "closure" not in R.judge_requests(res, CASE_C, REF_C, FULL)
    r = R.compute(res, CASE_C, REF_C, FULL, {})
    assert r["dimensions"]["outcome"] == -1 and r["components"]["structure_missing_marker"] == -2.0


def test_hypothesis_program_branches():
    res = run(CASE_C, three_fetches(FINAL_C))
    reqs = R.judge_requests(res, CASE_C, REF_C, FULL)
    assert set(reqs) == {"hypothesis", "closure"} and reqs["hypothesis"][2] == [["H1", "H2"], ["H1", "H2"]]
    assert "Hypotheses to assess (in this order)\nH1, H2" in reqs["hypothesis"][0]
    clo = {"conclusion": conc(), "gap": None}

    def hyp(items):
        return R.compute(res, CASE_C, REF_C, FULL, {"hypothesis": {"items": items}, "closure": clo})
    good = [upd([("H1", True, "supports"), ("H2", False, "unrelated")], "H1"),
            upd([("H1", True, "supports"), ("H2", True, "weakens")], "H1")]
    r = hyp(good)
    assert [p["s"] for p in r["hyp_program"]] == [1.0, 1.0] and close(r["dimensions"]["hypothesis"], 1.0)
    r = hyp([good[0], upd([("H1", True, "supports"), ("H2", False, "unrelated")], "H1")])
    assert r["hyp_program"][1]["H4"] is True and close(r["hyp_items"][1], 0.85)
    r = hyp([upd([("H1", True, "supports"), ("H2", True, "weakens")], "H1"), good[1]])
    assert r["hyp_program"][0]["H3"] is True and close(r["hyp_items"][0], 0.7)
    r = hyp([upd([("H1", True, "weakens"), ("H2", False, "unrelated")], "H1"), good[1]])
    assert r["hyp_program"][0]["H2"] is False and close(r["hyp_items"][0], 0.6)
    r = hyp([upd([("H1", False, "unrelated"), ("H2", False, "unrelated")], None, True),
             upd([("H1", False, "unrelated"), ("H2", False, "unrelated")], None, False)])
    assert r["hyp_program"][0]["H1"] is False and close(r["hyp_items"][0], 0.3) and close(r["hyp_items"][1], 0.0)
    assert close(r["dimensions"]["hypothesis"], 2 * 0.15 - 1)
    missed = R.hyp_item_score({"H1": True, "H2": False, "H3": True, "H4": False, "H5": True})
    no_news_ok = R.hyp_item_score({"H1": False, "H2": None, "H3": None, "H4": False, "H5": True})
    assert missed < no_news_ok
    assert R.compute(res, CASE_C, REF_C, FULL, {"hypothesis": {"items": good}, "closure": clo},
                     excluded={"hypothesis"})["dimensions"]["hypothesis"] == 0.0
    r = R.compute(run(CASE_C, one_fetch(FINAL_C)), CASE_C, REF_C, FULL, {"closure": clo})
    assert r["dimensions"]["hypothesis"] == 0.0 and "hypothesis" not in R.judge_requests(
        run(CASE_C, one_fetch(FINAL_C)), CASE_C, REF_C, FULL)


def test_parser():
    ctx = [["H1", "H2"]]

    def hobj(hyps, main="H1"):
        return {"updates": [{"update": 1, "hypotheses": hyps, "main": main, "H5": yn(True)}]}
    ok = hobj([{"id": "H1", "why": "x", "touched": "yes", "direction": "supports"},
               {"id": "H2", "why": "x", "touched": "no", "direction": "unrelated"}])
    p = R.parse_judge("hypothesis", ok, 1, None, ctx)
    assert p["items"][0]["main"] == "H1" and p["items"][0]["hyps"][1]["touched"] is False
    bad = [
        (hobj([{"id": "H1", "why": "x", "touched": "yes", "direction": "unrelated"},
               {"id": "H2", "why": "x", "touched": "no", "direction": "unrelated"}]), "out_of_range"),
        (hobj([{"id": "H1", "why": "x", "touched": "yes", "direction": "supports"},
               {"id": "H2", "why": "x", "touched": "no", "direction": "unrelated"}], main="H2"), "out_of_range"),
        (hobj([{"id": "H1", "why": "x", "touched": "no", "direction": "unrelated"},
               {"id": "H2", "why": "x", "touched": "no", "direction": "unrelated"}], main="H1"), "out_of_range"),
        (hobj([{"id": "H1", "why": "x", "touched": "yes", "direction": "supports"}]), "missing_items"),
        ({"updates": []}, "missing_items"),
    ]
    for obj, kind in bad:
        try:
            R.parse_judge("hypothesis", obj, 1, None, ctx)
            raise AssertionError(f"accepted {obj}")
        except JudgeFormatError as exc:
            assert exc.kind == kind, (exc.kind, kind)
    for g, kind in ((gap(True, True, True) | {"G2": {"answer": "n/a", "why": "x"}}, "out_of_range"),
                    (gap(False, None, True) | {"G2": yn(True)}, "out_of_range")):
        try:
            R.parse_judge("closure", {"conclusion": conc(), "gap": g}, 1, closed=False)
            raise AssertionError("G1/G2 inconsistency accepted")
        except JudgeFormatError as exc:
            assert exc.kind == kind
    c = conc()
    del c["C3"]
    for obj in ({"conclusion": c}, {"conclusion": conc(), "gap": None}):
        try:
            R.parse_judge("closure", obj, 1, closed=False if "C3" in obj["conclusion"] else True)
            raise AssertionError("accepted")
        except JudgeFormatError:
            pass
    try:
        R.parse_judge("fetch", {}, 1)
        raise AssertionError("fetch dimension accepted")
    except ValueError:
        pass


def test_fetch_kcov_and_gate():
    clo = {"conclusion": conc(), "gap": None}
    r = R.compute(run(CASE_C, three_fetches(FINAL_C)), CASE_C, REF_C, FULL, {"closure": clo})
    assert close(r["components"]["Kcov"], 1.0) and close(r["dimensions"]["fetch"], 0.5)
    res = run(CASE_C, [CALL("x", ["E2.2", "E2.1", "E2.3", "E2.4", "E2.6", "E2.8", "E2.9", "E2.10"], "one big request"),
                       FINAL_C])
    r = R.compute(res, CASE_C, REF_C, FULL, {"closure": clo})
    assert close(r["dimensions"]["fetch"], 0.5 / 3)
    res = run(CASE_C, [CALL("x", ["E1.1", "E1.2"], "re-read the record"), FINAL_C])
    assert R.compute(res, CASE_C, REF_C, FULL, {"closure": clo})["dimensions"]["fetch"] == 0.0
    r = R.compute(run(CASE_C, [FINAL_C]), CASE_C, REF_C, FULL, {"closure": clo})
    assert r["dimensions"]["fetch"] == 0.0 and r["components"]["n_requests"] == 0
    good = [upd([("H1", True, "supports"), ("H2", False, "unrelated")], "H1"),
            upd([("H1", True, "supports"), ("H2", True, "weakens")], "H1")]
    right = run(CASE_C, three_fetches(FINAL_C))
    wrong = run(CASE_N, three_fetches(FINAL_C))
    r1 = R.compute(right, CASE_C, REF_C, FULL, {"hypothesis": {"items": good}, "closure": clo})
    r2 = R.compute(wrong, CASE_N, REF_N, FULL, {"hypothesis": {"items": good}, "closure": clo})
    assert close(r1["total"], 2.0 + 1.0 * (0.5 + 1.0) + r1["dimensions"]["structure"])
    assert close(r2["total"], 0.0 + 0.3 * (0.5 + 1.0) + r2["dimensions"]["structure"])
    assert close(r2["components"]["g_fetch"], 0.15) and close(r2["components"]["g_hyp"], 0.3)


def test_outcome_variant_and_structure():
    res = run(CASE_C, three_fetches(FINAL_C))
    assert set(R.judge_requests(res, CASE_C, REF_C, OUTC)) == {"closure"}
    assert set(R.judge_requests(res, CASE_C, REF_C, FULL)) == {"hypothesis", "closure"}
    good = [upd([("H1", True, "supports"), ("H2", False, "unrelated")], "H1")] * 2
    r = R.compute(res, CASE_C, REF_C, OUTC, {"hypothesis": {"items": good}, "closure": {"conclusion": conc(c1b="partial"), "gap": None}})
    assert set(r["dimensions"]) == {"outcome", "structure"} and close(r["dimensions"]["outcome"], 1 + (2 * 0.9 - 1))
    assert close(r["total"], r["dimensions"]["outcome"] + r["dimensions"]["structure"])
    assert close(r["components"]["fetch_kcov_term_unrewarded"], 0.5) and r["hyp_items"] == []
    long = run(CASE_C, [CALL("x", ["E1.1"], "re-reading the record item") for _ in range(15)])
    assert close(R.compute(long, CASE_C, REF_C, FULL, {})["dimensions"]["structure"],
                 R1.structure(long, CASE_C, R1.RewardConfig.from_dict({"phase": 1}))["total"])


def test_decision_variant():
    DEC = R.RewardV2Config.from_dict({"version": "2.2", "variant": "decision", "phase": 2})
    assert DEC.dims == ()
    for case, ref, final, D in ((CASE_C, REF_C, FINAL_C, 1.0), (CASE_N, REF_N, FINAL_C, -1.0),
                                (CASE_C, REF_C, FINAL_N, -1.0), (CASE_N, REF_N, FINAL_N, 1.0)):
        res = run(case, three_fetches(final))
        assert R.judge_requests(res, case, ref, DEC) == {}
        r = R.compute(res, case, ref, DEC, {})
        r2 = R.compute(res, case, ref, DEC, {"closure": {"conclusion": conc(False, "no", False, (False,), False),
                                                       "gap": gap(False, False, False)}})
        assert r["components"]["D"] == D and close(r["dimensions"]["outcome"], D)
        assert set(r["dimensions"]) == {"outcome", "structure"}
        assert close(r["total"], D + r["dimensions"]["structure"]) and close(r2["total"], r["total"])
        assert close(r["components"]["judge_part"], 0.0) and r["variant"] == "decision"


def test_quality_only_if_correct():
    G = R.RewardV2Config.from_dict({"version": "2.2", "variant": "outcome", "phase": 2, "quality_only_if_correct": True})
    judged = {"closure": {"conclusion": conc(), "gap": None}}
    res = run(CASE_N, one_fetch(FINAL_C))
    r = R.compute(res, CASE_N, REF_N, G, judged)
    assert r["components"]["D"] == -1 and close(r["dimensions"]["outcome"], -1.0)
    assert close(r["total"], -1.0 + r["dimensions"]["structure"])
    r_old = R.compute(res, CASE_N, REF_N, OUTC, judged)
    assert close(r_old["dimensions"]["outcome"], 0.0)
    res = run(CASE_C, one_fetch(FINAL_C))
    assert close(R.compute(res, CASE_C, REF_C, G, judged)["total"], R.compute(res, CASE_C, REF_C, OUTC, judged)["total"])


def test_weights_override():
    cfg = R.RewardV2Config.from_dict({"variant": "full", "weights_v2": {"kcov": 0.3, "gate_wrong": 0.5}})
    r = R.compute(run(CASE_N, three_fetches(FINAL_C)), CASE_N, REF_N, cfg, {"closure": {"conclusion": conc(), "gap": None}})
    assert close(r["dimensions"]["fetch"], 0.3) and r["components"]["gate"] == 0.5
    try:
        R.RewardV2Config.from_dict({"weights_v2": {"F1": 0.2}})
        raise AssertionError("unknown weight accepted")
    except ValueError:
        pass


def test_parity_with_replay_fixtures():
    if not FIX.exists():
        import pytest
        pytest.skip("replay fixtures are not included in this repository")
    fx = json.loads(FIX.read_text())
    assert len(fx["trajectories"]) >= 6
    for t in fx["trajectories"]:
        res, case, ref, j, e = t["trajectory"], t["case"], t["ref"], t["judged"], t["expected"]
        a = R.compute(res, case, ref, FULL, j)
        b = R.compute(res, case, ref, OUTC, j)
        assert close(a["total"], e["total_full"]) and close(b["total"], e["total_outcome"]), (t["key"], a["total"], e)
        assert close(a["dimensions"]["hypothesis"], e["hypothesis"]) and close(a["dimensions"]["fetch"], e["fetch_kcov_term"])
        assert close(a["dimensions"]["outcome"], e["outcome"]) and a["components"]["gate"] == e["gate"]
        reqs = R.judge_requests(res, case, ref, FULL)
        assert {k: hashlib.sha256(v[0].encode()).hexdigest() for k, v in reqs.items()} == t["expected_request_sha"], t["key"]
        assert {k: v[2] for k, v in reqs.items()} == t["expected_contexts"]
    for x in fx["raw_answers"]:
        got = R.parse_judge(x["dimension"], json.loads(x["raw_text"]), x["n"], x["closed"], x["context"])
        assert got == x["expected"], x["dimension"]


def test_judge_client_v22():
    from nautil_rlvr.common import sha256_file
    from nautil_rlvr.judge import PROMPT_DIR, JudgeClient, JudgeConfig
    c = JudgeClient(JudgeConfig(model="mock", schema="v22", per_key_concurrency=1, retry_backoff=[],
                                format_retry_feedback=True),
                    keys={"API_KEY": "k1"})
    assert set(c.prompts) == {"hypothesis", "closure"}
    assert c.prompt_sha["hypothesis"] == sha256_file(PROMPT_DIR / "hypothesis_v21.txt")
    assert c.prompt_sha["closure"] == sha256_file(PROMPT_DIR / "closure_v21.txt")
    bad = {"updates": [{"update": 1, "hypotheses": [{"id": "H1", "why": "x", "touched": "yes", "direction": "unrelated"}],
                        "main": "H1", "H5": yn(True)}]}
    good = {"updates": [{"update": 1, "hypotheses": [{"id": "H1", "why": "x", "touched": "yes", "direction": "supports"}],
                         "main": "H1", "H5": yn(True)}]}
    seen = []

    def fake_post(messages):
        seen.append(messages)
        return {"choices": [{"message": {"content": json.dumps(bad if len(seen) == 1 else good)}, "finish_reason": "stop"}],
                "usage": {"cost": 0.0}}, "key1"
    c._post = fake_post
    out = c.submit("hypothesis", "user text", 1, None, [["H1"]]).result()
    assert "items" in out and out["items"][0]["main"] == "H1" and len(seen) == 2
    assert "touched=yes but direction=unrelated" in seen[1][-1]["content"]
    c.close()


def judged_by_mock(res, case, ref, cfg):
    out = {}
    for dim, (text, n, ctx) in R.judge_requests(res, case, ref, cfg).items():
        out[dim] = R.parse_judge(dim, mock_judge_v22.answer(dim, text), n, R1.model_closed(res), ctx)
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
    assert m["random hypothesis flips"] < m["wrong decision, good process"], m
    for tag in ("refC", "refN"):
        assert rows["good investigation"][f"full_{tag}"] >= rows["fetch everything"][f"full_{tag}"]


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
    for r in rows:
        print(f"{r['behaviour']:<30} full C/N/mean {r['full_refC']:>6} {r['full_refN']:>6} {r['full_mean']:>6} | "
              f"outcome {r['outcome_refC']:>6} {r['outcome_refN']:>6} {r['outcome_mean']:>6}")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps({"tests": len(tests), "failed": failed,
                                                                             "ranking": rows}, indent=1))
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
