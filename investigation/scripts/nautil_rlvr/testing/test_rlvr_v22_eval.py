from __future__ import annotations

import json
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nautil_rlvr import evaluation_v22 as E
from nautil_rlvr import reward_v22 as R


def close(a, b, tol=1e-9):
    return a is not None and abs(a - b) <= tol


def turn(kind):
    if kind == "ok":
        return {"call": {"evidence_ids": ["E2.1"]}, "tool_return": {"evidence_items": []}}
    if kind == "bad":
        return {"call": None, "call_errors": ["not JSON"]}
    return {"call": None}


def rec(cid, src, marker, sample=0, turns=("ok", "none"), cites=True, final="CASE CLOSED x (E2.1)"):
    return {"case_id": cid, "source": src, "conclusion_marker": marker, "job": {"sample_index": sample},
            "turns": [turn(t) for t in turns], "assistant_turns": len(turns), "status": "final",
            "final_citations_all_disclosed": cites, "final_answer": final}


LABELS = {"H1": "closed", "H2": "not_closed", "B1": "closed", "B2": "not_closed", "N1": "not_closed"}


def test_program_summary():
    rs = [rec("H1", "host", "CASE CLOSED"), rec("H2", "host", "CASE CLOSED"),
          rec("B1", "boards", "CASE NOT CLOSED", turns=("ok", "bad", "none")),
          rec("B2", "boards", "CASE NOT CLOSED", cites=False),
          rec("N1", "nhtsa", None, final="no marker")]
    s = E.program_summary(rs, LABELS)
    assert s["records"] == 5 and close(s["closure_acc"], 2 / 5)
    assert close(s["closure_acc_should_close"], 1 / 2) and close(s["closure_acc_should_not_close"], 1 / 3)
    assert close(s["closure_balanced_acc"], (1 / 2 + 1 / 3) / 2)
    h, rp = s["host_teacher_label"], s["report_label"]
    assert h["n"] == 2 and close(h["should_close"], 1.0) and close(h["should_not_close"], 0.0) and close(h["balanced"], 0.5)
    assert rp["n"] == 3 and close(rp["should_close"], 0.0) and close(rp["should_not_close"], 0.5) and close(rp["balanced"], 0.25)
    assert close(s["by_source"]["boards"]["acc"], 0.5) and s["by_source"]["nhtsa"]["n"] == 1
    assert close(s["invalid_fetch_rate"], 1 / 6)
    assert close(s["valid_citation_rate"], 4 / 5) and close(s["missing_marker_rate"], 1 / 5)
    assert close(s["turns_per_case"], 11 / 5) and close(s["closed_rate"], 2 / 5)
    assert close(s["answer_chars_mean"], (4 * len("CASE CLOSED x (E2.1)") + len("no marker")) / 5)


def test_counterfactual_summary():
    val = [rec(c, "host", "CASE CLOSED") for c in ("A", "B", "C", "D")] + [rec("Z", "host", "CASE CLOSED")]
    removed = [rec("A", "host", "CASE NOT CLOSED"), rec("B", "host", "CASE NOT CLOSED"),
               rec("C", "host", "CASE CLOSED"), rec("D", "host", None)]
    control = [rec(c, "host", "CASE CLOSED") for c in ("A", "B", "C")] + [rec("D", "host", "CASE NOT CLOSED")]
    s = E.counterfactual_summary(val, removed, control, ["A", "B", "C", "D"])
    assert s["records"] == {"full": 4, "removed": 4, "control": 4}
    assert close(s["closed_rate_full"], 1.0) and close(s["closed_rate_removed"], 0.25)
    assert close(s["closed_rate_control"], 0.75) and close(s["removed_minus_control"], -0.5)
    assert close(s["missing_marker_rate_removed"], 0.25)


def test_scheduler_config():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for n in ("val.jsonl", "rm.jsonl", "ct.jsonl", "sys.txt"):
            (d / n).write_text("{}\n")
        ecfg = {"counterfactual": {"removed_bundle": str(d / "rm.jsonl"), "control_bundle": str(d / "ct.jsonl")},
                "val_case_ids": ["A", "B"], "cf_case_ids": None, "concurrency": 7}
        paths = {"val_bundle": str(d / "val.jsonl"), "system_prompt": str(d / "sys.txt"), "base_model": "m"}
        c = E.scheduler_config(3, "lora", d, ecfg, paths, d / "p.json", ["http://a", "http://b"], d / "o", 3)
        assert set(c["datasets"]) == set(E.DATASETS) and all(v["split"] == "validation" for v in c["datasets"].values())
        assert c["datasets"]["validation_v1"]["case_ids"] == ["A", "B"] and "case_ids" not in c["datasets"]["val_cf_removed"]
        cond = c["conditions"][0]
        assert cond["samples"] == 3 and cond["datasets"] == list(E.DATASETS) and cond["model"] == "lora"
        assert [e["target_concurrency"] for e in c["backends"]["gpu"]["endpoints"]] == [7, 7]
        (d / "test_x").mkdir()
        (d / "test_x" / "rm.jsonl").write_text("{}\n")
        ecfg["counterfactual"]["removed_bundle"] = str(d / "test_x" / "rm.jsonl")
        try:
            E.scheduler_config(3, "lora", d, ecfg, paths, d / "p.json", ["http://a"], d / "o", 1)
            raise AssertionError("test-looking path accepted")
        except PermissionError:
            pass


class FakeJudge:
    def __init__(self, c1a="yes"):
        self.c1a, self.n = c1a, 0

    def submit(self, dim, text, n, closed, *ctx):
        from concurrent.futures import Future
        assert dim == "closure" and "CONCLUSION checklist" in text
        self.n += 1
        yn = lambda a: {"answer": a, "why": "x"}
        out = {"conclusion": {"C1a": yn(self.c1a), "C1b": yn("partial"), "C2": yn("yes"), "C4": yn("no"),
                              "C3": {"claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}],
                                     "why": "x"}},
               "gap": None if closed else {"G1": yn("yes"), "G2": yn("no"), "G3": yn("yes")}}
        f = Future()
        f.set_result(out)
        return f

    def cumulative(self):
        return {"calls": self.n, "cost_usd": 0.0, "errors": {}}


def test_llm_scores_and_compare():
    from nautil_rlvr.testing.test_reward_v2 import CALL, CASE_C, CASE_N, FINAL_C, FINAL_N, KEYS, REF_C, REF_N, run
    rc = run(CASE_C, [CALL("## Current hypotheses\nH1 - a", KEYS, "logs?"), FINAL_C])
    rn = run(CASE_N, [CALL("## Current hypotheses\nH1 - a", KEYS, "logs?"), FINAL_N])
    rc.update({"case_id": "C", "source": "host", "job": {"sample_index": 0}})
    rn.update({"case_id": "N", "source": "ntsb", "job": {"sample_index": 0}})
    cases, refs = {"C": CASE_C, "N": CASE_N}, {"C": REF_C, "N": REF_N}
    cfg = R.RewardV2Config.from_dict({"version": "2.2", "variant": "outcome", "phase": 2})
    a = E.llm_scores([rc, rn], FakeJudge("yes"), cases, refs, cfg)
    b = E.llm_scores([rc, rn], FakeJudge("no"), cases, refs, cfg)
    qa = 0.2 + 0.2 * 0.5 + 0.2 + 0.2 * 0.5 + 0.0
    s = a["summary"]
    assert s["scored"] == 2 and close(s["q_conc_mean"], qa) and close(s["q_gap_mean"], 0.3 + 0.3)
    assert close(s["yes_C1a"], 1.0) and close(s["partial_C1b"], 1.0) and close(s["C3_supported_share_mean"], 0.5)
    row_c = next(r for r in a["rows"] if r["case_id"] == "C")
    assert close(row_c["outcome"], 1 + (2 * qa - 1))
    cmp = E.compare_judges(a, b)
    assert cmp["paired"] == 2 and close(cmp["luna_minus_sol_q_conc"], 0.2) and close(cmp["agree_C1a"], 0.0)
    assert close(cmp["agree_C2"], 1.0)


def stub_run(tmp: Path, total=10, every_eval=10, every_ck=5):
    from nautil_rlvr.train import Run
    calls = {"checkpoint": [], "eval": [], "metrics": []}
    st = SimpleNamespace(
        cfg={"checkpoint": {"every_steps": every_ck}, "eval": {"every_steps": every_eval, "at_end": True},
             "monitor": {"rank_sync_check_every": 0}},
        total_steps=total, stop=threading.Event(), stop_reason=None, d=SimpleNamespace(world=1),
        stop_files=[tmp / "run" / "STOP", tmp / "STOP_ALL"])
    st.checkpoint = lambda step, adapter, sampler, kind: calls["checkpoint"].append((step, kind))
    st.prune_adapters = lambda v: None
    st.start_eval = lambda v: calls["eval"].append(v)
    st.step_metrics = lambda *a: {}
    st.write_metrics = lambda step, batch, m: calls["metrics"].append(step)
    st.poll_stop = lambda new_step: Run.poll_stop(st, new_step)
    st.after_step = lambda *a: Run.after_step(st, *a)
    (tmp / "run").mkdir(exist_ok=True)
    return st, calls


def test_stop_file():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        st, calls = stub_run(tmp)
        batch, res = {"sampler_after": {}}, {"adapter_dir": tmp}
        assert st.poll_stop(3) is False and not st.stop.is_set()
        assert st.after_step(2, batch, res, 0, 1.0) == 3 and calls["checkpoint"] == [] and calls["eval"] == []
        (tmp / "run" / "STOP").write_text("")
        assert st.poll_stop(4) is True and st.stop.is_set() and st.stop_reason.endswith("STOP")
        assert not (tmp / "run" / "STOP").exists()
        assert list((tmp / "run").glob("STOP.honoured_step00004_*"))
        st.after_step(3, batch, res, 0, 1.0)
        assert calls["checkpoint"] == [(4, "stop_file")] and calls["metrics"] == [3, 4]
        st2, calls2 = stub_run(tmp)
        (tmp / "STOP_ALL").write_text("")
        assert st2.poll_stop(10)
        st2.after_step(9, batch, res, 0, 1.0)
        assert calls2["checkpoint"] == [(10, "stop_file")] and calls2["eval"] == []
        st3, calls3 = stub_run(tmp, total=12)
        for s in range(12):
            st3.after_step(s, batch, res, 0, 1.0)
        assert calls3["checkpoint"] == [(5, "periodic"), (10, "periodic"), (12, "final")]
        assert calls3["eval"] == [10, 12]


def test_sync_loop_stops_after_the_step():
    from nautil_rlvr.train import Run
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        st, calls = stub_run(tmp, total=50)
        st.mode, st.start_step, st.published, st.version_names, st.resume = "sync", 0, 0, {0: "n"}, False
        st.cfg["eval"]["at_start"] = False
        steps = []
        st.make_batch = lambda b, v, n: {"sampler_before": {}, "sampler_after": {}, "index": b}
        st.judge_alert = lambda step, batch: False

        def train_step(step, batch):
            steps.append(step)
            st.published = step + 1
            st.version_names[step + 1] = "n"
            if step == 2:
                (tmp / "run" / "STOP").write_text("")
            return {"adapter_dir": tmp}
        st.train_step = train_step
        shut = []
        st.shutdown = lambda: shut.append(1)
        st.train_async = None
        assert Run.train(st) == 0
        assert steps == [0, 1, 2] and calls["checkpoint"] == [(3, "stop_file")] and shut == [1]


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
    print(f"{len(tests) - failed}/{len(tests)} passed")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps({"tests": len(tests), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
