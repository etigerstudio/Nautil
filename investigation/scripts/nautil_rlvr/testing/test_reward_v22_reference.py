from __future__ import annotations

import glob
import gzip
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from nautil_common import paths

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
REPLAY = Path(os.environ.get("NAUTIL_REPLAY_DIR", paths.EXPERIMENTS / "rlvr/reward_v2_replay"))
ROLLOUTS = Path(os.environ.get("NAUTIL_REPLAY_ROLLOUTS", paths.EXPERIMENTS / "rlvr/diagnosis_luna20/data/rollouts"))
TRAIN = Path(os.environ.get("NAUTIL_REPLAY_TRAIN", paths.RUN / "results/rlvr_v1/train_data_v1"))
sys.path.insert(0, str(REPLAY))

try:
    import reward_v21 as REF
except ImportError:
    import pytest
    pytest.skip("reference replay (reward_v21 and its data) is not included in this repository",
                allow_module_level=True)

from nautil_rlvr import reward_v22 as R

OUTC = R.RewardV2Config.from_dict({"version": "2.2", "variant": "outcome", "phase": 2, "citation_scope": "final"})
FULL = R.RewardV2Config.from_dict({"version": "2.2", "variant": "full", "phase": 2, "citation_scope": "final"})
TOL = 1e-9


def load():
    judged = {}
    for line in open(REPLAY / "results" / "judged_v21.jsonl"):
        r = json.loads(line)
        judged[(r["step"], r["case_id"], r["sample"])] = r["judged"]
    rows = {}
    for f in sorted(glob.glob(str(ROLLOUTS / "batch_*.jsonl.gz"))):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            key = (r["batch"] + 1, r["case_id"], r["sample"])
            if key in judged and r.get("group_status") == "kept" and "reward" in r:
                rows[key] = r
    ids = {k[1] for k in judged}
    cases = {c["case_id"]: c for c in map(json.loads, open(TRAIN / "train_bundle.jsonl")) if c["case_id"] in ids}
    refs = {c["case_id"]: c for c in map(json.loads, open(TRAIN / "train_refs.jsonl")) if c["case_id"] in ids}
    return judged, rows, cases, refs


def same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= TOL


def main() -> int:
    judged, rows, cases, refs = load()
    missing = sorted(set(judged) - set(rows))
    problems, n, per = [], 0, []
    for key in sorted(judged):
        if key not in rows:
            continue
        row, case, ref = rows[key], cases[key[1]], refs[key[1]]
        res = row["trajectory"]
        j = {d: v for d, v in judged[key].items() if "error" not in v}
        ref_c = REF.compute(row, case, ref, j)
        o, s = ref_c["dimensions"]["outcome"], ref_c["dimensions"]["structure"]
        g, h = ref_c["components"]["gate"], ref_c["dimensions"]["hypothesis"]
        fc = 0.5 * ref_c["components"]["Kcov"] if ref_c["components"]["n_requests"] else 0.0
        expect_outcome_only, expect_full = o + s, o + g * (fc + h) + s
        got = R.compute(res, case, ref, OUTC, j)
        got_full = R.compute(res, case, ref, FULL, j)
        checks = {
            "outcome_only_total": (got["total"], expect_outcome_only),
            "outcome": (got["dimensions"]["outcome"], o),
            "structure": (got["dimensions"]["structure"], s),
            "D": (got["components"]["D"], ref_c["components"]["D"]),
            "q_conc": (got["components"]["q_conc"], ref_c["components"]["q_conc"]),
            "q_gap": (got["components"]["q_gap"], ref_c["components"]["q_gap"]),
            "full_v22_total": (got_full["total"], expect_full),
        }
        bad = {k: v for k, v in checks.items() if not same(*v)}
        rq_ref = REF.render_closure(res, case, ref)
        rq_got = R.render_closure(res, case, ref, OUTC.base)
        if (rq_ref is None) != (rq_got is None) or (rq_ref and rq_ref[0] != rq_got[0]):
            bad["closure_request_text"] = "differs"
        if bad:
            problems.append({"key": list(key), "diff": {k: v for k, v in bad.items()}})
        n += 1
        per.append({"key": list(key), "outcome_only": got["total"], "reference": expect_outcome_only})
    ok = not problems and not missing and n == len(judged)
    report = {"trajectories_in_judged_v21": len(judged), "compared": n, "missing_rollouts": missing,
              "mismatches": problems[:20], "n_mismatches": len(problems), "passed": ok,
              "outcome_only_mean": sum(p["outcome_only"] for p in per) / max(n, 1),
              "distinct_outcome_only_values": len({round(p["outcome_only"], 9) for p in per})}
    print(json.dumps({k: v for k, v in report.items() if k != "mismatches"}))
    for p in problems[:10]:
        print("MISMATCH", json.dumps(p))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(report, indent=1))
    print("PASS reference equality" if ok else "FAIL reference equality")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
