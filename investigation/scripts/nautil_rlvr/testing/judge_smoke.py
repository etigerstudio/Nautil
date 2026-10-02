from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from nautil_harness_v2_2 import run_case

from nautil_rlvr.common import read_jsonl
from nautil_rlvr.judge import JudgeClient, JudgeConfig
from nautil_rlvr.reward import RewardConfig, compute, judge_requests, model_closed
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # repository root
from nautil_common import paths

RUN = paths.RUN


def call_text(msg: dict) -> str:
    text = (msg.get("content") or "").strip()
    for call in msg.get("tool_calls") or []:
        args = call["function"]["arguments"]
        body = "".join(f"<parameter={k}>\n{json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}\n</parameter>\n"
                       for k, v in args.items())
        text += ("\n\n" if text else "") + f"<tool_call>\n<function={call['function']['name']}>\n{body}</function>\n</tool_call>"
    return text


def replay(case: dict, texts: list[str]) -> dict:
    it = iter(texts)
    return run_case(None, None, "system", case, None, 32768, 15,
                    generate_fn=lambda *a: (next(it), 100, False))


def flip(final: str) -> str:
    if "CASE NOT CLOSED" in final:
        return final.replace("CASE NOT CLOSED", "CASE CLOSED")
    return final.replace("CASE CLOSED", "CASE NOT CLOSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--budget", type=float, default=0.8)
    ap.add_argument("--train-model", default="glm-5.3-flash")
    ap.add_argument("--eval-cases", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = RUN / "results/rlvr_v1/train_data_v1"
    cases = {c["case_id"]: c for c in read_jsonl(data / "train_bundle.jsonl")}
    refs = {r["case_id"]: r for r in read_jsonl(data / "train_refs.jsonl")}
    teach = {r["case_id"]: r for r in read_jsonl(RUN / "results/sft_content_passed_nonmedical_v2_2/split_v1/train_model_visible.jsonl")}
    rng = random.Random(7)
    by_stratum = {}
    for cid in sorted(cases):
        by_stratum.setdefault((refs[cid]["source"], refs[cid]["closure"]), []).append(cid)
    pick = []
    for key in sorted(by_stratum):
        pick.append(rng.choice(by_stratum[key]))
    pick = pick[:args.cases]
    rcfg = RewardConfig.from_dict({"phase": 2})
    judge = JudgeClient(JudgeConfig(model=args.train_model, env_file=str(paths.ENV_FILE), budget_usd=args.budget,
                                    max_tokens=8192))
    rows, began = [], time.time()
    for cid in pick:
        texts = [call_text(m) for m in teach[cid]["messages"] if m["role"] == "assistant"]
        for variant, tx in (("teacher", texts), ("closure_flipped", texts[:-1] + [flip(texts[-1])])):
            res = replay(cases[cid], tx)
            reqs = judge_requests(res, cases[cid], refs[cid], rcfg)
            if variant != "teacher":
                reqs = {"closure": reqs["closure"]}
            futs = {d: judge.submit(d, t, n, model_closed(res)) for d, (t, n) in reqs.items()}
            judged = {d: f.result() for d, f in futs.items()}
            r = compute(res, cases[cid], refs[cid], rcfg, judged)
            rows.append({"case_id": cid, "source": refs[cid]["source"], "reference": refs[cid]["closure"],
                         "variant": variant, "marker": res["conclusion_marker"], "items": {d: n for d, (_, n) in reqs.items()},
                         "raw": r["raw_judge_scores"], "failed": [d for d, j in judged.items() if "error" in j],
                         "total": round(r["total"], 3), "dimensions": {k: round(v, 3) for k, v in r["dimensions"].items()},
                         "justifications": {d: (j.get("items") or [j.get("conclusion")])[:2] for d, j in judged.items()
                                            if "error" not in j}})
    train_health = judge.cumulative()
    judge.close()
    ejudge = JudgeClient(JudgeConfig(model="gpt-6-sol", routes_file=str(paths.CONFIGS / "api_routes.json"),
                                     route_names=["api"],
                                     env_file=str(paths.ENV_FILE), max_tokens=None, temperature=None,
                                     response_format_json=False, timeout_seconds=300, retry_backoff=[5, 20]))
    erows = []
    for cid in pick[:args.eval_cases]:
        texts = [call_text(m) for m in teach[cid]["messages"] if m["role"] == "assistant"]
        res = replay(cases[cid], texts)
        text, n = judge_requests(res, cases[cid], refs[cid], rcfg)["closure"]
        out = ejudge.call("closure", text, n, model_closed(res))
        erows.append({"case_id": cid, "ok": "error" not in out, "provider": out.get("provider"),
                      "conclusion": (out.get("conclusion") or {}).get("score"),
                      "gap": (out.get("gap") or {}).get("score") if out.get("gap") else None,
                      "format_errors": out.get("format_errors")})
    eval_health = ejudge.cumulative()
    ejudge.close()
    report = {"train_judge_model": args.train_model, "cases": pick, "seconds": round(time.time() - began, 1),
              "train_judge_health": train_health, "rows": rows,
              "eval_judge": {"model": "gpt-6-sol", "rows": erows,
                             "health": {k: v for k, v in eval_health.items() if k != "per_key"},
                             "per_route": eval_health["per_key"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"train_judge": {k: train_health[k] for k in ("calls", "success_rate", "parse_failures", "errors",
                                                                   "latency_p50", "latency_max", "cost_usd", "prompt_tokens",
                                                                   "completion_tokens", "providers")},
                      "rows": [{k: r[k] for k in ("case_id", "variant", "reference", "marker", "raw", "total", "failed")} for r in rows],
                      "eval_judge": erows, "eval_cost_usd": eval_health["cost_usd"],
                      "eval_tokens": [eval_health["prompt_tokens"], eval_health["completion_tokens"]]}, indent=1))


if __name__ == "__main__":
    main()
