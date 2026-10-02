#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_sft_trajectories_v1 as generation
import structure_sft_scale60_evidence_v2 as ledger
import audit_sft_scale60_evidence_v2 as audit_wrapper
import revise_sft_final_only_v1 as revision

AUDIT = audit_wrapper.audit
AUDIT.MODEL = "gpt-6-sol"
MANIFEST = paths.CONFIGS / "sft_full_flat_v1.json"
MODEL = "gpt-6-sol"


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def audit_case(case_id: str, trace_path: Path, out: Path, meta: dict,
               routes: list[tuple[str, str]], semaphores: dict[str, threading.Semaphore]) -> dict:
    trace = json.loads(trace_path.read_text())
    if trace.get("training_approved") is not False:
        return {"case_id": case_id, "error": "source trace not review-only"}
    prompt = {"case_id": case_id,
              "trusted_reference_reviewer_only": AUDIT.reviewer_reference(case_id, meta),
              "candidate_model_visible_trace": {"tools": trace["tools"], "messages": trace["messages"]}}
    warnings = (trace.get("provenance") or {}).get("ledger_grounding_warnings") or []
    if warnings:
        prompt["teacher_grounding_warnings_reviewer_only"] = warnings
    user_text = compact(prompt)
    prompt_hash = hashlib.sha256((AUDIT.SYSTEM + user_text).encode()).hexdigest()
    raw_path = out / "raw" / f"{case_id}.json"
    if raw_path.exists():
        return AUDIT.parse_saved_audit(case_id, raw_path, prompt_hash)
    inflight = out / "inflight" / f"{case_id}.json"
    if inflight.exists():
        return {"case_id": case_id, "error": "uncertain_inflight_needs_manual_retry"}
    route, api_key = routes[int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % len(routes)]
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": AUDIT.SYSTEM},
                         {"role": "user", "content": user_text}],
            "response_format": {"type": "json_object"}}
    request = urllib.request.Request("https://example.com/v1/chat/completions",
        data=compact(body).encode(), method="POST",
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json",
                 "HTTP-Referer": "https://localhost/nautil", "X-Title": "Nautil SFT Audit"})
    started = time.monotonic()
    with semaphores[route]:
        atomic_json(inflight, {"case_id": case_id, "route": route,
                               "prompt_sha256": prompt_hash, "started_unix": time.time()})
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            inflight.unlink(missing_ok=True)
            return {"case_id": case_id, "audit_route": route,
                    "error": f"HTTP {exc.code}: {exc.read(300).decode(errors='replace')[:250]}"}
        except Exception as exc:
            inflight.unlink(missing_ok=True)
            return {"case_id": case_id, "audit_route": route,
                    "error": f"{type(exc).__name__}: {str(exc)[:250]}"}
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    if isinstance(content, list):
        content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
    atomic_json(raw_path, {"case_id": case_id, "route": route, "model_requested": MODEL,
                           "model_returned": data.get("model"), "provider_generation_id": data.get("id"),
                           "prompt_sha256": prompt_hash, "elapsed_seconds": round(time.monotonic()-started, 2),
                           "usage": data.get("usage"), "finish_reason": choice.get("finish_reason"),
                           "teacher_text": content})
    inflight.unlink(missing_ok=True)
    return AUDIT.parse_saved_audit(case_id, raw_path, prompt_hash)


class StreamingRun:
    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.items = manifest["items"]
        self.out = ROOT / manifest["output"]
        self.stage1 = self.out / "stage1"
        self.stage2 = self.out / "stage2"
        self.primary_dir = self.out / "primary_audit"
        self.revision_dir = self.out / "final_revision"
        self.reaudit_dir = self.out / "reaudit"
        self.quality_dir = self.out / "quality_gate"
        self.route_specs = {item["name"]: item for item in
                            json.loads((ROOT / manifest["route_config"]).read_text())["routes"]}
        self.values = generation.env()
        self.route_sems = {name: threading.Semaphore(spec["max_concurrent"])
                           for name, spec in self.route_specs.items()}
        self.audit_routes = [(name, self.values[self.route_specs[name]["key_env"]])
                             for name in ("api",)]
        self.meta = {row["case_id"]: row for row in
                     json.loads((RUN / "docs/SFT_POOL_AUDIT_20260924.json").read_text())["rows"]}
        self.lock = threading.Lock()
        self.first_pass_done = threading.Event()
        self.second_pass_done = threading.Event()
        self.first_results: dict[str, dict] = {}
        self.second_results: dict[str, dict] = {}
        self.primary_results: dict[str, dict] = {}
        self.revision_results: dict[str, dict] = {}
        self.reaudit_results: dict[str, dict] = {}
        self.first_terminal: dict[str, str] = {}
        self.second_terminal: dict[str, str] = {}
        self.stage1_pool = concurrent.futures.ThreadPoolExecutor(max_workers=paths.CONCURRENCY)
        self.stage2_pool = concurrent.futures.ThreadPoolExecutor(max_workers=paths.CONCURRENCY)
        self.audit_pool = concurrent.futures.ThreadPoolExecutor(max_workers=paths.CONCURRENCY)
        self.revision_pool = concurrent.futures.ThreadPoolExecutor(max_workers=paths.CONCURRENCY)
        ledger.OUT = self.stage2
        revision.OUT = self.revision_dir

    def checkpoint(self, stage: str, case_id: str, value: dict) -> None:
        atomic_json(self.out / stage / "checkpoints" / f"{case_id}.json", value)

    def attach(self, future: concurrent.futures.Future, case_id: str, method,
               phase: str, *args) -> None:
        def invoke(finished: concurrent.futures.Future) -> None:
            try:
                method(*args, finished)
            except Exception as exc:
                error = {"case_id": case_id, "error":
                         f"callback_exception: {type(exc).__name__}: {str(exc)[:250]}"}
                atomic_json(self.out / "callback_errors" / f"{case_id}.json", error)
                if phase == "first":
                    self.first_finish(case_id, "callback_error")
                else:
                    self.second_finish(case_id, "callback_error")
        future.add_done_callback(invoke)

    def first_finish(self, case_id: str, status: str) -> None:
        with self.lock:
            self.first_terminal[case_id] = status
            if len(self.first_terminal) == len(self.items):
                self.first_pass_done.set()

    def second_finish(self, case_id: str, status: str) -> None:
        with self.lock:
            self.second_terminal[case_id] = status
            if len(self.second_terminal) == self.plus3_total:
                self.second_pass_done.set()

    @staticmethod
    def result(future: concurrent.futures.Future, case_id: str) -> dict:
        try:
            return future.result()
        except Exception as exc:
            return {"case_id": case_id, "error": f"local_exception: {type(exc).__name__}: {str(exc)[:250]}"}

    def submit_first_pass(self) -> None:
        for item in self.items:
            cid = item["case_id"]
            spec = self.route_specs[item["route"]]
            future = self.stage1_pool.submit(generation.call, cid, spec,
                self.values[spec["key_env"]], self.stage1, self.route_sems[spec["name"]],
                item.get("reuse_stage1_raw_path"))
            self.attach(future, cid, self.after_stage1, "first", item)

    def after_stage1(self, item: dict, future: concurrent.futures.Future) -> None:
        cid = item["case_id"]
        result = self.result(future, cid)
        with self.lock:
            self.first_results[cid] = result
        self.checkpoint("stage1", cid, result)
        if result.get("status") != "compiled_review_draft":
            self.first_finish(cid, "stage1_" + str(result.get("status") or "error"))
            return
        next_item = {"case_id": cid, "route": result["route"], "trace": result["trace"]}
        future2 = self.stage2_pool.submit(self.stage2_task, next_item, item.get("reuse_stage2_raw_path"))
        self.attach(future2, cid, self.after_stage2, "first", next_item)

    def stage2_task(self, item: dict, reuse_path: str | None) -> dict:
        if reuse_path:
            source = ROOT / reuse_path
            target = self.stage2 / "raw" / f"{item['case_id']}.json"
            if source.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        spec = self.route_specs[item["route"]]
        return ledger.run(item, spec, self.values[spec["key_env"]], self.route_sems[spec["name"]])

    def after_stage2(self, item: dict, future: concurrent.futures.Future) -> None:
        cid = item["case_id"]
        result = self.result(future, cid)
        with self.lock:
            self.second_results[cid] = result
        self.checkpoint("stage2", cid, result)
        if result.get("status") != "structured_review_draft":
            self.first_finish(cid, "stage2_" + str(result.get("status") or "error"))
            return
        audit_future = self.audit_pool.submit(audit_case, cid, ROOT / result["trace"],
            self.primary_dir, self.meta[cid], self.audit_routes, self.route_sems)
        self.attach(audit_future, cid, self.after_primary, "first", cid)

    def after_primary(self, cid: str, future: concurrent.futures.Future) -> None:
        result = self.result(future, cid)
        with self.lock:
            self.primary_results[cid] = result
        self.checkpoint("primary_audit", cid, result)
        self.first_finish(cid, "scored" if isinstance(result.get("score"), int) else "audit_error")

    def submit_second_pass(self) -> None:
        plus3 = [cid for cid, value in self.primary_results.items()
                 if value.get("score") == 3 and value.get("hard_failure") is False]
        self.plus3_total = len(plus3)
        if not plus3:
            self.second_pass_done.set()
            return
        key = self.values[self.route_specs["api"]["key_env"]]
        for cid in plus3:
            future = self.revision_pool.submit(revision.run, cid, key,
                self.route_sems["api"], self.primary_results[cid], self.stage2 / "traces")
            self.attach(future, cid, self.after_revision, "second", cid)

    def after_revision(self, cid: str, future: concurrent.futures.Future) -> None:
        result = self.result(future, cid)
        with self.lock:
            self.revision_results[cid] = result
        self.checkpoint("final_revision", cid, result)
        if result.get("status") != "final_only_review_draft":
            self.second_finish(cid, "revision_" + str(result.get("status") or "error"))
            return
        next_future = self.audit_pool.submit(audit_case, cid, ROOT / result["trace"],
            self.reaudit_dir, self.meta[cid], self.audit_routes, self.route_sems)
        self.attach(next_future, cid, self.after_reaudit, "second", cid)

    def after_reaudit(self, cid: str, future: concurrent.futures.Future) -> None:
        result = self.result(future, cid)
        with self.lock:
            self.reaudit_results[cid] = result
        self.checkpoint("reaudit", cid, result)
        self.second_finish(cid, "rescored" if isinstance(result.get("score"), int) else "reaudit_error")

    def dashboard(self, phase: str) -> dict:
        with self.lock:
            terminal = len(self.first_terminal)
            secondary_terminal = len(self.second_terminal)
            stage1_values = list(self.first_results.values())
            stage2_values = list(self.second_results.values())
            primary_values = list(self.primary_results.values())
            revision_values = list(self.revision_results.values())
            reaudit_values = list(self.reaudit_results.values())
        def count(values: list[dict], key: str) -> dict:
            return dict(Counter(value.get(key, "error") for value in values))
        return {"phase": phase, "cases": len(self.items), "first_pass_terminal": terminal,
                "stage1": count(stage1_values, "status"),
                "stage2": count(stage2_values, "status"),
                "primary_scores": dict(Counter(str(value.get("score")) for value in primary_values)),
                "plus3_total": getattr(self, "plus3_total", None),
                "second_pass_terminal": secondary_terminal,
                "revision": count(revision_values, "status"),
                "reaudit_scores": dict(Counter(str(value.get("score")) for value in reaudit_values))}

    def wait_phase(self, phase: str, event: threading.Event) -> None:
        while not event.wait(30):
            snapshot = self.dashboard(phase)
            atomic_json(self.out / "status.json", snapshot)
            print(compact(snapshot), flush=True)
        snapshot = self.dashboard(phase)
        atomic_json(self.out / "status.json", snapshot)
        print(compact(snapshot), flush=True)

    def export(self) -> dict:
        decisions = []
        clean = []
        for item in self.items:
            cid = item["case_id"]
            first = self.primary_results.get(cid)
            second = self.reaudit_results.get(cid)
            decision = {"case_id": cid, "family": item["family"],
                        "stage1_status": self.first_results.get(cid, {}).get("status"),
                        "stage2_status": self.second_results.get(cid, {}).get("status"),
                        "initial_score": first.get("score") if first else None,
                        "revised_score": second.get("score") if second else None}
            trace_path = None
            if first and first.get("score", -5) >= 4 and first.get("hard_failure") is False and first.get("action") == "keep":
                trace_path = ROOT / self.second_results[cid]["trace"]
                decision["decision"] = "content_pass"
            elif first and first.get("score") == 3:
                revised = self.revision_results.get(cid)
                if (revised and revised.get("status") == "final_only_review_draft" and second
                        and second.get("score", -5) >= 4 and second.get("hard_failure") is False
                        and second.get("action") == "keep"):
                    trace_path = ROOT / revised["trace"]
                    decision["decision"] = "content_pass_after_one_final_only_revision"
                else:
                    decision["decision"] = "hold_after_one_final_only_revision"
            elif first:
                decision["decision"] = "content_revise_or_skip"
            else:
                decision["decision"] = "structural_or_transport_unscored"
            if trace_path:
                trace = json.loads(trace_path.read_text())
                if trace.get("training_approved") is not False:
                    raise ValueError(f"unexpected approved source trace: {cid}")
                decision["trace"] = str(trace_path.relative_to(ROOT))
                clean.append({"case_id": cid, "source": trace["source"],
                              "tools": trace["tools"], "messages": trace["messages"]})
            decisions.append(decision)
        atomic_json(self.quality_dir / "summary.json",
            {"cases": len(decisions), "decisions": dict(Counter(x["decision"] for x in decisions)),
             "medical_excluded": True, "model_visible_export_excludes_reference_and_judge": True,
             "qwen_wire_tokenization_verified": False, "training_started": False})
        (self.quality_dir / "decisions.jsonl").write_text("".join(compact(x) + "\n" for x in decisions))
        (self.quality_dir / "content_passed_model_visible.jsonl").write_text(
            "".join(compact(x) + "\n" for x in clean))
        return {"cases": len(decisions), "content_passed": len(clean),
                "decisions": dict(Counter(x["decision"] for x in decisions))}

    def shutdown(self) -> None:
        self.stage1_pool.shutdown(wait=True)
        self.stage2_pool.shutdown(wait=True)
        self.audit_pool.shutdown(wait=True)
        self.revision_pool.shutdown(wait=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    items = manifest["items"]
    routes = json.loads((ROOT / manifest["route_config"]).read_text())["routes"]
    if len(items) != 1345 or any(x["family"] == "medical" for x in items):
        raise ValueError("flat queue must contain only the 1,345 eligible nonmedical cases")
    print(compact({"cases": len(items), "scheduling": "single global case queue; no batches",
                   "route_limits": {x["name"]: x["max_concurrent"] for x in routes},
                   "pass_order": manifest["pass_order"], "execute": args.execute}), flush=True)
    if not args.execute:
        return
    runner = StreamingRun(manifest)
    runner.out.mkdir(parents=True, exist_ok=True)
    try:
        runner.submit_first_pass()
        runner.wait_phase("first_pass", runner.first_pass_done)
        runner.submit_second_pass()
        runner.wait_phase("final_only_revisions", runner.second_pass_done)
        result = runner.export()
        completed_status = runner.dashboard("complete")
        completed_status["quality_gate"] = result
        atomic_json(runner.out / "status.json", completed_status)
        print(compact({"completed": result}), flush=True)
    finally:
        runner.shutdown()


if __name__ == "__main__":
    main()
