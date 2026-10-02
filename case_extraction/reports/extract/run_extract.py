#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from nautil_common import paths

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from extract import assemble, pipeline
from extract.adapters import ADAPTERS, PREFIX
from extract.client import STOP, Client
from nautil_common.case_contract import package_digest, validate_case


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_config(path: Path, endpoint: str | None) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if endpoint:
        config["endpoint"] = endpoint
    config["models"] = config["model_sets"][config["endpoint"]]
    return config


class Work:

    def __init__(self, root: Path, case_id: str, redo: bool):
        self.dir = root / case_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.redo = redo

    def get(self, name: str) -> Any:
        path = self.dir / f"{name}.json"
        if self.redo or not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if isinstance(value, dict) and value.get("ok") is False:
            return None
        return value

    def put(self, name: str, value: Any) -> Any:
        (self.dir / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return value


def run_case(*, source: str, native_id: str, case_id: str, client: Client, gate: pipeline.Gate,
             out_dir: Path, work_root: Path, redo: bool, block_workers: int,
             board: str | None = None) -> dict[str, Any]:
    adapter = ADAPTERS[source]
    started = time.monotonic()
    work = Work(work_root, case_id, redo)
    record: dict[str, Any] = {"case_id": case_id, "source": source, "native_id": native_id, "started_at": now()}
    try:
        raw = adapter.load(native_id, board) if source == "boards" else adapter.load(native_id)
    except Exception as exc:
        return {**record, "status": "failed", "step": "load", "error": f"{type(exc).__name__}: {exc}"}
    steps: dict[str, Any] = {}

    if raw.needs_block_labelling:
        label = work.get("label") or work.put("label", pipeline.step_label(raw, client, gate))
        if not label.get("ok"):
            return {**record, "status": "failed", "step": "label", "error": label.get("error")}
        for index, block in enumerate(raw.blocks):
            block.role = label["roles"].get(str(index), "appendix")
        steps["label"] = label
        raw.analysis_text = "\n\n".join(f"{b.label}\n{b.text}" for b in raw.blocks
                                        if b.role in ("analysis", "mixed"))
        raw.conclusion_text = "\n\n".join(f"{b.label}\n{b.text}" for b in raw.blocks if b.role == "conclusion")
        if not any(b.role in ("factual", "mixed") for b in raw.blocks):
            hints = tuple(raw.extra.get("factual_headings", ()))
            for block in raw.blocks:
                if hints and any(h in block.label.upper() for h in hints):
                    block.role = "factual"
            steps.setdefault("label", {})["factual_fallback"] = "no factual block; heading hints applied"
        if not any(b.role in ("factual", "mixed") for b in raw.blocks):
            return {**record, "status": "failed", "step": "label",
                    "error": "every block was labelled analysis or conclusion"}
        if not raw.analysis_text.strip():
            raw.analysis_text = "\n\n".join(f"{b.label}\n{b.text}" for b in raw.blocks if b.role == "conclusion")
            steps.setdefault("label", {})["analysis_fallback"] = "no analysis block; used the conclusion blocks"
        if not raw.analysis_text.strip():
            return {**record, "status": "failed", "step": "label", "error": "no analysis or conclusion block found"}
        if not raw.conclusion_text.strip():
            raw.conclusion_text = raw.analysis_text[-4000:]
            steps.setdefault("label", {})["conclusion_fallback"] = "no conclusion block; used the tail of the analysis"
    steps["analysis_text"] = raw.analysis_text

    atom = work.get("atomize")
    if atom is None:
        atom = work.put("atomize", pipeline.step_atomize(raw, client, gate, block_workers))
    if not atom.get("ok"):
        return {**record, "status": "failed", "step": "atomize", "error": atom.get("error")}
    steps["atomize"] = atom
    items = atom["items"]
    items[:] = [i for i in items if i["quote_verified"] or i["overlap"] >= 0.6]
    if len(items) < int(raw.extra.get("min_items", 10)):
        return {**record, "status": "failed", "step": "atomize",
                "error": f"only {len(items)} verified evidence items"}

    src = work.get("sources")
    if src is None:
        src = work.put("sources", pipeline.step_sources(items, client, gate))
    if not src.get("ok"):
        return {**record, "status": "failed", "step": "sources", "error": src.get("error")}
    by_id = {i["evidence_id"]: i for i in items}
    for saved in src.get("assigned", []):
        if saved["evidence_id"] in by_id:
            by_id[saved["evidence_id"]]["source_id"] = saved["source_id"]
    if any("source_id" not in i for i in items):
        src = work.put("sources", pipeline.step_sources(items, client, gate))
    src["assigned"] = [{"evidence_id": i["evidence_id"], "source_id": i["source_id"]} for i in items]
    work.put("sources", src)
    steps["sources"] = src

    scen = work.get("scenario")
    if scen is None:
        scen = work.put("scenario", pipeline.step_scenario(raw, items, src["sources"], client, gate))
    if not scen.get("ok"):
        return {**record, "status": "failed", "step": "scenario", "error": scen.get("error")}
    steps["scenario"] = scen

    anon = work.get("anonymize")
    if anon is None:
        anon = work.put("anonymize", pipeline.step_anonymize(raw, items, scen["initial_context"], client, gate))
    else:
        import re as _re
        pairs = [(a, b) for a, b in anon.get("replacements", [])]
        pairs.sort(key=lambda p: -len(p[0]))
        for item in items:
            item["text"] = pipeline.swap_pairs(pairs, item["text"])
            item["neutral_title"] = pipeline.swap_pairs(pairs, item["neutral_title"])
    if not anon.get("ok"):
        return {**record, "status": "failed", "step": "anonymize", "error": anon.get("error")}
    steps["anonymize"] = anon

    ref = work.get("reference")
    if ref is None:
        ref = work.put("reference", pipeline.step_reference(raw, items, raw.analysis_text,
                                                            raw.conclusion_text, client, gate))
    if not ref.get("ok"):
        return {**record, "status": "failed", "step": "reference", "error": ref.get("error")}
    steps["reference"] = ref

    rev = work.get("review")
    if rev is None:
        rev = work.put("review", pipeline.step_review(items, ref, raw.analysis_text, raw.conclusion_text,
                                                      client, gate, raw.extra.get("reference_note", "")))
    steps["review"] = rev

    built = assemble.build(case_id, raw, steps)
    package = built["package"]
    package["package_hash"] = package_digest(package)
    try:
        validate_case(package)
    except Exception as exc:
        return {**record, "status": "failed", "step": "contract", "error": f"{type(exc).__name__}: {exc}"}
    (out_dir / "runtime").mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime" / f"{case_id}.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review_dir = out_dir / "review" / case_id
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "reference.json").write_text(
        json.dumps(built["reference"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (review_dir / "provenance.md").write_text(assemble.provenance(case_id, raw, steps, built), encoding="utf-8")
    checks = built["checks"]
    if checks["counterevidence"] == 0:
        return {**record, "status": "failed", "step": "reference",
                "error": "no rejected explanation survived the review; the case cannot be scored",
                "checks": checks, "seconds": round(time.monotonic() - started, 1)}
    problems = []
    if checks["withheld_still_present"]:
        problems.append(f"withheld strings still in the package: {checks['withheld_still_present'][:5]}")
    if checks.get("residual_withheld_tokens"):
        problems.append(f"identifier tokens still in the package: {checks['residual_withheld_tokens'][:6]}")
    if checks["conclusion_words_in_package"]:
        problems.append(f"conclusion words in the package: {sorted(set(checks['conclusion_words_in_package']))[:5]}")

    return {**record, "status": "ok" if not problems else "attention", "problems": problems,
            "checks": checks, "seconds": round(time.monotonic() - started, 1),
            "finished_at": now()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--endpoint", choices=("api",))
    ap.add_argument("--serial-start", type=int, help="first serial for new case ids (parallel processes on one output)")
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--out", default=str(paths.CASES / "auto"))
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--case-workers", type=int, default=8)
    ap.add_argument("--block-workers", type=int, default=4)
    ap.add_argument("--gate", type=int, help="max simultaneous model calls (default from config)")
    ap.add_argument("--native-id", action="append", default=[], help="run exactly these ids")
    ap.add_argument("--native-id-file", help="one native id per line")
    ap.add_argument("--redo", action="store_true", help="ignore checkpoints")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-assemble cases that already have output, reusing every step file still on disk")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    config = load_config(Path(args.config), args.endpoint)
    pipeline.WHOLE_WORD = bool(config.get("anonymize_whole_word", False))
    out_dir = Path(args.out) / args.run_name / args.source
    work_root = out_dir / "_work"
    adapter = ADAPTERS[args.source]
    wanted = list(args.native_id)
    if args.native_id_file:
        wanted += [line.strip() for line in Path(args.native_id_file).read_text().splitlines() if line.strip()]
    if wanted:
        chosen = [{"native_id": n} for n in wanted]
        if args.source == "boards":
            index = {x["rid"]: x["board"] for x in adapter.index()}
            for item in chosen:
                item["board"] = index.get(item["native_id"])
    else:
        chosen = adapter.candidates(args.limit, seed=args.seed)
    serial_start = 1
    existing = sorted((out_dir / "runtime").glob("*.json")) if (out_dir / "runtime").exists() else []
    if existing:
        serial_start = max(int(p.stem.split("-")[-1]) for p in existing) + 1
    done = {json.loads(p.read_text())["case_id"]: p for p in existing}
    by_native: dict[str, str] = {}
    for path in existing:
        ref = out_dir / "review" / path.stem / "reference.json"
        if ref.exists():
            by_native[json.loads(ref.read_text())["source"]["native_id"]] = path.stem
    plan = []
    if args.serial_start:
        serial_start = args.serial_start
    serial = serial_start
    for item in chosen:
        native = item["native_id"]
        if native in by_native:
            plan.append({**item, "case_id": by_native[native], "skip": not (args.redo or args.rebuild)})
            continue
        plan.append({**item, "case_id": f"{PREFIX[args.source]}-{serial:04d}", "skip": False})
        serial += 1
    todo = [p for p in plan if not p["skip"]]
    for entry in todo:
        clash = work_root / entry["case_id"]
        if args.serial_start and clash.exists() and not (args.redo or args.rebuild):
            raise SystemExit(f"case id {entry['case_id']} already has a work folder; choose another --serial-start")
    header = {"source": args.source, "endpoint": config["endpoint"], "models": {k: v["id"] for k, v in config["models"].items()},
              "selected": len(plan), "already_done": len(plan) - len(todo), "to_run": len(todo),
              "case_workers": args.case_workers, "block_workers": args.block_workers,
              "gate": args.gate or config.get("gate", 50), "out": str(out_dir)}
    print(json.dumps(header, ensure_ascii=False, indent=2), flush=True)
    if args.plan or not todo:
        return 0

    client = Client(config)
    gate = pipeline.Gate(args.gate or config.get("gate", 50))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: STOP.set())
    lock = threading.Lock()
    counts = {"ok": 0, "attention": 0, "failed": 0}
    results: list[dict[str, Any]] = []

    def one(entry: dict[str, Any]) -> dict[str, Any]:
        return run_case(source=args.source, native_id=entry["native_id"], case_id=entry["case_id"],
                        client=client, gate=gate, out_dir=out_dir, work_root=work_root, redo=args.redo,
                        block_workers=args.block_workers, board=entry.get("board"))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.case_workers) as pool:
        futures = {pool.submit(one, entry): entry for entry in todo}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                result = future.result()
            except BaseException as exc:
                result = {"case_id": entry["case_id"], "native_id": entry["native_id"], "status": "failed",
                          "step": "unhandled", "error": f"{type(exc).__name__}: {exc}"}
            with lock:
                results.append(result)
                counts[result["status"]] = counts.get(result["status"], 0) + 1
                checks = result.get("checks") or {}
                print(f"[{now()}] {len(results)}/{len(todo)} {result['status']:9} {result['case_id']} "
                      f"{result['native_id'][:44]:44} items={checks.get('items')} cov={checks.get('mean_coverage')} "
                      f"rej={checks.get('counterevidence')} {result.get('seconds', '')}s"
                      + (f" | {result.get('step')}: {str(result.get('error'))[:120]}" if result["status"] == "failed" else "")
                      + (f" | {result.get('problems')}" if result.get("problems") else ""), flush=True)
                (out_dir / "run_log.jsonl").parent.mkdir(parents=True, exist_ok=True)
                with (out_dir / "run_log.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    usage = client.usage.snapshot()
    client.close()
    summary = {"finished_at": now(), "wall_seconds": round(time.time() - t0, 1), **counts,
               "usage": usage, "usd_per_case": round(usage["cost_usd"] / max(1, len(results)), 4)}
    (out_dir / "run_summary.json").write_text(json.dumps({**header, **summary}, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
