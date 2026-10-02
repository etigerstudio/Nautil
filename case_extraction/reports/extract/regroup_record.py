#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import statistics as st
import sys
from pathlib import Path

from nautil_common.case_contract import package_digest, validate_case
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from nautil_common import paths

RECORD_KINDS = {"observation", "measurement"}
RECORD_MAX = 4
ID_RE = re.compile(r"\bE\d+\.\d+\b")


def record_ids(items: list[dict], k: int = RECORD_MAX) -> list[str]:
    primary = items[0]["source_id"]
    chosen = [
        i["evidence_id"] for i in items
        if i["source_id"] == primary and i["kind"] in RECORD_KINDS
    ][:k]
    return chosen or [items[0]["evidence_id"]]


def regroup(items: list[dict], k: int = RECORD_MAX) -> tuple[list[dict], dict[str, str]]:
    record = set(record_ids(items, k))
    ordered: list[dict] = [i for i in items if i["evidence_id"] in record]
    by_source: dict[str, list[dict]] = {}
    for item in items:
        if item["evidence_id"] in record:
            continue
        by_source.setdefault(item["source_id"], []).append(item)
    id_map: dict[str, str] = {}
    out: list[dict] = []
    for position, item in enumerate(ordered, 1):
        new = dict(item)
        id_map[item["evidence_id"]] = f"E1.{position}"
        new["evidence_id"] = f"E1.{position}"
        out.append(new)
    for group, (_source, group_items) in enumerate(by_source.items(), start=2):
        for position, item in enumerate(group_items, 1):
            new = dict(item)
            id_map[item["evidence_id"]] = f"E{group}.{position}"
            new["evidence_id"] = f"E{group}.{position}"
            out.append(new)
    return out, id_map


def _rewrite(match: re.Match, id_map: dict[str, str]) -> str:
    old = match.group(0)
    return id_map.get(old, f"dropped:{old}")


def remap(value, id_map: dict[str, str]):
    if isinstance(value, str):
        return ID_RE.sub(lambda m: _rewrite(m, id_map), value)
    if isinstance(value, list):
        return [remap(v, id_map) for v in value]
    if isinstance(value, dict):
        return {remap(k, id_map): remap(v, id_map) for k, v in value.items()}
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", default=str(paths.CASES / "auto/scale"))
    ap.add_argument("--out", default=str(paths.CASES / "auto/scale_regrouped_v1"))
    ap.add_argument("--record-max", type=int, default=RECORD_MAX)
    ap.add_argument("--sources", nargs="*", default=["boards", "ntsb", "nhtsa", "medical"])
    args = ap.parse_args()

    src_root, out_root = Path(args.src), Path(args.out)
    report: dict[str, dict] = {}
    for source in args.sources:
        runtime = src_root / source / "runtime"
        if not runtime.exists():
            continue
        (out_root / source / "runtime").mkdir(parents=True, exist_ok=True)
        rec_sizes, fetch_sizes = [], []
        fallbacks = invalid = 0
        before_fetch = []
        for path in sorted(runtime.glob("*.json")):
            package = json.loads(path.read_text())
            items = package["evidence_items"]
            before_fetch.append(sum(1 for i in items if i["evidence_id"].split(".")[0] != "E1"))
            new_items, id_map = regroup(items, args.record_max)
            chosen = record_ids(items, args.record_max)
            if len(chosen) == 1 and items[0]["kind"] not in RECORD_KINDS:
                fallbacks += 1
            package["evidence_items"] = new_items
            package["package_hash"] = ""
            package["package_hash"] = package_digest(package)
            try:
                validate_case(package)
            except Exception as exc:
                invalid += 1
                print(f"  INVALID {path.stem}: {type(exc).__name__}: {exc}", flush=True)
                continue
            (out_root / source / "runtime" / path.name).write_text(
                json.dumps(package, ensure_ascii=False, indent=1), encoding="utf-8")
            rec_sizes.append(sum(1 for i in new_items if i["evidence_id"].startswith("E1.")))
            fetch_sizes.append(len(new_items) - rec_sizes[-1])

            review_in = src_root / source / "review" / path.stem
            review_out = out_root / source / "review" / path.stem
            if review_in.exists():
                review_out.mkdir(parents=True, exist_ok=True)
                for item in review_in.iterdir():
                    if item.suffix == ".json":
                        review_out.joinpath(item.name).write_text(
                            json.dumps(remap(json.loads(item.read_text()), id_map),
                                       ensure_ascii=False, indent=1), encoding="utf-8")
                    elif item.suffix == ".md":
                        review_out.joinpath(item.name).write_text(
                            ID_RE.sub(lambda m: _rewrite(m, id_map), item.read_text()),
                            encoding="utf-8")
                    else:
                        shutil.copy2(item, review_out / item.name)
        report[source] = {
            "packages": len(rec_sizes),
            "invalid": invalid,
            "record_median": int(st.median(rec_sizes)) if rec_sizes else 0,
            "requestable_before": int(st.median(before_fetch)) if before_fetch else 0,
            "requestable_after_min_med_max": [min(fetch_sizes), int(st.median(fetch_sizes)), max(fetch_sizes)] if fetch_sizes else [],
            "packages_under_10_requestable": sum(1 for f in fetch_sizes if f < 10),
            "record_fell_back_to_first_item": fallbacks,
        }
        print(json.dumps({source: report[source]}, ensure_ascii=False), flush=True)
    (out_root / "REGROUP_REPORT.json").write_text(
        json.dumps({"record_max": args.record_max, "rule": "first K observation/measurement items of the primary source",
                    "sources": report}, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
