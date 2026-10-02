from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from . import SCRIPTS_DIR
from .common import (EVIDENCE_ID, atomic_json, initial_record_ids, read_jsonl, refuse_test_path,
                     sha256_file, write_jsonl)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from nautil_common import paths

RUN = paths.RUN
ROOT = paths.DATA
SPLIT = RUN / "results/sft_content_passed_nonmedical_v2_2/split_v1"
DEFAULT_OUT = RUN / "results/rlvr_v1/train_data_v1"


def closure_from_expected(expected: str) -> str:
    if expected in ("teacher_closed",):
        return "closed"
    if expected in ("teacher_not_closed",):
        return "not_closed"
    if expected.startswith("determined"):
        return "closed"
    if expected.startswith("undetermined") and expected != "undetermined_by_source":
        return "not_closed"
    raise ValueError(f"cannot map expected_closure {expected[:60]!r}")


def build(output: Path) -> dict:
    from prepare_sft_v2_2_validation_bundle import valid_package
    ids = (SPLIT / "train_ids.txt").read_text().split()
    train_file = refuse_test_path(SPLIT / "train_model_visible.jsonl")
    examples = {r["case_id"]: r for r in read_jsonl(train_file)}
    meta = {r["case_id"]: r for r in read_jsonl(SPLIT / "per_case_metadata.jsonl")
            if r["split"] == "train"}
    if len(ids) != 545 or set(ids) != set(examples) or set(ids) != set(meta):
        raise ValueError("train split is not the frozen 545 cases")
    work = {r["case_id"]: r for r in read_jsonl(RUN / "results/multiturn_worklist_v2/cases.jsonl")}
    host = {r["case_id"]: r for r in
            read_jsonl(RUN / "results/host_teacher_closure_v1/host_teacher_closure.jsonl")}
    bundle, refs, problems = [], [], []
    chosen = collections.Counter()
    for cid in ids:
        ex = examples[cid]
        candidates = []
        if cid in work:
            candidates.append(ROOT / work[cid]["package_path"])
        if ex["source"] == "host":
            candidates.append(paths.HOST_CASES / f"runtime/{cid}.json")
        candidates.append(RUN / f"results/sft_extra_v1/audit_repair/packages/{cid}.json")
        selected = None
        for path in candidates:
            if path.is_file():
                package = json.loads(path.read_text())
                if valid_package(ex, package):
                    selected = path, package
                    break
        if selected is None:
            problems.append({"case_id": cid, "problem": "no full evidence store matches"})
            continue
        path, package = selected
        chosen[str(path.relative_to(ROOT)).split("/")[0]] += 1
        final = ex["messages"][-1]
        if final["role"] != "assistant" or final.get("tool_calls"):
            problems.append({"case_id": cid, "problem": "teacher trajectory has no final answer"})
            continue
        teacher_final = final["content"]
        marker = ("not_closed" if "CASE NOT CLOSED" in teacher_final else
                  "closed" if "CASE CLOSED" in teacher_final else None)
        key = sorted(set(EVIDENCE_ID.findall(teacher_final)),
                     key=lambda x: tuple(int(p) for p in x[1:].split(".")))
        store = {item["evidence_id"] for item in package["evidence_items"]}
        if not set(key) <= store:
            problems.append({"case_id": cid, "problem": "teacher cites ids outside the store"})
        initial = initial_record_ids(ex["messages"][0]["content"])
        if ex["source"] == "host":
            label = host[cid]
            if label["split"] != "train":
                raise ValueError(f"host label split mismatch {cid}")
            closure = closure_from_expected(label["expected_closure"])
            provenance = "host_teacher_closure_v1 (teacher judgement, not ground truth)"
            ref_path = paths.HOST_CASES / f"review/{cid}/reference.json"
            reference = json.loads(ref_path.read_text()) if ref_path.is_file() else {}
            official = None
            alternatives = reference.get("alternatives_left_open") or []
        else:
            ref_path = ROOT / work[cid]["review_only_reference_path"]
            reference = json.loads(ref_path.read_text())
            if reference["case_id"] != cid:
                raise ValueError(f"reference identity mismatch {cid}")
            closure = closure_from_expected(reference["expected_closure"])
            provenance = "reference.json expected_closure (official report)"
            official = reference.get("official_conclusion_verbatim")
            alternatives = reference.get("alternatives_left_open") or []
        if closure != meta[cid]["closure"]:
            problems.append({"case_id": cid, "problem": f"label {closure} != split metadata "
                             f"{meta[cid]['closure']}"})
        bundle.append({"case_id": cid, "source": ex["source"], "user_message": ex["messages"][0]["content"],
                       "tools": ex["tools"], "evidence_items": package["evidence_items"],
                       "package_hash": package["package_hash"]})
        refs.append({"case_id": cid, "source": ex["source"], "category": meta[cid]["category"],
                     "closure": closure, "closure_provenance": provenance,
                     "teacher_marker": marker, "teacher_final_answer": teacher_final,
                     "key_evidence_ids": key,
                     "key_fetchable_ids": [k for k in key if k not in initial],
                     "official_conclusion": official,
                     "alternatives_left_open": alternatives,
                     "reference_path": str(ref_path.relative_to(ROOT)) if ref_path.is_file() else None,
                     "teacher_fetch_rounds": sum(bool(m.get("tool_calls")) for m in ex["messages"])})
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "train_bundle.jsonl", bundle)
    write_jsonl(output / "train_refs.jsonl", refs)
    strata = collections.Counter(f"{r['source']}/{r['closure']}" for r in refs)
    manifest = {"schema_version": "nautil.rlvr.train_data.v1", "split": "train",
                "cases": len(bundle), "train_ids_sha256": sha256_file(SPLIT / "train_ids.txt"),
                "source_file_sha256": sha256_file(train_file),
                "bundle_sha256": sha256_file(output / "train_bundle.jsonl"),
                "refs_sha256": sha256_file(output / "train_refs.jsonl"),
                "strata": dict(sorted(strata.items())), "package_roots": dict(chosen),
                "teacher_marker_vs_label_disagree": sum(r["teacher_marker"] != r["closure"] for r in refs),
                "key_evidence": {"mean": round(sum(len(r["key_evidence_ids"]) for r in refs) / len(refs), 2),
                                 "mean_fetchable": round(sum(len(r["key_fetchable_ids"]) for r in refs) / len(refs), 2),
                                 "cases_without_fetchable_key": sum(not r["key_fetchable_ids"] for r in refs)},
                "problems": problems,
                "test_data_read": False, "bundle_model_visible": True, "refs_model_visible": False}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def build_val_refs(output: Path) -> dict:
    val_file = refuse_test_path(SPLIT / "val_model_visible.jsonl")
    rows = {r["case_id"]: r for r in read_jsonl(val_file)}
    refs_v2 = {r["case_id"]: r for r in
               read_jsonl(RUN / "results/sft_v2_2_eval/validation_judge_refs_v2/references.jsonl")}
    bundle = {r["case_id"]: r for r in
              read_jsonl(RUN / "results/sft_v2_2_eval/validation_bundle_v1/validation_prompts_and_evidence.jsonl")}
    out = []
    for cid in (SPLIT / "val_ids.txt").read_text().split():
        final = rows[cid]["messages"][-1]["content"]
        key = sorted(set(EVIDENCE_ID.findall(final)), key=lambda x: tuple(int(p) for p in x[1:].split(".")))
        initial = initial_record_ids(bundle[cid]["user_message"])
        ref = refs_v2[cid]
        out.append({"case_id": cid, "source": rows[cid]["source"],
                    "closure": closure_from_expected(ref["expected_closure"]),
                    "closure_provenance": "validation_judge_refs_v2",
                    "teacher_final_answer": final, "key_evidence_ids": key,
                    "key_fetchable_ids": [k for k in key if k not in initial],
                    "official_conclusion": None if cid.startswith("HOST") else ref.get("official_conclusion_verbatim"),
                    "alternatives_left_open": ref.get("alternatives_left_open") or []})
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "val_refs.jsonl", out)
    manifest = {"schema_version": "nautil.rlvr.val_refs.v1", "cases": len(out), "split": "validation",
                "file_sha256": sha256_file(output / "val_refs.jsonl"), "model_visible": False,
                "closure": dict(collections.Counter(r["closure"] for r in out))}
    atomic_json(output / "manifest.json", manifest)
    return manifest


class StratifiedSampler:

    def __init__(self, refs: list[dict], alpha: float = 0.0, seed: int = 20260925,
                 exclude: set[str] | None = None):
        exclude = exclude or set()
        strata = collections.defaultdict(list)
        for r in refs:
            if r["case_id"] not in exclude:
                strata[(r["source"], r["closure"])].append(r["case_id"])
        self.keys = sorted(strata)
        self.members = {k: sorted(strata[k]) for k in self.keys}
        sizes = [len(self.members[k]) for k in self.keys]
        self.weights = [s ** alpha for s in sizes]
        self.alpha, self.seed = alpha, seed
        self.reset()

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self.order = {k: [] for k in self.keys}
        self.draws = 0

    def _next_in(self, key) -> str:
        if not self.order[key]:
            cycle = list(self.members[key])
            self.rng.shuffle(cycle)
            self.order[key] = cycle
        return self.order[key].pop()

    def draw(self, n: int, avoid: set[str] | None = None) -> list[str]:
        out: list[str] = []
        avoid = set(avoid or ())
        attempts = 0
        while len(out) < n:
            key = self.rng.choices(self.keys, weights=self.weights)[0]
            cid = self._next_in(key)
            self.draws += 1
            attempts += 1
            if (cid in out or cid in avoid) and attempts < 50 * n:
                continue
            out.append(cid)
        return out

    def state(self) -> dict:
        return {"seed": self.seed, "alpha": self.alpha, "draws": self.draws}

    def restore(self, state: dict) -> None:
        if state["seed"] != self.seed or state["alpha"] != self.alpha:
            raise ValueError("sampler settings changed across resume")
        self.reset()
        while self.draws < state["draws"]:
            key = self.rng.choices(self.keys, weights=self.weights)[0]
            self._next_in(key)
            self.draws += 1

    def describe(self) -> dict:
        total = sum(self.weights)
        return {f"{s}/{c}": {"cases": len(self.members[(s, c)]),
                             "share_of_draws": round(w / total, 4)}
                for (s, c), w in zip(self.keys, self.weights)}


class SubsetEpochSampler:

    def __init__(self, refs: list[dict], fraction: float, subset_seed: int, seed: int,
                 exclude: set[str] | None = None):
        exclude = exclude or set()
        strata = collections.defaultdict(list)
        for r in refs:
            if r["case_id"] not in exclude:
                strata[(r["source"], r["closure"])].append(r["case_id"])
        rng = random.Random(subset_seed)
        self.subset = {}
        for key in sorted(strata):
            members = sorted(strata[key])
            k = max(1, round(fraction * len(members)))
            self.subset[key] = sorted(rng.sample(members, k))
        self.fraction, self.subset_seed, self.seed = fraction, subset_seed, seed
        self.alpha = None
        self.reset()

    def case_ids(self) -> list[str]:
        return sorted(c for v in self.subset.values() for c in v)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)
        self.stream: list[str] = []
        self.draws = 0
        self.epoch = 0

    def _new_epoch(self) -> None:
        keyed = []
        for key in sorted(self.subset):
            members = list(self.subset[key])
            self.rng.shuffle(members)
            for i, cid in enumerate(members):
                keyed.append(((i + self.rng.random()) / len(members), cid))
        keyed.sort()
        self.stream = [cid for _, cid in keyed]
        self.epoch += 1

    def _next(self) -> str:
        if not self.stream:
            self._new_epoch()
        self.draws += 1
        return self.stream.pop(0)

    def draw(self, n: int, avoid: set[str] | None = None) -> list[str]:
        avoid = set(avoid or ())
        out, skipped = [], []
        while len(out) < n:
            cid = self._next()
            if cid in avoid or cid in out:
                skipped.append(cid)
                if len(skipped) > 4 * len(self.case_ids()):
                    break
                continue
            out.append(cid)
        return out

    def state(self) -> dict:
        return {"mode": "subset_epoch", "seed": self.seed, "subset_seed": self.subset_seed,
                "fraction": self.fraction, "draws": self.draws, "epoch": self.epoch}

    def restore(self, state: dict) -> None:
        if (state.get("mode"), state["seed"], state.get("subset_seed"), state.get("fraction")) != \
                ("subset_epoch", self.seed, self.subset_seed, self.fraction):
            raise ValueError("sampler settings changed across resume")
        self.reset()
        while self.draws < state["draws"]:
            self._next()

    def describe(self) -> dict:
        return {f"{s}/{c}": {"subset_cases": len(v)} for (s, c), v in sorted(self.subset.items())}


def make_sampler(refs: list[dict], spec: dict, exclude: set[str]):
    if spec.get("mode", "stratified") == "subset_epoch":
        return SubsetEpochSampler(refs, float(spec["subset_fraction"]), int(spec.get("subset_seed", 20260925)),
                                  int(spec.get("seed", 20260925)), exclude)
    return StratifiedSampler(refs, float(spec.get("alpha", 0.0)), int(spec.get("seed", 20260925)), exclude)


def load_extra_sources(spec: dict) -> list[dict]:
    cf = (spec or {}).get("counterfactual_pairs", {})
    if cf.get("enabled"):
        raise NotImplementedError("counterfactual training pairs are not built yet; "
                                  "set sampler.extra_sources.counterfactual_pairs.enabled=false")
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--output", type=Path, default=DEFAULT_OUT)
    v = sub.add_parser("build-val-refs")
    v.add_argument("--output", type=Path, default=RUN / "results/rlvr_v1/val_refs_v1")
    args = ap.parse_args()
    if args.cmd == "build-val-refs":
        print(json.dumps(build_val_refs(args.output), indent=1))
        return
    if args.cmd == "build":
        manifest = build(args.output)
        print(json.dumps({k: manifest[k] for k in ("cases", "strata", "key_evidence",
                                                   "teacher_marker_vs_label_disagree")}, indent=1))
        print(json.dumps(manifest["problems"][:20], indent=1))


if __name__ == "__main__":
    main()
