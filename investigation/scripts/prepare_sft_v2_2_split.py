#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

ROOT = paths.DATA
RUN = paths.RUN
SOURCE_DIR = RUN / "results/sft_content_passed_nonmedical_v2_2"
SOURCE = SOURCE_DIR / "content_passed_model_visible.jsonl"
OUTPUT = SOURCE_DIR / "split_v1"
SEED = 20260925
SPLITS = ("train", "val", "test")

TARGETS = {
    ("host", "closed"): (194, 26, 39),
    ("host", "not_closed"): (45, 6, 9),
    ("CSB", "closed"): (17, 2, 4),
    ("CSB", "not_closed"): (7, 1, 1),
    ("ATSB", "closed"): (15, 2, 4),
    ("ATSB", "not_closed"): (4, 1, 1),
    ("MAIB", "closed"): (27, 4, 5),
    ("MAIB", "not_closed"): (9, 1, 2),
    ("RAIB", "closed"): (22, 3, 5),
    ("RAIB", "not_closed"): (1, 1, 1),
    ("nhtsa", "closed"): (4, 1, 2),
    ("nhtsa", "not_closed"): (93, 12, 18),
    ("ntsb", "closed"): (14, 2, 3),
    ("ntsb", "not_closed"): (93, 12, 18),
}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def conclusion(row: dict) -> str:
    final = next(message.get("content", "") for message in reversed(row["messages"])
                 if message["role"] == "assistant").lstrip()
    if final.startswith("CASE NOT CLOSED"):
        return "not_closed"
    if final.startswith("CASE CLOSED"):
        return "closed"
    raise ValueError(f"missing final closure marker: {row['case_id']}")


def metadata(rows: list[dict], origins: dict, worklist: dict) -> dict[str, dict]:
    result = {}
    for row in rows:
        cid = row["case_id"]
        item = worklist.get(cid)
        if row["source"] != "host" and item is None:
            raise ValueError(f"missing source worklist for {cid}")
        category = item["dataset"] if row["source"] == "boards" else row["source"]
        if category == "host":
            native = cid
        else:
            reference = ROOT / item["review_only_reference_path"]
            ref = json.loads(reference.read_text())
            native = ref["source"]["native_id"]
            if row["source"] == "boards" and ref["source"]["dataset"] != category:
                raise ValueError(f"board subtype disagreement: {cid}")
        if not native:
            raise ValueError(f"missing native grouping key: {cid}")
        group_raw = f"{category}:{native}"
        result[cid] = {"case_id": cid, "category": category,
                       "closure": conclusion(row), "component": origins[cid],
                       "fetch_rounds": sum(message["role"] == "tool" for message in row["messages"]),
                       "group_id": category + ":" + hashlib.sha256(group_raw.encode()).hexdigest()[:20]}
    if len(result) != len(rows):
        raise ValueError("duplicate case IDs")
    return result


def prior_development_ids(all_ids: set[str]) -> tuple[set[str], dict[str, list[str]]]:
    pilot = {row["case_id"] for row in read_jsonl(
        RUN / "results/sft_score_policy_v1/content_passed_model_visible_nonmedical.jsonl")}
    smoke = {row["case_id"] for row in json.loads(
        (paths.CONFIGS / "sft_scale60_v1.json").read_text())["items"]}
    eval8 = set(json.loads((paths.CONFIGS / "sft_full_lora_plan_v1.json").read_text())
                ["eval_case_ids"])
    by_reason = {"pilot21": sorted(pilot & all_ids),
                 "scale60_used": sorted(smoke & all_ids),
                 "prior_evaluation8": sorted(eval8 & all_ids)}
    return (pilot | smoke | eval8) & all_ids, by_reason


def counts_for(ids: list[str], meta: dict[str, dict]) -> collections.Counter:
    return collections.Counter((meta[cid]["category"], meta[cid]["closure"]) for cid in ids)


def candidate(meta: dict[str, dict], groups: dict[str, list[str]],
              forced_groups: set[str], trial: int) -> dict[str, str] | None:
    rng = random.Random(SEED + trial)
    assignments = {group: "train" for group in forced_groups}
    counts = {split: counts_for([cid for group in forced_groups for cid in groups[group]], meta)
              if split == "train" else collections.Counter() for split in SPLITS}
    multi_host = [group for group, ids in groups.items()
                  if group not in forced_groups and len(ids) > 1]
    rng.shuffle(multi_host)
    for group in multi_host:
        stratum = counts_for(groups[group], meta)
        choices = rng.choices(SPLITS, weights=(75, 10, 15), k=1)[0]
        if choices != "train" and any(
            counts[choices][cell] + number > TARGETS[cell][SPLITS.index(choices)]
            for cell, number in stratum.items()
        ):
            choices = "train"
        assignments[group] = choices
        counts[choices].update(stratum)
    for cell, targets in TARGETS.items():
        singles = [group for group, ids in groups.items()
                   if group not in assignments and len(ids) == 1 and
                   (meta[ids[0]]["category"], meta[ids[0]]["closure"]) == cell]
        rng.shuffle(singles)
        needed = {split: targets[index] - counts[split][cell]
                  for index, split in enumerate(SPLITS)}
        if any(number < 0 for number in needed.values()) or sum(needed.values()) != len(singles):
            return None
        cursor = 0
        for split in ("val", "test", "train"):
            for group in singles[cursor:cursor + needed[split]]:
                assignments[group] = split
            cursor += needed[split]
        if cursor != len(singles):
            return None
    if len(assignments) != len(groups):
        return None
    return assignments


def score(assignments: dict[str, str], groups: dict[str, list[str]], meta: dict[str, dict]) -> float:
    selected = {split: [] for split in SPLITS}
    for group, split in assignments.items():
        selected[split].extend(groups[group])
    penalty = 0.0
    for category in sorted({item["category"] for item in meta.values()}):
        source_ids = [cid for cid, item in meta.items() if item["category"] == category]
        for key in ("component", "fetch_rounds"):
            all_counts = collections.Counter(meta[cid][key] for cid in source_ids)
            for split in ("val", "test"):
                subset = [cid for cid in selected[split] if meta[cid]["category"] == category]
                observed = collections.Counter(meta[cid][key] for cid in subset)
                for value, overall in all_counts.items():
                    expected = overall * len(subset) / len(source_ids)
                    penalty += (observed[value] - expected) ** 2 / (expected + 1.0)
    multi = {split: sum(len(groups[group]) for group, chosen in assignments.items()
                        if chosen == split and len(groups[group]) > 1) for split in SPLITS}
    penalty += (multi["val"] - 10) ** 2 / 10 + (multi["test"] - 15) ** 2 / 15
    return penalty


def main() -> None:
    upstream = json.loads((SOURCE_DIR / "manifest.json").read_text())
    if sha256(SOURCE) != upstream["export_sha256"]:
        raise ValueError("v2.2 source hash differs from frozen manifest")
    rows = read_jsonl(SOURCE)
    if len(rows) != 731:
        raise ValueError("expected exactly 731 v2.2 cases")
    origins = {row["case_id"]: row["component"] for row in read_jsonl(SOURCE_DIR / "origins.jsonl")}
    worklist = {row["case_id"]: row for row in read_jsonl(
        RUN / "results/multiturn_worklist_v2/cases.jsonl")}
    meta = metadata(rows, origins, worklist)
    overall = counts_for(list(meta), meta)
    if any(overall[cell] != sum(target) for cell, target in TARGETS.items()):
        raise ValueError("board subtype or closure totals changed")
    groups = collections.defaultdict(list)
    for cid, item in meta.items():
        groups[item["group_id"]].append(cid)
    prior, prior_reasons = prior_development_ids(set(meta))
    forced_groups = {meta[cid]["group_id"] for cid in prior}
    best = None
    for trial in range(3000):
        proposal = candidate(meta, groups, forced_groups, trial)
        if proposal is None:
            continue
        value = score(proposal, groups, meta)
        if best is None or value < best[0]:
            best = (value, trial, proposal)
    if best is None:
        raise RuntimeError("could not satisfy grouped stratification targets")
    objective, trial, assignments = best
    by_split = {split: sorted(cid for group, ids in groups.items()
                              if assignments[group] == split for cid in ids)
                for split in SPLITS}
    if set(by_split["train"]) & set(by_split["val"]) or \
       set(by_split["train"]) & set(by_split["test"]) or \
       set(by_split["val"]) & set(by_split["test"]):
        raise AssertionError("split overlap")
    if set().union(*(set(ids) for ids in by_split.values())) != set(meta):
        raise AssertionError("split coverage incomplete")
    if not prior <= set(by_split["train"]):
        raise AssertionError("earlier development cases escaped training split")
    for index, split in enumerate(SPLITS):
        observed = counts_for(by_split[split], meta)
        if any(observed[cell] != targets[index] for cell, targets in TARGETS.items()):
            raise AssertionError(f"stratum quota failed: {split}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_lines = {json.loads(line)["case_id"]: line for line in SOURCE.read_text().splitlines(True)
                 if line.strip()}
    hashes = {}
    for split in SPLITS:
        path = OUTPUT / f"{split}_model_visible.jsonl"
        path.write_text("".join(raw_lines[cid] for cid in by_split[split]))
        hashes[split] = sha256(path)
        (OUTPUT / f"{split}_ids.txt").write_text("\n".join(by_split[split]) + "\n")
    audit = {"schema_version": "nautil.sft.v2_2.split.v1",
             "status": "pending_qwen_wire_validation",
             "source_sha256": upstream["export_sha256"],
             "seed": SEED, "chosen_trial": trial, "balance_penalty": round(objective, 4),
             "cases": {split: len(ids) for split, ids in by_split.items()},
             "hashes": hashes,
             "prior_development_cases_forced_to_train": prior_reasons,
             "groups": len(groups),
             "multi_case_host_groups": 0,
             "group_leakage": 0,
             "by_category_closure": {
                 category: {closure: {split: counts_for(by_split[split], meta)[(category, closure)]
                                      for split in SPLITS}
                            for closure in ("closed", "not_closed")}
                 for category in ("host", "CSB", "ATSB", "MAIB", "RAIB", "nhtsa", "ntsb")},
             "by_category_component": {
                 category: {split: dict(collections.Counter(
                     meta[cid]["component"] for cid in by_split[split]
                     if meta[cid]["category"] == category)) for split in SPLITS}
                 for category in ("host", "CSB", "ATSB", "MAIB", "RAIB", "nhtsa", "ntsb")},
             "known_risks": [
                 "All 731 task questions still use the generic wording what caused this?",
                 "81 remaining MAIB/RAIB cases have not received per-case human identity review",
                 "Rare closed NHTSA/NTSB and not-closed RAIB/ATSB test cells are too small for stable standalone rates",
             ]}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (OUTPUT / "per_case_metadata.jsonl").write_text("".join(
        json.dumps({**meta[cid], "split": next(split for split in SPLITS if cid in by_split[split])},
                   ensure_ascii=False) + "\n" for cid in sorted(meta)))
    (OUTPUT / "test_maib_raib_review_queue.txt").write_text("\n".join(
        cid for cid in by_split["test"] if meta[cid]["category"] in {"MAIB", "RAIB"}) + "\n")
    lines = ["# SFT v2.2 grouped split", "",
             "Source: 731 nonmedical cases, SHA-256 `" + upstream["export_sha256"] + "`.",
             "The split is event/report disjoint; host cases are split by case ID, as requested.",
             "Prior pilot, smoke and eight-case evaluation examples are train-only.",
             "The Qwen wire/loss mask must be validated before training. A separate case-identity",
             "audit remains desirable before treating the test set as a locked paper benchmark.", "",
             "| Category | Train | Validation | Test | Closed / not closed in test |",
             "|---|---:|---:|---:|---:|"]
    for category in ("host", "CSB", "ATSB", "MAIB", "RAIB", "nhtsa", "ntsb"):
        cell = audit["by_category_closure"][category]
        totals = [sum(cell[label][split] for label in cell) for split in SPLITS]
        lines.append(f"| {category} | {totals[0]} | {totals[1]} | {totals[2]} | "
                     f"{cell['closed']['test']} / {cell['not_closed']['test']} |")
    lines += ["", f"Totals: {len(by_split['train'])} train, {len(by_split['val'])} validation, "
              f"{len(by_split['test'])} test. Balance search trial {trial}, score {objective:.3f}.", "",
              "Test MAIB/RAIB review queue: [test_maib_raib_review_queue.txt](test_maib_raib_review_queue.txt).",
              "Each `*_model_visible.jsonl` contains exactly the original v2.2 rows for that split;",
              "closure labels and group hashes are kept separately in `per_case_metadata.jsonl`."]
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cases": audit["cases"], "categories": audit["by_category_closure"],
                      "groups": len(groups), "trial": trial, "score": round(objective, 3)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
