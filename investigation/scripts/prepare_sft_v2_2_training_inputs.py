#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids(path: Path) -> set[str]:
    values = path.read_text().splitlines()
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate ID in {path}")
    return set(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--wire-cases", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    split = {part: ids(args.split_dir / f"{part}_ids.txt")
             for part in ("train", "val", "test")}
    if list(map(len, split.values())) != [545, 74, 112]:
        raise ValueError("unexpected v2.2 split sizes")
    if any(split[a] & split[b] for a, b in (("train", "val"), ("train", "test"),
                                           ("val", "test"))):
        raise ValueError("split overlap")
    wire = {}
    for line in args.wire_cases.open():
        row = json.loads(line)
        if row["case_id"] in wire or row["errors"] or not row["official_token_ids_match"]:
            raise ValueError(f"invalid wire case {row['case_id']}")
        wire[row["case_id"]] = row
    if len(wire) != 731 or set(wire) != set.union(*split.values()):
        raise ValueError("wire cases and split do not match")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    trainval_path = args.output_dir / "trainval_tokenized.jsonl"
    test_path = args.output_dir / "test_tokenized.jsonl"
    seen = set()
    with args.tokenized.open() as source, trainval_path.open("w") as trainval, \
            test_path.open("w") as test:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row["case_id"]
            if cid in seen or cid not in wire:
                raise ValueError(f"duplicate or unexpected tokenized case {cid}")
            seen.add(cid)
            length = len(row["input_ids"])
            targets = sum(label != -100 for label in row["labels"])
            if length != wire[cid]["tokens"] or targets != wire[cid]["assistant_target_tokens"]:
                raise ValueError(f"token counts changed for {cid}")
            if length > 32768 or len(row["labels"]) != length or len(row["attention_mask"]) != length:
                raise ValueError(f"context overflow or malformed row {cid}")
            (test if cid in split["test"] else trainval).write(line)
    if seen != set(wire):
        raise ValueError("tokenized file missing cases")
    train = sorted(split["train"], key=lambda cid: (wire[cid]["tokens"], cid))
    singleton = train.pop(0)
    pairs = [[train[index], train[index + 1]] for index in range(0, len(train), 2)]
    random.Random(20260925).shuffle(pairs)
    pairs.append([singleton, None])
    plan = {
        "schema_version": "nautil.sft.v2_2.lora_plan.v1",
        "source_export_sha256": json.loads((args.split_dir / "audit.json").read_text())["source_sha256"],
        "split_audit_sha256": sha256(args.split_dir / "audit.json"),
        "all_tokenized_sha256": sha256(args.tokenized),
        "data_sha256": sha256(trainval_path),
        "test_tokenized_sha256": sha256(test_path),
        "eval_case_ids": sorted(split["val"], key=lambda cid: (wire[cid]["tokens"], cid)),
        "test_case_ids_sha256": sha256(args.split_dir / "test_ids.txt"),
        "train_pairs": pairs,
        "shadow_case_id": singleton,
        "case_token_lengths": {cid: wire[cid]["tokens"] for cid in split["train"] | split["val"]},
        "training_case_count": 545,
        "evaluation_case_count": 74,
        "test_case_count": 112,
        "optimizer_steps": len(pairs),
        "epochs": 1,
        "max_sequence_length": 32768,
        "lora_rank": 16,
        "lora_alpha": 32,
        "base_learning_rate": 5e-5,
        "warmup_steps": 8,
        "eval_every_steps": 100000,
        "max_grad_norm": 1.0,
        "seed": 20260925,
        "pairing": "adjacent lengths, shuffled pairs; one zero-weight shadow forward",
        "evaluation_schedule": "step 0 and 25%, 50%, 100%; test untouched",
    }
    plan_path = args.output_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    audit = {"train": 545, "val": 74, "test": 112, "optimizer_steps": len(pairs),
             "max_tokens": max(row["tokens"] for row in wire.values()),
             "source_sha256": plan["source_export_sha256"],
             "trainval_tokenized_sha256": plan["data_sha256"],
             "test_tokenized_sha256": plan["test_tokenized_sha256"],
             "plan_sha256": sha256(plan_path),
             "test_loaded_by_trainer": False}
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
