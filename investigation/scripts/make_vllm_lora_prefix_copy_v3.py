#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC_PREFIX = "base_model.model.model.layers."
DST_PREFIX = "base_model.model.model.language_model.layers."


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refuse to overwrite {args.output}")
    src_file = args.source / "adapter_model.safetensors"
    tensors = {}
    with safe_open(str(src_file), "pt") as stream:
        metadata = stream.metadata()
        for key in stream.keys():
            if not key.startswith(SRC_PREFIX):
                raise ValueError(f"unexpected adapter key {key}")
            tensors[DST_PREFIX + key[len(SRC_PREFIX):]] = stream.get_tensor(key)
    args.output.mkdir(parents=True)
    out_file = args.output / "adapter_model.safetensors"
    save_file(tensors, str(out_file), metadata=metadata)
    shutil.copyfile(args.source / "adapter_config.json", args.output / "adapter_config.json")
    mismatched = []
    with safe_open(str(src_file), "pt") as old, safe_open(str(out_file), "pt") as new:
        for key in old.keys():
            a = old.get_tensor(key)
            b = new.get_tensor(DST_PREFIX + key[len(SRC_PREFIX):])
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                mismatched.append(key)
    manifest = {"schema_version": "nautil.vllm.lora_prefix_copy.v3",
                "source_dir": str(args.source),
                "source_safetensors_sha256": sha256(src_file),
                "source_config_sha256": sha256(args.source / "adapter_config.json"),
                "output_safetensors_sha256": sha256(out_file),
                "output_config_sha256": sha256(args.output / "adapter_config.json"),
                "rename": {SRC_PREFIX: DST_PREFIX},
                "tensors": len(tensors), "bit_exact_mismatches": mismatched}
    (args.output / "prefix_copy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))
    if mismatched:
        raise ValueError("tensor copy is not bit-exact")


if __name__ == "__main__":
    main()
