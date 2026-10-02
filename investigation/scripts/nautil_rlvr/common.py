from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

EVIDENCE_ID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")
INDEX_SPLIT = "## Index of the remaining"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_ids(ids) -> str:
    return hashlib.sha256((",".join(map(str, ids))).encode()).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                               for r in rows))


def atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial")
    pending.write_text(text)
    os.replace(pending, path)


def atomic_json(path: Path, obj) -> None:
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


FORBIDDEN_PREFIXES = ("test_", "counterfactual_test_")


def looks_like_test_data(path) -> bool:
    return any(part.startswith(FORBIDDEN_PREFIXES) for part in Path(str(path)).parts)


def refuse_test_path(path) -> Path:
    if looks_like_test_data(path):
        raise PermissionError(f"refusing to touch test-set data: {path}")
    return Path(path)


def initial_record_ids(user_message: str) -> set[str]:
    return set(EVIDENCE_ID.findall(user_message.split(INDEX_SPLIT, 1)[0]))
