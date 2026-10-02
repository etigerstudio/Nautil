#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

RUN = paths.RUN
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_sft_format_v1 import validate as validate_sft

CLAIM = re.compile(r"^(H[1-9][0-9]*) — (.+)$")
STATE = re.compile(r"^(H[1-9][0-9]*) \[(open|favored|weakened) → (open|favored|weakened)\] — (.+)$")
EID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")


def section(content: str, heading: str, following: str | None) -> str:
    marker = f"## {heading}\n"
    if marker not in content:
        raise ValueError(f"missing {heading} heading")
    value = content.split(marker, 1)[1]
    if following:
        next_marker = f"\n## {following}"
        if next_marker not in value:
            raise ValueError(f"missing {following} heading")
        value = value.split(next_marker, 1)[0]
    return value.strip()


def validate(row: dict) -> dict:
    lossless = row.get("schema_version") == "nautil.sft.v1.1-lossless-draft"
    shadow = copy.deepcopy(row) if lossless else row
    if lossless:
        shadow["schema_version"] = "nautil.sft.v1"
    base = validate_sft(shadow)
    stable_claims = None
    previous = None
    updates = 0
    for turn, pos in enumerate(range(1, len(row["messages"]) - 1, 2), 1):
        assistant = row["messages"][pos]
        content = assistant["content"]
        claims_text = section(content, "Current hypotheses", "Evidence and hypothesis update")
        claims = dict(CLAIM.fullmatch(line).groups() for line in claims_text.splitlines() if line.strip() and CLAIM.fullmatch(line))
        if len(claims) < 2 or len(claims) != len([line for line in claims_text.splitlines() if line.strip()]):
            raise ValueError(f"turn {turn}: malformed stable hypotheses")
        if stable_claims is None:
            stable_claims = claims
            previous = {hid: "open" for hid in claims}
        elif claims != stable_claims:
            raise ValueError(f"turn {turn}: hypothesis claims changed")
        ledger = section(content, "Hypothesis ledger", None if lossless else "Next discriminating check")
        lines = [line for line in ledger.splitlines() if line.strip()]
        if len(lines) != len(stable_claims):
            raise ValueError(f"turn {turn}: missing ledger line")
        seen = set()
        for line in lines:
            match = STATE.fullmatch(line)
            if not match:
                raise ValueError(f"turn {turn}: malformed ledger line")
            hid, before, after, basis = match.groups()
            if hid not in stable_claims or hid in seen or before != previous[hid]:
                raise ValueError(f"turn {turn}: hypothesis ID or previous status mismatch")
            if before != after:
                if not EID.search(basis):
                    raise ValueError(f"turn {turn} {hid}: state changed without evidence citation")
                updates += 1
            previous[hid] = after
            seen.add(hid)
        reason = assistant["tool_calls"][0]["function"]["arguments"]["reason"]
        if lossless:
            if "## Next discriminating check" in content:
                raise ValueError(f"turn {turn}: duplicate next-check paragraph remains")
        else:
            question = section(content, "Next discriminating check", None)
            if question != reason:
                raise ValueError(f"turn {turn}: next check differs from native tool reason")
    return {**base, "hypotheses": len(stable_claims or {}), "explicit_state_updates": updates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, nargs="+")
    args = ap.parse_args()
    for path in args.jsonl:
        for line in path.read_text().splitlines():
            if line.strip():
                print(json.dumps(validate(json.loads(line)), ensure_ascii=False))


if __name__ == "__main__":
    main()
