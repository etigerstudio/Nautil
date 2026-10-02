"""Evidence-package contract: content digest and strict shape check.

A package is one case as the investigator sees it: the question, the initial
context and a list of evidence items. `package_digest` fingerprints a package so
that any later edit is detectable; `validate_case` rejects packages whose shape,
identifiers or fingerprint are wrong. Neither function reads reference answers.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

CASE_FIELDS = frozenset({
    "case_id", "domain", "task_question", "initial_context", "evidence_items",
    "version", "package_hash",
})
EVIDENCE_FIELDS = frozenset({
    "evidence_id", "neutral_title", "text", "source_id", "locator",
})
# Shuffled packages carry the evidence type as `kind` instead of a locator.
EVIDENCE_FIELDS_KIND = frozenset({
    "evidence_id", "neutral_title", "text", "source_id", "kind",
})
# Optional fields, allowed but not required:
#   sources  {source_id: description of that source}
#   family   metric family of a host metric item, e.g. "CPU"
#   meaning  one line explaining a metric whose name does not say what it counts
OPTIONAL_CASE_FIELDS = frozenset({"sources"})
OPTIONAL_EVIDENCE_FIELDS = frozenset({"family", "meaning"})


class CaseValidationError(ValueError):
    """The package is malformed or does not match its content digest."""


def package_digest(case: dict[str, Any]) -> str:
    """SHA256 of all fields except package_hash, as canonical UTF-8 JSON."""
    if not isinstance(case, dict):
        raise CaseValidationError("case must be a JSON object")
    payload = {k: v for k, v in case.items() if k != "package_hash"}
    try:
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaseValidationError("case cannot be canonically serialized") from exc
    return hashlib.sha256(encoded).hexdigest()


def _shape(
    value: Any, fields: frozenset[str], location: str,
    optional: frozenset[str] = frozenset(),
) -> None:
    """Every required field present; nothing beyond required plus optional."""
    if not isinstance(value, dict):
        raise CaseValidationError(f"{location} must be a JSON object")
    present = set(value)
    missing = sorted(fields - present)
    extra = sorted(present - fields - optional)
    if missing or extra:
        raise CaseValidationError(
            f"{location} fields mismatch: missing={missing}; extra={extra}"
        )


def _text(value: Any, location: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CaseValidationError(f"{location} must be a nonempty string")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
        raise CaseValidationError(f"{location} contains unsupported control characters")


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Validate strictly; unknown fields, duplicate IDs and stale hashes fail."""
    _shape(case, CASE_FIELDS, "case", OPTIONAL_CASE_FIELDS)
    for field in CASE_FIELDS - {"evidence_items"}:
        _text(case[field], field)
    if "sources" in case:
        sources = case["sources"]
        if not isinstance(sources, dict) or not sources:
            raise CaseValidationError("case.sources must be a nonempty JSON object")
        for source_id, description in sources.items():
            if not re.fullmatch(r"S[0-9]+", str(source_id)):
                raise CaseValidationError("case.sources keys must be S followed by digits")
            _text(description, f"case.sources[{source_id}]")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", case["case_id"]):
        raise CaseValidationError("case_id must be an opaque alphanumeric identifier")
    items = case["evidence_items"]
    if not isinstance(items, list) or not items:
        raise CaseValidationError("evidence_items must be a nonempty list")
    seen: set[str] = set()
    for index, item in enumerate(items):
        location = f"evidence_items[{index}]"
        fields = EVIDENCE_FIELDS_KIND if isinstance(item, dict) and "kind" in item else EVIDENCE_FIELDS
        _shape(item, fields, location, OPTIONAL_EVIDENCE_FIELDS)
        for field in fields:
            _text(item[field], f"{location}.{field}")
        for field in OPTIONAL_EVIDENCE_FIELDS & set(item):
            _text(item[field], f"{location}.{field}")
        # Every item must cite a declared source when the package declares them.
        if "sources" in case and item["source_id"] not in case["sources"]:
            raise CaseValidationError(
                f"{location}.source_id {item['source_id']} is not declared in case.sources"
            )
        # Flat (E12) and hierarchical (E1.3, E1.3.2) evidence IDs are accepted.
        if not re.fullmatch(r"E[0-9]+(?:\.[0-9]+)*", item["evidence_id"]):
            raise CaseValidationError(
                f"{location}.evidence_id must be E followed by digits, optionally dot-separated (E1, E1.3)"
            )
        if not re.fullmatch(r"S[0-9]+", item["source_id"]):
            raise CaseValidationError(f"{location}.source_id must be S followed by digits")
        if item["evidence_id"] in seen:
            raise CaseValidationError(f"duplicate evidence_id: {item['evidence_id']}")
        seen.add(item["evidence_id"])
    if not re.fullmatch(r"[0-9a-f]{64}", case["package_hash"]):
        raise CaseValidationError("package_hash must be a lowercase SHA256 hex digest")
    if case["package_hash"] != package_digest(case):
        raise CaseValidationError("package_hash does not match the evidence package")
    return case
