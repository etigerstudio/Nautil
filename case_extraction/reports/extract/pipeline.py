from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import verify
from .adapters.base import Block, RawCase, squash
from .client import Client

PROMPTS = Path(__file__).resolve().parent / "prompts"
_TEMPLATES: dict[str, str] = {}


def template(name: str) -> str:
    if name not in _TEMPLATES:
        _TEMPLATES[name] = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    return _TEMPLATES[name]


def fill(name: str, **values: str) -> str:
    text = template(name)
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


class Gate:

    def __init__(self, limit: int):
        self.sem = threading.Semaphore(limit)

    def call(self, client: Client, **kw: Any) -> dict[str, Any]:
        with self.sem:
            return client.complete(**kw)


def step_label(raw: RawCase, client: Client, gate: Gate) -> dict[str, Any]:
    listing = []
    for index, block in enumerate(raw.blocks):
        head = squash(block.text)[:280]
        hint = f" | this source always treats it as {block.role}" if block.role in ("factual", "analysis") else ""
        listing.append(f"[{index}] heading: {block.label!r} | {len(block.text)} chars{hint}\n     starts: {head}")
    result = gate.call(client, role="extract",
                       prompt=fill("label_blocks", record_kind=raw.record_kind, blocks="\n".join(listing)),
                       max_tokens=8000)
    if not result.get("ok") or not isinstance(result.get("value"), dict):
        return {"ok": False, "error": result.get("error") or "reply was not a JSON object",
                "raw": str(result.get("raw", ""))[:500]}
    roles = {int(item["index"]): str(item["role"]) for item in (result["value"].get("blocks") or [])
             if isinstance(item, dict) and "index" in item and "role" in item}
    for index, block in enumerate(raw.blocks):
        if block.role in ("factual", "analysis", "conclusion"):
            roles[index] = block.role
    missing = [i for i in range(len(raw.blocks)) if i not in roles]
    for index, block in enumerate(raw.blocks):
        block.role = roles.get(index, "appendix")
    return {"ok": True, "roles": {str(k): v for k, v in roles.items()}, "unlabelled": missing}


MIXED_WARNING = (
    "This block states observed fact and the investigators' reasoning in the same passage. Take the facts and "
    "leave the reasoning. Any sentence that weighs one explanation against another, says why something happened, "
    "rules a possibility in or out, or states a cause, finding or diagnosis goes in `withheld_sentences`, never in "
    "`items`. A measurement, a test result, a witness account or a description of a component's condition is a "
    "fact even when it appears in the middle of an argument.\n\n")


def step_atomize(raw: RawCase, client: Client, gate: Gate, block_workers: int) -> dict[str, Any]:
    factual = [b for b in raw.blocks if b.role in ("factual", "mixed")]
    if not factual:
        return {"ok": False, "error": "no factual block after labelling"}
    source_hint = (f"For this record the origin is usually: {raw.blocks[0].source_hint}."
                   if raw.blocks[0].source_hint else "")

    def one(block: Block) -> dict[str, Any]:
        out = gate.call(client, role="extract",
                        prompt=fill("atomize", domain=raw.domain, block_label=block.label,
                                    block=block.text, source_hint=source_hint,
                                    granularity=raw.extra.get("granularity", ""),
                                    mixed_warning=MIXED_WARNING if block.role == "mixed" else ""),
                        max_tokens=20000)
        if not out.get("ok"):
            return {"label": block.label, "ok": False, "error": out.get("error"), "items": [], "withheld": []}
        value = out["value"]
        items = [i for i in (value.get("items") or []) if isinstance(i, dict) and squash(i.get("text", ""))]
        for item in items:
            item["quote_verified"] = verify.contains(block.text, str(item.get("quote", "")))
            item["overlap"] = round(verify.overlap(str(item["text"]), block.text), 3)
        return {"label": block.label, "ok": True, "items": items,
                "withheld": [w for w in (value.get("withheld_sentences") or []) if isinstance(w, dict)],
                "coverage": round(verify.coverage([str(i["text"]) for i in items], block.text), 3),
                "block_chars": len(block.text)}

    with ThreadPoolExecutor(max_workers=max(1, block_workers)) as pool:
        results = list(pool.map(one, factual))

    items: list[dict[str, Any]] = []
    withheld: list[dict[str, Any]] = []
    for group_index, result in enumerate(results, 1):
        for item_index, item in enumerate(result["items"], 1):
            items.append({"evidence_id": f"E{group_index}.{item_index}", "group_label": result["label"],
                          "neutral_title": squash(item.get("neutral_title", "")) or "Record item",
                          "text": squash(item["text"]), "kind": squash(item.get("kind", "")).lower(),
                          "source_proposed": squash(item.get("source", "")) or "investigation record",
                          "quote": squash(item.get("quote", "")), "quote_verified": item["quote_verified"],
                          "overlap": item["overlap"]})
        withheld.extend(result["withheld"])
    duplicates = verify.near_duplicates([i["text"] for i in items])
    dropped = [items[d]["evidence_id"] for d in duplicates]
    if duplicates:
        keep = set(range(len(items))) - set(duplicates)
        items = [items[i] for i in sorted(keep)]
    failed = [r["label"] for r in results if not r["ok"]]
    return {"ok": bool(items), "items": items, "withheld_sentences": withheld,
            "near_duplicates_dropped": dropped,
            "blocks_failed": failed,
            "coverage_by_block": {r["label"]: r.get("coverage") for r in results if r["ok"]},
            "error": f"blocks failed: {failed}" if failed and not items else None}


KINDS = {"observation", "measurement", "statement", "test result", "examination conclusion"}


def step_sources(items: list[dict[str, Any]], client: Client, gate: Gate) -> dict[str, Any]:
    from collections import Counter
    counts = Counter(i["source_proposed"] for i in items)
    listing = "\n".join(f"- {name}  ({n} items)" for name, n in counts.most_common())
    result = gate.call(client, role="extract", prompt=fill("sources", proposed=listing), max_tokens=6000)
    if not result.get("ok") or not isinstance(result.get("value"), dict):
        return {"ok": False, "error": result.get("error") or "reply was not a JSON object"}
    value = result["value"]
    table = [{"id": squash(s.get("id", "")), "description": squash(s.get("description", ""))}
             for s in (value.get("sources") or []) if isinstance(s, dict)]
    mapping = {squash(k): squash(v) for k, v in (value.get("map") or {}).items()}
    ids = {s["id"] for s in table}
    fallback = table[0]["id"] if table else "S1"
    unmapped = []
    for item in items:
        target = mapping.get(item["source_proposed"])
        if target not in ids:
            unmapped.append(item["source_proposed"])
            target = fallback
        item["source_id"] = target
    used = {i["source_id"] for i in items}
    table = [s for s in table if s["id"] in used]
    renumber = {s["id"]: f"S{n}" for n, s in enumerate(sorted(table, key=lambda s: min(
        idx for idx, i in enumerate(items) if i["source_id"] == s["id"])), 1)}
    for item in items:
        item["source_id"] = renumber[item["source_id"]]
    table = sorted(({"id": renumber[s["id"]], "description": s["description"]} for s in table),
                   key=lambda s: int(s["id"][1:]))
    return {"ok": bool(table), "sources": table, "unmapped": sorted(set(unmapped))}


def step_scenario(raw: RawCase, items: list[dict[str, Any]], sources: list[dict[str, str]],
                  client: Client, gate: Gate) -> dict[str, Any]:
    sample = "\n".join(f"- {i['neutral_title']}: {i['text'][:160]}" for i in items[:6])
    table = "\n".join(f"{s['id']} = {s['description']}" for s in sources)
    result = gate.call(client, role="extract",
                       prompt=fill("scenario", domain=raw.domain, record_kind=raw.record_kind,
                                   sample=sample, sources=table), max_tokens=3000)
    if not result.get("ok") or not isinstance(result.get("value"), dict):
        return {"ok": False, "error": result.get("error") or "reply was not a JSON object"}
    context = squash(str(result["value"].get("initial_context", "")))
    if "S1" not in context:
        context = context.rstrip(".") + ". Sources: " + "; ".join(
            f"{s['id']} = {s['description']}" for s in sources) + "."
    return {"ok": bool(context), "initial_context": context}


STRUCTURAL_ID = re.compile(r"^[SE]\d+(\.\d+)?$", re.I)


WHOLE_WORD = False


def _pattern(old: str) -> re.Pattern[str]:
    left = "(?<![A-Za-z0-9])" if old[:1].isalnum() else ""
    right = "(?![A-Za-z0-9])" if old[-1:].isalnum() else ""
    single = len(old.split()) == 1
    return re.compile(left + re.escape(old) + right, 0 if single else re.I)


def swap_pairs(pairs: list[tuple[str, str]], text: str) -> tuple[str, int] | str:
    if not WHOLE_WORD:
        for old, new in pairs:
            text = re.sub(re.escape(old), new.replace("\\", ""), text, flags=re.I)
        return text
    for old, new in pairs:
        text = _pattern(old).sub(new.replace("\\", ""), text)
    for old, new in pairs:
        if len(old) >= 5 and (" " in old.strip() or any(ch.isdigit() for ch in old)):
            text = re.sub(re.escape(old), new.replace("\\", ""), text, flags=re.I)
    return text


def _apply_pairs(pairs: list[tuple[str, str]], context: str, items: list[dict[str, Any]]) -> tuple[str, int]:
    applied = 0
    pairs = [(old, new) for old, new in pairs if not STRUCTURAL_ID.fullmatch(old.strip())]
    ordered = sorted(pairs, key=lambda p: -len(p[0]))

    def swap(text: str) -> str:
        nonlocal applied
        if WHOLE_WORD:
            before = text
            text = swap_pairs([p for p in ordered if p[0].strip()], text)
            applied += int(before != text)
            return text
        for old, new in ordered:
            if not old.strip():
                continue
            text, n = re.subn(re.escape(old), new.replace("\\", ""), text, flags=re.I)
            applied += n
        return text

    context = squash(swap(context))
    for item in items:
        item["text"] = squash(swap(item["text"]))
        item["neutral_title"] = squash(swap(item["neutral_title"]))
    return context, applied


def step_anonymize(raw: RawCase, items: list[dict[str, Any]], context: str,
                   client: Client, gate: Gate, *, passes: int = 2) -> dict[str, Any]:
    if not raw.anonymize:
        return {"ok": True, "replacements": [], "identity": dict(raw.identity_seed),
                "withheld_identifiers": [v for k, v in raw.identity_seed.items() if k in ("title", "pmcid") and v],
                "applied": 0, "residual": {}, "initial_context": context, "skipped": "medical: not anonymised"}
    all_pairs: list[tuple[str, str]] = []
    identity: dict[str, str] = {}
    applied_total = 0
    second_pass = ""
    sweeps: list[list[str]] = []
    for attempt in range(passes):
        payload = context + "\n\n" + "\n".join(
            f"{i['evidence_id']} | {i['neutral_title']} | {i['text']}" for i in items)
        result = gate.call(client, role="extract",
                           prompt=fill("anonymize", text=payload, keep_extra=raw.keep_extra,
                                       second_pass=second_pass), max_tokens=12000)
        if not result.get("ok") or not isinstance(result.get("value"), dict):
            if attempt == 0:
                return {"ok": False, "error": result.get("error") or "reply was not a JSON object"}
            break
        value = result["value"]
        pairs = [(str(a), str(b)) for a, b in (value.get("replacements") or [])
                 if isinstance(a, str) and isinstance(b, str) and a.strip()]
        for key, val in (value.get("identity") or {}).items():
            if squash(str(val)) and not identity.get(key):
                identity[key] = squash(str(val))
        context, applied = _apply_pairs(pairs, context, items)
        all_pairs.extend(pairs)
        applied_total += applied
        package_text = context + " " + " ".join(f"{i['neutral_title']} {i['text']}" for i in items)
        survivors = [(n, c) for n, c in verify.proper_nouns(package_text, min_count=1)
                     if not STRUCTURAL_ID.fullmatch(n.strip())]
        sweeps.append([f"{n} ({c})" for n, c in survivors[:40]])
        if attempt + 1 >= passes or not survivors:
            break
        listed = "\n".join(f'- "{name}" (appears {count} times)' for name, count in survivors[:40])
        second_pass = (
            "\nA first pass already ran. These capitalised strings are STILL in the text below. "
            "For each one decide: is it a series-produced make, model, component or standard term (leave it), "
            "or does it identify this particular case, machine, place, person or body (replace it)? "
            "List only the ones to replace.\n" + listed + "\n")
    for key, seed in raw.identity_seed.items():
        if seed and not identity.get(key):
            identity[key] = seed
    keys = raw.extra.get("identity_keys_withheld",
                         ("date", "location", "operator", "registration", "native_id"))
    withheld = verify.identifier_hygiene(
        [old for old, _ in all_pairs] + [v for k, v in identity.items() if k in keys and v])
    package_text = context + " " + " ".join(f"{i['neutral_title']} {i['text']}" for i in items)
    survivors = verify.leakage(package_text, withheld)
    forced: list[str] = []
    if survivors:
        by_value = {v: k for k, v in identity.items()}
        for value in survivors:
            neutral = verify.NEUTRAL_FOR_KEY.get(by_value.get(value, ""), "the subject")
            context, hit = verify.force_remove(context, [value], neutral)
            for item in items:
                item["text"], h2 = verify.force_remove(item["text"], [value], neutral)
                item["neutral_title"], h3 = verify.force_remove(item["neutral_title"], [value], neutral)
                hit = hit or h2 or h3
            if hit:
                forced.append(value)
                all_pairs.append((value, neutral))
        context = squash(context)
        for item in items:
            item["text"] = squash(item["text"])
            item["neutral_title"] = squash(item["neutral_title"])
        package_text = context + " " + " ".join(f"{i['neutral_title']} {i['text']}" for i in items)
    acronyms = verify.residual_withheld_tokens(package_text, withheld)
    if acronyms:
        context, _ = verify.force_remove(context, acronyms, "the authority")
        for item in items:
            item["text"], _ = verify.force_remove(item["text"], acronyms, "the authority")
            item["neutral_title"], _ = verify.force_remove(item["neutral_title"], acronyms, "the authority")
        forced.extend(acronyms)
        context = squash(context)
        for item in items:
            item["text"] = squash(item["text"])
            item["neutral_title"] = squash(item["neutral_title"])
        package_text = context + " " + " ".join(f"{i['neutral_title']} {i['text']}" for i in items)
    return {"ok": True, "replacements": all_pairs, "identity": identity, "forced_removals": forced,
            "withheld_identifiers": withheld, "applied": applied_total, "passes_run": len(sweeps),
            "still_present": verify.leakage(package_text, withheld),
            "proper_nouns_remaining": sweeps[-1] if sweeps else [],
            "residual": verify.residual_identifiers(package_text), "initial_context": context}


def step_reference(raw: RawCase, items: list[dict[str, Any]], analysis: str, conclusion: str,
                   client: Client, gate: Gate) -> dict[str, Any]:
    evidence = "\n".join(f"{i['evidence_id']} | {i['neutral_title']} | {i['text']}" for i in items)
    prompt = fill("reference", evidence=evidence, analysis=analysis, conclusion=conclusion,
                  domain_note=raw.extra.get("reference_note", ""))
    result = gate.call(client, role="extract", prompt=prompt, max_tokens=26000)
    if not result.get("ok") and "JSON" in str(result.get("error", "")):
        result = gate.call(client, role="extract", max_tokens=26000,
                           prompt=prompt + "\n\nYour previous reply was not valid JSON. Return valid JSON only. "
                                           "Inside a JSON string write any quotation mark of the quoted text as '.")
    if not result.get("ok") or not isinstance(result.get("value"), dict):
        return {"ok": False, "error": result.get("error") or "reply was not a JSON object",
                "raw_head": str(result.get("raw", ""))[:1500], "raw_tail": str(result.get("raw", ""))[-800:]}
    value = result["value"]
    ids = {i["evidence_id"] for i in items}
    rejected = []
    for index, item in enumerate(value.get("rejected_explanations") or [], 1):
        if not isinstance(item, dict) or not squash(item.get("mechanism", "")):
            continue
        quote = squash(str(item.get("quote", "")))
        cited = [str(e) for e in (item.get("evidence_ids") or []) if str(e) in ids]
        dropped = [str(e) for e in (item.get("evidence_ids") or []) if str(e) not in ids]
        rejected.append({"id": f"C{len(rejected) + 1}", "contradicted_mechanism": squash(item["mechanism"]),
                         "text": quote, "evidence_ids": cited, "note": squash(item.get("note", "")),
                         "quote_verbatim_in_analysis": verify.contains(analysis, quote),
                         "invented_evidence_ids": dropped})
    determination = squash(str(value.get("determination", ""))).lower()
    if determination not in ("determined", "undetermined"):
        determination = "undetermined" if re.search(
            r"not (be )?(determined|established)|could not|undetermined|no defect", conclusion, re.I) else "determined"
    return {"ok": True, "determination": determination,
            "closure_object": squash(str(value.get("closure_object", ""))),
            "official_conclusion_verbatim": squash(str(value.get("official_conclusion_verbatim", ""))) or squash(conclusion),
            "rejected_explanations": rejected,
            "alternatives_left_open": [squash(str(a)) for a in (value.get("alternatives_left_open") or []) if squash(str(a))]}


def step_review(items: list[dict[str, Any]], reference: dict[str, Any], analysis: str, conclusion: str,
                client: Client, gate: Gate, domain_note: str = "") -> dict[str, Any]:
    evidence = "\n".join(f"{i['evidence_id']} | {i['neutral_title']} | {i['text']}" for i in items)
    rejected = "\n".join(
        f"[{n}] mechanism: {c['contradicted_mechanism']}\n     quote: {c['text']}\n     evidence: {c['evidence_ids']}"
        for n, c in enumerate(reference["rejected_explanations"]))
    prompt = fill("review", determination=reference["determination"],
                  closure_object=reference["closure_object"], rejected=rejected or "(none)",
                  evidence=evidence, analysis=analysis, conclusion=conclusion, domain_note=domain_note)
    result = gate.call(client, role="review", prompt=prompt, max_tokens=26000)
    if not result.get("ok") and "JSON" in str(result.get("error", "")):
        result = gate.call(client, role="review", prompt=prompt + "\n\nYour previous reply was not valid JSON. "
                           "Return valid JSON only. Inside a JSON string write any quotation mark as '.",
                           max_tokens=26000)
    if not result.get("ok") or not isinstance(result.get("value"), dict):
        return {"ok": False, "error": result.get("error") or "reply was not a JSON object"}
    return {"ok": True, **result["value"]}
