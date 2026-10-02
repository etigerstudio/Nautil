#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repository root
from nautil_common import paths

sys.dont_write_bytecode = True

RUN = paths.RUN
ROOT = paths.DATA
SPLIT = RUN / "results/sft_content_passed_nonmedical_v2_2/split_v1"
TEST_BUNDLE = RUN / "results/sft_v2_2_eval/test_bundle_v1/test_prompts_and_evidence.jsonl"
BURNED = RUN / "results/sft_v2_2_eval/BURNED_TEST_CASES_long16_probe.json"
HOST_CLOSURE = RUN / "results/host_teacher_closure_v1/host_teacher_closure.jsonl"
WORKLIST = RUN / "results/multiturn_worklist_v2/cases.jsonl"
OUT = RUN / "results/counterfactual_test_v2"
BUNDLE_DIR = OUT / "test_bundles"
SEED = 20260925
LEVELS = [("level_040", 0.40), ("level_075", 0.75), ("level_100", 1.00)]
PROTECTED_RULE = {"all_sources_id_prefix": "E1.", "host_families": ["alert record"]}


def protected_ids(source: str, items: list[dict]) -> set[str]:
    ids = {x["evidence_id"] for x in items
           if x["evidence_id"].startswith(PROTECTED_RULE["all_sources_id_prefix"])}
    if source == "host":
        ids |= {x["evidence_id"] for x in items if x.get("family") in PROTECTED_RULE["host_families"]}
    return ids

ROLE_LABELS = {
    "ABRD-0010": "GGGAN", "ABRD-0028": "GGGGGGA", "ABRD-0054": "GGGGGGAGGGGGGGNA",
    "ABRD-0069": "GGGGGGGGGGGGGGAGGGGGGGNNNN", "ABRD-0097": "GNGAGA",
    "ABRD-0116": "GGGGGAGGGGGGGGNNAA", "ABRD-0231": "GGGGAGGAAAN", "ABRD-0284": "GGGGANA",
    "ABRD-0322": "GGGG", "ABRD-0351": "GGGGGGGGNNNNNNAA", "ABRD-0411": "GGGGGA",
    "ABRD-0415": "GGGGGGGAN", "ABRD-0433": "GGGGGGGGAAA", "ABRD-0453": "GGNNA",
    "ABRD-0464": "GGAAAAA", "ABRD-0467": "GGGGGGAA", "ABRD-0492": "GGNGGAA",
    "ABRD-0637": "GGGGGG",
    "ANHTSA-0044": "GAAAA", "ANHTSA-0051": "GGGAAGA",
    "ANTSB-0136": "GGGGN", "ANTSB-0155": "GGGNA", "ANTSB-0289": "GGGGNA",
    "HOSTW-0102": "GGANN", "HOSTW-0112": "GGGNNNNA", "HOSTW-0128": "GGGAAAN",
    "HOSTW-0139": "GGNNNA", "HOSTW-0186": "GGNNA", "HOSTW-0324": "GGNNN",
    "HOSTW-0332": "GNGGNNA", "HOSTW-0400": "GGGGNAA", "HOSTW-0408": "GGANNA",
    "HOSTW-0459": "GGGGNNN", "HOSTW-0500": "GGGNNNNA", "HOSTW-0548": "GGNNNNA",
    "HOSTW-0567": "GNAGGAN", "HOSTW-0571": "GNNNNA", "HOSTW-0581": "GGGNNNN",
    "HOSTW-0676": "GGGGANNANA", "HOSTW-0752": "GGGNNNNNNGA", "HOSTW-0814": "GANNNAG",
    "HOSTW-0815": "GGANNNN", "HOSTW-0826": "GGGNN", "HOSTW-0847": "GGGGGAGAANNNANNNNNNNNN",
    "HOSTW-0886": "GAGANNA", "HOSTW-0891": "GGNNNGA", "HOSTW-0914": "GGGNNAN",
    "HOSTW-0972": "GGGNNNA", "HOSTW-0997": "GGGNNA", "HOSTW-1001": "GGNANN",
    "HOSTW-1090": "GGGAANNNN", "HOSTW-1191": "GGGANNA", "HOSTW-1208": "GGANN",
    "HOSTW-1300": "GANNA", "HOSTW-1320": "GGGNNA", "HOSTW-1329": "GGGANNNA",
    "HOSTW-1413": "GGNNAA", "HOSTW-1430": "GNANA", "HOSTW-1473": "GGGGNNA",
    "HOSTW-1474": "GGGNNN", "HOSTW-1496": "GGAGAANNA",
    "HOSTW-1524": "GGGGGNGGNNNNNNNNNG",
}
ROLE_NAME = {"G": "grounds", "N": "not_grounds", "A": "ambiguous"}

EXPECTED_NOTE = {
    "level_000": "full evidence (unmodified): baseline closure rate on the same cases",
    "level_040": "no fixed expectation; one point of the closure-rate curve",
    "level_075": "no fixed expectation; one point of the closure-rate curve",
    "level_100": "100%: grounds of the teacher's conclusion removed -> closing is expected to "
                 "become unsupported",
    "control_nonkey_100": "control: same number of items removed as level_100 but none cited in the "
                          "teacher's final answer -> a closure drop here is not attributable to "
                          "losing the grounds",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_digest(package: dict) -> str:
    payload = {k: v for k, v in package.items() if k != "package_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def opening_brief(package: dict) -> str:
    record = [x for x in package["evidence_items"] if x["evidence_id"].startswith("E1.")]
    indexed = [x for x in package["evidence_items"] if x not in record]
    shown = "\n".join(f'- {x["evidence_id"]} | {x["neutral_title"]} | {x.get("kind") or x.get("locator") or ""} | source {x["source_id"]}\n  {x["text"]}' for x in record)
    index = "\n".join(f'- {x["evidence_id"]} | {x["neutral_title"]} | {x.get("kind") or str(x.get("locator", "")).rsplit(" | ", 1)[-1]} | source {x["source_id"]}' for x in indexed)
    return (f'# Case brief — {package["case_id"]}\n\n## Task question\n{package["task_question"]}\n\n'
            f'## Initial context\n{package["initial_context"]}\n\n'
            f'## Case record E1 ({len(record)} items, full text)\n{shown}\n\n'
            f'## Index of the remaining {len(indexed)} evidence items (titles only)\n{index}\n\n'
            'Fetch exact text with request_evidence when needed. Finish with a direct answer '
            'that separates the supported mechanism, discounted alternatives, and unresolved details.')


def find_package(row: dict, worklist: dict) -> tuple[Path, dict]:
    cid = row["case_id"]
    candidates = []
    if cid in worklist:
        candidates.append(ROOT / worklist[cid]["package_path"])
    if row["source"] == "host":
        candidates.append(paths.HOST_CASES / f"runtime/{cid}.json")
    candidates.append(RUN / f"results/sft_extra_v1/audit_repair/packages/{cid}.json")
    candidates += sorted(RUN.glob(f"results/sft_supplement_v1/*/v2_repair/packages/{cid}.json"))
    for path in candidates:
        if path.is_file():
            package = json.loads(path.read_text())
            if (package.get("package_hash") == row["package_hash"]
                    and package["evidence_items"] == row["evidence_items"]):
                if package_digest(package) != package["package_hash"]:
                    raise ValueError(f"{cid}: cannot reproduce package hash")
                if opening_brief(package) != row["user_message"]:
                    raise ValueError(f"{cid}: cannot reproduce the brief")
                return path, package
    raise ValueError(f"{cid}: no package matches the test bundle row")


ID = re.compile(r"E(\d+)\.(\d+)")
GROUP = re.compile(r"[\(\[]([^()\[\]]*?E\d+\.\d+[^()\[\]]*?)[\)\]]")
RANGE = re.compile(r"E(\d+)\.(\d+)\s*(?:–|-|to)\s*E(\d+)\.(\d+)")


def expand_ids(text: str) -> list[str]:
    out, spans = [], []
    for m in RANGE.finditer(text):
        a, b, c, d = map(int, m.groups())
        out += [f"E{a}.{k}" for k in range(b, d + 1)] if a == c and d >= b else [f"E{a}.{b}", f"E{c}.{d}"]
        spans.append(m.span())
    for s, e in reversed(spans):
        text = text[:s] + " " * (e - s) + text[e:]
    out += [m.group(0) for m in ID.finditer(text)]
    return list(dict.fromkeys(out))


def citation_units(final: str) -> list[dict]:
    body = final[len("CASE CLOSED"):]
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    units = []
    for pi, para in enumerate(paragraphs):
        last = 0
        for m in GROUP.finditer(para):
            units.append({"unit": len(units), "paragraph": pi, "text": para[last:m.start()].strip(),
                          "citation": m.group(0), "ids": expand_ids(m.group(1))})
            last = m.end()
    if set(expand_ids(final)) != {i for u in units for i in u["ids"]}:
        raise ValueError("some cited ids fall outside citation groups")
    return units


def round_half_up(x: float) -> int:
    return int(x + 0.5)


def level_counts(k: int) -> dict[str, int]:
    n = {name: max(1, round_half_up(p * k)) for name, p in LEVELS}
    n["level_100"] = k
    if k >= 3:
        n["level_075"] = min(n["level_075"], k - 1)
        n["level_040"] = min(n["level_040"], n["level_075"] - 1)
    return n


STOP = set("""a about above after again against all also although among an and any are around as at
be because been before being below between both but by can could did do does during each either
from had has have having he her his how however if in into is it its itself least less more most
much near no nor not of off on once only or other our out over per same she should since so some
such than that the their them then there these they this those through thus to too under until
up upon very was were what when where whether which while who whom why will with within would
event day record report reported recorded shows showed found""".split())
UNIT = (r"(?:%|percent|per cent|bytes?|[kKMGT]i?B|GB/s|MB/s|mph|km/?h|knots?|kts?|ft|feet|m|mm|cm|km|"
        r"miles?|nm|psi|psig|kPa|MPa|bar|°C|°F|C|F|lb|lbs|pounds?|kg|tonnes?|tons?|N|kN|V|A|Hz|rpm|"
        r"seconds?|s|minutes?|min|hours?|h|degrees?|°|gallons?|litres?|liters?|/s|mg|ml)")
NUM_UNIT = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?(?:e[+-]?\d+)?)\s?(" + UNIT + r")(?![A-Za-z])")
LONG_NUM = re.compile(r"(?<![\w.:])(\d[\d,]*\.\d+(?:e[+-]?\d+)?|\d{1,3}(?:,\d{3})+|\d{4,}(?:\.\d+)?)(?![\w:])")
TIME = re.compile(r"^\d{1,2}:\d{2}")
QUOTE = re.compile(r"[‘'\"“]([^‘'\"“”’]{12,120})[’'\"”]")
WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*[A-Za-z0-9]")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'").replace("‘", "'")).lower()


def words(text: str) -> list[str]:
    return [w.lower().rstrip(".") for w in WORD.findall(text)]


def raw_numbers(text: str) -> set[str]:
    nums = set()
    for m in NUM_UNIT.finditer(text):
        value = m.group(1).replace(",", "")
        if len(value.replace(".", "")) >= 2 or m.group(2) in ("%", "percent"):
            nums.add(value)
    for m in LONG_NUM.finditer(text):
        value = m.group(1).replace(",", "")
        if not re.fullmatch(r"(19|20)\d\d", value):
            nums.add(value)
    return nums


TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*[A-Za-z0-9]|\d[\d,.:]*\d|\d")


def raw_shingles(text: str) -> set[str]:
    ws = [w for w in (t.lower().rstrip(".") for t in TOKEN.findall(text)) if w not in STOP]
    return {" ".join(ws[i:i + 6]) for i in range(len(ws) - 5)}


MAX_DF = 3


def features(item: dict, df: dict) -> dict:
    text = item["text"]
    quotes = {norm(q.strip()) for q in QUOTE.findall(text) if len(q.split()) >= 3}
    return {"numbers": {v for v in raw_numbers(text) if df["numbers"][v] <= MAX_DF},
            "quotes": quotes,
            "rare_tokens": {w for w in words(text) if len(w) >= 5 and w not in STOP
                            and df["tokens"][w] <= MAX_DF},
            "shingles": {s for s in raw_shingles(text) if df["shingles"][s] <= MAX_DF
                         and sum(t not in df["template"] for t in s.split()) >= 3},
            "title": norm(item["neutral_title"]),
            "times": set(TIME_TOKEN.findall(text))}


TIME_TOKEN = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")


def number_in(value: str, text: str) -> bool:
    variants = {value}
    if "." not in value and "e" not in value and len(value) >= 4:
        variants.add(f"{int(value):,}")
    return any(re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d])", text) for v in variants)


def match_target(feat: dict, target_text: str) -> dict:
    t_norm = norm(target_text)
    t_words = set(words(target_text))
    t_sh = raw_shingles(target_text)
    return {"numbers": sorted(v for v in feat["numbers"] if number_in(v, target_text)),
            "quotes": sorted(q for q in feat["quotes"] if q in t_norm),
            "rare_tokens": sorted(feat["rare_tokens"] & t_words),
            "shingles": sorted(feat["shingles"] & t_sh),
            "title": len(feat["title"]) >= 8 and feat["title"] in t_norm,
            "times": sorted(t for t in feat["times"] if t in set(TIME_TOKEN.findall(target_text)))}


def is_likely_restatement(hit: dict, where: str = "") -> bool:
    return ((where == "brief" and len(hit["times"]) >= 2) or len(hit["numbers"]) >= 2 or bool(hit["quotes"]) or len(hit["shingles"]) >= 2
            or (len(hit["numbers"]) >= 1 and len(hit["rare_tokens"]) >= 2))


def leak_check(removed: list[dict], remaining: list[dict], brief: str, all_items: list[dict]) -> list[dict]:
    df = {"numbers": collections.Counter(), "tokens": collections.Counter(),
          "shingles": collections.Counter()}
    for item in all_items:
        df["numbers"].update(raw_numbers(item["text"]))
        df["tokens"].update(set(words(item["text"])))
        df["shingles"].update(raw_shingles(item["text"]))
    tok_df = collections.Counter()
    for item in all_items:
        tok_df.update({t for s in raw_shingles(item["text"]) for t in s.split()})
    df["template"] = {t for t, n in tok_df.items() if n > 0.2 * len(all_items)}
    results = []
    for item in removed:
        feat = features(item, df)
        hits = []
        targets = [("brief", brief)] + [(x["evidence_id"], x["neutral_title"] + " " + x["text"])
                                         for x in remaining]
        for where, text in targets:
            hit = match_target(feat, text)
            if where != "brief":
                hit["times"] = []
            strong_hit = is_likely_restatement(hit, where)
            if strong_hit or hit["title"]:
                hits.append({"where": where, "likely_restatement": strong_hit, **hit})
        strong = [h for h in hits if h["likely_restatement"]]
        results.append({
            "evidence_id": item["evidence_id"],
            "n_features": {k: (len(v) if isinstance(v, set) else 1) for k, v in feat.items()},
            "restated_in_brief_by_clock_times": any(h["where"] == "brief" and len(h["times"]) >= 2
                                                    for h in strong),
            "likely_restated": bool(strong),
            "restated_in": [h["where"] for h in strong],
            "restated_in_brief": any(h["where"] == "brief" for h in strong),
            "title_mentioned_in": [h["where"] for h in hits if h["title"]],
            "hits": sorted(strong, key=lambda h: -(len(h["numbers"]) + 2 * len(h["quotes"])
                                                    + len(h["shingles"])))[:5],
        })
    return results


def build_variant(package: dict, removed_ids: list[str]) -> tuple[dict, str]:
    removed = set(removed_ids)
    new_package = dict(package)
    new_package["evidence_items"] = [x for x in package["evidence_items"] if x["evidence_id"] not in removed]
    new_package["package_hash"] = ""
    new_package["package_hash"] = package_digest(new_package)
    brief = opening_brief(new_package)
    old_lines, new_lines = opening_brief(package).split("\n"), brief.split("\n")
    header = re.compile(r"^## (Case record E1 \(\d+ items|Index of the remaining \d+ evidence items)")
    by_id = {x["evidence_id"]: x for x in package["evidence_items"]}
    dropped = []
    for x in package["evidence_items"]:
        if x["evidence_id"] in removed:
            dropped.append(f'- {x["evidence_id"]} |')
    j = 0
    for line in old_lines:
        if j < len(new_lines) and line == new_lines[j]:
            j += 1
            continue
        if header.match(line) and j < len(new_lines) and header.match(new_lines[j]):
            j += 1
            continue
        if any(line.startswith(p) for p in dropped):
            continue
        if any(by_id[i]["evidence_id"].startswith("E1.") and line in ("  " + by_id[i]["text"]).split("\n")
               for i in removed):
            continue
        raise ValueError(f"{package['case_id']}: unexpected brief change: {line[:80]!r}")
    if j != len(new_lines):
        raise ValueError(f"{package['case_id']}: brief diff did not consume the new brief")
    return new_package, brief


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    BUNDLE_DIR.mkdir(parents=False, exist_ok=False)
    test_ids = (SPLIT / "test_ids.txt").read_text().splitlines()
    examples = {r["case_id"]: r for r in read_jsonl(SPLIT / "test_model_visible.jsonl")}
    bundle_rows = {r["case_id"]: r for r in read_jsonl(TEST_BUNDLE)}
    bundle_lines = {json.loads(l)["case_id"]: l for l in TEST_BUNDLE.read_text().splitlines() if l.strip()}
    meta = {r["case_id"]: r for r in read_jsonl(SPLIT / "per_case_metadata.jsonl")}
    host_label = {r["case_id"]: r["expected_closure"] for r in read_jsonl(HOST_CLOSURE)}
    burned = set(json.loads(BURNED.read_text())["case_ids"])
    worklist = {r["case_id"]: r for r in read_jsonl(WORKLIST)}
    review_queue = set((SPLIT / "test_maib_raib_review_queue.txt").read_text().split())
    assert len(test_ids) == 112 and set(test_ids) == set(examples) == set(bundle_rows)

    eligibility, eligible = [], []
    for cid in test_ids:
        ex = examples[cid]
        final = ex["messages"][-1]
        assert final["role"] == "assistant" and not final.get("tool_calls")
        text = final["content"]
        marker = ("CASE CLOSED" if text.startswith("CASE CLOSED") else
                  "CASE NOT CLOSED" if text.startswith("CASE NOT CLOSED") else "none")
        entry = {"case_id": cid, "source": ex["source"], "category": meta[cid]["category"],
                 "teacher_marker": marker, "metadata_closure": meta[cid]["closure"],
                 "host_teacher_label": host_label.get(cid), "burned_long16": cid in burned}
        closed = marker == "CASE CLOSED"
        if ex["source"] == "host":
            if (host_label[cid] == "teacher_closed") != closed:
                raise ValueError(f"{cid}: host label disagrees with the final answer")
        if (meta[cid]["closure"] == "closed") != closed:
            raise ValueError(f"{cid}: metadata closure disagrees with the final answer")
        entry["eligible"] = closed
        entry["reason"] = "eligible" if entry["eligible"] else "teacher CASE NOT CLOSED"
        eligibility.append(entry)
        if closed:
            eligible.append(cid)

    unit_rows, manifest, bundles = [], [], collections.defaultdict(list)
    leak_rows, per_case = [], []
    for cid in eligible:
        ex = examples[cid]
        row = bundle_rows[cid]
        _, package = find_package(row, worklist)
        store = {x["evidence_id"]: x for x in package["evidence_items"]}
        final = ex["messages"][-1]["content"]
        units = citation_units(final)
        labels = ROLE_LABELS[cid]
        if len(labels) != len(units):
            raise ValueError(f"{cid}: {len(units)} units but {len(labels)} labels")
        roles = collections.defaultdict(set)
        for unit, lab in zip(units, labels):
            unit["role"] = ROLE_NAME[lab]
            for i in unit["ids"]:
                roles[i].add(lab)
            unit_rows.append({"case_id": cid, **unit})
        cited = set(roles)
        unknown = sorted(cited - set(store))
        protected = protected_ids(ex["source"], package["evidence_items"])
        key_all = sorted((i for i in cited if "G" in roles[i] and i in store), key=lambda e: tuple(map(int, e[1:].split("."))))
        key = [i for i in key_all if i not in protected]
        ambiguous_only = sorted(i for i in cited if roles[i] == {"A"})
        not_grounds_only = sorted(i for i in cited if roles[i] == {"N"})
        a_and_n = sorted(i for i in cited if roles[i] == {"A", "N"})
        seen = {i for i in store if i.startswith("E1.")}
        for m in ex["messages"]:
            if m["role"] == "tool":
                seen |= {x["evidence_id"] for x in json.loads(m["content"])["evidence_items"]}
        key_unseen = sorted(set(key) - seen)
        k = len(key)
        rng = random.Random(f"{SEED}:{cid}:key")
        order = list(key_all)
        rng.shuffle(order)
        order = [i for i in order if i not in protected]
        counts = level_counts(k)
        variants = {"level_000": []}
        for name, _ in LEVELS:
            variants[name] = order[:counts[name]]
        pool = sorted(i for i in store if i not in cited)
        n_e1_key = sum(i.startswith("E1.") for i in key)
        rng_c = random.Random(f"{SEED}:{cid}:control")
        pool_e1 = [i for i in pool if i.startswith("E1.")]
        pool_ix = [i for i in pool if not i.startswith("E1.")]
        rng_c.shuffle(pool_e1)
        rng_c.shuffle(pool_ix)
        pool_e1 = [i for i in pool_e1 if i not in protected]
        pool_ix = [i for i in pool_ix if i not in protected]
        if pool_e1 or n_e1_key:
            raise ValueError(f"{cid}: E1 item left in key set or control pool")
        take_e1 = 0
        control = pool_ix[:k]
        variants["control_nonkey_100"] = control
        n_brief_items = sum(i.startswith("E1.") for i in store)
        case_info = {"case_id": cid, "source": ex["source"], "category": meta[cid]["category"],
                     "burned_long16": cid in burned, "maib_raib_review_queue": cid in review_queue,
                     "n_items": len(store), "n_units": len(units), "k": k, "key_ids": key,
                     "key_ids_in_brief": [i for i in key if i.startswith("E1.")],
                     "ambiguous_only_ids": ambiguous_only, "not_grounds_only_ids": not_grounds_only,
                     "ambiguous_and_not_grounds_ids": a_and_n, "cited_ids_not_in_store": unknown,
                     "key_ids_never_shown_to_teacher": key_unseen,
                     "protected_ids": sorted(protected),
                     "protected_ids_dropped_from_key": sorted(set(key_all) - set(key)),
                     "level_counts": counts, "control_pool_size": len(pool_e1) + len(pool_ix),
                     "control_short_by": k - len(control),
                     "control_items_fetched_by_teacher": sum(i in seen for i in control)}
        per_case.append(case_info)
        for vname, removed_ids in variants.items():
            if vname == "level_000":
                new_row_line = bundle_lines[cid]
                new_hash = row["package_hash"]
                leak = []
                all_e1_removed = False
            else:
                new_package, brief = build_variant(package, removed_ids)
                new_row = {"case_id": cid, "source": row["source"], "user_message": brief,
                           "tools": row["tools"], "evidence_items": new_package["evidence_items"],
                           "package_hash": new_package["package_hash"]}
                new_row_line = json.dumps(new_row, ensure_ascii=False, separators=(",", ":"))
                new_hash = new_package["package_hash"]
                removed_items = [store[i] for i in removed_ids]
                leak = leak_check(removed_items, new_package["evidence_items"], brief, package["evidence_items"])
                all_e1_removed = n_brief_items > 0 and all(
                    not i.startswith("E1.") for i in (x["evidence_id"] for x in new_package["evidence_items"]))
            bundles[vname].append(new_row_line)
            n_leak = sum(x["likely_restated"] for x in leak)
            manifest.append({
                "case_id": cid, "source": row["source"], "category": meta[cid]["category"],
                "split": "test", "variant": vname, "k": k,
                "removed_ids": removed_ids, "n_removed": len(removed_ids),
                "n_remaining": len(store) - len(removed_ids),
                "removed_key_ids_in_brief": [i for i in removed_ids if i.startswith("E1.")] if vname != "control_nonkey_100" else [],
                "removed_ids_in_brief": [i for i in removed_ids if i.startswith("E1.")],
                "key_ids_in_brief_flag": any(i.startswith("E1.") for i in removed_ids) and vname != "control_nonkey_100",
                "all_case_record_items_removed": all_e1_removed,
                "leak_n_removed_likely_restated": n_leak,
                "leak_ids_likely_restated": [x["evidence_id"] for x in leak if x["likely_restated"]],
                "leak_ids_restated_in_brief": [x["evidence_id"] for x in leak if x["restated_in_brief"]],
                "leak_flag": n_leak > 0,
                "original_package_hash": row["package_hash"], "new_package_hash": new_hash,
                "expected_note": EXPECTED_NOTE[vname]})
            for x in leak:
                leak_rows.append({"case_id": cid, "variant": vname, **x})

    files = {}
    for vname, lines in bundles.items():
        path = BUNDLE_DIR / f"test_cf_{vname}.jsonl"
        path.write_text("".join(line + "\n" for line in lines))
        files[vname] = {"path": str(path.relative_to(RUN)), "sha256": sha256_file(path), "cases": len(lines)}
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))
    (OUT / "per_case_key_sets.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_case))
    (OUT / "final_answer_citation_units.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unit_rows))
    (OUT / "leak_check_details.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in leak_rows))
    (OUT / "eligibility.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in eligibility))
    summary = {
        "schema_version": "nautil.counterfactual_test.v2", "split": "test", "seed": SEED,
        "script": "scripts/build_counterfactual_test_v2.py",
        "inputs": {str(p.relative_to(RUN)): sha256_file(p) for p in
                   (SPLIT / "test_model_visible.jsonl", SPLIT / "per_case_metadata.jsonl", TEST_BUNDLE,
                    BURNED, HOST_CLOSURE)},
        "bundles": files,
        "scheduler_note": "declare every dataset from test_bundles/ with split: test; the test_ path "
                          "prefix also trips the scheduler name guard (config.looks_like_test_file). "
                          "For the PrebuiltBundle transform use level_000 as the dataset bundle and a "
                          "variant file as prebuilt_bundles[dataset].",
        "protected_rule": PROTECTED_RULE,
        "protected_counts_by_source": dict(collections.Counter(
            f'{r["source"]}:{len(r["protected_ids"])} protected' for r in per_case)),
        "control_rule": "first k items of the seeded pool of non-cited, non-protected items; the "
                        "earlier 'match the number of E1 items' stratification is dropped because "
                        "no E1 item can be removed any more",
        "cases_with_small_key_set": [{"case_id": r["case_id"], "k": r["k"]}
                                     for r in per_case if r["k"] < 2],
        "no_inference_run": True, "evidence_ids_renumbered": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"eligible_closed": len(eligible), "bundles": {k: v["cases"] for k, v in files.items()}}))


if __name__ == "__main__":
    main()
