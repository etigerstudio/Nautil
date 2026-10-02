from __future__ import annotations

import re

ID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")
LEDGER = re.compile(r"H(\d+) \[([^\]]*?)(?:->|→)\s*([a-z]+)\]")


def _yn(b: bool, why: str = "rule") -> dict:
    return {"why": why, "answer": "yes" if b else "no"}


def _blocks(text: str, head: str) -> list[str]:
    return re.split(rf"^# {head} \d+\n", text, flags=re.M)[1:]


def _ledger_end(text: str) -> dict:
    return {h: after for h, _before, after in LEDGER.findall(text)}


def answer(dimension: str, user: str) -> dict:
    if dimension == "fetch":
        out = []
        for n, blk in enumerate(_blocks(user, "Request"), 1):
            req = blk.split("## Requested items\n", 1)[1].split("\n\n## Stated reason", 1)[0].splitlines()
            reason = blk.split("## Stated reason\n", 1)[1].split("\n\n## KEY items", 1)[0]
            unread_key = set(ID.findall(blk.split("## KEY items still UNREAD", 1)[1]))
            asked = [ln.split(" | ")[0] for ln in req]
            unrelated = sum(("background" in ln) or ln.endswith("ALREADY READ") for ln in req)
            out.append({"request": n,
                        "F1": _yn(bool(re.search(r"\bH[12]\b|hypothes", reason, re.I))),
                        "F2": _yn(unrelated * 2 > len(req)),
                        "F3": _yn(bool(unread_key) and not (set(asked) & unread_key))})
        return {"requests": out}
    if dimension == "hypothesis":
        out = []
        for n, blk in enumerate(_blocks(user, "Update"), 1):
            before = blk.split("## Evidence JUST RETURNED", 1)[0]
            returned = blk.split("## Evidence JUST RETURNED\n", 1)[1].split("## The investigator's update", 1)[0]
            update = blk.split("## The investigator's update\n", 1)[1]
            lb, lu = _ledger_end(before), _ledger_end(update)
            supports, inconc = "supports H1" in returned, "inconclusive" in returned
            bears = supports or inconc
            changed = lb != lu
            if supports:
                h2 = lu.get("1") == "favored"
            elif inconc:
                h2 = lu.get("1") == "open" and lu.get("2") == "open"
            else:
                h2 = None
            missed = "support" in update.lower() and lu.get("1") == lb.get("1") and supports
            nothing_new = "(no new evidence" in returned or "(the fetch failed" in returned or "(no tool result" in returned
            ids_ret = set(ID.findall(returned))
            h5 = ("no new evidence" in update.lower()) if nothing_new else bool(set(ID.findall(update)) & ids_ret)
            out.append({"update": n, "H1": _yn(bears),
                        "H2": {"why": "rule", "answer": "n/a" if h2 is None else ("yes" if h2 else "no")},
                        "H3": _yn(missed), "H4": _yn((not bears) and changed), "H5": _yn(h5)})
        return {"updates": out}
    if dimension == "closure":
        final = user.split("# Final answer\n", 1)[1].split("\n# ", 1)[0]
        ref = user.split("# REFERENCE", 1)[1]
        ref_closed = "## Reference closure decision\nCASE CLOSED" in ref
        closed = "# Investigator's decision\nCASE CLOSED" in user
        main = final.split("Discounted")[0].lower()
        right = ("bearing seizure (h1)" in main) if ref_closed else ("inconclusive" in main)
        wrong = "electrical fault (h2)" in main
        c1 = "no" if wrong or not ID.findall(final) else "yes" if right else "partial"
        read = user.split("# Titles of items the investigator READ\n", 1)[1].split("\n# ", 1)[0]
        unread = user.split("# Titles of items the investigator did NOT READ\n", 1)[1].split("\n# ", 1)[0]
        read_ids = set(ID.findall(read))
        key_read = "(decisive)" in read
        cited = set(ID.findall(final))
        inconclusive_read = not ref_closed and key_read
        c2 = not (closed and (inconclusive_read or not key_read))
        c3 = bool(cited) and cited <= read_ids and "NEVER READ" not in user and "DOES NOT EXIST" not in user
        c4 = bool(re.search(r"H2|electrical fault|Discounted", final)) and not wrong
        gap = None
        if not closed:
            g1 = "teardown" in final
            gap = {"G1": _yn(g1), "G2": _yn(g1 and "(decisive)" not in unread),
                   "G3": _yn(g1 and not ref_closed)}
        return {"conclusion": {"C1": {"why": "rule", "answer": c1}, "C2": _yn(c2), "C3": _yn(c3),
                               "C4": _yn(c4)}, "gap": gap}
    raise ValueError(dimension)
