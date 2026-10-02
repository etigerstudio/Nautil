from __future__ import annotations

import re

ID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")


def _yn(b: bool) -> dict:
    return {"why": "rule", "answer": "yes" if b else "no"}


def answer(dimension: str, user: str) -> dict:
    if dimension == "hypothesis":
        out = []
        for n, blk in enumerate(re.split(r"^# Update \d+\n", user, flags=re.M)[1:], 1):
            returned = blk.split("## Evidence JUST RETURNED\n", 1)[1].split("## The investigator's update", 1)[0]
            update = blk.split("## The investigator's update\n", 1)[1].split("## Hypotheses to assess", 1)[0]
            ids = [x.strip() for x in blk.split("## Hypotheses to assess (in this order)\n", 1)[1]
                   .split("\n", 1)[0].split(",")]
            hyps = []
            for h in ids:
                t = h == "H1" and "supports H1" in returned
                hyps.append({"id": h, "why": "rule", "touched": "yes" if t else "no",
                             "direction": "supports" if t else "unrelated"})
            touched = [x["id"] for x in hyps if x["touched"] == "yes"]
            nothing = "(no new evidence" in returned or "(the fetch failed" in returned or "(no tool result" in returned
            h5 = ("no new evidence" in update.lower()) if nothing else bool(set(ID.findall(update)) & set(ID.findall(returned)))
            out.append({"update": n, "hypotheses": hyps, "main": touched[0] if touched else None, "H5": _yn(h5)})
        return {"updates": out}
    if dimension == "closure":
        final = user.split("# Final answer\n", 1)[1].split("\n# ", 1)[0]
        ref = user.split("# REFERENCE", 1)[1]
        ref_closed = "## Reference closure decision\nCASE CLOSED" in ref
        closed = "# Investigator's decision\nCASE CLOSED" in user
        main = final.split("Discounted")[0].lower()
        right = ("bearing seizure (h1)" in main) if ref_closed else ("inconclusive" in main)
        wrong = "electrical fault (h2)" in main
        cited = set(ID.findall(final))
        read = user.split("# Titles of items the investigator READ\n", 1)[1].split("\n# ", 1)[0]
        unread = user.split("# Titles of items the investigator did NOT READ\n", 1)[1].split("\n# ", 1)[0]
        read_ids, key_read = set(ID.findall(read)), "(decisive)" in read
        c1b = "no" if wrong or not cited else "yes" if right else "partial"
        claims = [{"claim": s.strip()[:60], "supported": "yes" if (set(ID.findall(s)) and set(ID.findall(s)) <= read_ids) else "no"}
                  for s in re.split(r"(?<=[.;])\s+", final) if re.search(r"\(H[12]\)|H[12]\b", s)]
        con = {"C1a": _yn(right and not wrong), "C1b": {"why": "rule", "answer": c1b},
               "C2": _yn(not (closed and ((not ref_closed and key_read) or not key_read))),
               "C3": {"why": "rule", "claims": claims},
               "C4": _yn(bool(re.search(r"H2|electrical fault|Discounted", final)) and not wrong)}
        gap = None
        if not closed:
            g1 = "teardown" in final
            gap = {"G1": _yn(g1), "G2": ({"why": "rule", "answer": "n/a"} if not g1 else _yn("(decisive)" not in unread)),
                   "G3": _yn(g1 and not ref_closed)}
        return {"conclusion": con, "gap": gap}
    raise ValueError(dimension)
