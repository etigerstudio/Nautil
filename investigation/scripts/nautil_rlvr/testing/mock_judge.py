from __future__ import annotations

import argparse
import json
import random
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ID = re.compile(r"\bE[1-9][0-9]*\.[1-9][0-9]*\b")
LEDGER_END = re.compile(r"H(\d+) \[[^\]]*?(?:->|→)\s*([a-z]+)\]")


def _blocks(text: str, head: str) -> list[str]:
    parts = re.split(rf"^# {head} \d+\n", text, flags=re.M)
    return parts[1:]


def _key_ids(text: str) -> set[str]:
    sec = text.split("# KEY EVIDENCE", 1)[1].split("\n# ", 1)[0] if "# KEY EVIDENCE" in text else ""
    return set(ID.findall(sec))


def _ledger(text: str) -> dict:
    return {h: s for h, s in LEDGER_END.findall(text)}


def score(dimension: str, user: str) -> dict:
    if dimension == "fetch":
        key = _key_ids(user)
        steps = []
        for n, blk in enumerate(_blocks(user, "Step"), 1):
            req = blk.split("## Requested items\n", 1)[1].split("\n\n## Stated reason", 1)[0].splitlines()
            new = [ln for ln in req if ln.endswith("| NEW")]
            key_new = sum(ln.split(" | ")[0] in key for ln in new)
            s = 0 if not new else round(2 + 8 * key_new / max(len(req), 1))
            steps.append({"step": n, "justification": f"{key_new} key of {len(req)} requested", "score": s})
        return {"steps": steps}
    if dimension == "hypothesis":
        ups = []
        for n, blk in enumerate(_blocks(user, "Update"), 1):
            before = blk.split("## Evidence JUST RETURNED", 1)[0]
            returned = blk.split("## Evidence JUST RETURNED\n", 1)[1].split("## The investigator's update", 1)[0]
            update = blk.split("## The investigator's update\n", 1)[1]
            lb, lu = _ledger(before), _ledger(update)
            changed = lb != lu
            if "(no new evidence" in returned or "(the fetch failed" in returned:
                s, new = (0 if changed else 6), False
            elif "supports H1" in returned:
                s, new = (9 if lu.get("1") == "favored" else 2), True
            elif "inconclusive" in returned:
                s, new = (8 if lu.get("1") == "open" and lu.get("2") == "open" else 3), True
            else:
                s, new = (6 if not changed else 2), True
            ups.append({"update": n, "new_information": new, "justification": "rule", "score": s})
        return {"updates": ups}
    if dimension == "closure":
        final = user.split("# Final answer\n", 1)[1].split("\n# ", 1)[0]
        ref = user.split("# REFERENCE", 1)[1]
        key = set(ID.findall(ref.split("## Reference final answer", 1)[1].split("## Official", 1)[0]))
        cites = set(ID.findall(final))
        right = "bearing seizure (H1)" in final or ("H1" in final and "H2" not in final.split("Discounted")[0])
        wrong = "electrical fault (H2)" in final.split("Discounted")[0]
        c = 1 if wrong else 9 if right and len(cites & key) >= 2 else 4
        if "inconclusive" in final and "inconclusive" in ref:
            c = 9 if len(cites & key) >= 2 else 4
        closed = "# Investigator's decision\nCASE CLOSED" in user
        gap = None
        if not closed:
            unread = user.split("# Titles of items the investigator did NOT READ\n", 1)[1].split("\n# ", 1)[0]
            missing = "(decisive)" in unread
            gap = {"justification": "rule", "claims_missing_evidence_that_was_available": missing,
                   "score": 1 if missing else 8}
        return {"conclusion": {"justification": "rule", "score": c}, "gap": gap}
    raise ValueError(dimension)


def dimension_of(system: str) -> str:
    if "evidence requests" in system:
        return "fetch"
    if "hypothesis state BEFORE" in system or "updates its hypotheses" in system:
        return "hypothesis"
    return "closure"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        return

    def do_POST(self):
        st = self.server.state
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        with st["lock"]:
            st["calls"] += 1
            st["keys"][self.headers.get("Authorization", "")[-6:]] = st["keys"].get(self.headers.get("Authorization", "")[-6:], 0) + 1
            r = st["rng"].random()
        if r < st["fail_rate"]:
            out = json.dumps({"error": "injected"}).encode()
            self.send_response(500)
        else:
            msgs = body["messages"]
            parsed = score(dimension_of(msgs[0]["content"]), msgs[1]["content"])
            content = "not json {" if st["rng"].random() < st["garbage_rate"] else json.dumps(parsed)
            out = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                              "usage": {"prompt_tokens": len(msgs[1]["content"]) // 4, "completion_tokens": 50,
                                        "cost": 0.0001}, "provider": "mock"}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        out = json.dumps({"calls": self.server.state["calls"], "keys": self.server.state["keys"]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--fail-rate", type=float, default=0.0)
    ap.add_argument("--garbage-rate", type=float, default=0.0)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.state = {"lock": threading.Lock(), "calls": 0, "keys": {}, "rng": random.Random(7),
                    "fail_rate": args.fail_rate, "garbage_rate": args.garbage_rate}
    print(json.dumps({"mock_judge": args.port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
