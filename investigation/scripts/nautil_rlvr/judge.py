from __future__ import annotations

import collections
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
import heapq
import itertools
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .common import atomic_json, sha256_bytes, sha256_file
from .reward import JudgeFormatError, parse_judge

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILES = {"fetch": "fetch_judge_v1.txt", "hypothesis": "hypothesis_judge_v1.txt",
                "closure": "closure_judge_v1.txt"}
PROMPT_FILES_V2 = {"fetch": "fetch_v2.txt", "hypothesis": "hypothesis_v2.txt", "closure": "closure_v2.txt"}
PROMPT_FILES_V22 = {"hypothesis": "hypothesis_v21.txt", "closure": "closure_v21.txt"}
SCHEMAS = {"v1": PROMPT_FILES, "v2": PROMPT_FILES_V2, "v22": PROMPT_FILES_V22}


def prompt_path(name) -> Path:
    p = Path(name)
    return p if p.is_absolute() else PROMPT_DIR / p


def read_keys_from_stdin(names: list[str]) -> None:
    wanted = set(names)
    for line in sys.stdin:
        line = line.strip()
        if "=" in line:
            name, value = line.split("=", 1)
            if name in wanted:
                os.environ[name] = value.strip().strip('"').strip("'")
        if all(os.environ.get(n) for n in wanted):
            break
    try:
        sys.stdin.close()
    except OSError:
        pass
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"judge keys missing on stdin: {missing}")


def keys_from_env(names: list[str], env_file: Path | None = None) -> list[str]:
    values = {n: os.environ.get(n) for n in names}
    if env_file and Path(env_file).is_file():
        for line in Path(env_file).read_text().splitlines():
            for n in names:
                if line.startswith(n + "=") and not values[n]:
                    values[n] = line.split("=", 1)[1].strip().strip('"').strip("'")
    missing = [n for n, v in values.items() if not v]
    if missing:
        raise RuntimeError(f"judge API keys not available: {missing}")
    return [values[n] for n in names]


@dataclass
class JudgeConfig:
    model: str
    base_url: str = "https://example.com/v1"
    key_env: list = field(default_factory=lambda: ["API_KEY"])
    env_file: str | None = None
    per_key_concurrency: int = 4
    routes: list = field(default_factory=list)
    routes_file: str | None = None
    route_names: list = field(default_factory=list)
    stagger_seconds: float = 0.01
    timeout_seconds: float = 180.0
    max_tokens: int | None = 4096
    temperature: float | None = 0.0
    response_format_json: bool = True
    extra_body: dict = field(default_factory=dict)
    retry_backoff: list = field(default_factory=lambda: [2, 5, 15, 30, 60])
    format_retries: int = 2
    cache_dir: str | None = None
    price_input_per_m: float = 0.0
    price_output_per_m: float = 0.0
    budget_usd: float | None = None
    schema: str = "v1"
    prompt_files: dict | None = None
    format_retry_feedback: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "JudgeConfig":
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    def resolved_routes(self) -> list[dict]:
        routes = list(self.routes)
        if self.routes_file:
            listed = json.loads(Path(self.routes_file).read_text())["routes"]
            routes += [r for r in listed if not self.route_names or r["name"] in self.route_names]
        if not routes:
            routes = [{"name": f"key{i + 1}", "base_url": self.base_url, "key_env": k,
                       "wire_model": self.model, "max_concurrent": self.per_key_concurrency}
                      for i, k in enumerate(self.key_env)]
        return routes


class BudgetExceeded(RuntimeError):
    pass


def _new_counters() -> dict:
    return {"requests": 0, "calls_ok": 0, "calls_failed": 0, "cache_hits": 0, "http_attempts": 0,
            "retries": 0, "errors": collections.Counter(), "parse_failures": collections.Counter(),
            "latency": [], "per_key": collections.Counter(), "providers": collections.Counter(),
            "cost_usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0}


class JudgeClient:
    def __init__(self, cfg: JudgeConfig, keys: dict | None = None):
        self.cfg = cfg
        self.routes = cfg.resolved_routes()
        if keys is None:
            values = keys_from_env([r["key_env"] for r in self.routes], cfg.env_file)
            keys = {r["key_env"]: v for r, v in zip(self.routes, values)}
        self.keys = keys
        self.sems = [threading.Semaphore(int(r["max_concurrent"])) for r in self.routes]
        slots = [i for i, r in enumerate(self.routes) for _ in range(int(r["max_concurrent"]))]
        random.Random(20260925).shuffle(slots)
        self.slots = slots
        self.pq: list = []
        self.pq_cond = threading.Condition()
        self.pq_seq = itertools.count()
        self.pq_closed = False
        self.workers = [threading.Thread(target=self._worker, daemon=True, name=f"judge-{i}")
                        for i in range(len(slots))]
        for w in self.workers:
            w.start()
        self.stagger_lock = threading.Lock()
        self.last_start = 0.0
        self.rr = 0
        self.lock = threading.Lock()
        self.total = _new_counters()
        self.window = _new_counters()
        self.submitted = 0
        self.finished = 0
        if cfg.schema not in SCHEMAS:
            raise ValueError(f"judge schema must be one of {sorted(SCHEMAS)}, got {cfg.schema!r}")
        files = {**SCHEMAS[cfg.schema], **(cfg.prompt_files or {})}
        self.prompt_paths = {d: str(prompt_path(f)) for d, f in files.items()}
        self.prompts = {d: prompt_path(f).read_text() for d, f in files.items()}
        self.prompt_sha = {d: sha256_file(prompt_path(f)) for d, f in files.items()}
        self.cache = Path(cfg.cache_dir) if cfg.cache_dir else None
        self.sessions = threading.local()

    def _count(self, fn) -> None:
        with self.lock:
            fn(self.total)
            fn(self.window)

    def reset_window(self) -> dict:
        with self.lock:
            w, self.window = self.window, _new_counters()
            backlog = self.submitted - self.finished
        return self.summarize(w, backlog)

    @staticmethod
    def summarize(c: dict, backlog: int | None = None) -> dict:
        lat = sorted(c["latency"])
        q = lambda p: lat[min(len(lat) - 1, int(p * (len(lat) - 1)))] if lat else None
        n = c["calls_ok"] + c["calls_failed"]
        attempts = max(c["http_attempts"], 1)
        return {"requests": c["requests"], "calls": n, "success_rate": c["calls_ok"] / n if n else None,
                "failed_calls": c["calls_failed"], "failure_rate": c["calls_failed"] / n if n else None,
                "cache_hits": c["cache_hits"], "http_attempts": c["http_attempts"], "retries": c["retries"],
                "errors": dict(c["errors"]), "parse_failures": dict(c["parse_failures"]),
                "parse_failure_rate": (sum(v for k, v in c["parse_failures"].items()
                                           if k not in ("truncated", "inconsistent")) / attempts
                                       if c["http_attempts"] else None),
                "inconsistent_rate": c["parse_failures"].get("inconsistent", 0) / attempts if c["http_attempts"] else None,
                "truncated_rate": c["parse_failures"].get("truncated", 0) / attempts if c["http_attempts"] else None,
                "missing_item_answers": c["parse_failures"].get("missing_items", 0),
                "out_of_range_answers": c["parse_failures"].get("out_of_range", 0),
                "latency_p50": q(0.5), "latency_p95": q(0.95), "latency_max": lat[-1] if lat else None,
                "per_key": dict(c["per_key"]), "providers": dict(c["providers"]),
                "cost_usd": c["cost_usd"], "prompt_tokens": c["prompt_tokens"],
                "completion_tokens": c["completion_tokens"], "backlog": backlog}

    def cumulative(self) -> dict:
        with self.lock:
            return self.summarize(self.total, self.submitted - self.finished)

    def _session(self) -> requests.Session:
        if not hasattr(self.sessions, "s"):
            self.sessions.s = requests.Session()
        return self.sessions.s

    def _pick_route(self, exclude: int | None = None) -> int:
        with self.stagger_lock:
            for _ in range(len(self.slots)):
                ri = self.slots[self.rr % len(self.slots)]
                self.rr += 1
                if ri != exclude or len(self.routes) == 1:
                    return ri
            return self.slots[0]

    def _stagger(self) -> None:
        with self.stagger_lock:
            wait = self.last_start + self.cfg.stagger_seconds - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last_start = time.monotonic()

    def cache_key(self, dimension: str, user: str) -> str:
        payload = json.dumps([self.cfg.model, self.prompt_sha[dimension], self.cfg.extra_body,
                              self.cfg.temperature, user], ensure_ascii=False)
        return sha256_bytes(payload.encode())

    def _cache_path(self, key: str) -> Path | None:
        return self.cache / key[:2] / f"{key}.json" if self.cache else None

    def _post(self, messages: list[dict]) -> tuple[dict, str]:
        last, ri = None, None
        for attempt, backoff in enumerate([0] + list(self.cfg.retry_backoff)):
            if backoff:
                time.sleep(backoff * (0.8 + 0.4 * random.random()))
                self._count(lambda c: c.__setitem__("retries", c["retries"] + 1))
            ri = self._pick_route(exclude=ri)
            route = self.routes[ri]
            body = {"model": route["wire_model"], "messages": messages, **self.cfg.extra_body}
            if self.cfg.temperature is not None:
                body["temperature"] = self.cfg.temperature
            if self.cfg.max_tokens is not None:
                body["max_tokens"] = self.cfg.max_tokens
            if self.cfg.response_format_json:
                body["response_format"] = {"type": "json_object"}
            headers = {"Authorization": f"Bearer {self.keys[route['key_env']]}",
                       "Content-Type": "application/json"}
            with self.sems[ri]:
                self._stagger()
                self._count(lambda c, n=route["name"]: (c.__setitem__("http_attempts", c["http_attempts"] + 1),
                                                         c["per_key"].update([n])))
                try:
                    reply = self._session().post(route["base_url"].rstrip("/") + "/chat/completions",
                                                 json=body, headers=headers, timeout=self.cfg.timeout_seconds)
                except requests.Timeout:
                    last = "timeout"
                    self._count(lambda c: c["errors"].update(["timeout"]))
                    continue
                except requests.ConnectionError:
                    last = "connection"
                    self._count(lambda c: c["errors"].update(["connection"]))
                    continue
            if reply.status_code == 429 or reply.status_code >= 500:
                kind = "http_429" if reply.status_code == 429 else "http_5xx"
                last = kind
                self._count(lambda c: c["errors"].update([kind]))
                continue
            if reply.status_code != 200:
                kind = f"http_{reply.status_code}"
                last = kind
                self._count(lambda c: c["errors"].update([kind]))
                if len(self.routes) > 1:
                    continue
                raise RuntimeError(f"judge HTTP {reply.status_code}: {reply.text[:300]}")
            try:
                data = reply.json()
            except ValueError:
                last = "bad_http_body"
                self._count(lambda c: c["errors"].update(["bad_http_body"]))
                continue
            if "error" in data and not data.get("choices"):
                last = "api_error"
                self._count(lambda c: c["errors"].update(["api_error"]))
                continue
            return data, route["name"]
        raise RuntimeError(f"judge request failed after retries: {last}")

    def call(self, dimension: str, user: str, n_items: int, closed: bool | None = None, context=None) -> dict:
        self._count(lambda c: c.__setitem__("requests", c["requests"] + 1))
        key = self.cache_key(dimension, user)
        path = self._cache_path(key)
        if path and path.is_file():
            cached = json.loads(path.read_text())
            self._count(lambda c: (c.__setitem__("cache_hits", c["cache_hits"] + 1),
                                   c.__setitem__("calls_ok", c["calls_ok"] + 1)))
            return {**cached["parsed"], "cache_hit": True, "cost_usd": 0.0}
        if self.cfg.budget_usd is not None and self.total["cost_usd"] >= self.cfg.budget_usd:
            raise BudgetExceeded(f"judge budget {self.cfg.budget_usd} USD reached")
        messages = [{"role": "system", "content": self.prompts[dimension]},
                    {"role": "user", "content": user}]
        errors, cost_total = [], 0.0
        for _ in range(self.cfg.format_retries + 1):
            started = time.monotonic()
            data, route = self._post(messages)
            elapsed = time.monotonic() - started
            usage = data.get("usage") or {}
            cost = usage.get("cost")
            if cost is None:
                cost = (int(usage.get("prompt_tokens", 0) or 0) * self.cfg.price_input_per_m +
                        int(usage.get("completion_tokens", 0) or 0) * self.cfg.price_output_per_m) / 1e6
            cost_total += float(cost)
            provider = f"{route}:{data.get('provider') or 'unknown'}"

            def acc(c, cost=float(cost), usage=usage, elapsed=elapsed, provider=provider):
                c["cost_usd"] += cost
                c["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
                c["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
                c["latency"].append(round(elapsed, 3))
                c["providers"].update([provider])
            self._count(acc)
            choice = (data.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or ""
            if isinstance(content, list):
                content = "".join(x.get("text", "") for x in content if isinstance(x, dict))
            text = content.strip()
            try:
                if choice.get("finish_reason") == "length":
                    raise JudgeFormatError("answer truncated (finish_reason=length)", "truncated")
                if text.startswith("```"):
                    text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise JudgeFormatError(f"not JSON: {exc}", "json") from exc
                if self.cfg.schema == "v2":
                    from .reward_v2 import parse_judge as parse_v2
                    parsed = parse_v2(dimension, obj, n_items, closed)
                elif self.cfg.schema == "v22":
                    from .reward_v22 import parse_judge as parse_v22
                    parsed = parse_v22(dimension, obj, n_items, closed, context)
                else:
                    parsed = parse_judge(dimension, obj, n_items, closed)
            except JudgeFormatError as exc:
                errors.append(f"{exc.kind}: {str(exc)[:200]}")
                self._count(lambda c, k=exc.kind: c["parse_failures"].update([k]))
                if self.cfg.format_retry_feedback:
                    messages = messages[:2]
                    if exc.kind != "truncated":
                        messages += [{"role": "assistant", "content": content if isinstance(content, str) else text}]
                    messages += [{"role": "user", "content":
                                  f"Your previous answer was rejected by the validator: {str(exc)[:300]}. "
                                  "Reply again with the complete JSON object in exactly the required shape, "
                                  "following every rule in the instructions."}]
                continue
            if path:
                atomic_json(path, {"dimension": dimension, "model": self.cfg.model,
                                   "prompt_sha256": self.prompt_sha[dimension],
                                   "user_sha256": sha256_bytes(user.encode()),
                                   "parsed": parsed, "raw_text": text, "usage": usage,
                                   "provider": provider})
            self._count(lambda c: c.__setitem__("calls_ok", c["calls_ok"] + 1))
            return {**parsed, "cache_hit": False, "cost_usd": cost_total,
                    "format_errors": errors, "provider": provider}
        self._count(lambda c: c.__setitem__("calls_failed", c["calls_failed"] + 1))
        return {"error": "format", "format_errors": errors, "cost_usd": cost_total}

    def submit(self, dimension: str, user: str, n_items: int, closed: bool | None = None, context=None):
        with self.lock:
            self.submitted += 1

        def run():
            try:
                return self.call(dimension, user, n_items, closed) if context is None else \
                    self.call(dimension, user, n_items, closed, context)
            except BudgetExceeded:
                raise
            except Exception as exc:
                self._count(lambda c: c.__setitem__("calls_failed", c["calls_failed"] + 1))
                return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            finally:
                with self.lock:
                    self.finished += 1
        fut: Future = Future()
        priority = -(len(self.prompts[dimension]) + len(user))
        with self.pq_cond:
            heapq.heappush(self.pq, (priority, next(self.pq_seq), fut, run))
            self.pq_cond.notify()
        return fut

    def _worker(self) -> None:
        while True:
            with self.pq_cond:
                while not self.pq and not self.pq_closed:
                    self.pq_cond.wait()
                if not self.pq:
                    return
                _, _, fut, fn = heapq.heappop(self.pq)
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:
                fut.set_exception(exc)

    def queued(self) -> int:
        with self.pq_cond:
            return len(self.pq)

    def close(self) -> None:
        with self.pq_cond:
            self.pq_closed = True
            self.pq_cond.notify_all()
        for w in self.workers:
            w.join()
