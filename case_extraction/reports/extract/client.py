from __future__ import annotations

import json
import re
import ssl
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from nautil_common import paths

ENV_FILE = paths.ENV_FILE

ENDPOINTS = {
    "api": {"base": "https://example.com/v1", "key_var": "API_KEY"},
}
RETRY_ON = ("http_403", "http_429", "http_500", "http_502", "http_503", "http_504", "http_524",
            "connect_error", "remote_protocol_error", "ssl_eof", "read_timeout")
BACKOFF = (10.0, 30.0, 60.0)
STOP = threading.Event()


def read_key(var: str) -> str:
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{var}="):
            value = line.split("=", 1)[1].strip()
            if value:
                return value
    raise RuntimeError(f"{var} is absent from the key file")


def classify(exc: BaseException) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"http_{status}"
    text = str(exc)
    if isinstance(exc, (httpx.ReadTimeout, httpx.PoolTimeout, httpx.TimeoutException)):
        return "read_timeout"
    if isinstance(exc, httpx.RemoteProtocolError):
        return "remote_protocol_error"
    if isinstance(exc, ssl.SSLError) or "SSL" in text:
        return "ssl_eof"
    if isinstance(exc, httpx.TransportError):
        return "connect_error"
    return "other"


FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


def parse_json(text: str) -> tuple[Any, bool]:
    body = FENCE.sub("", (text or "").strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = body.find(opener), body.rfind(closer)
        if i == -1:
            continue
        if j <= i:
            salvaged = _close_open_json(body[i:])
            if salvaged is not None:
                return salvaged, True
            continue
        inner = body[i:j + 1]
        try:
            return json.loads(inner), False
        except Exception:
            fixed = re.sub(r'"\s*;\s*(\n\s*)?,', '",', inner)
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            fixed = re.sub(r'"\s*;\s*\n', '"\n', fixed)
            try:
                return json.loads(fixed), True
            except Exception:
                pass
            try:
                return json.loads(_escape_inner_quotes(fixed)), True
            except Exception:
                pass
            salvaged = _close_open_json(inner)
            if salvaged is not None:
                return salvaged, True
    return None, False


def _close_open_json(text: str) -> Any:
    stack: list[str] = []
    in_string = False
    escaped = False
    last_safe = 0
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if in_string:
            if char == '"':
                in_string = False
                last_safe = index + 1
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if stack:
                stack.pop()
            last_safe = index + 1
        elif char == ",":
            last_safe = index
    body = text[:last_safe].rstrip().rstrip(",")
    for _ in range(len(stack) + 2):
        for closing in ("", "}", "]", "}]", "]}", "}}", "]]", "}]}", "]}]"):
            try:
                value = json.loads(body + closing)
                if isinstance(value, (dict, list)):
                    return value
            except Exception:
                continue
        body = body[: body.rfind(",")].rstrip() if "," in body else body
        if not body:
            break
    return None


def _escape_inner_quotes(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            out.append(char)
            escaped = False
            continue
        if char == "\\":
            out.append(char)
            escaped = True
            continue
        if char == '"':
            if not in_string:
                in_string = True
                out.append(char)
                continue
            following = text[index + 1: index + 40].lstrip()
            previous = "".join(out[-40:]).rstrip()
            structural = (following[:1] in {",", "}", "]", ":"} or following == ""
                          or previous.endswith(("{", "[", ",", ":")))
            if structural:
                in_string = False
                out.append(char)
            else:
                out.append("'")
            continue
        out.append(char)
    return "".join(out)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    lock: Any = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: dict[str, Any], price_in: float, price_out: float) -> None:
        with self.lock:
            p = int(usage.get("prompt_tokens") or 0)
            c = int(usage.get("completion_tokens") or 0)
            d = usage.get("completion_tokens_details") or {}
            self.prompt_tokens += p
            self.completion_tokens += c
            self.reasoning_tokens += int(d.get("reasoning_tokens") or 0)
            self.calls += 1
            reported = usage.get("cost")
            self.cost_usd += float(reported) if reported is not None else (p * price_in + c * price_out) / 1e6

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens, "reasoning_tokens": self.reasoning_tokens,
                    "cost_usd": round(self.cost_usd, 4)}


class Client:
    def __init__(self, config: dict[str, Any]):
        self.endpoint_name = config["endpoint"]
        endpoint = ENDPOINTS[self.endpoint_name]
        self.base = endpoint["base"]
        self.api_key = read_key(endpoint["key_var"])
        self.models = config["models"]
        self.timeout = float(config.get("http_timeout_seconds", 900))
        self.max_attempts = int(config.get("max_attempts", 3))
        self.usage = Usage()
        self.http = httpx.Client(timeout=self.timeout,
                                 limits=httpx.Limits(max_connections=128, max_keepalive_connections=64))

    def close(self) -> None:
        self.http.close()

    def complete(self, *, role: str, prompt: str, max_tokens: int = 16000,
                 images: list[str] | None = None, expect: str = "json") -> dict[str, Any]:
        model = self.models[role]
        content: Any = prompt
        if images:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": url}} for url in images]
        body: dict[str, Any] = {"model": model["id"],
                                "messages": [{"role": "user", "content": content}],
                                "max_tokens": max_tokens}
        if model.get("reasoning_effort"):
            body["reasoning_effort"] = model["reasoning_effort"]
        attempts: list[dict[str, Any]] = []
        for attempt in range(self.max_attempts + 1):
            if STOP.is_set():
                return {"ok": False, "error": "interrupted", "attempts": attempts}
            t0 = time.monotonic()
            try:
                response = self.http.post(f"{self.base}/chat/completions",
                                          headers={"Authorization": f"Bearer {self.api_key}",
                                                   "Content-Type": "application/json"},
                                          json=body)
                response.raise_for_status()
                data = response.json()
                if "error" in data and not data.get("choices"):
                    raise RuntimeError(f"provider error: {json.dumps(data['error'])[:200]}")
                choice = (data.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                self.usage.add(usage, model.get("price_in", 0.0), model.get("price_out", 0.0))
                attempts.append({"attempt": attempt, "seconds": round(time.monotonic() - t0, 2), "outcome": "ok"})
                if choice.get("finish_reason") == "length":
                    return {"ok": False, "error": "finish_reason=length (output cap hit)", "raw": text,
                            "usage": usage, "attempts": attempts}
                if expect == "text":
                    return {"ok": bool(text.strip()), "text": text, "raw": text, "usage": usage, "attempts": attempts}
                value, repaired = parse_json(text)
                if value is None:
                    return {"ok": False, "error": "unparseable JSON", "raw": text, "usage": usage, "attempts": attempts}
                return {"ok": True, "value": value, "repaired": repaired, "raw": text,
                        "usage": usage, "attempts": attempts}
            except BaseException as exc:
                category = classify(exc)
                attempts.append({"attempt": attempt, "seconds": round(time.monotonic() - t0, 2),
                                 "outcome": f"failed: {category}", "error": str(exc)[:300]})
                if category not in RETRY_ON or attempt >= self.max_attempts or STOP.is_set():
                    return {"ok": False, "error": f"{category}: {str(exc)[:300]}", "attempts": attempts}
                STOP.wait(BACKOFF[min(attempt, len(BACKOFF) - 1)])
        return {"ok": False, "error": "retries exhausted", "attempts": attempts}
