from __future__ import annotations

import copy
import threading
import time

import requests

from nautil_harness_v2_2 import run_case

from .common import sha256_ids

_local = threading.local()
_tok_lock = threading.Lock()


def session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def stable_seed(case_id: str, turn: int, base: int) -> int:
    import hashlib
    payload = f"{base}:{case_id}:{turn}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def render_prompt_ids(tokenizer, messages, tools) -> list[int]:
    with _tok_lock:
        rendered = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        return list(tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _parse_logprobs(choice: dict, n: int) -> list[float] | None:
    lp = choice.get("logprobs")
    if not lp:
        return None
    if isinstance(lp, dict) and lp.get("token_logprobs") is not None:
        vals = lp["token_logprobs"]
    elif isinstance(lp, dict) and lp.get("content") is not None:
        vals = [c.get("logprob") for c in lp["content"]]
    else:
        return None
    vals = [float(v) if v is not None else None for v in vals]
    return vals if len(vals) == n else None


def generate_turn(endpoint: str, served_model: str, tokenizer, end_id: int, messages, tools,
                  protocol: dict, case_id: str, turn: int, seed_base: int,
                  remaining_case_tokens: int, remaining_seconds: float,
                  want_logprobs: bool = True):
    prompt_ids = render_prompt_ids(tokenizer, messages, tools)
    prompt_tokens = len(prompt_ids)
    context_limit = protocol["context_limit"]
    base_info = {"prompt_len": prompt_tokens, "prompt_sha": sha256_ids(prompt_ids),
                 "n_messages": len(messages), "gen_ids": [], "gen_logprobs": None}
    if prompt_tokens >= context_limit:
        return "", prompt_tokens, True, {"generated_tokens": 0, "seconds": 0.0,
                                          "finish_reason": "context_limit", "rl": base_info}
    if remaining_case_tokens <= 0:
        return "", prompt_tokens, True, {"generated_tokens": 0, "seconds": 0.0,
                                          "finish_reason": "case_token_budget", "rl": base_info}
    limit = min(protocol["max_tokens_per_turn"], remaining_case_tokens, context_limit - prompt_tokens)
    sampling = protocol["sampling"]
    body = {"model": served_model, "prompt": prompt_ids, "max_tokens": limit,
            "temperature": sampling["temperature"], "top_p": sampling["top_p"],
            "top_k": sampling["top_k"], "min_p": sampling["min_p"],
            "presence_penalty": sampling["presence_penalty"],
            "repetition_penalty": sampling["repetition_penalty"],
            "seed": stable_seed(case_id, turn, seed_base),
            "stop_token_ids": [end_id], "skip_special_tokens": False,
            "ignore_eos": False, "return_token_ids": True}
    if want_logprobs:
        body["logprobs"] = 0
    started = time.perf_counter()
    response = session().post(f"{endpoint}/v1/completions", json=body,
                              timeout=max(remaining_seconds, 0) + 1800)
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    if choice.get("prompt_token_ids") not in (None, prompt_ids):
        raise ValueError("server changed prompt token IDs")
    raw_ids = list(choice["token_ids"])
    logprobs = _parse_logprobs(choice, len(raw_ids)) if want_logprobs else None
    if want_logprobs and logprobs is None:
        raise ValueError("server returned no per-token logprobs aligned with token_ids")
    stopped = choice["finish_reason"] == "stop"
    generated = list(raw_ids)
    if stopped and generated and generated[-1] == end_id:
        generated.pop()
    if choice["finish_reason"] not in ("stop", "length"):
        raise RuntimeError(f"unexpected finish reason {choice['finish_reason']}")
    with _tok_lock:
        text = tokenizer.decode(generated, skip_special_tokens=False)
    if stopped:
        reason = "stop"
    elif limit == remaining_case_tokens:
        reason = "case_token_budget"
    elif limit == context_limit - prompt_tokens:
        reason = "context_limit"
    else:
        reason = "turn_token_budget"
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    rl = dict(base_info)
    rl["gen_ids"] = raw_ids
    rl["gen_logprobs"] = logprobs
    rl["stopped_with_end"] = bool(stopped and raw_ids and raw_ids[-1] == end_id)
    metrics = {"generated_tokens": len(generated),
               "seconds": round(time.perf_counter() - started, 3),
               "finish_reason": reason, "cached_prompt_tokens": details.get("cached_tokens"), "rl": rl}
    return text, prompt_tokens, not stopped, metrics


def run_trajectory(endpoint: str, served_model: str, tokenizer, end_id: int, system: str,
                   case: dict, protocol: dict, seed_base: int, want_logprobs: bool = True,
                   abort=None) -> dict:
    turn_no, tokens = 0, 0
    started = time.perf_counter()
    captured = {"messages": None}
    turn_rl: list[dict] = []

    def one_turn(model, tok, messages, tools, device, context_limit):
        nonlocal turn_no, tokens
        captured["messages"] = messages
        turn_no += 1
        if abort is not None and abort.is_set():
            turn_rl.append({"prompt_len": None, "gen_ids": [], "gen_logprobs": None,
                            "n_messages": len(messages), "aborted": True})
            return "", 0, True, {"generated_tokens": 0, "seconds": 0, "finish_reason": "aborted"}
        elapsed = time.perf_counter() - started
        if elapsed >= protocol["max_case_seconds"]:
            turn_rl.append({"prompt_len": None, "gen_ids": [], "gen_logprobs": None,
                            "n_messages": len(messages)})
            return "", 0, True, {"generated_tokens": 0, "seconds": 0, "finish_reason": "wall_budget"}
        result = generate_turn(endpoint, served_model, tokenizer, end_id, messages, tools, protocol,
                               case["case_id"], turn_no, seed_base,
                               protocol["max_case_generated_tokens"] - tokens,
                               protocol["max_case_seconds"] - elapsed, want_logprobs)
        tokens += result[3]["generated_tokens"]
        turn_rl.append(result[3].pop("rl"))
        return result

    result = run_case(None, tokenizer, system, case, None, protocol["context_limit"],
                      protocol["max_turns"], generate_fn=one_turn)
    messages = copy.deepcopy(captured["messages"]) if captured["messages"] is not None else None
    return {"case_id": case["case_id"], "source": case["source"], "result": result,
            "messages": messages, "turn_rl": turn_rl, "served_model": served_model,
            "endpoint": endpoint, "seed_base": seed_base, "generated_tokens": tokens,
            "runtime_error": result["status"] == "runtime_error",
            "aborted": any(t.get("aborted") for t in turn_rl)}
