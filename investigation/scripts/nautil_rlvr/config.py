from __future__ import annotations

import json
import os
from pathlib import Path

from .common import looks_like_test_data, sha256_file
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repository root
from nautil_common import paths

FIXED_PROTOCOL = {"max_turns": 15, "context_limit": 32768, "max_tokens_per_turn": 4096,
                  "max_case_generated_tokens": 12000, "max_case_seconds": 900}
FIXED_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
                  "presence_penalty": 1.5, "repetition_penalty": 1.0}
FIXED_RL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                     "presence_penalty": 0.0, "repetition_penalty": 1.0}
FORBIDDEN_KEYS = {"sampling", "temperature", "top_p", "top_k", "max_turns", "context_limit",
                  "max_tokens_per_turn", "max_case_generated_tokens", "seed_override"}


class ConfigError(ValueError):
    pass


def _walk_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k, f"{path}.{k}"
            yield from _walk_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_keys(v, f"{path}[{i}]")


def load(path: Path, overrides: dict | None = None) -> dict:
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    for key, value in (overrides or {}).items():
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    base = path.parent

    def resolve(v):
        if v is None or not isinstance(v, str) or v.startswith(("http://", "https://", "mock:")):
            return v
        p = Path(os.path.expandvars(os.path.expanduser(v)))
        return str(p if p.is_absolute() else (base / p).resolve())

    for key in list(cfg.get("paths", {})):
        cfg["paths"][key] = resolve(cfg["paths"][key])
    for key in ("scheduler_scripts_dir",):
        if key in cfg.get("eval", {}):
            cfg["eval"][key] = resolve(cfg["eval"][key])
    for name in ("train", "eval"):
        j = cfg.get("judges", {}).get(name)
        if j and j.get("cache_dir"):
            j["cache_dir"] = resolve(j["cache_dir"])
        if j and j.get("env_file"):
            j["env_file"] = resolve(j["env_file"])
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = sha256_file(path)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    bad = [p for k, p in _walk_keys({k: v for k, v in cfg.items() if not k.startswith("_")})
           if k in FORBIDDEN_KEYS and not p.startswith(".judges")]
    if bad:
        raise ConfigError(f"generation settings may only come from the protocol file: {bad}")
    for key, value in cfg["paths"].items():
        if isinstance(value, str) and looks_like_test_data(value):
            raise ConfigError(f"paths.{key} looks like test-set data: {value}")
    for key, sampling in (("rollout_protocol", FIXED_RL_SAMPLING), ("eval_protocol", FIXED_SAMPLING)):
        protocol = json.loads(Path(cfg["paths"][key]).read_text())
        for k, v in FIXED_PROTOCOL.items():
            if protocol.get(k) != v:
                raise ConfigError(f"{key} {k}={protocol.get(k)} differs from the fixed value {v}")
        if protocol.get("sampling") != sampling or protocol.get("mode") != "non_thinking":
            raise ConfigError(f"{key} sampling / mode differ from the fixed values {sampling}")
    mode = cfg.get("pipeline", {}).get("mode", "sync")
    if mode not in ("sync", "async"):
        raise ConfigError("pipeline.mode must be sync or async")
    if mode == "async" and cfg.get("pipeline", {}).get("colocate"):
        raise ConfigError("colocation (vLLM sleep during training) only works in sync mode")
    g = cfg["grpo"]
    if g["group_size"] < 2 or g["cases_per_step"] < 1:
        raise ConfigError("group_size >= 2 and cases_per_step >= 1 required")
    if not 0 <= g.get("max_staleness", 1) <= 4:
        raise ConfigError("max_staleness must be 0..4")
    if cfg["reward"]["phase"] == 2 and "train" not in cfg.get("judges", {}):
        raise ConfigError("phase 2 needs judges.train")
    version = str(cfg["reward"].get("version", 1))
    schema = cfg.get("judges", {}).get("train", {}).get("schema", "v1")
    need = {"1": "v1", "2": "v2", "2.2": "v22"}
    if version not in need:
        raise ConfigError(f"reward.version must be one of {sorted(need)}")
    if version != "1":
        if cfg["reward"]["phase"] != 2 or schema != need[version]:
            raise ConfigError(f"reward.version {version} needs reward.phase 2 and judges.train.schema {need[version]!r}")
        if cfg["reward"].get("variant", "full") not in ("full", "outcome", "decision"):
            raise ConfigError("reward.variant must be full, outcome or decision")
    elif "train" in cfg.get("judges", {}) and schema != "v1":
        raise ConfigError(f"judges.train.schema {schema!r} needs a matching reward.version")
    if cfg.get("sampler", {}).get("extra_sources", {}).get("counterfactual_pairs", {}).get("enabled"):
        raise ConfigError("counterfactual pairs are not built yet (keep the flag off)")
