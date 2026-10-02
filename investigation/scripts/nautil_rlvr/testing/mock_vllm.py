from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from nautil_rlvr.testing.byte_tokenizer import ByteTokenizer
from nautil_rlvr.testing.synthetic import VARIANTS, scripted_reply

CASE = re.compile(r"# Case brief .{1,3}?(\S+)")
STRAT = re.compile(r"Strategy: (\w+)")
RESP = re.compile(r"<tool_response>\n(.*?)\n</tool_response>", re.S)


class State:
    def __init__(self, args):
        import torch
        from transformers import Qwen3_5ForCausalLM
        torch.set_num_threads(2)
        self.args = args
        self.tok = ByteTokenizer()
        self.base = Qwen3_5ForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
        self.base.eval()
        self.loras: dict[str, tuple] = {}
        self.lock = threading.Lock()
        self.sleeping = False
        self.gen_tokens = 0
        self.skipped_keys: dict[str, int] = {}

    def model_for(self, name: str):
        if name == self.args.served_name:
            return self.base
        return self.loras[name][0]

    def load_lora(self, name: str, path: str) -> None:
        import torch
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        cfg = json.loads((Path(path) / "adapter_config.json").read_text())
        tensors = load_file(str(Path(path) / "adapter_model.safetensors"))
        want = "base_model.model.model.language_model.layers."
        mapped, skipped = {}, 0
        for k, v in tensors.items():
            if k.startswith(want):
                mapped["base_model.model.model.layers." + k[len(want):]] = v
            else:
                skipped += 1
        model = get_peft_model(copy.deepcopy(self.base),
                               LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"],
                                          target_modules=cfg["target_modules"], lora_dropout=0.0,
                                          bias="none", task_type="CAUSAL_LM"))
        if mapped:
            set_peft_model_state_dict(model, mapped)
        model.eval()
        with self.lock:
            self.loras[name] = (model, path)
            self.skipped_keys[name] = skipped

    def logprobs(self, model, ids: list[int], start: int) -> list[float]:
        import torch
        with self.lock, torch.no_grad():
            t = torch.tensor([ids])
            logits = model(input_ids=t).logits[0].float()
            lp = torch.log_softmax(logits, -1)
            return [float(lp[p - 1, ids[p]]) for p in range(start, len(ids))]


def scripted(text: str, seed: int) -> str:
    brief = text.split("<|im_start|>user\n", 1)[1]
    returned = RESP.findall(text)
    texts = []
    for r in returned:
        try:
            texts += [i["text"] for i in json.loads(r).get("evidence_items", [])]
        except json.JSONDecodeError:
            pass
    turn = text.count("<|im_start|>assistant")
    m = STRAT.search(text)
    variant = m.group(1) if m else VARIANTS[seed % len(VARIANTS)]
    del brief
    return scripted_reply(variant, turn, texts, None, random.Random(seed))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        return

    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        st: State = self.server.state
        if self.path.rstrip("/") == "/v1/models":
            data = [{"id": st.args.served_name, "object": "model", "root": st.args.model_dir, "parent": None,
                     "max_model_len": 32768}]
            data += [{"id": n, "object": "model", "root": p, "parent": st.args.served_name, "max_model_len": 32768}
                     for n, (_, p) in st.loras.items()]
            return self._send(200, {"object": "list", "data": data})
        if self.path == "/is_sleeping":
            return self._send(200, {"is_sleeping": st.sleeping})
        if self.path == "/metrics":
            return self._send(200, f'vllm:generation_tokens_total{{model_name="m"}} {st.gen_tokens}\n'.encode(), "text/plain")
        if self.path == "/mock/skipped":
            return self._send(200, st.skipped_keys)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        st: State = self.server.state
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        path = self.path.split("?")[0]
        if path == "/sleep":
            st.sleeping = True
            return self._send(200, {})
        if path == "/wake_up":
            st.sleeping = False
            return self._send(200, {})
        if st.sleeping:
            return self._send(503, {"error": "asleep"})
        if path == "/v1/load_lora_adapter":
            if body["lora_name"] in st.loras:
                return self._send(400, {"error": "already loaded"})
            st.load_lora(body["lora_name"], body["lora_path"])
            return self._send(200, {"ok": True})
        if path == "/v1/unload_lora_adapter":
            st.loras.pop(body["lora_name"], None)
            return self._send(200, {"ok": True})
        if path == "/v1/completions":
            return self.completions(st, body)
        return self._send(404, {"error": "not found"})

    def completions(self, st: State, body: dict):
        name = body["model"]
        if name != st.args.served_name and name not in st.loras:
            return self._send(404, {"error": f"model {name} not found"})
        ids = list(body["prompt"])
        model = st.model_for(name)
        if body.get("prompt_logprobs") is not None:
            lps = st.logprobs(model, ids, 1)
            entries = [None] + [{str(ids[p]): {"logprob": lps[p - 1], "rank": 1}} for p in range(1, len(ids))]
            return self._send(200, {"choices": [{"index": 0, "text": "", "token_ids": [], "finish_reason": "length",
                                                 "prompt_logprobs": entries}]})
        bytes3 = any(i >= 300 for i in ids)
        if bytes3:
            sys.path.insert(0, str(HERE.parents[1]))
            from nautil_scheduler.testing.fake_tokenizer import decode_ids, encode_text
            text = decode_ids(ids)
            reply = scripted(text.replace("Strategy:", "Strat:"), 0)
            out = encode_text(reply)
            end = 2
        else:
            text = st.tok.decode(ids)
            reply = scripted(text, int(body.get("seed", 0)))
            out = st.tok("" + reply)["input_ids"]
            end = st.tok.convert_tokens_to_ids("<|im_end|>")
        limit = int(body["max_tokens"])
        if len(out) + 1 > limit:
            out, finish = out[:limit], "length"
        else:
            out, finish = out + [end], "stop"
        st.gen_tokens += len(out)
        choice = {"index": 0, "text": "", "token_ids": out, "finish_reason": finish,
                  "prompt_token_ids": ids if body.get("return_token_ids") else None}
        if body.get("logprobs") is not None and not bytes3:
            choice["logprobs"] = {"token_logprobs": st.logprobs(model, ids + out, len(ids))}
        return self._send(200, {"choices": [choice], "usage": {"prompt_tokens": len(ids),
                                                               "completion_tokens": len(out)}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--served-name", default="mock-base")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.state = State(args)
    print(json.dumps({"mock_vllm": args.port}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
