from __future__ import annotations

import json
import re

SPECIALS = {"<|im_start|>": 1, "<|im_end|>": 2, "<|endoftext|>": 3}
BY_ID = {v: k for k, v in SPECIALS.items()}
SPLIT = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")
VOCAB = 300


def char_id(ch: str) -> int:
    o = ord(ch)
    return 10 + o if o < 128 else 140 + (o % 150)


def id_char(i: int) -> str:
    if 10 <= i < 138:
        return chr(i - 10)
    return "?"


def _render_call(call: dict) -> str:
    fn = call.get("function", call)
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        args = json.loads(args)
    out = f"<tool_call>\n<function={fn['name']}>\n"
    for k, v in args.items():
        v = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
        out += f"<parameter={k}>\n{v}\n</parameter>\n"
    return out + "</function>\n</tool_call>"


class ByteTokenizer:
    name_or_path = "mock:bytes"
    vocab_size = VOCAB

    def convert_tokens_to_ids(self, token: str) -> int:
        return SPECIALS.get(token, -1)

    def encode_with_offsets(self, text: str):
        ids, offs, pos = [], [], 0
        for piece in SPLIT.split(text):
            if not piece:
                continue
            if piece in SPECIALS:
                ids.append(SPECIALS[piece])
                offs.append((pos, pos + len(piece)))
            else:
                for j, ch in enumerate(piece):
                    ids.append(char_id(ch))
                    offs.append((pos + j, pos + j + 1))
            pos += len(piece)
        return ids, offs

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None):
        ids, offs = self.encode_with_offsets(text)
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def decode(self, ids, skip_special_tokens=False) -> str:
        out = []
        for i in ids:
            if i in BY_ID:
                if not skip_special_tokens:
                    out.append(BY_ID[i])
            else:
                out.append(id_char(int(i)))
        return "".join(out)

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False,
                            enable_thinking=None, return_dict=False, **_):
        out = []
        sys_content = messages[0]["content"].strip() if messages and messages[0]["role"] == "system" else None
        if tools:
            body = "# Tools\n\n<tools>" + "".join("\n" + json.dumps(t, ensure_ascii=False) for t in tools) + "\n</tools>"
            if sys_content:
                body += "\n\n" + sys_content
            out.append(f"<|im_start|>system\n{body}<|im_end|>\n")
        elif sys_content is not None:
            out.append(f"<|im_start|>system\n{sys_content}<|im_end|>\n")
        last_query = max(i for i, m in enumerate(messages) if m["role"] == "user")
        for i, m in enumerate(messages):
            content = (m.get("content") or "").strip()
            if m["role"] == "system":
                continue
            if m["role"] == "user":
                out.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif m["role"] == "assistant":
                head = "<|im_start|>assistant\n" + ("<think>\n\n</think>\n\n" if i > last_query else "")
                body = content
                for n, call in enumerate(m.get("tool_calls") or []):
                    sep = ("\n\n" if content else "") if n == 0 else "\n"
                    body += sep + _render_call(call)
                out.append(head + body + "<|im_end|>\n")
            elif m["role"] == "tool":
                prev = messages[i - 1]["role"] if i else None
                if prev != "tool":
                    out.append("<|im_start|>user")
                out.append(f"\n<tool_response>\n{content}\n</tool_response>")
                nxt = messages[i + 1]["role"] if i + 1 < len(messages) else None
                if nxt != "tool":
                    out.append("<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
            out.append("<think>\n\n</think>\n\n" if enable_thinking is False else "<think>\n")
        text = "".join(out)
        if not tokenize:
            return text
        ids = self.encode_with_offsets(text)[0]
        return {"input_ids": ids} if return_dict else ids
