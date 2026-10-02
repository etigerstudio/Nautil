from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from .grpo import clipped_token_loss


def _chunk_logp(lm_head, hidden, targets):
    logits = lm_head(hidden).float()
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        entropy = -(logp_all.exp() * logp_all).sum(-1)
    return logp, entropy


def _dist():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() else None


class Policy:
    def __init__(self, model_dir: Path, init_adapter: Path, device: str = "cuda:0",
                 dtype: str = "bfloat16", lr: float = 5e-6, weight_decay: float = 0.0,
                 betas=(0.9, 0.999), eps: float = 1e-8, max_grad_norm: float = 1.0,
                 ref_adapter: Path | None = None, chunk_size: int = 2048,
                 expect_rank: int | None = 16, expect_alpha: int | None = 32,
                 gradient_checkpointing: bool = True, attn_implementation: str = "sdpa",
                 pack_tokens: int = 32768):
        import transformers
        from peft import PeftModel
        from transformers import AutoConfig, Qwen3_5ForCausalLM
        transformers.logging.set_verbosity_error()
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        cfg_raw = json.loads((Path(init_adapter) / "adapter_config.json").read_text())
        if expect_rank is not None and (cfg_raw.get("r") != expect_rank or
                                        cfg_raw.get("lora_alpha") != expect_alpha):
            raise ValueError(f"adapter rank/alpha {cfg_raw.get('r')}/{cfg_raw.get('lora_alpha')} "
                             f"!= expected {expect_rank}/{expect_alpha}")
        config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
        config = getattr(config, "text_config", config)
        base, info = Qwen3_5ForCausalLM.from_pretrained(
            str(model_dir), config=config, dtype=self.dtype,
            device_map={"": self.device.index if self.device.type == "cuda" else "cpu"},
            low_cpu_mem_usage=True, local_files_only=True, output_loading_info=True,
            attn_implementation=attn_implementation)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")):
            raise ValueError("base checkpoint did not map cleanly to the text backbone")
        base.config.use_cache = False
        self.model = PeftModel.from_pretrained(base, str(init_adapter), is_trainable=True)
        self.has_ref = ref_adapter is not None
        if self.has_ref:
            self.model.load_adapter(str(ref_adapter), adapter_name="ref", is_trainable=False)
            self.model.set_adapter("default")
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.model.train()
        self._freeze_ref()
        self.trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.trainable_names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if not self.trainable or any(".ref." in n for n in self.trainable_names):
            raise ValueError("unexpected trainable parameter set")
        self.optimizer = torch.optim.AdamW(self.trainable, lr=lr, betas=tuple(betas), eps=eps,
                                           weight_decay=weight_decay)
        self.max_grad_norm = max_grad_norm
        self.chunk_size = chunk_size
        self.pack_tokens = pack_tokens
        self.packing = False
        causal = self.model.base_model.model
        self.backbone, self.lm_head = causal.model, causal.lm_head

    def _freeze_ref(self) -> None:
        for name, p in self.model.named_parameters():
            if ".ref." in name:
                p.requires_grad_(False)

    def _forward_rows(self, seqs: list[dict], grad: bool, packed: bool):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        outs = []
        with ctx:
            if packed and len(seqs) > 1:
                ids = torch.tensor([sum((s["input_ids"] for s in seqs), [])], dtype=torch.long, device=self.device)
                pos = torch.cat([torch.arange(len(s["input_ids"]), device=self.device) for s in seqs])[None]
                lens = torch.tensor([len(s["input_ids"]) for s in seqs], device=self.device)
                cu = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=self.device)
                cu[1:] = torch.cumsum(lens, 0)
                mx = int(lens.max())
                hidden = self.backbone(input_ids=ids, position_ids=pos, use_cache=False,
                                       cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=mx,
                                       max_length_k=mx).last_hidden_state[0]
                offsets = [0] + [int(x) for x in torch.cumsum(lens, 0)[:-1]]
            else:
                hidden_list = []
                for s in seqs:
                    ids = torch.tensor([s["input_ids"]], dtype=torch.long, device=self.device)
                    hidden_list.append(self.backbone(input_ids=ids, attention_mask=torch.ones_like(ids),
                                                     use_cache=False).last_hidden_state[0])
            for n, s in enumerate(seqs):
                mask = torch.tensor(s["loss_mask"], dtype=torch.bool, device=self.device)
                positions = torch.nonzero(mask[1:]).flatten()
                ids_t = torch.tensor(s["input_ids"], dtype=torch.long, device=self.device)
                targets = ids_t[positions + 1]
                if packed and len(seqs) > 1:
                    hs = hidden.index_select(0, positions + offsets[n])
                else:
                    hs = hidden_list[n].index_select(0, positions)
                lps, ents = [], []
                for start in range(0, len(positions), self.chunk_size):
                    h, t = hs[start:start + self.chunk_size], targets[start:start + self.chunk_size]
                    if grad:
                        lp, ent = checkpoint(_chunk_logp, self.lm_head, h, t, use_reentrant=False)
                    else:
                        lp, ent = _chunk_logp(self.lm_head, h, t)
                    lps.append(lp)
                    ents.append(ent)
                outs.append((torch.cat(lps) if lps else torch.zeros(0, device=self.device),
                             torch.cat(ents) if ents else torch.zeros(0, device=self.device), positions))
        return outs

    def target_logprobs(self, input_ids, loss_mask, grad: bool):
        return self._forward_rows([{"input_ids": input_ids, "loss_mask": loss_mask}], grad, False)[0]

    def ref_logprobs(self, seq: dict):
        self.model.set_adapter("ref")
        try:
            lp, _, _ = self.target_logprobs(seq["input_ids"], seq["loss_mask"], grad=False)
        finally:
            self.model.set_adapter("default")
            self._freeze_ref()
        return lp

    def probe(self, input_ids, loss_mask) -> dict:
        was = self.model.training
        self.model.eval()
        lp, _, _ = self.target_logprobs(input_ids, loss_mask, grad=False)
        if was:
            self.model.train()
        return {"loss": float(-lp.double().mean().item()), "tokens": int(lp.numel()),
                "logprobs": [float(x) for x in lp.float().cpu()]}

    def packing_equivalence(self, seqs: list[dict], tol_max: float = 0.05, tol_mean: float = 2e-3,
                            tol_hidden: float = 1e-2) -> dict:
        was = self.model.training
        self.model.eval()
        try:
            single = self._forward_rows(seqs, False, False)
            packed = self._forward_rows(seqs, False, True)
            diffs = torch.cat([(a[0] - b[0]).abs().float() for a, b in zip(single, packed)])
            out = {"max_abs": float(diffs.max()), "mean_abs": float(diffs.mean()), "tokens": int(diffs.numel())}
            with torch.no_grad():
                hs = [self.backbone(input_ids=torch.tensor([s["input_ids"]], device=self.device),
                                    use_cache=False).last_hidden_state[0].float() for s in seqs]
                ids = torch.tensor([sum((s["input_ids"] for s in seqs), [])], device=self.device)
                pos = torch.cat([torch.arange(len(s["input_ids"]), device=self.device) for s in seqs])[None]
                lens = torch.tensor([len(s["input_ids"]) for s in seqs], device=self.device)
                cu = torch.zeros(len(seqs) + 1, dtype=torch.int32, device=self.device)
                cu[1:] = torch.cumsum(lens, 0)
                hp = self.backbone(input_ids=ids, position_ids=pos, use_cache=False, cu_seq_lens_q=cu,
                                   cu_seq_lens_k=cu, max_length_q=int(lens.max()),
                                   max_length_k=int(lens.max())).last_hidden_state[0].float()
                ref = torch.cat(hs)
                out["hidden_rel_mean_abs"] = float((hp - ref).abs().mean() / ref.abs().mean())
        except Exception as exc:
            out = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        finally:
            if was:
                self.model.train()
        out["passed"] = ("error" not in out and out["max_abs"] <= tol_max and out["mean_abs"] <= tol_mean
                         and out["hidden_rel_mean_abs"] <= tol_hidden)
        return out

    def kl_to_ref(self, seqs: list[dict]) -> dict | None:
        if not self.has_ref or not seqs:
            return None
        tot, n = 0.0, 0
        was = self.model.training
        self.model.eval()
        for s in seqs:
            ref = self.ref_logprobs(s)
            cur, _, _ = self.target_logprobs(s["input_ids"], s["loss_mask"], grad=False)
            diff = (ref - cur).double()
            tot += float((torch.exp(diff) - diff - 1).sum())
            n += int(diff.numel())
        if was:
            self.model.train()
        return {"kl_to_e2": tot / max(n, 1), "tokens": n}

    def _micro_batches(self, items: list[dict]) -> list[list[dict]]:
        if not self.packing:
            return [[it] for it in items]
        rows, cur, size = [], [], 0
        for it in sorted(items, key=lambda x: -len(x["input_ids"])):
            n = len(it["input_ids"])
            if cur and size + n > self.pack_tokens:
                rows.append(cur)
                cur, size = [], 0
            cur.append(it)
            size += n
        if cur:
            rows.append(cur)
        return rows

    def _allreduce_grads(self) -> None:
        dist = _dist()
        if dist is None or dist.get_world_size() == 1:
            return
        grads = []
        for p in self.trainable:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat)
        for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)

    def _global_sum(self, values: list[float], op: str = "sum") -> list[float]:
        dist = _dist()
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        if dist is not None and dist.get_world_size() > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
        return [float(x) for x in t.cpu()]

    def update(self, minibatches: list[list[dict]], clip_low: float, clip_high: float,
               kl_coef: float = 0.0, lr: float | None = None) -> dict:
        if lr is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        keys = ["loss", "tokens", "clipped", "ratio_sum", "kl_b", "absdiff", "entropy", "kl_ref", "seq_tokens", "seqs"]
        agg = dict.fromkeys(keys, 0.0)
        first = None
        ratio_max = 0.0
        norms = []
        began = time.perf_counter()
        for m, items in enumerate(minibatches):
            items = [it for it in items if sum(it["loss_mask"][1:]) > 0]
            local_tokens = float(sum(sum(it["loss_mask"][1:]) for it in items))
            total = self._global_sum([local_tokens])[0]
            self.optimizer.zero_grad(set_to_none=True)
            mb = dict.fromkeys(keys, 0.0)
            for row in self._micro_batches(items):
                refs = [self.ref_logprobs(it) if (self.has_ref and kl_coef > 0) else None for it in row]
                outs = self._forward_rows(row, grad=True, packed=self.packing)
                loss_row = 0.0
                for it, (logp, ent, positions), ref in zip(row, outs, refs):
                    old = torch.tensor(it["behaviour_logprobs"], dtype=torch.float32,
                                       device=self.device)[positions + 1]
                    loss_tok, st = clipped_token_loss(logp, old, float(it["advantage"]), clip_low,
                                                      clip_high, ref, kl_coef)
                    loss_row = loss_row + loss_tok.sum()
                    mb["loss"] += float(loss_tok.detach().sum())
                    mb["tokens"] += logp.numel()
                    mb["clipped"] += float(st["clipped"].sum())
                    mb["ratio_sum"] += float(st["ratio"].sum())
                    ratio_max = max(ratio_max, float(st["ratio"].max()) if logp.numel() else 0.0)
                    mb["kl_b"] += float(st["approx_kl_behaviour"].sum())
                    mb["absdiff"] += float((logp.detach() - old).abs().sum())
                    mb["entropy"] += float(ent.sum())
                    mb["kl_ref"] += float(st["kl_ref"].sum()) if "kl_ref" in st else 0.0
                    mb["seq_tokens"] += len(it["input_ids"])
                    mb["seqs"] += 1
                (loss_row / max(total, 1.0)).backward()
            self._allreduce_grads()
            norm = torch.nn.utils.clip_grad_norm_(self.trainable, self.max_grad_norm)
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite gradient norm")
            norms.append(float(norm))
            self.optimizer.step()
            g = dict(zip(keys, self._global_sum([mb[k] for k in keys])))
            if first is None and g["tokens"]:
                first = {"ratio_mean": g["ratio_sum"] / g["tokens"],
                         "mean_abs_logp_diff": g["absdiff"] / g["tokens"]}
            for k in keys:
                agg[k] += g[k]
        ratio_max = self._global_sum([ratio_max], "max")[0]
        seconds = time.perf_counter() - began
        peak = torch.cuda.max_memory_allocated(self.device) / 1e9 if self.device.type == "cuda" else None
        self.release_memory()
        t = max(agg["tokens"], 1.0)
        return {"policy_loss": agg["loss"] / t, "target_tokens": agg["tokens"],
                "sequence_tokens": agg["seq_tokens"], "sequences": agg["seqs"],
                "clip_fraction": agg["clipped"] / t, "ratio_mean": agg["ratio_sum"] / t,
                "ratio_max": ratio_max, "approx_kl_behaviour": agg["kl_b"] / t,
                "mean_abs_logp_diff_vs_behaviour": agg["absdiff"] / t,
                "first_minibatch_ratio_mean": first["ratio_mean"] if first else None,
                "first_minibatch_abs_logp_diff": first["mean_abs_logp_diff"] if first else None,
                "entropy": agg["entropy"] / t, "kl_ref_in_loss": agg["kl_ref"] / t,
                "grad_norm": norms, "optimizer_steps": len(norms), "update_seconds": seconds,
                "tokens_per_second": agg["seq_tokens"] / seconds if seconds else None,
                "packing": self.packing,
                "peak_memory_gb": peak}

    def release_memory(self) -> None:
        import gc
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

    def save_adapter(self, directory: Path) -> Path:
        self.model.save_pretrained(str(directory), selected_adapters=["default"], safe_serialization=True)
        return Path(directory)

    def trainer_state(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "trainable_names": self.trainable_names}

    def load_trainer_state(self, state: dict) -> None:
        if state["trainable_names"] != self.trainable_names:
            raise ValueError("trainable parameter order changed; cannot restore optimizer")
        self.optimizer.load_state_dict(state["optimizer"])

    def adapter_fingerprint(self) -> float:
        with torch.no_grad():
            return float(sum(p.detach().double().abs().sum().item()
                             for n, p in self.model.named_parameters()
                             if "lora_B" in n and ".ref." not in n))
