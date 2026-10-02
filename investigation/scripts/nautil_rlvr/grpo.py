from __future__ import annotations

import math


def group_advantages(rewards: list[float], std_normalize: bool = True, eps: float = 1e-6) -> list[float]:
    mean = sum(rewards) / len(rewards)
    if not std_normalize:
        return [r - mean for r in rewards]
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    return [(r - mean) / (std + eps) for r in rewards]


def center_by_decision(totals: list[float], decisions: list, D: list[float]) -> list[float]:
    rest = [t - d for t, d in zip(totals, D)]
    out = list(totals)
    for cls in (True, False):
        idx = [i for i, c in enumerate(decisions) if c is cls]
        if idx:
            m = sum(rest[i] for i in idx) / len(idx)
            for i in idx:
                out[i] = D[i] + rest[i] - m
    return out


def group_is_degenerate(rewards: list[float], tol: float = 1e-9) -> bool:
    return max(rewards) - min(rewards) <= tol


def group_std(rewards: list[float]) -> float:
    mean = sum(rewards) / len(rewards)
    return math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))


def clipped_token_loss(logp, behaviour_logp, advantage, clip_low: float, clip_high: float,
                       ref_logp=None, kl_coef: float = 0.0):
    import torch
    log_ratio = logp - behaviour_logp
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high) * advantage
    loss = -torch.minimum(unclipped, clipped)
    stats = {"clipped": (clipped < unclipped).float().detach(),
             "ratio": ratio.detach(),
             "approx_kl_behaviour": (-log_ratio).detach()}
    if ref_logp is not None and kl_coef > 0:
        diff = ref_logp - logp
        kl = torch.exp(diff) - diff - 1.0
        loss = loss + kl_coef * kl
        stats["kl_ref"] = kl.detach()
    return loss, stats
