"""Frontier-scale open LLMs at their true size, without their weights.

Each entry has two builds of the *same* architecture code:
  * the **true model** on PyTorch's meta device: real shapes, exact parameter counts, zero memory, never executed on data;
  * a **twin** with identical layer structure (same number of blocks, same module names) but tiny widths and random
    weights, which actually runs so the viewer gets activations and attention maps to draw.
After tracing the twin, `true_scale` runs the same trace on the meta model and overwrites every node's shape, size,
parameter count and FLOPs with the real values. Activations stay illustrative (random, scaled down) and are labelled so.

Configurations follow the published model configs (Hugging Face `config.json`).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zoo import RMSNorm
from fake_outputs import PROMPT_NAMES

T_TOKENS = 16


class Attention(nn.Module):
    """Grouped-query attention with explicit head size, optional sliding window, RoPE, optional attention sinks."""

    def __init__(self, d, heads, kv_heads, head_dim, T, window=None, bias=False, sinks=False):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim, self.window = heads, kv_heads, head_dim, window
        self.qkv = nn.Linear(d, (heads + 2 * kv_heads) * head_dim, bias=bias)
        self.o = nn.Linear(heads * head_dim, d, bias=bias)
        self.sinks = nn.Parameter(torch.zeros(heads)) if sinks else None
        i = torch.arange(T)
        m = i[:, None] >= i[None, :]
        if window: m = m & (i[:, None] - i[None, :] < window)
        self.register_buffer("mask", m, persistent=False)
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        ang = torch.outer(torch.arange(T).float(), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        self.last_attn = None

    def _rope(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        c, s = self.cos[:x.shape[2]], self.sin[:x.shape[2]]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def forward(self, x):
        B, T, _ = x.shape
        h, hk, dh = self.heads, self.kv_heads, self.head_dim
        q, k, v = self.qkv(x).split([h * dh, hk * dh, hk * dh], -1)
        q = self._rope(q.view(B, T, h, dh).transpose(1, 2))
        k = self._rope(k.view(B, T, hk, dh).transpose(1, 2)).repeat_interleave(h // hk, 1)
        v = v.view(B, T, hk, dh).transpose(1, 2).repeat_interleave(h // hk, 1)
        s = (q @ k.transpose(-2, -1) / math.sqrt(dh)).masked_fill(~self.mask[:T, :T], float("-inf"))
        if self.sinks is not None:
            a = torch.cat([s, self.sinks.view(1, h, 1, 1).expand(B, h, T, 1)], -1).softmax(-1)[..., :-1]
        else:
            a = s.softmax(-1)
        self.last_attn = a.detach()
        return self.o((a @ v).transpose(1, 2).reshape(B, T, h * dh))


class SwiGLU(nn.Module):
    """Dense feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, d, ff):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """Top-k routed SwiGLU experts (computed densely on the twin, masked by the sparse gates)."""

    def __init__(self, d, ff, experts, top_k, bias=False):
        super().__init__()
        self.num_experts, self.top_k = experts, top_k
        self.router = nn.Linear(d, experts, bias=bias)
        self.w_gate = nn.Parameter(torch.randn(experts, d, ff) * 0.02)
        self.w_up = nn.Parameter(torch.randn(experts, d, ff) * 0.02)
        self.w_down = nn.Parameter(torch.randn(experts, ff, d) * 0.02)
        self.b_gate_up = nn.Parameter(torch.zeros(experts, 2 * ff)) if bias else None
        self.b_down = nn.Parameter(torch.zeros(experts, d)) if bias else None
        self.last_gates = None

    def forward(self, x):
        logits = self.router(x)
        top, idx = logits.topk(self.top_k, -1)
        gates = torch.zeros_like(logits).scatter(-1, idx, top.softmax(-1))
        self.last_gates = gates.detach()
        h = F.silu(torch.einsum("btd,edf->btef", x, self.w_gate)) * torch.einsum("btd,edf->btef", x, self.w_up)
        y = torch.einsum("btef,efd->bted", h, self.w_down)
        return (y * gates.unsqueeze(-1)).sum(2)


class Block(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        window = c["window"] if (c.get("window") and (not c.get("alternate") or i % 2 == 0)) else None
        self.attn_norm = RMSNorm(c["d"])
        self.attn = Attention(c["d"], c["heads"], c["kv_heads"], c["head_dim"], T_TOKENS, window, c.get("bias", False), c.get("sinks", False))
        self.ffn_norm = RMSNorm(c["d"])
        self.ffn = MoE(c["d"], c["ff"], c["experts"], c["top_k"], c.get("bias", False)) if c.get("experts") else SwiGLU(c["d"], c["ff"])

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class LLM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed = nn.Embedding(c["vocab"], c["d"])
        self.blocks = nn.ModuleList([Block(c, i) for i in range(c["layers"])])
        self.norm, self.lm_head = RMSNorm(c["d"]), nn.Linear(c["d"], c["vocab"], bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm(x))


def twin_config(c):
    """Same depth, experts and attention pattern; tiny widths so it runs in milliseconds on a laptop CPU."""
    heads = 8
    kv = max(1, heads * c["kv_heads"] // c["heads"])
    return {**c, "vocab": 512, "d": 64, "heads": heads, "kv_heads": kv, "head_dim": 8, "ff": 96,
            "window": min(c["window"], T_TOKENS // 2) if c.get("window") else None}


# ------------------------------------------------------------------ published configurations
CONFIGS = {
    "mistral-7b": dict(org="Mistral AI", vocab=32000, d=4096, layers=32, heads=32, kv_heads=8, head_dim=128, ff=14336,
                       window=4096, desc="Mistral 7B (2023) · dense · GQA 32/8 · sliding-window attention 4,096"),
    "mixtral-8x7b": dict(org="Mistral AI", vocab=32000, d=4096, layers=32, heads=32, kv_heads=8, head_dim=128, ff=14336,
                         experts=8, top_k=2, desc="Mixtral 8×7B (2023) · sparse MoE, 8 experts, top-2 · ≈13B active per token"),
    "mistral-large-2": dict(org="Mistral AI", vocab=32768, d=12288, layers=88, heads=96, kv_heads=8, head_dim=128, ff=28672,
                            desc="Mistral Large 2 (2024) · dense · 88 layers · GQA 96/8"),
    "gpt-oss-20b": dict(org="OpenAI", vocab=201088, d=2880, layers=24, heads=64, kv_heads=8, head_dim=64, ff=2880,
                        experts=32, top_k=4, window=128, alternate=True, bias=True, sinks=True,
                        desc="gpt-oss-20b (2025) · MoE 32 experts, top-4 · alternating full / 128-token sliding window · attention sinks"),
    "gpt-oss-120b": dict(org="OpenAI", vocab=201088, d=2880, layers=36, heads=64, kv_heads=8, head_dim=64, ff=2880,
                         experts=128, top_k=4, window=128, alternate=True, bias=True, sinks=True,
                         desc="gpt-oss-120b (2025) · MoE 128 experts, top-4 · ≈5.1B active per token"),
    "llama-3.1-405b": dict(org="Meta", vocab=128256, d=16384, layers=126, heads=128, kv_heads=8, head_dim=128, ff=53248,
                           desc="Llama 3.1 405B (2024) · dense · 126 layers · size reference for the largest open dense model"),
}


def build_true(c):
    with torch.device("meta"):
        return LLM(c).eval()


def true_scale(net, model_id):
    """Overwrite the twin's per-node numbers with the real model's (traced on the meta device)."""
    from extract import FxTrace, attn_parts, attrs_of, first_tensor, flops_of, kind_of, strip_batch
    c = CONFIGS[model_id]
    big = build_true(c)
    ex = torch.zeros(1, T_TOKENS, dtype=torch.long, device="meta")
    tr = FxTrace(big, ex)
    vals, _ = tr.run(ex)
    mods = {nd["name"]: nd["mod"] for nd in tr.nodes}
    for node in net.graph["nodes"]:
        name = node["name"]
        t = first_tensor(vals.get(name))
        if node["kind"] == "output":
            t = first_tensor(vals.get("lm_head"))
        if t is None:
            continue
        mod = mods.get(name)
        tt = strip_batch(t)
        node["shape"], node["tshape"], node["numel"] = list(t.shape), list(tt.shape), int(tt.numel())
        node["sampled"] = True
        if mod is not None:
            node["params"] = sum(p.numel() for p in mod.parameters())
            node["attrs"] = attrs_of(mod)
            parts = attn_parts(mod) if kind_of(mod) == "attention" else None
            node["flops"] = flops_of(mod, kind_of(mod), tt, node["label"], parts)
            if node["kind"] == "attention":   # flops_of assumes head_dim = d / heads; use the exact formula
                T, D = tt.shape
                q, kv = mod.heads * mod.head_dim, mod.kv_heads * mod.head_dim
                node["flops"] = int(2 * T * D * (q + 2 * kv) + 4 * T * T * q + 2 * T * q * D)
            if node["kind"] == "moe":
                node["attrs"] = {"experts": mod.num_experts, "top_k": mod.top_k, "d_model": c["d"], "ff_per_expert": c["ff"]}
        else:
            node["flops"] = int(tt.numel())
        if node.get("weights") and mod is not None and hasattr(mod, "weight"):
            node["weights"]["shape"] = list(mod.weight.shape)
    total = sum(p.numel() for p in big.parameters())
    net.graph["params"] = total
    net.graph["true_scale"] = {"params": total, "active": active_params(c), "bytes_bf16": 2 * total, "org": c["org"],
                               "note": "Exact architecture, shapes and sizes. Weights and activations are illustrative "
                                       "(random, computed on a scaled-down twin with the same layers)."}
    net.total_params = total


def active_params(c):
    """Parameters used per token (MoE: only top-k experts)."""
    total = sum(p.numel() for p in build_true(c).parameters())
    if not c.get("experts"):
        return total
    per_expert = 3 * c["d"] * c["ff"] + (3 * c["ff"] + c["d"] if c.get("bias") else 0)
    return total - c["layers"] * (c["experts"] - c["top_k"]) * per_expert


def _token_input(cls, seed):
    g = torch.Generator().manual_seed(seed + 1000 * cls)
    return torch.randint(0, 512, (1, T_TOKENS), generator=g)


def _entry(mid, c):
    return {"group": "Frontier LLMs (true scale)", "build": lambda pretrained=False, c=c: LLM(twin_config(c)).eval(),
            "make_input": _token_input, "classes": list(PROMPT_NAMES), "desc": c["desc"], "pretrained": False,
            "count": lambda c=c: sum(p.numel() for p in build_true(c).parameters()),
            "post": lambda net, mid=mid: true_scale(net, mid)}


LLMS = {mid: _entry(mid, c) for mid, c in CONFIGS.items()}
