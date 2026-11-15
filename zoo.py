"""Model registry: famous architectures (random init unless pretrained=True) + synthetic inputs."""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from toynet import trained_model, make_sample, ToyNet, CLASSES as TOY_CLASSES


# ---------------------------------------------------------------- LLM building blocks (fx-traceable)
class CausalSelfAttention(nn.Module):
    """Fused-QKV multi-head attention (GPT-2 style); optional RoPE (Llama style). Exposes qkv/o/heads for nnviz."""

    def __init__(self, dim, heads, seq_len, rope=False):
        super().__init__()
        self.heads, self.dim, self.rope = heads, dim, rope
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)), persistent=False)
        if rope:
            dh = dim // heads
            inv = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
            ang = torch.outer(torch.arange(seq_len).float(), inv)
            self.register_buffer("cos", ang.cos(), persistent=False)
            self.register_buffer("sin", ang.sin(), persistent=False)
        self.last_attn = None

    def _rope(self, x):  # x: B,h,T,dh
        x1, x2 = x[..., 0::2], x[..., 1::2]
        T = x.shape[2]
        c, s = self.cos[:T], self.sin[:T]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def forward(self, x):
        B, T, D = x.shape
        h = self.heads
        q, k, v = self.qkv(x).view(B, T, 3, h, D // h).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = self._rope(q), self._rope(k)
        s = q @ k.transpose(-2, -1) / math.sqrt(D // h)
        s = s.masked_fill(~self.mask[:T, :T], float("-inf"))
        a = s.softmax(-1)
        self.last_attn = a.detach()
        return self.o((a @ v).transpose(1, 2).reshape(B, T, D))


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class GPTBlock(nn.Module):
    def __init__(self, d, h, T):
        super().__init__()
        self.ln_1, self.attn, self.ln_2 = nn.LayerNorm(d), CausalSelfAttention(d, h, T), nn.LayerNorm(d)
        self.c_fc, self.act, self.c_proj = nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.c_proj(self.act(self.c_fc(self.ln_2(x))))


class GPT(nn.Module):
    """GPT-2 architecture, fixed sequence length so it traces with torch.fx."""

    def __init__(self, vocab=50257, d=768, layers=12, heads=12, T=32):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(T, d)
        self.register_buffer("pos", torch.arange(T), persistent=False)
        self.blocks = nn.ModuleList([GPTBlock(d, heads, T) for _ in range(layers)])
        self.ln_f, self.lm_head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)
        self.lm_head.weight = self.wte.weight      # tied embeddings, as in GPT-2 (124 M instead of 162 M)

    def forward(self, ids):
        x = self.wte(ids) + self.wpe(self.pos)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.ln_f(x))


class LlamaBlock(nn.Module):
    def __init__(self, d, h, T, ff):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(d), CausalSelfAttention(d, h, T, rope=True)
        self.ffn_norm = RMSNorm(d)
        self.gate_proj, self.up_proj, self.down_proj = nn.Linear(d, ff, bias=False), nn.Linear(d, ff, bias=False), nn.Linear(ff, d, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        h = self.ffn_norm(x)
        return x + self.down_proj(self.act(self.gate_proj(h)) * self.up_proj(h))


class Llama(nn.Module):
    """Llama-style (RMSNorm, RoPE, SwiGLU). Plain MHA (no GQA) so the fused-QKV viewer applies."""

    def __init__(self, vocab=32000, d=512, layers=8, heads=8, ff=1376, T=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([LlamaBlock(d, heads, T, ff) for _ in range(layers)])
        self.norm, self.lm_head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm(x))


# ---------------------------------------------------------------- synthetic inputs
PHOTOS = [("cat", "Photo: cat"), ("coffee", "Photo: coffee cup"), ("astronaut", "Photo: astronaut"), ("rocket", "Photo: rocket launch")]
_IMNET_MEAN, _IMNET_STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def photo(name, size, seed=0):
    """CC0 / public-domain sample photo (assets/images), ImageNet-normalised. seed > 0: light augmentation."""
    from PIL import Image
    import numpy as np
    im = Image.open(Path(__file__).parent / "assets" / "images" / f"{name}.jpg").convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255
    if seed:
        g = torch.Generator().manual_seed(seed)
        if torch.rand(1, generator=g) < 0.5: x = x.flip(-1)
        x = (x + 0.04 * torch.randn(x.shape, generator=g)).clamp(0, 1)
    return ((x - _IMNET_MEAN) / _IMNET_STD).unsqueeze(0)


def imagenet_labels():
    from torchvision.models import ResNet18_Weights
    return list(ResNet18_Weights.DEFAULT.meta["categories"])


def image_input(size=224):
    def make(cls, seed):
        g = torch.Generator().manual_seed(seed)
        if cls < len(PHOTOS):
            return photo(PHOTOS[cls][0], size, seed)
        if cls == len(PHOTOS) + 1:                    # noise
            return torch.randn(1, 3, size, size, generator=g)
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        cx, cy = (torch.rand(2, generator=g) - 0.5) * 0.8
        disk = (((xx - cx) ** 2 + (yy - cy) ** 2) < 0.18).float()
        stripes = (torch.sin(xx * 14 + torch.rand(1, generator=g) * 6) > 0).float()
        img = torch.stack([disk, stripes * (1 - disk), xx * 0.5 + 0.5])
        img = img + 0.05 * torch.randn(3, size, size, generator=g)
        return ((img - 0.45) / 0.3).unsqueeze(0)       # roughly ImageNet-normalised
    return make, [label for _, label in PHOTOS] + ["Synthetic scene", "Noise"]


def token_input(vocab, T=32):
    """Token ids for one of the example prompts (fake_outputs.PROMPTS); the ids are stand-ins (no tokenizer)."""
    from fake_outputs import PROMPT_NAMES

    def make(cls, seed):
        g = torch.Generator().manual_seed(seed + 1000 * cls)
        return torch.randint(0, vocab, (1, T), generator=g)
    return make, list(PROMPT_NAMES)


def toy_input(cls, seed):
    return make_sample(cls, torch.Generator().manual_seed(seed)).view(1, 1, 8, 8)


# ---------------------------------------------------------------- registry
def _tv(name, **kw):
    def build(pretrained=False):
        import torchvision.models as M
        fn = getattr(M, name)
        return fn(weights="DEFAULT" if pretrained else None, **kw).eval()
    return build


def _vit(**cfg):
    def build(pretrained=False):
        import torchvision.models as M
        return M.VisionTransformer(image_size=224, patch_size=16, **cfg).eval()
    return build


def _entry(group, build, inp, desc, pretrained=False, labels=None):
    make, classes = inp
    return {"group": group, "build": build, "make_input": make, "classes": classes, "desc": desc, "pretrained": pretrained,
            "labels": labels}


REGISTRY = {
    "toy-mixed": {"group": "Toy", "build": lambda pretrained=False: trained_model(), "make_input": toy_input,
                  "classes": TOY_CLASSES, "desc": "Conv + attention toy net trained on line images", "pretrained": False},
    "resnet18": _entry("CNN", _tv("resnet18"), image_input(), "ResNet-18 · 11.7M params · residual blocks", True, labels=imagenet_labels),
    "resnet50": _entry("CNN", _tv("resnet50"), image_input(), "ResNet-50 · 25.6M · bottleneck residual blocks", True, labels=imagenet_labels),
    "vgg16": _entry("CNN", _tv("vgg16"), image_input(), "VGG-16 · 138M · plain conv stack + huge MLP head", True, labels=imagenet_labels),
    "alexnet": _entry("CNN", _tv("alexnet"), image_input(), "AlexNet · 61M", True, labels=imagenet_labels),
    "mobilenet_v3_small": _entry("CNN", _tv("mobilenet_v3_small"), image_input(), "MobileNetV3-Small · 2.5M · depthwise + SE", True, labels=imagenet_labels),
    "efficientnet_b0": _entry("CNN", _tv("efficientnet_b0"), image_input(), "EfficientNet-B0 · 5.3M · MBConv", True, labels=imagenet_labels),
    "densenet121": _entry("CNN", _tv("densenet121"), image_input(), "DenseNet-121 · 8M · dense concatenation", True, labels=imagenet_labels),
    "vit_tiny_16": _entry("Vision Transformer", _vit(num_layers=12, num_heads=3, hidden_dim=192, mlp_dim=768),
                          image_input(), "ViT-Tiny/16 · 5.7M · 197 tokens", labels=imagenet_labels),
    "vit_b_16": _entry("Vision Transformer", _tv("vit_b_16"), image_input(), "ViT-Base/16 · 86M · 12 layers, 12 heads", True, labels=imagenet_labels),
    "gpt2-small": _entry("LLM", lambda pretrained=False: GPT().eval(), token_input(50257),
                         "GPT-2 small architecture · 124M · 12 layers, 12 heads, T=32 (random weights)"),
    "gpt2-tiny": _entry("LLM", lambda pretrained=False: GPT(vocab=2048, d=128, layers=4, heads=4).eval(), token_input(2048),
                        "4-layer GPT, d=128 · fast to explore"),
    "llama-style": _entry("LLM", lambda pretrained=False: Llama().eval(), token_input(32000),
                          "Llama-style · RMSNorm + RoPE + SwiGLU · 8 layers (MHA, no GQA)"),
}

# more architectures + model cards (imported last: they use helpers defined above)
from zoo_extra import EXTRA          # noqa: E402
from zoo_llm import LLMS             # noqa: E402
from modelinfo import INFO           # noqa: E402

REGISTRY.update(EXTRA)
REGISTRY.update(LLMS)
for _k, _v in INFO.items():
    if _k in REGISTRY:
        REGISTRY[_k]["info"] = _v
REGISTRY["toy-mixed"]["count"] = lambda: sum(p.numel() for p in ToyNet().parameters())


def param_count(model_id):
    """Parameter count without allocating weights (meta device); cached."""
    if model_id in _COUNTS:
        return _COUNTS[model_id]
    e = REGISTRY[model_id]
    try:
        if "count" in e:
            n = e["count"]()
        else:
            with torch.device("meta"):
                n = sum(p.numel() for p in e["build"]().parameters())
    except Exception:
        n = None
    _COUNTS[model_id] = n
    return n


_COUNTS = {}
