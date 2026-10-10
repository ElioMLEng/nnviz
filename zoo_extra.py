"""More architectures: U-Net, VAE, DCGAN, DDPM U-Net, GPT-OSS-style MoE Transformer."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from zoo import image_input, token_input, RMSNorm


# ------------------------------------------------------------------ U-Net (Ronneberger et al. 2015)
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1, self.b1, self.r1 = nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()
        self.c2, self.b2, self.r2 = nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()

    def forward(self, x):
        return self.r2(self.b2(self.c2(self.r1(self.b1(self.c1(x))))))


class UNet(nn.Module):
    def __init__(self, cin=1, ncls=2, base=32, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([DoubleConv(cin if i == 0 else ch[i - 1], ch[i]) for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in range(depth)])
        self.dec = nn.ModuleList([DoubleConv(ch[i] * 2, ch[i]) for i in range(depth)])
        self.head = nn.Conv2d(ch[0], ncls, 1)
        self.depth = depth

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = self.pool(x)
        x = self.enc[self.depth](x)
        for i in reversed(range(self.depth)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], 1))
        return self.head(x)


# ------------------------------------------------------------------ VAE (Kingma & Welling 2013)
class VAE(nn.Module):
    def __init__(self, latent=16):
        super().__init__()
        self.e1, self.ea = nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU()
        self.e2, self.eb = nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU()
        self.flat = nn.Flatten()
        self.fc_mu, self.fc_logvar = nn.Linear(64 * 7 * 7, latent), nn.Linear(64 * 7 * 7, latent)
        self.fc_dec, self.da = nn.Linear(latent, 64 * 7 * 7), nn.ReLU()
        self.unflat = nn.Unflatten(1, (64, 7, 7))
        self.d1, self.db = nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU()
        self.d2, self.out = nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid()

    def forward(self, x):                                   # x: (B,1,28,28)
        h = self.flat(self.eb(self.e2(self.ea(self.e1(x)))))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterization trick
        h = self.unflat(self.da(self.fc_dec(z)))
        return self.out(self.d2(self.db(self.d1(h)))), mu, logvar


# ------------------------------------------------------------------ DCGAN (Radford et al. 2015)
def _gen_block(cin, cout, k, s, p):
    return [nn.ConvTranspose2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]


class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            *_gen_block(nz, nf * 8, 4, 1, 0), *_gen_block(nf * 8, nf * 4, 4, 2, 1),
            *_gen_block(nf * 4, nf * 2, 4, 2, 1), *_gen_block(nf * 2, nf, 4, 2, 1),
            nn.ConvTranspose2d(nf, 3, 4, 2, 1, bias=False), nn.Tanh())

    def forward(self, z):
        return self.net(z)


class DCGANDiscriminator(nn.Module):
    def __init__(self, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, nf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 2), nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 4), nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(nf * 8), nn.LeakyReLU(0.2),
            nn.Conv2d(nf * 8, 1, 4, 1, 0, bias=False), nn.Sigmoid(), nn.Flatten())

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ DDPM U-Net (Ho et al. 2020), scaled down
class BiAttention(nn.Module):
    """Fused-QKV multi-head self-attention without a causal mask (exposes qkv/o/heads for the viewer)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.qkv, self.o = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.last_attn = None

    def forward(self, x):
        B, T, D = x.shape
        h = self.heads
        q, k, v = self.qkv(x).view(B, T, 3, h, D // h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1) / math.sqrt(D // h)).softmax(-1)
        self.last_attn = a.detach()
        return self.o((a @ v).transpose(1, 2).reshape(B, T, D))


class SinusoidalEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        f = torch.exp(-math.log(10000) * torch.arange(dim // 2).float() / (dim // 2))
        self.register_buffer("freqs", f, persistent=False)

    def forward(self, t):
        a = t[:, None].float() * self.freqs[None]
        return torch.cat([a.sin(), a.cos()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.norm1, self.act, self.conv1 = nn.GroupNorm(8, cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1)
        self.t_proj = nn.Linear(tdim, cout)
        self.norm2, self.conv2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.t_proj(self.act(temb))[:, :, None, None]     # timestep conditioning
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm, self.attn = nn.GroupNorm(8, c), BiAttention(c, heads)

    def forward(self, x):
        B, C, H, W = x.shape
        t = self.attn(self.norm(x).flatten(2).transpose(1, 2))
        return x + t.transpose(1, 2).reshape(B, C, H, W)


class DDPMUNet(nn.Module):
    def __init__(self, base=32, mults=(1, 2, 2), tdim=128):
        super().__init__()
        ch = [base * m for m in mults]
        self.t_emb, self.t_fc1, self.t_act, self.t_fc2 = SinusoidalEmb(base), nn.Linear(base, tdim), nn.SiLU(), nn.Linear(tdim, tdim)
        self.inc = nn.Conv2d(3, base, 3, padding=1)
        cin = [base] + ch[:-1]
        self.down_res = nn.ModuleList([ResBlock(cin[i], ch[i], tdim) for i in range(len(ch))])
        self.down_conv = nn.ModuleList([nn.Conv2d(ch[i], ch[i], 3, stride=2, padding=1) for i in range(len(ch) - 1)])
        self.mid1, self.mid_attn, self.mid2 = ResBlock(ch[-1], ch[-1], tdim), AttnBlock(ch[-1]), ResBlock(ch[-1], ch[-1], tdim)
        rev = ch[::-1]
        nxt = rev[1:] + [base]
        self.up_res = nn.ModuleList([ResBlock(rev[i] * 2, nxt[i], tdim) for i in range(len(ch))])
        self.up_samp = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2.0), nn.Conv2d(nxt[i], nxt[i], 3, padding=1))
                                      for i in range(len(ch) - 1)])
        self.norm, self.act, self.out = nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1)

    def forward(self, x, t):
        temb = self.t_fc2(self.t_act(self.t_fc1(self.t_emb(t))))
        h = self.inc(x)
        skips = []
        for i in range(len(self.down_res)):
            h = self.down_res[i](h, temb)
            skips.append(h)
            if i < len(self.down_conv):
                h = self.down_conv[i](h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for i in range(len(self.up_res)):
            h = self.up_res[i](torch.cat([h, skips.pop()], 1), temb)
            if i < len(self.up_samp):
                h = self.up_samp[i](h)
        return self.out(self.act(self.norm(h)))


# ------------------------------------------------------------------ GPT-OSS-style (MoE + GQA + sliding window)
class GQAttention(nn.Module):
    """Grouped-query attention, optional sliding window, RoPE and learned per-head attention sinks."""

    def __init__(self, dim, heads, kv_heads, T, window=None):
        super().__init__()
        self.heads, self.kv_heads, self.dim, self.window = heads, kv_heads, dim, window
        dh = dim // heads
        self.qkv = nn.Linear(dim, dim + 2 * kv_heads * dh)
        self.o = nn.Linear(dim, dim)
        self.sinks = nn.Parameter(torch.zeros(heads))
        i = torch.arange(T)
        m = i[:, None] >= i[None, :]
        if window: m = m & (i[:, None] - i[None, :] < window)
        self.register_buffer("mask", m, persistent=False)
        inv = 1.0 / (10000 ** (torch.arange(0, dh, 2).float() / dh))
        ang = torch.outer(torch.arange(T).float(), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        self.last_attn = None

    def _rope(self, x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        T = x.shape[2]
        c, s = self.cos[:T], self.sin[:T]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def forward(self, x):
        B, T, D = x.shape
        h, hk = self.heads, self.kv_heads
        dh = D // h
        q, k, v = self.qkv(x).split([D, hk * dh, hk * dh], -1)
        q = self._rope(q.view(B, T, h, dh).transpose(1, 2))
        k = self._rope(k.view(B, T, hk, dh).transpose(1, 2)).repeat_interleave(h // hk, 1)
        v = v.view(B, T, hk, dh).transpose(1, 2).repeat_interleave(h // hk, 1)
        s = (q @ k.transpose(-2, -1) / math.sqrt(dh)).masked_fill(~self.mask[:T, :T], float("-inf"))
        sink = self.sinks.view(1, h, 1, 1).expand(B, h, T, 1)
        a = torch.cat([s, sink], -1).softmax(-1)[..., :-1]       # the sink absorbs probability mass
        self.last_attn = a.detach()
        return self.o((a @ v).transpose(1, 2).reshape(B, T, D))


class MoE(nn.Module):
    """Top-k routed SwiGLU experts (computed densely here, masked by the sparse gates)."""

    def __init__(self, d, ff, experts, top_k):
        super().__init__()
        self.num_experts, self.top_k = experts, top_k
        self.router = nn.Linear(d, experts)
        self.w_gate = nn.Parameter(torch.randn(experts, d, ff) * 0.02)
        self.w_up = nn.Parameter(torch.randn(experts, d, ff) * 0.02)
        self.w_down = nn.Parameter(torch.randn(experts, ff, d) * 0.02)
        self.last_gates = None

    def forward(self, x):
        logits = self.router(x)
        top, idx = logits.topk(self.top_k, -1)
        gates = torch.zeros_like(logits).scatter(-1, idx, top.softmax(-1))
        self.last_gates = gates.detach()
        h = F.silu(torch.einsum("btd,edf->btef", x, self.w_gate)) * torch.einsum("btd,edf->btef", x, self.w_up)
        y = torch.einsum("btef,efd->bted", h, self.w_down)
        return (y * gates.unsqueeze(-1)).sum(2)


class OSSBlock(nn.Module):
    def __init__(self, d, h, hk, T, ff, E, k, window):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(d), GQAttention(d, h, hk, T, window)
        self.ffn_norm, self.moe = RMSNorm(d), MoE(d, ff, E, k)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.moe(self.ffn_norm(x))


class GPTOSS(nn.Module):
    def __init__(self, vocab=4096, d=256, layers=8, heads=8, kv_heads=2, ff=256, experts=8, top_k=2, T=32, window=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([OSSBlock(d, heads, kv_heads, T, ff, experts, top_k, window if i % 2 == 0 else None)
                                     for i in range(layers)])
        self.norm, self.lm_head = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.lm_head(self.norm(x))


# ------------------------------------------------------------------ inputs + registry entries
def _gray(size):
    base, classes = image_input(size)
    return (lambda cls, seed: base(cls, seed).mean(1, keepdim=True)), classes


def _slerp(a, b, t):
    na, nb = a / a.norm(), b / b.norm()
    om = torch.acos((na * nb).sum().clamp(-1 + 1e-6, 1 - 1e-6))
    return (torch.sin((1 - t) * om) * a + torch.sin(t * om) * b) / torch.sin(om)


def _latent(cls, seed, alpha=0.0, scale=1.0):
    if cls == 1:
        return torch.zeros(1, 100, 1, 1)
    za = torch.randn(1, 100, 1, 1, generator=torch.Generator().manual_seed(seed))
    zb = torch.randn(1, 100, 1, 1, generator=torch.Generator().manual_seed(seed + 1000))
    return _slerp(za.flatten(), zb.flatten(), float(alpha)).view(1, 100, 1, 1) * float(scale)


# DDPM linear noise schedule (Ho et al. 2020): alpha_bar_t = prod(1 - beta_s)
_ALPHA_BAR = torch.cumprod(1 - torch.linspace(1e-4, 0.02, 1000), 0)


class _Pair(tuple):
    """(x_t, t) that also remembers how it was made (clean image, true noise) for the gallery."""


def _diffusion_input(cls, seed, t=500):
    t = int(max(0, min(999, float(t))))
    g = torch.Generator().manual_seed(seed)
    base, _ = image_input(32)
    x0 = base(0, seed).clamp(-3, 3) / 3 if cls == 0 else torch.zeros(1, 3, 32, 32)
    eps = torch.randn(1, 3, 32, 32, generator=g)
    ab = _ALPHA_BAR[t]
    p = _Pair((ab.sqrt() * x0 + (1 - ab).sqrt() * eps, torch.tensor([t])))
    p.meta = {"x0": x0, "eps": eps, "ab": float(ab)}
    return p


def _diffusion_extras(x, out):
    m = getattr(x, "meta", None)
    if m is None: return {}
    ab = m["ab"]
    est = ((x[0] - (1 - ab) ** 0.5 * out) / ab ** 0.5).clamp(-1.5, 1.5)
    return {"clean image x₀": m["x0"], "true noise ε": m["eps"], "predicted noise ε̂": out, "x₀ estimate from ε̂": est}


def _e(group, build, make, classes, desc, controls=None, extras=None):
    return {"group": group, "build": build, "make_input": make, "classes": classes, "desc": desc, "pretrained": False,
            "controls": controls or [], "extras": extras}


EXTRA = {
    "unet": _e("Segmentation", lambda pretrained=False: UNet().eval(), *_gray(64), "U-Net · encoder–decoder with skip concatenations · 64×64 input"),
    "vae": _e("Autoencoder / GAN", lambda pretrained=False: VAE().eval(), *_gray(28), "Convolutional VAE · μ/logvar + reparameterization"),
    "dcgan-generator": _e("Autoencoder / GAN", lambda pretrained=False: DCGANGenerator().eval(), _latent,
                          ["z ~ N(0, I)", "z = 0"], "DCGAN generator · 100-d noise → 64×64 image",
                          controls=[{"id": "alpha", "label": "interpolate latent z: A → B", "min": 0, "max": 1, "step": 0.05, "default": 0},
                                    {"id": "scale", "label": "latent scale", "min": 0, "max": 2, "step": 0.1, "default": 1}]),
    "dcgan-discriminator": _e("Autoencoder / GAN", lambda pretrained=False: DCGANDiscriminator().eval(), *image_input(64),
                              "DCGAN discriminator · real/fake classifier"),
    "ddpm-unet": _e("Diffusion", lambda pretrained=False: DDPMUNet().eval(), _diffusion_input,
                    ["synthetic scene x₀", "blank image x₀ = 0"], "DDPM noise-prediction U-Net (scaled down) · inputs x_t and t",
                    controls=[{"id": "t", "label": "timestep t (0 = clean, 999 = pure noise)", "min": 0, "max": 999, "step": 10, "default": 500}],
                    extras=_diffusion_extras),
    "gpt-oss-style": _e("LLM", lambda pretrained=False: GPTOSS().eval(), *token_input(4096),
                        "GPT-OSS-style · MoE (8 experts, top-2) + GQA + sliding-window attention (scaled down)"),
}
