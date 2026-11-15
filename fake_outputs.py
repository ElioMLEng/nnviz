"""Illustrative outputs for models that run with random weights.

A random network's real output is noise, which makes the viewer feel broken. For untrained models nnviz therefore
shows a *plausible* output that matches the chosen input (what a trained model would typically answer), and the UI
labels it "Illustrative output". Internal activations are untouched: they are the real values of the random network.
Trained models (pretrained torchvision classifiers, the toy net) always show their real outputs.
"""
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- image classifiers (input order = zoo.PHOTOS + synthetic + noise)
CLASSIFIER = [
    [("tabby", .46), ("tiger cat", .27), ("Egyptian cat", .19), ("lynx", .03), ("Persian cat", .01)],
    [("espresso", .71), ("cup", .14), ("coffee mug", .09), ("eggnog", .02), ("chocolate sauce", .01)],
    [("space shuttle", .38), ("military uniform", .17), ("crash helmet", .12), ("parachute", .06), ("wig", .03)],
    [("missile", .52), ("space shuttle", .24), ("projectile", .15), ("drilling platform", .03), ("crane", .02)],
    [("pinwheel", .21), ("shower curtain", .14), ("window shade", .09), ("jigsaw puzzle", .07), ("velvet", .05)],
    [("window screen", .08), ("chain mail", .06), ("velvet", .05), ("dishrag", .04), ("stole", .04)],
]

# ---------------------------------------------------------------- language models: prompt -> plausible next tokens
PROMPTS = [
    ("The capital of France is", [(" Paris", .62), (" a", .07), (" the", .05), (" located", .03), (" known", .02)]),
    ("The cat sat on the", [(" mat", .41), (" floor", .13), (" couch", .09), (" bed", .06), (" window", .03)]),
    ("1, 1, 2, 3, 5, 8,", [(" 13", .88), (" 21", .03), (" 12", .02), (" 11", .01), (" 14", .01)]),
    ("def fibonacci(n):", [("⏎", .57), (" return", .14), (" if", .09), (" #", .05), (" \"\"\"", .04)]),
    ("Neural networks learn by", [(" adjusting", .31), (" minimizing", .18), (" updating", .12), (" example", .06), (" back", .04)]),
]
PROMPT_NAMES = [p for p, _ in PROMPTS]


def _jitter(items, seed):
    """Small seed-dependent variation so 'New noise' changes the numbers a little, renormalised to the same mass."""
    if not seed:
        return items
    g = torch.Generator().manual_seed(seed)
    ps = torch.tensor([p for _, p in items]) * (1 + 0.15 * (torch.rand(len(items), generator=g) - 0.5))
    ps = ps * (sum(p for _, p in items) / float(ps.sum()))
    order = ps.argsort(descending=True).tolist()
    return [(items[i][0], float(ps[i])) for i in order]


def classifier_top(cls, seed, labels):
    items = _jitter(CLASSIFIER[min(cls, len(CLASSIFIER) - 1)], seed)
    return [{"i": labels.index(n) if labels and n in labels else -1, "label": n, "p": p} for n, p in items]


def token_top(cls, seed):
    items = _jitter(PROMPTS[min(cls, len(PROMPTS) - 1)][1], seed)
    return [{"i": -1, "label": t, "p": p} for t, p in items]


# ---------------------------------------------------------------- generative / dense-output models: images derived from the input
def _blur(x, k=5):
    c = x.shape[1]
    w = torch.ones(c, 1, k, k) / (k * k)
    return F.conv2d(F.pad(x, [k // 2] * 4, mode="replicate"), w, groups=c)


def vae_images(x, seed):
    return {"Output": _blur(x, 3) * 0.9}          # a slightly soft reconstruction, like a real VAE's


def unet_images(x, seed):
    g = _blur(x.mean(1, keepdim=True), 7)
    mask = (g > g.mean()).float()                # foreground / background segmentation of the input
    return {"Output": torch.cat([1 - mask, mask], 1)}


def ddpm_images(x, meta):
    if not meta:
        return {}
    eps, x0, ab = meta["eps"], meta["x0"], meta["ab"]
    pred = 0.85 * eps + 0.15 * _blur(eps, 3)      # a good denoiser predicts the true noise closely
    est = ((x[0] - (1 - ab) ** 0.5 * pred) / ab ** 0.5).clamp(-1.5, 1.5)
    return {"Output": pred, "Predicted noise ε̂": pred, "x₀ estimate from ε̂": est}


def dcgan_images(cls, seed, params, photo):
    """Generator: the latent z walks between two images (alpha slider)."""
    a = float(params.get("alpha", 0)) if params else 0.0
    names = ["cat", "coffee", "astronaut", "rocket"]
    i = (seed + cls) % len(names)
    A, B = photo(names[i], 64), photo(names[(i + 1) % len(names)], 64)
    s = float(params.get("scale", 1)) if params else 1.0
    return {"Output": ((1 - a) * A + a * B) * min(1.0, max(0.15, s))}
