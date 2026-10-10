"""Toy mixed network: Conv -> Pool -> Conv -> tokens -> LayerNorm + SelfAttention (residual) -> MLP."""
import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.dim = heads, dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)
        self.last_attn = None

    def forward(self, x):
        B, T, D = x.shape
        h = self.heads
        q, k, v = self.qkv(x).view(B, T, 3, h, D // h).permute(2, 0, 3, 1, 4)
        a = (q @ k.transpose(-2, -1) / math.sqrt(D // h)).softmax(-1)
        self.last_attn = a.detach()
        return self.o((a @ v).transpose(1, 2).reshape(B, T, D))


class ToyNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.act1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.act2 = nn.ReLU()
        self.norm = nn.LayerNorm(8)
        self.attn = SelfAttention(8, 2)
        self.fc1 = nn.Linear(8, 16)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(16, n_classes)

    def forward(self, x):                        # x: (B,1,8,8)
        x = self.pool(self.act1(self.conv1(x)))  # (B,4,4,4)
        x = self.act2(self.conv2(x))             # (B,8,4,4)
        t = x.flatten(2).transpose(1, 2)         # (B,16,8) tokens
        t = t + self.attn(self.norm(t))          # residual attention block
        z = t.mean(1)                            # (B,8)
        return self.fc2(self.act3(self.fc1(z)))  # (B,3)


CLASSES = ["horizontal", "vertical", "diagonal"]


def make_sample(cls, gen, noise=0.25):
    img = torch.randn(8, 8, generator=gen) * noise
    r = int(torch.randint(0, 8, (1,), generator=gen))
    if cls == 0:
        img[r, :] += 1
    elif cls == 1:
        img[:, r] += 1
    else:
        for i in range(8):
            img[i, i] += 1
    return img


def make_batch(n, gen):
    ys = torch.randint(0, 3, (n,), generator=gen)
    xs = torch.stack([make_sample(int(y), gen) for y in ys]).unsqueeze(1)
    return xs, ys


def trained_model(steps=300, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = ToyNet()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(steps):
        x, y = make_batch(64, gen)
        loss = nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    x, y = make_batch(256, gen)
    acc = (m(x).argmax(1) == y).float().mean().item()
    print(f"toy model trained: final loss {loss.item():.3f}, acc {acc:.2f}")
    return m
