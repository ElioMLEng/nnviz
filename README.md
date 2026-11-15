# nnviz — explore neural networks in 3D

A local tool that traces a PyTorch model and shows it as an interactive 3D scene: real tensors as volumes,
layers as the operations acting on them (sliding conv kernels, activation gates, per-head attention maps, MoE routers…),
with live activations for a chosen input.

![status](https://img.shields.io/badge/runs-locally-blue)

## Features
- **Built-in tutorial**: the **?** button (or the `?` key) opens an 8-step guide covering navigation, models, inspection, layouts, display options, importing your own model and shortcuts. `?help=1` in the URL opens it directly.
- **Explain NNs** (button left of **?**, or the `L` key): a 10-chapter illustrated course. *Foundations*: machine learning, deep learning, neural networks, tensors, GPUs, computation. *Layers*: fully connected, convolution, attention and the other essential layers, each with purpose, analogy, mechanics, strengths/limits and real-world uses. `?learn=8` opens chapter 8. Content lives in `static/learn.js`.
- **Loading indicator** to the right of the panel, naming what is loading (model build, graph, activations).
- **Tabbed control panel** (Model / View / Display) that remembers its tab and collapsed state; `?tab=view` selects a tab from the URL.
- **Model zoo**: ResNet-18/50, VGG-16, AlexNet, MobileNetV3, EfficientNet-B0, DenseNet-121, ViT-B/16, ViT-Tiny, GPT-2 (small/tiny),
  Llama-style, GPT-OSS-style (MoE + GQA + sliding window), U-Net, VAE, DCGAN (G/D), DDPM U-Net. Random weights unless
  *pretrained* is ticked (torchvision models; needs internet).
- **Import your own model**: a `.py` file defining `build()` (optionally `example_input()` and `CLASSES`), or a whole
  pickled module `.pt`/`.pth`. See `examples/my_model.py`.
- **Layer inspection**: click a layer for a plain-language description, shapes, parameters, FLOPs, kernels / weight matrices,
  attention maps, router gates.
- **Architecture analysis**: bottleneck ("deepest level") detection for autoencoders / U-Nets, encoder–decoder stages,
  skip connections (residual vs concatenation vs bypassing), tensor-size envelope and profile.
- **Depth levels** (line view): every layer hangs lower the more its feature map has been downsampled, so a U-Net draws
  a U with the bottleneck at the bottom and a ResNet a descending staircase. A vertical ruler lists each level
  (resolution, channels, layer count). Offered only for models with 3+ resolution levels; toggle with *depth levels*.
- **Skip-connection highlight**: select a layer or block and its skip links light up in green, with the layer at the
  other end outlined and labelled (also listed in the side panel).
- **Colour layers by** parameters, FLOPs, activation size, sparsity, mean |activation| or residual change.
- **Collapsible blocks** with animated fold / unfold, and layouts: line, folded grid, spiral.
- **Model cards** for each zoo model (what it is known for, what it introduced, what to look for).
- **Image gallery and controls** where they make sense (DDPM timestep, GAN latent interpolation).

## Run
```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python server.py 8765         # optional 2nd argument: a model id, e.g. resnet18
```
Open http://127.0.0.1:8765. three.js is vendored in `static/vendor/`, so no CDN is needed.

Options (command line or environment variable):

| Flag / argument | Environment variable | Default | Meaning |
|---|---|---|---|
| `PORT` (positional) | `NNVIZ_PORT` | `8765` | listening port |
| `MODEL` (positional) | `NNVIZ_MODEL` | `toy-mixed` | model loaded first |
| `--host` | `NNVIZ_HOST` | `127.0.0.1` | interface to bind |
| `--no-browser` | `NNVIZ_OPEN_BROWSER=0` | open a tab | do not open a browser |
| — | `NNVIZ_ALLOW_UPLOAD=0` | `1` | disable model import |
| — | `NNVIZ_UPLOAD_DIR` | `./uploads` | where imported files are stored |
| — | `NNVIZ_MAX_UPLOAD_MB` | `500` | upload size limit |

## Docker (Linux)
CPU-only image based on `python:3.12-slim`, running as an unprivileged user.

```bash
docker compose up --build            # then open http://127.0.0.1:8765
# or, without compose:
docker build -t nnviz .
docker run --rm -p 127.0.0.1:8765:8765 -v nnviz-data:/data nnviz
docker run --rm -p 127.0.0.1:8765:8765 -e NNVIZ_MODEL=resnet50 -e NNVIZ_ALLOW_UPLOAD=0 nnviz
```

- Publish the port to **localhost only** (`127.0.0.1:8765:8765`, as above). Inside the container the server binds
  `0.0.0.0`; importing a model executes its code in the container, so do not expose the port to a network.
- `/data` is a volume holding imported models (`/data/uploads`) and downloaded pretrained weights (`/data/torch`).
- Pretrained weights are downloaded on demand (torchvision models); the container needs internet access for that.
- GPU is not used. The image is CPU-only to stay small.

### Opening the tool
- **Container on the same machine:** open http://127.0.0.1:8765 in your browser.
- **Container on a remote or headless server:** your browser cannot reach it directly. From *your own computer*,
  open an SSH tunnel:

  ```bash
  ssh -L 8765:127.0.0.1:8765 your_user@your_server
  ```

  Leave that terminal open, then open http://127.0.0.1:8765 on your own computer. This is also the safe way to do it:
  the tool is never exposed to the internet.

### Useful commands
| What you want | Command |
|---|---|
| Run in the background | `docker compose up -d` |
| Follow the logs | `docker compose logs -f` |
| Stop and remove the container | `docker compose down` |
| Update to a new version | `git pull` then `docker compose up --build` |

### Troubleshooting
| Symptom | Likely fix |
|---|---|
| `permission denied` when running `docker` | Use `sudo docker …`, or add yourself to the `docker` group and log in again. |
| "port is already allocated" | Change the left side of the port mapping in `docker-compose.yml`, e.g. `"127.0.0.1:9000:8765"`. |
| Blank page right after start | Wait about 20 seconds (the toy model trains at start-up), then reload. |
| Large models are slow or crash | GPT-2 / ViT-B need a few GB of RAM; try a smaller model such as `resnet18`. |
| "pretrained" fails | Weights are downloaded on demand, so the container needs internet access. |

## Importing your own model
Click **import…** in the *Model* panel and choose a file. Two kinds of file are accepted.

### 1. A Python file defining `build()` (recommended)
```python
# my_model.py
import torch, torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 10)
    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))

def build():                       # required: return an nn.Module
    return Net().eval()

def example_input():               # optional but recommended: one tensor, batch size 1
    return torch.randn(1, 3, 32, 32)

CLASSES = ["example", "noise"]     # optional: labels for the input selector
```
- `build()` is required. If it is missing, the file's single `nn.Module` subclass is used, provided it can be
  constructed with no arguments.
- `example_input()` tells the tool what to feed the model. Without it, give an *input spec* under
  **import options** (see below).
- To use trained weights, load them inside `build()` (`model.load_state_dict(torch.load(...))`).
- A working example is in `examples/my_model.py`.

### 2. A whole saved model: `.pt` / `.pth`
Created with `torch.save(model, "model.pt")` — the **entire module**, not just its weights.

- The file does not say what to feed the model, so fill in **import options → input spec** before importing:
  `1,3,224,224` for an image, or `int:1,32:50257` for 32 token ids with a vocabulary of 50 257.
- A pickled module refers to its class by name and only loads if that class's code is importable where the tool runs.
  Standard `torch.nn` / torchvision models work; your own classes usually do not (especially inside the Docker
  container). For custom models use the `.py` route.

### What is not supported
| File | Why / what to do instead |
|---|---|
| `state_dict` only (`.pt`, `.pth`, `.safetensors`) | It has weights but no architecture. Write a `.py` with `build()` that creates the model and loads them. |
| TorchScript (`torch.jit.save`, `torch.jit.trace` / `script`) | Layers run inside the TorchScript interpreter and cannot be traced layer by layer. Import the original model instead. |
| ONNX, TensorFlow / Keras, JAX | PyTorch `nn.Module`s only. |
| Hugging Face checkpoints | Same as a `state_dict`: define the model class in a `.py`. |

### Good to know
- **Batch size 1**: the model must accept a batch of 1 (`1,3,224,224`); that is what is displayed.
- **One input tensor** works best. Several inputs are possible if `example_input()` returns a tuple (like the DDPM
  U-Net), but the "random noise" input choice is not offered for those.
- **Level of detail depends on tracing**: most models are traced with `torch.fx` (residual adds and tensor ops are
  visible). Models with data-dependent control flow fall back to a simpler trace of the layer calls only; the status
  line then says "hooks".
- **Richer attention view** for `nn.MultiheadAttention`, or for your own class whose name ends in `Attention` and
  exposes `qkv` and `o` linear layers plus `heads`.
- **Security**: the file is executed by the tool. Only import code you trust. Import can be disabled with
  `NNVIZ_ALLOW_UPLOAD=0`.

## How it works
- `extract.py` traces the model with `torch.fx` (falling back to forward hooks), builds the graph, samples large tensors
  for display (≤24 channels/features, ≤12×12 cells, ≤4 attention heads — true shapes are always reported), and runs
  the architecture analysis.
- `server.py` is a small standard-library HTTP server (`/api/models`, `/api/graph`, `/api/activations`, `/api/load`, `/api/upload`).
- `static/index.html` is the three.js front-end; `static/info.js` holds the layer descriptions;
  `modelinfo.py` holds the model cards.

## Security note
Importing a `.py` or `.pt` file **executes arbitrary code** on your machine. The server binds to `127.0.0.1` only;
do not expose it to a network and only import files you trust.

## Tests
`python selftest.py [model-id …]` extracts every registry model and prints graph sizes;
`python analysis_test.py` prints the bottleneck / stage analysis.

## Status
Personal research tool. Model facts in the cards come from the original papers/model cards as best known; check
them before citing. The GPT-OSS-style and DDPM models are scaled-down, randomly initialised illustrations of the
architecture, not the released weights.
