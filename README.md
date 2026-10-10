# nnviz — explore neural networks in 3D

A local tool that traces a PyTorch model and shows it as an interactive 3D scene: real tensors as volumes,
layers as the operations acting on them (sliding conv kernels, activation gates, per-head attention maps, MoE routers…),
with live activations for a chosen input.

![status](https://img.shields.io/badge/runs-locally-blue)

## Features
- **Model zoo**: ResNet-18/50, VGG-16, AlexNet, MobileNetV3, EfficientNet-B0, DenseNet-121, ViT-B/16, ViT-Tiny, GPT-2 (small/tiny),
  Llama-style, GPT-OSS-style (MoE + GQA + sliding window), U-Net, VAE, DCGAN (G/D), DDPM U-Net. Random weights unless
  *pretrained* is ticked (torchvision models; needs internet).
- **Import your own model**: a `.py` file defining `build()` (optionally `example_input()` and `CLASSES`), or a whole
  pickled module `.pt`/`.pth`. See `examples/my_model.py`.
- **Layer inspection**: click a layer for a plain-language description, shapes, parameters, FLOPs, kernels / weight matrices,
  attention maps, router gates.
- **Architecture analysis**: bottleneck ("deepest level") detection for autoencoders / U-Nets, encoder–decoder stages,
  skip connections (residual vs concatenation vs bypassing), tensor-size envelope and profile.
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
