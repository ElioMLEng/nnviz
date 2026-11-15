# nnviz

Explore neural networks in 3D, from a toy CNN to Llama 3.1 405B, with a built-in course (**Explain NNs**) and tutorial (**?**).

**Live: https://eliomleng.github.io/nnviz/** (nothing to install)

## Quick start (no install)

```bash
git clone https://github.com/ElioMLEng/nnviz.git
```

Then open `nnviz/site/index.html` in your browser (double-click it).

## Full version (import your own PyTorch models)

```bash
cd nnviz
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

It opens http://127.0.0.1:8765 in your browser.

More: [docs/DETAILS.md](docs/DETAILS.md)
