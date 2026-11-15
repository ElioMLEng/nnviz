import argparse
import gc
import importlib.util
import json
import os
import re
import sys
import threading
import time
import traceback
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch
import torch.nn as nn

from extract import NetGraph
from zoo import REGISTRY, param_count

HERE = Path(__file__).parent
ROOT = (HERE / "static").resolve()
UPLOADS = Path(os.environ.get("NNVIZ_UPLOAD_DIR", HERE / "uploads"))
ALLOW_UPLOAD = os.environ.get("NNVIZ_ALLOW_UPLOAD", "1") != "0"
MAX_UPLOAD = int(os.environ.get("NNVIZ_MAX_UPLOAD_MB", 500)) * 1024 * 1024
UPLOADS.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
STATE = {"net": None, "id": None, "name": None, "make_input": None, "classes": [], "load_s": 0.0,
         "controls": [], "extras": None, "key": None, "build": None, "graph": None}

# ---------------------------------------------------------------- disk cache
# Graphs and activations of zoo models are computed once and kept in .cache/: switching models or inputs afterwards
# only reads a file, the network is never rebuilt. The key includes a hash of the code that produces the data.
import hashlib
CACHE = Path(os.environ.get("NNVIZ_CACHE_DIR", HERE / ".cache"))
_code = hashlib.sha1(b"".join((HERE / f).read_bytes() for f in
                              ("extract.py", "zoo.py", "zoo_extra.py", "zoo_llm.py", "toynet.py", "server.py") if (HERE / f).exists()))
CACHE_VER = _code.hexdigest()[:10]


def cache_path(key, name):
    return CACHE / CACHE_VER / key / f"{name}.json" if key else None


def cache_get(key, name):
    f = cache_path(key, name)
    return f.read_bytes() if f and f.exists() else None


def cache_put(key, name, body):
    f = cache_path(key, name)
    if f is None: return
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp"); tmp.write_bytes(body); tmp.replace(f)


def ensure_net():
    """Build + trace the model only when something must actually be computed (cache miss)."""
    if STATE["net"] is None and STATE["build"] is not None:
        STATE["build"]()
    return STATE["net"]


def to_image(t):
    """Tensor -> {c,h,w,data} (uint8, HWC) when it looks like an image (1/2/3 channels), else None."""
    from extract import first_tensor
    import torch.nn.functional as F
    t = first_tensor(t)
    if t is None: return None
    t = t.detach().float()
    if t.ndim == 4: t = t[0]
    if t.ndim != 3 or t.shape[0] not in (1, 2, 3) or min(t.shape[1:]) < 4: return None
    if t.shape[0] == 2: t = (t[1] - t[0])[None]                  # 2-class map -> difference
    h, w = t.shape[1:]
    if max(h, w) > 96: t = F.adaptive_avg_pool2d(t[None], (min(h, 96), min(w, 96)))[0]
    lo, hi = t.min(), t.max()
    t = (t - lo) / (hi - lo + 1e-9)
    return {"c": int(t.shape[0]), "h": int(t.shape[1]), "w": int(t.shape[2]),
            "data": (t.permute(1, 2, 0).reshape(-1) * 255).round().int().tolist()}


def parse_input_spec(spec):
    """'1,3,224,224' -> float randn; 'int:1,32:50257' -> random token ids (vocab optional)."""
    if spec.startswith("int:"):
        parts = spec[4:].split(":")
        shape = [int(v) for v in parts[0].split(",")]
        vocab = int(parts[1]) if len(parts) > 1 else 1000
        return lambda cls, seed: torch.randint(0, vocab, shape, generator=torch.Generator().manual_seed(seed))
    shape = [int(v) for v in spec.split(",")]
    return lambda cls, seed: torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def install(model, make_input, classes, model_id, name, controls=None, extras=None, post=None, labels=None, trained=False, key=None):
    t0 = time.time()
    example = make_input(0, 0)
    net = NetGraph(model.eval(), example)
    if post: post(net)                      # e.g. true-scale numbers for the frontier LLM twins
    with lock:
        old = STATE["net"]
        STATE.update(net=net, id=model_id, name=name, make_input=make_input, classes=classes, load_s=time.time() - t0,
                     labels=labels, trained=trained, key=key, build=None, graph=None,
                     controls=controls or [], extras=extras)
    del old
    gc.collect()


def load_registry(model_id, pretrained):
    e = REGISTRY[model_id]
    key = model_id + ("-pretrained" if pretrained else "")
    name = model_id + (" (pretrained)" if pretrained else "")
    labels = e["labels"]() if e.get("labels") else None
    trained = bool(pretrained) or model_id == "toy-mixed"

    def build():
        model = e["build"](pretrained=pretrained) if pretrained else e["build"]()
        install(model, e["make_input"], e["classes"], model_id, name, controls=e.get("controls"), extras=e.get("extras"),
                post=e.get("post"), labels=labels, trained=trained, key=key)

    t0 = time.time()
    graph = cache_get(key, "graph")
    if graph is None:                       # first time: build, trace, and remember the graph
        build()
        STATE["build"] = build
        return
    with lock:                              # cached: nothing is built until an uncached activation is requested
        old = STATE["net"]
        STATE.update(net=None, id=model_id, name=name, make_input=e["make_input"], classes=e["classes"], labels=labels,
                     trained=trained, key=key, build=build, graph=graph, controls=e.get("controls") or [],
                     extras=e.get("extras"), load_s=time.time() - t0)
    del old
    gc.collect()


def load_upload(path, input_spec):
    suffix = path.suffix.lower()
    make_input, classes = None, ["example input", "random noise"]
    if suffix == ".py":
        spec = importlib.util.spec_from_file_location("user_model_" + re.sub(r"\W", "_", path.stem), path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "build"):
            model = mod.build()
        else:
            cands = [c for c in vars(mod).values() if isinstance(c, type) and issubclass(c, nn.Module)
                     and c.__module__ == mod.__name__]
            if len(cands) != 1:
                raise ValueError("define build() returning an nn.Module (and optionally example_input())")
            model = cands[0]()
        if hasattr(mod, "example_input"):
            ex = mod.example_input()
            make_input = lambda cls, seed, ex=ex: ex if cls == 0 else torch.randn_like(ex.float()).to(ex.dtype) \
                if ex.is_floating_point() else ex
        classes = getattr(mod, "CLASSES", classes)
    elif suffix in (".pt", ".pth"):
        try:
            model = torch.load(path, weights_only=False, map_location="cpu")
        except Exception:
            model = torch.jit.load(str(path), map_location="cpu")
        if isinstance(model, dict):
            raise ValueError("this file is a state_dict (weights only, no architecture). "
                             "Upload a .py defining build() instead, or torch.save(model) the whole module.")
        if isinstance(model, torch.jit.ScriptModule):
            raise ValueError("TorchScript archives (torch.jit.save / torch.jit.trace) are not supported: their layers run "
                             "inside the TorchScript interpreter, so they cannot be traced layer by layer. Import the "
                             "original model instead: a .py defining build(), or torch.save(model) of the nn.Module.")
    else:
        raise ValueError("supported: .py (defining build()) or .pt/.pth (a whole torch.save(model) pickled nn.Module)")
    if make_input is None:
        if not input_spec:
            raise ValueError("input shape required, e.g. 1,3,224,224 or int:1,32:50257")
        make_input = parse_input_spec(input_spec)
    install(model, make_input, classes, "upload:" + path.name, path.name)


def top_k(out, labels, k=5):
    """Top-k output classes (or next tokens) with probabilities, for the output card."""
    from extract import first_tensor
    out = first_tensor(out)
    if out is None: return None
    l = out[0] if out.ndim >= 2 else out
    if l.ndim == 2: l = l[-1]
    if l.ndim != 1 or l.numel() < 2: return None
    p = l.float().softmax(-1)
    v, i = p.topk(min(k, p.numel()))
    seq = out.ndim == 3
    name = lambda j: labels[j] if labels and len(labels) == p.numel() else (f"token #{j}" if seq else f"class #{j}")
    return [{"i": int(j), "label": name(int(j)), "p": float(x)} for x, j in zip(v, i)]


def prediction(out):
    l = out[0] if out.ndim >= 2 else out
    if l.ndim == 2: l = l[-1]                      # sequence model: next-token logits at last position
    if l.ndim != 1 or l.numel() < 2: return None, None, 0
    p = l.float().softmax(-1)
    return int(p.argmax()), float(p.max()), int(l.numel())


class H(BaseHTTPRequestHandler):
    def _raw(self, body, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, f):
        types = {".html": "text/html; charset=utf-8", ".js": "text/javascript"}
        body = f.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", types.get(f.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/models":
                return self._json({"current": STATE["id"], "upload": ALLOW_UPLOAD, "models": [
                    {"id": k, "group": v["group"], "desc": v["desc"], "pretrained": v["pretrained"],
                     "params": param_count(k), "info": v.get("info")}
                    for k, v in REGISTRY.items()]})
            if u.path == "/api/graph":
                if STATE["graph"] is None:
                    net = ensure_net()
                    STATE["graph"] = json.dumps({**net.graph, "classes": STATE["classes"], "model": STATE["id"],
                                                 "name": STATE["name"], "load_s": round(STATE["load_s"], 1),
                                                 "controls": STATE["controls"]}).encode()
                    cache_put(STATE["key"], "graph", STATE["graph"])
                    return self._raw(STATE["graph"])
                g = json.loads(STATE["graph"])     # cached: report this (fast) load time
                g["load_s"], g["cached"] = round(STATE["load_s"], 2), True
                return self._json(g)
            if u.path == "/api/activations":
                cls, seed = int(q.get("cls", [0])[0]), int(q.get("seed", [0])[0])
                params = {k[2:]: float(v[0]) for k, v in q.items() if k.startswith("p_")}
                ckey = f"a_{cls}_{seed}" + "".join(f"_{k}={params[k]:g}" for k in sorted(params))
                hit = cache_get(STATE["key"], ckey)
                if hit is not None:
                    return self._raw(hit)
                ensure_net()
                with lock:
                    x = STATE["make_input"](cls, seed, **params) if params else STATE["make_input"](cls, seed)
                    acts, out = STATE["net"].activations(x)
                    images = []
                    xin = x[0] if isinstance(x, tuple) else x
                    for name, tens in [("Input", xin), ("Output", out)]:
                        im = to_image(tens)
                        if im: images.append({"name": name, **im})
                    if STATE["extras"]:
                        for name, tens in STATE["extras"](x, out).items():
                            im = to_image(tens)
                            if im: images.append({"name": name, **im})
                pred, prob, n = prediction(out)
                body = json.dumps({"acts": acts, "pred": pred, "prob": prob, "n_out": n, "images": images,
                                   "top": top_k(out, STATE.get("labels") or (STATE["classes"] if STATE["id"] == "toy-mixed" else None)),
                                   "trained": STATE.get("trained", False)}).encode()
                cache_put(STATE["key"], ckey, body)
                return self._raw(body)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        f = (ROOT / ("index.html" if u.path == "/" else u.path.lstrip("/"))).resolve()
        if f.is_file() and ROOT in f.parents:
            return self._file(f)
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/load":
                if q["model"] not in REGISTRY: raise ValueError("unknown model")
                load_registry(q["model"], q.get("pretrained") == "1")
            elif u.path == "/api/upload":
                if not ALLOW_UPLOAD:
                    raise PermissionError("model import is disabled on this server (NNVIZ_ALLOW_UPLOAD=0)")
                name = re.sub(r"[^\w.\-]", "_", q.get("name", "model.py"))
                size = int(self.headers.get("Content-Length", 0))
                if size > MAX_UPLOAD:
                    raise ValueError(f"file too large ({size} bytes, limit {MAX_UPLOAD})")
                data = self.rfile.read(size)
                p = UPLOADS / name
                p.write_bytes(data)
                load_upload(p, q.get("input", ""))
            else:
                return self.send_error(404)
            return self._json({"ok": True, "model": STATE["id"]})
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description="nnviz: interactive 3D neural network explorer")
    ap.add_argument("port", nargs="?", type=int, default=int(os.environ.get("NNVIZ_PORT", 8765)))
    ap.add_argument("model", nargs="?", default=os.environ.get("NNVIZ_MODEL", "toy-mixed"),
                    help="model id to load first (see /api/models)")
    ap.add_argument("--host", default=os.environ.get("NNVIZ_HOST", "127.0.0.1"),
                    help="interface to bind; use 0.0.0.0 inside a container (publish the port to localhost only)")
    ap.add_argument("--no-browser", action="store_true",
                    default=os.environ.get("NNVIZ_OPEN_BROWSER", "1") == "0", help="do not open a browser tab")
    a = ap.parse_args()
    if a.model not in REGISTRY:
        sys.exit(f"unknown model '{a.model}'. choices: {', '.join(REGISTRY)}")
    load_registry(a.model, False)
    loopback = a.host in ("127.0.0.1", "localhost", "::1")
    shown = "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host
    print(f"nnviz on http://{shown}:{a.port}  (bound to {a.host}, model: {a.model}, "
          f"import {'enabled' if ALLOW_UPLOAD else 'disabled'})", flush=True)
    if not loopback and ALLOW_UPLOAD:
        print("WARNING: model import executes uploaded code. Only publish this port to localhost, "
              "or set NNVIZ_ALLOW_UPLOAD=0.", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), H)
    if not a.no_browser and loopback:
        threading.Timer(0.5, lambda: webbrowser.open(f"http://{shown}:{a.port}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
