"""Export a Python-free, static copy of nnviz into ./site (open site/index.html directly, or host it on any static host).

Needs the normal server running once:  python server.py 8765   then   python export_static.py [http://127.0.0.1:8765]
For each zoo model it records the traced graph and a grid of activations (a few inputs × noise seeds, plus a coarse grid
for the model-specific sliders). Data files are JS (`__nnvizData(key, {...})`) so they load from file:// without a server.
"""
import itertools, json, shutil, sys, time, urllib.request
from pathlib import Path

_args = [a for a in sys.argv[1:] if not a.startswith("--")]
BASE = (_args[0] if _args else "http://127.0.0.1:8765").rstrip("/")
HERE = Path(__file__).parent
OUT = HERE / "site"
MAX_CLASSES, SEEDS, GRID_POINTS = 6, 2, 5   # inputs per model, noise seeds, values per slider


def call(path, method="GET"):
    req = urllib.request.Request(BASE + path, method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    if isinstance(j, dict) and j.get("error"):
        raise RuntimeError(j["error"])
    return j


def shrink(o):
    """Round floats to 3 significant digits: about half the size, invisible in the views."""
    if isinstance(o, float):
        return float(f"{o:.3g}")
    if isinstance(o, list):
        return [shrink(v) for v in o]
    if isinstance(o, dict):
        return {k: shrink(v) for k, v in o.items()}
    return o


def write(key, obj):
    f = OUT / "data" / f"{key}.js"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f"__nnvizData({json.dumps(key)},{json.dumps(shrink(obj), separators=(',', ':'), ensure_ascii=False)});",
                 encoding="utf-8")
    return f.stat().st_size


def grid(c):
    """Coarse, evenly spaced slider values; the slider step is changed to match so the UI only offers recorded values."""
    lo, hi = float(c["min"]), float(c["max"])
    vals = [round(lo + (hi - lo) * i / (GRID_POINTS - 1), 6) for i in range(GRID_POINTS)]
    default = min(vals, key=lambda v: abs(v - float(c["default"])))
    return vals, {**c, "step": (hi - lo) / (GRID_POINTS - 1), "default": default}


def copy_ui():
    """Copy the page and its scripts into site/, wiring in the static API shim. Leaves site/data untouched."""
    shutil.copytree(HERE / "static", OUT, ignore=shutil.ignore_patterns("2d.html", "README.md"), dirs_exist_ok=True)
    html = (OUT / "index.html").read_text(encoding="utf-8")
    html = html.replace('<script src="vendor/three.min.js">', '<script src="static-shim.js"></script>\n<script src="vendor/three.min.js">', 1)
    (OUT / "index.html").write_text(html, encoding="utf-8")
    shutil.copy(HERE / "static-shim.js", OUT / "static-shim.js")


def main():
    if "--ui-only" in sys.argv:   # UI changed, data didn't: no server needed
        copy_ui(); print(f"UI refreshed in {OUT}"); return
    if OUT.exists():
        shutil.rmtree(OUT)
    copy_ui()

    ms = call("/api/models")
    ids = [m["id"] for m in ms["models"]]
    total, done = 0, []
    for mid in ids:
        t = time.time()
        try:
            call(f"/api/load?model={mid}", "POST")
            g = call("/api/graph")
        except Exception as e:
            print(f"  skip {mid}: {e}")
            continue
        n_cls = max(1, min(MAX_CLASSES, len(g.get("classes") or [])))
        grids, ctrls = [], []
        for c in g.get("controls") or []:
            vals, c2 = grid(c)
            grids.append(vals); ctrls.append(c2)
        g["controls"] = ctrls
        g["static"] = {"classes": n_cls, "seeds": SEEDS, "grids": grids}
        size = write(f"{mid}/graph", g)
        combos = list(itertools.product(*[range(len(v)) for v in grids])) or [()]
        for cls, seed, combo in itertools.product(range(n_cls), range(SEEDS), combos):
            q = "".join(f"&p_{c['id']}={grids[i][k]}" for i, (c, k) in enumerate(zip(ctrls, combo)))
            a = call(f"/api/activations?cls={cls}&seed={seed}{q}")
            size += write(f"{mid}/a_{cls}_{seed}" + "".join(f"_{k}" for k in combo), a)
        total += size; done.append(mid)
        print(f"  {mid:22s} {n_cls} inputs × {SEEDS} seeds × {len(combos)} slider points  {size/1e6:6.1f} MB  {time.time()-t:5.1f}s", flush=True)

    models = {**ms, "upload": False, "current": "toy-mixed" if "toy-mixed" in done else done[0],
              "models": [{**m, "pretrained": False} for m in ms["models"] if m["id"] in done]}
    write("models", models)
    print(f"done: {len(done)} models, {total/1e6:.0f} MB in {OUT}")


if __name__ == "__main__":
    main()
