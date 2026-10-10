"""Turn any torch module into a JSON-able graph + per-input activations, sized for display.

Tracing: torch.fx first (gives residuals/functional ops), falling back to forward-hook tracing.
Large tensors are *sampled for display only* (channels / tokens / features / spatial cells);
true shapes and parameter counts are always reported.
"""
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

ACT = (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.Softmax, nn.Hardswish,
       nn.Hardsigmoid, nn.ELU)
POOL = (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)
NORM = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)
PASS = (nn.Dropout, nn.Identity, nn.Dropout2d)       # shown as nothing: wired straight through


def kind_of(m):
    n = type(m).__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)): return "conv"
    if isinstance(m, nn.Upsample) or n.startswith("Upsampling"): return "upsample"
    if n.endswith("MoE"): return "moe"
    if isinstance(m, nn.Linear): return "linear"
    if isinstance(m, nn.Embedding): return "embedding"
    if isinstance(m, NORM) or n.endswith("Norm"): return "norm"
    if isinstance(m, ACT): return "activation"
    if isinstance(m, POOL): return "pool"
    if n.endswith("Attention"): return "attention"
    return "op"


class Tracer(fx.Tracer):
    def is_leaf_module(self, m, qn):
        return kind_of(m) != "op" or isinstance(m, PASS) or super().is_leaf_module(m, qn)


def first_tensor(v):
    if isinstance(v, torch.Tensor): return v
    if isinstance(v, (tuple, list)):
        for e in v:
            t = first_tensor(e)
            if t is not None: return t
    return None


# ---------------------------------------------------------------- tracing back-ends
class Recorder(fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.vals = {}

    def run_node(self, n):
        out = super().run_node(n)
        self.vals[n.name] = out
        return out


class FxTrace:
    """nodes: [{name, op, mod, label, inputs}] in topological order."""

    def __init__(self, model, example):
        self.gm = fx.GraphModule(model, Tracer().trace(model)).eval()
        self.nodes = []
        for n in self.gm.graph.nodes:
            if n.op == "get_attr":
                continue
            mod = self.gm.get_submodule(n.target) if n.op == "call_module" else None
            label = type(mod).__name__ if mod is not None else (
                n.op if n.op in ("placeholder", "output") else getattr(n.target, "__name__", str(n.target)))
            self.nodes.append({"name": n.name, "op": n.op, "mod": mod, "label": label,
                               "path": str(n.target) if n.op == "call_module" else "",
                               "inputs": [i.name for i in n.all_input_nodes if i.op != "get_attr"],
                               "getitem": n.op == "call_function" and getattr(n.target, "__name__", "") == "getitem"
                                          and n.args and isinstance(n.args[1], int) and n.args[1] == 0})

    @torch.no_grad()
    def run(self, x):
        rec = Recorder(self.gm)
        out = rec.run(*x) if isinstance(x, tuple) else rec.run(x)
        return rec.vals, out


class HookTrace:
    """Fallback: record leaf-module calls; edges follow tensor identity (falls back to call order)."""

    def __init__(self, model, example):
        self.model = model.eval()
        self.leaves = {qn: m for qn, m in model.named_modules()
                       if qn and (not list(m.children()) or kind_of(m) != "op")}
        self.nodes = None
        self.run(example)

    @torch.no_grad()
    def run(self, x):
        xs = x if isinstance(x, tuple) else (x,)
        vals, produced, nodes, hooks, prev = {}, {id(xs[0]): "input"}, [], [], ["input"]
        vals["input"] = xs[0]
        nodes.append({"name": "input", "op": "placeholder", "mod": None, "label": "input", "inputs": [], "getitem": False})
        counts = {}

        def mk(qn, mod):
            def hook(m, inp, out):
                k = counts.get(qn, 0); counts[qn] = k + 1
                name = qn.replace(".", "_") + (f"_{k}" if k else "")
                ins = []
                for t in inp:
                    t = first_tensor(t)
                    if t is not None and id(t) in produced and produced[id(t)] not in ins: ins.append(produced[id(t)])
                if not ins: ins = [prev[0]]
                o = first_tensor(out)
                if o is not None: produced[id(o)] = name
                vals[name] = out; prev[0] = name
                nodes.append({"name": name, "op": "call_module", "mod": mod, "label": type(mod).__name__,
                              "path": qn, "inputs": ins, "getitem": False})
            return hook
        for qn, m in self.leaves.items():
            hooks.append(m.register_forward_hook(mk(qn, m)))
        try:
            out = self.model(*xs)
        finally:
            for h in hooks: h.remove()
        nodes.append({"name": "output", "op": "output", "mod": None, "label": "output", "inputs": [prev[0]], "getitem": False})
        vals["output"] = out
        if self.nodes is None: self.nodes = nodes
        return vals, out


# ---------------------------------------------------------------- display sampling
def pick(n, cap):
    if n <= cap: return list(range(n))
    return sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})


def strip_batch(t, batch=1):
    return t[0] if t.ndim >= 2 and t.shape[0] == batch else t


def display_info(t, budget, attn=None):
    """t: batch-free tensor. Returns how it is sampled for display."""
    if t.ndim == 3:
        C, H, W = t.shape
        c, s = min(C, 24), min(max(H, W), 12)
        while c * min(H, s) * min(W, s) > budget and (s > 4 or c > 4):
            if s > 4: s -= 1
            else: c -= 1
        return {"type": "vol", "sel": pick(C, c), "hw": (min(H, s), min(W, s)), "units": C}
    if t.ndim == 2:
        T, D = t.shape
        sel, tc = pick(D, 24), 24
        while len(sel) * min(T, tc) > budget and tc > 4: tc -= 1
        return {"type": "grid", "sel": sel, "tok": pick(T, tc), "units": D}
    v = t.reshape(-1)
    return {"type": "vec", "sel": pick(v.numel(), 32), "units": v.numel()}


def to_display(t, info):
    t = t.detach().float()
    if info["type"] == "vol":
        x = t[info["sel"]]
        if tuple(x.shape[1:]) != info["hw"]:
            x = F.adaptive_avg_pool2d(x[None], info["hw"])[0]
        return x
    if info["type"] == "grid":
        return t[info["tok"]][:, info["sel"]]
    return t.reshape(-1)[info["sel"]]


def unit_means(t, info):
    t = t.detach().float()
    if info["type"] == "vol": m = t.mean((1, 2))
    elif info["type"] == "grid": m = t.mean(0)
    else: m = t.reshape(-1)
    return m[info["sel"]]


def r3(x):
    return ((x.detach().float() * 1000).round() / 1000).tolist()


# ---------------------------------------------------------------- weights
def attn_parts(m):
    """(fused qkv weight, output weight, heads, kv_heads) or None."""
    if hasattr(m, "qkv") and hasattr(m, "o"):
        return m.qkv.weight, m.o.weight, m.heads, getattr(m, "kv_heads", m.heads)
    if getattr(m, "in_proj_weight", None) is not None and hasattr(m, "out_proj"):
        return m.in_proj_weight, m.out_proj.weight, m.num_heads, m.num_heads
    return None


def kv_rows(qkv, o, H, Hkv):
    """Return (Wq, Wk_expanded, Wv_expanded), each (D, in): K/V repeated across the query heads that share them."""
    D = o.shape[0]
    dh = D // H
    kvd = Hkv * dh
    q, k, v = qkv[:D], qkv[D:D + kvd], qkv[D + kvd:D + 2 * kvd]
    if Hkv != H:
        k = k.view(Hkv, dh, -1).repeat_interleave(H // Hkv, 0).reshape(D, -1)
        v = v.view(Hkv, dh, -1).repeat_interleave(H // Hkv, 0).reshape(D, -1)
    return q, k, v


def sub_matrix(m, rows, cols, src_units):
    """(len(rows) x len(cols)) signed matrix of how dst units (rows) read src units (cols), or None."""
    with torch.no_grad():
        if isinstance(m, nn.ConvTranspose2d):
            if m.in_channels != src_units or m.groups != 1: return None
            return m.weight[cols][:, rows].sum((2, 3)).T           # weight is (in, out, k, k)
        if isinstance(m, nn.Conv2d):
            if m.in_channels != src_units: return None
            if m.groups == 1:
                return m.weight[rows][:, cols].sum((2, 3))
            if m.groups == m.in_channels == m.out_channels:        # depthwise: diagonal
                w = m.weight[:, 0].sum((1, 2))
                M = torch.zeros(len(rows), len(cols))
                for i, r in enumerate(rows):
                    for j, c in enumerate(cols):
                        if r == c: M[i, j] = w[r]
                return M
            return None
        if isinstance(m, nn.Linear):
            if m.in_features != src_units: return None
            return m.weight[rows][:, cols]
        p = attn_parts(m)
        if p is not None:
            qkv, o, H, Hkv = p
            D = o.shape[0]
            if D != src_units: return None
            _, _, v = kv_rows(qkv, o, H, Hkv)
            return o[rows] @ v[:, cols]                      # value path W_o·W_v
    return None


def flops_of(mod, kind, tt, label, parts):
    """Rough forward FLOPs (multiply-add = 2) of one layer for the displayed (batch-1) tensor `tt`."""
    n = tt.numel()
    try:
        if isinstance(mod, nn.ConvTranspose2d):
            kh, kw = mod.kernel_size
            return int(2 * mod.in_channels * kh * kw * n / (mod.stride[0] * mod.stride[1]))
        if isinstance(mod, nn.Conv2d):
            kh, kw = mod.kernel_size
            return int(2 * n * (mod.in_channels // mod.groups) * kh * kw)
        if isinstance(mod, nn.Linear):
            return int(2 * mod.in_features * n)
        if parts is not None and tt.ndim == 2:
            qkv, o, H, Hkv = parts
            T, D = tt.shape
            Dkv = Hkv * (D // H)
            return int(2 * T * D * (D + 2 * Dkv) + 4 * T * T * D + 2 * T * D * D)
        if kind == "moe" and hasattr(mod, "w_gate"):
            T, D = tt.shape
            E, _, ff = mod.w_gate.shape
            return int(6 * T * mod.top_k * D * ff + 2 * T * D * E)
        if kind == "norm": return 5 * n
        if kind == "pool":
            k = getattr(mod, "kernel_size", 2)
            k = k if isinstance(k, int) else int(k[0])
            return n * k * k
        if kind == "embedding": return 0
    except Exception:
        pass
    return n


def attrs_of(m):
    keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding", "groups", "in_features", "out_features",
            "heads", "dim", "num_heads", "embed_dim", "normalized_shape", "num_features", "num_embeddings", "embedding_dim",
            "kv_heads", "window", "num_experts", "top_k", "output_padding", "scale_factor"]
    out = {}
    for k in keys:
        if hasattr(m, k):
            v = getattr(m, k)
            if isinstance(v, (int, float, str, tuple, list)): out[k] = list(v) if isinstance(v, tuple) else v
    return out


# ---------------------------------------------------------------- graph
class NetGraph:
    def __init__(self, model, example, max_edges=600, cell_budget=150000):
        self.model = model.eval()
        try:
            self.trace = FxTrace(model, example)
            self.tracer = "fx"
            self.trace.run(example)                     # verify the traced graph really runs
        except Exception as e:                          # untraceable control flow etc.
            self.trace = HookTrace(model, example)
            self.tracer = f"hooks ({type(e).__name__})"
        vals, _ = self.trace.run(example)
        raw = self.trace.nodes
        budget = int(min(6000, max(600, cell_budget / max(1, len(raw)))))
        alias, self.nodes, self.edges, self.mods, self.src_of, self.info = {}, [], [], {}, {}, {}
        self.srcs = {}
        res = lambda n: alias.get(n, n) if alias.get(n, n) is not None else None
        keep = {}

        for nd in raw:
            name, mod = nd["name"], nd["mod"]
            v = vals.get(name)
            t = first_tensor(v) if nd["op"] != "output" else None
            srcs = []
            for i in nd["inputs"]:
                r = i
                while r in alias: r = alias[r]
                if r is not None and r not in srcs: srcs.append(r)
            if nd["op"] == "output":
                if not srcs: continue
                nd = {**nd, "label": "output"}
                t = first_tensor(vals.get(srcs[0]))
            if t is None:                                # int/shape/None values: not a tensor node
                alias[name] = None
                continue
            if (mod is not None and isinstance(mod, PASS)) or (nd["getitem"] and srcs and isinstance(vals.get(srcs[0]), (tuple, list))):
                alias[name] = srcs[0] if srcs else None
                continue
            if nd["op"] == "placeholder": kind = "input"
            elif nd["op"] == "output": kind = "output"
            elif mod is not None: kind = kind_of(mod)
            else: kind = "op"
            batched = t.ndim >= 2 and t.shape[0] == 1
            tt = strip_batch(t)
            info = display_info(tt, budget)
            parts = attn_parts(mod) if (mod is not None and kind == "attention") else None
            if parts is not None and tt.ndim == 2:
                _, o, H, _ = parts
                D = o.shape[0]
                if D == tt.shape[1] and D % H == 0:
                    Hd, dh = min(H, 4), D // H
                    heads_ids = pick(H, Hd)
                    info["sel"] = [h * dh + j for h in heads_ids for j in range(min(dh, 4))]
                    info["heads_ids"], info["Hd"], info["dhd"] = heads_ids, len(heads_ids), min(dh, 4)
                else:
                    parts = None
            keep[name] = (t, tt, info)
            self.info[name] = info
            self.mods[name] = mod
            self.src_of[name] = srcs[0] if srcs else None
            dsh = {"vol": [len(info["sel"]), *info["hw"]] if info["type"] == "vol" else None,
                   "grid": [len(info["tok"]), len(info["sel"])] if info["type"] == "grid" else None,
                   "vec": [len(info["sel"])]}[info["type"]]
            node = {"name": name, "kind": kind, "label": nd["label"], "units": info["units"],
                    "shape": list(t.shape), "tshape": list(tt.shape), "dshape": dsh,
                    "sampled": any(a != b for a, b in zip(dsh, tt.shape)) or len(dsh) != tt.ndim,
                    "heads": info.get("Hd") if parts is not None else None,
                    "params": sum(p.numel() for p in mod.parameters()) if mod is not None else 0,
                    "attrs": attrs_of(mod) if mod is not None else {},
                    "path": nd.get("path", ""), "numel": int(tt.numel()),
                    "flops": flops_of(mod, kind, tt, nd["label"], parts)}
            self._weights(node, mod, kind, parts, srcs, info)
            self.nodes.append(node)
            self.srcs[name] = srcs
            for s in srcs:
                e = self._edge(s, name, mod, info, max_edges)
                if mod is None and nd["label"] in ("cat", "add", "mul", "iadd", "concat"):
                    e["join"] = "cat" if nd["label"] in ("cat", "concat") else ("mul" if nd["label"] == "mul" else "add")
                self.edges.append(e)
        self.total_params = sum(p.numel() for p in model.parameters())
        self.graph = {"nodes": self.nodes, "edges": self.edges, "params": self.total_params, "tracer": self.tracer,
                      "analysis": self._analyze()}
        del vals

    # --- architecture analysis: main path, bottleneck ("deepest level"), encoder/decoder stages, bypassing skips
    def _analyze(self):
        N = len(self.nodes)
        if N < 3: return None
        idx = {d["name"]: i for i, d in enumerate(self.nodes)}
        # main path = longest path from the *primary* input (largest tensor, e.g. the image, not the timestep)
        inputs = [i for i, d in enumerate(self.nodes) if d["kind"] == "input"] or [0]
        primary = max(inputs, key=lambda i: self.nodes[i]["numel"])
        NEG = -10 ** 9
        dist, par = [NEG] * N, [-1] * N
        dist[primary] = 0
        for e in self.edges:
            s, d = idx[e["src"]], idx[e["dst"]]
            if dist[s] > NEG and dist[s] + 1 > dist[d]: dist[d], par[d] = dist[s] + 1, s
        out = max((i for i, d in enumerate(self.nodes) if d["kind"] == "output" and dist[i] > NEG), default=None)
        if out is None: out = max(range(N), key=lambda i: dist[i])
        spine, i = [], out
        while i != -1:
            spine.append(i); i = par[i]
        spine.reverse()
        L = len(spine)
        nums = [self.nodes[i]["numel"] for i in spine]
        res = {"spine": [self.nodes[i]["name"] for i in spine], "kind": "flat", "bottleneck": None, "compression": None,
               "spatial_ratio": None, "numel_in": nums[0] if nums else None,
               "stochastic": any(d["label"] == "randn_like" for d in self.nodes),
               "stages": [], "pairs": [], "bypass": {"count": 0, "channels": 0}}
        if L < 5: return res

        def spatial(pos):
            ts = self.nodes[spine[pos]]["tshape"]
            return ts[1] * ts[2] if len(ts) == 3 else None

        mid = nums[1:L - 2] or nums
        bn_pos = None
        if max(mid) <= 2 * min(mid):
            res["kind"] = "flat"                                  # Transformer-like: the stream keeps its size
        else:
            lo_i, hi_i = max(1, int(L * 0.1)), max(2, int(L * 0.9))
            mn = min(nums[lo_i:hi_i])
            cand = [j for j in range(lo_i, hi_i) if nums[j] == mn]
            j = cand[len(cand) // 2]
            before, after = max(nums[:j] or [0]), max(nums[j + 1:] or [0])
            median = sorted(nums[1:-1])[(L - 2) // 2]
            if mn <= 0.5 * nums[0] and mn <= 0.5 * median and before >= 2 * mn and after >= 0.5 * before and 1 < j < L - 2:
                bn_pos, res["kind"] = j, "hourglass"              # information really is squeezed (autoencoder / VAE)
            else:
                # spatial hourglass (U-Net: channels grow while resolution collapses and comes back)
                sp = [(spatial(p), p) for p in range(L) if spatial(p)]
                if sp:
                    smin = min(v for v, _ in sp)
                    pos = [p for v, p in sp if v == smin]
                    first, last = min(pos), max(pos)
                    first_sp, after_sp = sp[0][0], max((v for v, p in sp if p > last), default=0)
                    if smin * 4 <= first_sp and after_sp >= 4 * smin and last < L - 2:
                        bn_pos, res["kind"] = last, "hourglass"
                if bn_pos is None:                                # compare the first and last quarter of the path
                    q = max(1, len(mid) // 4)
                    q1, q4 = sorted(mid[:q])[q // 2], sorted(mid[-q:])[q // 2]
                    res["kind"] = "funnel" if q1 >= 3 * q4 else "expansion" if q4 >= 3 * q1 else "flat"
        if bn_pos is not None:
            bnode = self.nodes[spine[bn_pos]]
            res["bottleneck"] = bnode["name"]
            res["compression"] = bnode["numel"] / max(1, nums[0])
            s0 = next((spatial(p) for p in range(L) if spatial(p)), None)
            sb = spatial(bn_pos)
            if s0 and sb: res["spatial_ratio"] = s0 / sb
        # stages = runs of equal spatial size along the spine (volumes only; other tensors inherit)
        keys, cur = [], None
        for i in spine:
            ts = self.nodes[i]["tshape"]
            if len(ts) == 3: cur = (ts[1], ts[2])
            keys.append(cur)
        stages, prev = [], object()
        for pos, i in enumerate(spine):
            k = keys[pos]
            if k != prev or not stages:
                stages.append({"hw": list(k) if k else None, "nodes": []})
                prev = k
            stages[-1]["nodes"].append(self.nodes[i]["name"])
        if 3 <= len(stages) <= 14:
            bi = res["bottleneck"]
            bstage = next((s for s in stages if bi in s["nodes"]), None) if bi else None
            enc = {}
            for s_i, s in enumerate(stages):
                key = tuple(s["hw"] or ())
                if bi is None or bstage is None: s["name"] = f"S{s_i + 1}"
                elif s is bstage: s["name"] = "bottleneck"
                elif s_i < stages.index(bstage): s["name"] = f"E{s_i + 1}"; enc[key] = s["name"]
                else: s["name"] = ("D" + enc[key][1:]) if key in enc else "D"
            res["stages"] = stages
            if bi:
                by_name = {s["name"][1:]: s for s in stages if s["name"].startswith("E")}
                res["pairs"] = [[by_name[s["name"][1:]]["nodes"][0], s["nodes"][0]]
                                for s in stages if s["name"].startswith("D") and s["name"][1:] in by_name]
        if res["bottleneck"]:
            bpos = idx[res["bottleneck"]]
            for e in self.edges:
                if e.get("join") == "cat" and idx[e["src"]] < bpos < idx[e["dst"]]:
                    res["bypass"]["count"] += 1
                    res["bypass"]["channels"] += int(self.nodes[idx[e["src"]]]["units"])
        return res

    # --- per-node weight payload for the side panel / 3D attention view (always small)
    def _weights(self, node, mod, kind, parts, srcs, info):
        if mod is None: return
        with torch.no_grad():
            if parts is not None:
                qkv, o, H, Hkv = parts
                D = o.shape[0]
                xs = self.info[srcs[0]]["sel"] if srcs and srcs[0] in self.info else pick(D, 24)
                sel = info["sel"]
                q, k, v = kv_rows(qkv, o, H, Hkv)           # K/V expanded across the heads that share them (GQA)
                node["weights"] = {"shape": list(qkv.shape), "qkv": r3(torch.cat([q[sel][:, xs], k[sel][:, xs], v[sel][:, xs]])),
                                   "o": r3(o[sel][:, sel]), "data": r3(o[sel][:, sel])}
            elif isinstance(mod, nn.ConvTranspose2d):
                ic, oc = pick(mod.in_channels, 12), pick(mod.out_channels, 12)
                node["weights"] = {"shape": [mod.out_channels, mod.in_channels, *mod.kernel_size],
                                   "data": r3(mod.weight[ic][:, oc].transpose(0, 1))}      # as (out, in, k, k)
            elif isinstance(mod, nn.Conv2d):
                oc, ic = pick(mod.out_channels, 12), pick(mod.weight.shape[1], 12)
                node["weights"] = {"shape": list(mod.weight.shape), "data": r3(mod.weight[oc][:, ic])}
            elif isinstance(mod, nn.Linear):
                oc, ic = pick(mod.out_features, 32), pick(mod.in_features, 32)
                node["weights"] = {"shape": list(mod.weight.shape), "data": r3(mod.weight[oc][:, ic])}

    def _edge(self, src, dst, mod, dinfo, max_edges):
        e = {"src": src, "dst": dst, "mode": "bundle"}
        sinfo = self.info[src]
        W = sub_matrix(mod, dinfo["sel"], sinfo["sel"], sinfo["units"]) if mod is not None else None
        if W is not None and tuple(W.shape) == (len(dinfo["sel"]), len(sinfo["sel"])):
            flat = W.abs().flatten()
            idx = flat.topk(min(max_edges, flat.numel())).indices
            m = flat.max().clamp_min(1e-9)
            n_src = W.shape[1]
            fw = W.flatten()
            e["mode"] = "dense"
            e["sparse"] = [[int(i % n_src), int(i // n_src), round(float(fw[i] / m), 3)] for i in idx]
        elif sinfo["units"] == dinfo["units"] and sinfo["sel"] == dinfo["sel"] and sinfo["type"] == dinfo["type"] \
                and (mod is None or kind_of(mod) in ("activation", "norm", "pool", "op", "upsample")):
            e["mode"] = "identity"
        return e

    # --- activations for one input
    @torch.no_grad()
    def activations(self, x):
        vals, out = self.trace.run(x)
        acts = {}
        for d in self.nodes:
            name = d["name"]
            v = out if d["kind"] == "output" else vals.get(name)
            t = first_tensor(v)
            if t is None: continue
            info = self.info[name]
            tt = strip_batch(t)
            a = {"units": [round(float(u), 3) for u in unit_means(tt, info)], "tensor": r3(to_display(tt, info))}
            tf = tt.detach().float()
            a["stats"] = {"sparsity": float((tf.abs() < 1e-6).float().mean()), "mean_abs": float(tf.abs().mean()),
                          "std": float(tf.std()) if tf.numel() > 1 else 0.0}
            ss = self.srcs.get(name, [])
            if d["label"] in ("add", "iadd") and len(ss) == 2 and all(s in vals for s in ss):
                x, y = first_tensor(vals[ss[0]]), first_tensor(vals[ss[1]])
                if x is not None and y is not None and x.shape == y.shape:
                    stream, branch = x.float(), y.float()          # earlier input = the stream, later = F(x)
                    a["delta"] = float(branch.norm() / (stream.norm() + 1e-9))
            mod = self.mods[name]
            if d.get("heads") and attn_parts(mod) is not None and self.src_of[name] in vals:
                a.update(self._attention(mod, first_tensor(vals[self.src_of[name]]), info, self.info[self.src_of[name]]))
            g = getattr(mod, "last_gates", None)
            if g is not None and info["type"] == "grid" and g.shape[-2] == tt.shape[0]:
                a["gates"] = r3(g[0][info["tok"]])         # router weights: tokens × experts
            acts[name] = a
        return acts, first_tensor(out)

    def _attention(self, mod, x_in, info, src_info):
        qkv_w, o_w, H, Hkv = attn_parts(mod)
        x = strip_batch(x_in).float()                    # (T, D)
        T, D = x.shape
        dh = D // H
        bias = getattr(mod, "qkv", None)
        b = bias.bias if bias is not None else getattr(mod, "in_proj_bias", None)
        qkv = F.linear(x, qkv_w, b)
        kvd = Hkv * dh
        q, k, v = qkv.split([D, kvd, kvd], -1)
        if Hkv != H:                                     # GQA: repeat each K/V head for the query heads sharing it
            k = k.view(T, Hkv, dh).repeat_interleave(H // Hkv, 1).reshape(T, D)
            v = v.view(T, Hkv, dh).repeat_interleave(H // Hkv, 1).reshape(T, D)
        if getattr(mod, "last_attn", None) is not None and mod.last_attn.shape[-1] == T:
            A = mod.last_attn[0].float()                  # (H,T,T), honours causal mask / RoPE / window
        else:
            qh, kh = q.view(T, H, dh).transpose(0, 1), k.view(T, H, dh).transpose(0, 1)
            A = (qh @ kh.transpose(-2, -1) / dh ** 0.5).softmax(-1)
        vh = v.view(T, H, dh).transpose(0, 1)
        ctx = (A @ vh).transpose(0, 1).reshape(T, D)
        tok, sel, hid = info["tok"], info["sel"], info["heads_ids"]
        return {"attn": r3(A[hid][:, tok][:, :, tok]),
                "qkv": {"q": r3(q[tok][:, sel]), "k": r3(k[tok][:, sel]), "v": r3(v[tok][:, sel]),
                        "ctx": r3(ctx[tok][:, sel])}}
