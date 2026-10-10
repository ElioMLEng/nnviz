"""Smoke-test the extractor on registry models: python selftest.py [id ...]"""
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import torch
from extract import NetGraph
from zoo import REGISTRY

ids = sys.argv[1:] or list(REGISTRY)
for mid in ids:
    t0 = time.time()
    try:
        e = REGISTRY[mid]
        m = e["build"]()
        net = NetGraph(m, e["make_input"](0, 0))
        t1 = time.time()
        acts, out = net.activations(e["make_input"](0, 1))
        g = json.dumps(net.graph)
        a = json.dumps(acts)
        kinds = {}
        for n in net.nodes:
            kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
        att = [n for n in net.nodes if n["heads"]]
        print(f"{mid:20s} OK  tracer={net.tracer:10s} nodes={len(net.nodes):4d} edges={len(net.edges):4d} "
              f"graph={len(g)/1e6:.2f}MB acts={len(a)/1e6:.2f}MB build+trace={t1-t0:.1f}s act={time.time()-t1:.1f}s "
              f"attn_rich={len(att)} kinds={kinds}")
    except Exception as ex:
        import traceback
        traceback.print_exc()
        print(f"{mid:20s} FAIL {type(ex).__name__}: {ex}")
