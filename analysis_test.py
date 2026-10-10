"""Print the architecture analysis (bottleneck / stages / skips) for some registry models."""
import sys
import warnings

warnings.filterwarnings("ignore")
from zoo import REGISTRY
from extract import NetGraph

ids = sys.argv[1:] or ["unet", "vae", "resnet18", "vit_tiny_16", "dcgan-generator", "ddpm-unet", "gpt-oss-style"]
for mid in ids:
    e = REGISTRY[mid]
    net = NetGraph(e["build"](), e["make_input"](0, 0))
    a = net.graph["analysis"]
    names = [s["name"] for s in a["stages"]]
    print(f"{mid:18s} kind={a['kind']:10s} bottleneck={a['bottleneck']} compression={a['compression']} "
          f"stages={names} pairs={len(a['pairs'])} bypass={a['bypass']} stochastic={a['stochastic']}")
