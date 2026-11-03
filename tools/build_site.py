"""Assemble docs/scenes.js (scene list, measured links, site config) from docs/img, docs/links.json and tools/site_config.json.
Run after build_scenes.py / estimate_all.py:  python3 tools/build_site.py"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "docs")
cfg = json.load(open(os.path.join(ROOT, "tools", "site_config.json")))
ids = sorted(f[:3] for f in os.listdir(os.path.join(D, "img")) if f.endswith("_preview.jpg"))
links = json.load(open(os.path.join(D, "links.json"))) if os.path.exists(os.path.join(D, "links.json")) else {}
import math
def dist(l): return math.hypot(l["p"][0], l["p"][1])
for k, l in links.items():
    a, b = k.split(">"); r = links.get(b + ">" + a)
    confident = l["ncc"] >= 0.2 and l["ncc"] - l["base"] >= 0.1
    agrees = r is None or max(dist(l), dist(r)) / max(0.1, min(dist(l), dist(r))) <= 1.6
    l["ok"] = bool(confident and agrees)
print(sum(1 for l in links.values() if l["ok"]), "links trusted of", len(links))
tour = dict(cfg, scenes=[{"id": i} for i in ids], links=links)
open(os.path.join(D, "scenes.js"), "w").write("window.TOUR = " + json.dumps(tour, separators=(",", ":")) + ";\n")
print(len(ids), "scenes,", len(links), "measured links")
