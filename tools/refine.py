"""Refine one measured link at higher resolution.  ES_W=1024 ES_H=512 python3 tools/refine.py A B"""
import sys, json, os, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("est", os.path.join(HERE, "estimate.py")); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
a, b = sys.argv[1:3]
links = json.load(open(os.path.join(ROOT, "docs", "links.json")))
init = np.array(links["%s>%s" % (a, b)]["p"])
A, B = E.load(os.path.join(ROOT, "docs/img/%s_preview.jpg" % a), E.W, E.H), E.load(os.path.join(ROOT, "docs/img/%s_preview.jpg" % b), E.W, E.H)
hpA, hpB = E.highpass(A), E.highpass(B)
best, bc = init.copy(), E.cost(init, hpA, hpB); print("start", round(-bc, 3), flush=True)
rng = np.random.default_rng(3); sig = np.array([0.25, 0.25, 2.5, 0.25, 1.0, 1.0, 1.5, 2.0])
for it in range(360):
    cand = np.clip(best + rng.normal(size=8) * sig, E.LO, E.HI); cc = E.cost(cand, hpA, hpB)
    if cc < bc: best, bc = cand, cc
    if it % 90 == 89: sig *= 0.55
print("refined", round(-bc, 3), np.round(best, 3).tolist())
json.dump({"p": [round(float(x), 4) for x in best], "ncc": round(-float(bc), 3)}, open(os.path.join(os.environ.get("OUTDIR", "/tmp"), "refined_%s_%s.json" % (a, b)), "w"))
