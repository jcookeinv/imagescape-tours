"""Measure the camera move between each pair of neighbouring photos (both directions).
Writes docs/links.json: {"NNN>MMM": {"p": [...8 params...], "ncc": confidence 0..1, "base": no-move score}}
Run: python3 tools/estimate_all.py [workers]
"""
import json, os, sys, importlib.util
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("est", os.path.join(HERE, "estimate.py"))
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
OUT = os.path.join(ROOT, "docs", "links.json")

def img(n):
    return E.load(os.path.join(ROOT, "docs", "img", "%03d_preview.jpg" % n))

def job(pair):
    a, b = pair
    p, c, base = E.optimise(img(a), img(b), verbose=False)
    return "%03d>%03d" % (a, b), {"p": [round(float(x), 3) for x in p], "ncc": round(-float(c), 3), "base": round(-float(base), 3)}

if __name__ == "__main__":
    nums = sorted(int(f[:3]) for f in os.listdir(os.path.join(ROOT, "docs", "img")) if f.endswith("_preview.jpg"))
    pairs = [(a, b) for a, b in zip(nums, nums[1:])]
    pairs += [(b, a) for a, b in pairs]
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    todo = [p for p in pairs if "%03d>%03d" % p not in res]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for k, v in pool.imap_unordered(job, todo):
            res[k] = v
            json.dump(res, open(OUT, "w"), indent=0, sort_keys=True)
            print(k, v["ncc"], v["base"], flush=True)
