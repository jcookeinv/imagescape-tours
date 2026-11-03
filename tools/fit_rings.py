"""Fit one camera move so recognisable bright features (here the circular lights) coincide in the two photos' views.
Cost = distance between each light's position as seen from photo A and from photo B, over several points of the move.
Usage: python3 tools/fit_rings.py out.json   (edit PAIR / WINDOWS for another move)"""
import sys, json, math, numpy as np
sys.argv = ['render_tour_video.py', 'x', '--w', '640', '--h', '360', '--src', 'web']
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("rv", os.path.join(HERE, "render_tour_video.py")); rv = importlib.util.module_from_spec(spec); rv.__name__ = "rv"; spec.loader.exec_module(rv)
OUT = os.environ.get("FIT_OUT", "/tmp/fit_out.json")
A_, B_ = "007", "008"
init = np.array(list(rv.G78) + [0.0, 0.0])
Y0, Y1 = rv.HEAD78, rv.Y8_END
WINS = {"wreath L": (40, 165, 80, 200), "wreath R": (390, 550, 80, 210), "big ring": (500, 640, 0, 85)}
def cen(g, x0, x1, y0, y1, thr):
    s = g[y0:y1, x0:x1]; ys, xs = np.nonzero(s > thr)
    return (xs.mean() + x0, ys.mean() + y0) if len(xs) > 12 else None
def cost(p, progs=(0.25, 0.4, 0.5, 0.6, 0.75)):
    rv.TILT[(A_, B_)] = (p[8], p[9]); tot = 0.0
    for prog in progs:
        pose = (Y0 + (Y1 - Y0) * prog, -3, 64)
        a, b = rv.travel_ab(A_, B_, list(p[:8]), pose, prog, 0.5); ga, gb = a.mean(2), b.mean(2)
        for (x0, x1, y0, y1) in WINS.values():
            thr = 0.86 * max(ga[y0:y1, x0:x1].max(), gb[y0:y1, x0:x1].max())
            ca, cb = cen(ga, x0, x1, y0, y1, thr), cen(gb, x0, x1, y0, y1, thr)
            tot += 40.0 if not (ca and cb) else math.hypot(ca[0] - cb[0], ca[1] - cb[1])
        tot += 0.05 * np.abs(ga[:200] - gb[:200]).mean()
    return tot / len(progs)
span = np.array([0.5, 0.8, 5, 1.0, 6, 6, 8, 16, 2.0, 2.0])
lo, hi = init - span, init + span; lo[3] = max(lo[3], 2.5)
best = init.copy(); bc = cost(best); print("start", round(bc, 2), flush=True)
rng = np.random.default_rng(5); sig = np.array([0.1, 0.15, 0.8, 0.25, 1.2, 1.2, 1.5, 3.0, 0.4, 0.4])
for it in range(450):
    cand = np.clip(best + rng.normal(size=10) * sig, lo, hi)
    if not (cand[4] + 0.5 < min(0, cand[0]) and cand[5] - 0.5 > max(0, cand[0]) and cand[6] + 0.5 < 0 and cand[7] - 0.5 > cand[1]): continue
    cc = cost(cand)
    if cc < bc: best, bc = cand, cc
    if it % 75 == 74: sig *= 0.65; print(it, round(bc, 2), flush=True)
print("refined", round(bc, 2), np.round(best, 3).tolist())
json.dump([round(float(x), 4) for x in best], open(OUT, "w"))
