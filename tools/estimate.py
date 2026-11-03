"""Estimate where photo B was taken relative to photo A, plus a rough room box, by reprojecting A into B's
viewpoint and maximising correlation with B. Output JSON params used by the viewer's transition shader.

Frame: A's equirect frame. x right, y up, z forward (yaw 0 = photo centre). Camera height fixed at 1.6 m (scale).
params = [bx, bz, psi, c, xl, xr, zb, zf]
  (bx,bz)  position of B in A's frame (metres)
  psi      yaw (deg) of B's photo centre in A's frame
  c        ceiling height above camera; xl/xr/zb/zf  room walls relative to A (xl,zb negative)
"""
import sys, json, math
import numpy as np
from PIL import Image, ImageFilter

H_CAM = 1.6
import os
W, H = int(os.environ.get('ES_W', 512)), int(os.environ.get('ES_H', 256))

def load(path, w=W, h=H):
    im = Image.open(path).convert("L").resize((w, h), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0

def highpass(a):
    im = Image.fromarray((a * 255).astype(np.uint8))
    lo = np.asarray(im.filter(ImageFilter.GaussianBlur(6)), dtype=np.float32) / 255.0
    sm = np.asarray(im.filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32) / 255.0
    hp = sm - lo
    sd = np.asarray(Image.fromarray((np.abs(hp) * 255 * 4).clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(10)), dtype=np.float32) / 255.0 / 4 + 1e-3
    return hp / sd

lon_g = (np.arange(W) + 0.5) / W * 2 * np.pi - np.pi
lat_g = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
LON, LAT = np.meshgrid(lon_g, lat_g)
DIR = np.stack([np.sin(LON) * np.cos(LAT), np.sin(LAT), np.cos(LON) * np.cos(LAT)], -1)
MASK = (np.abs(LAT) < math.radians(48)) & (LAT > math.radians(-50))

def rot_yaw(d, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.stack([c * d[..., 0] + s * d[..., 2], d[..., 1], -s * d[..., 0] + c * d[..., 2]], -1)

def box_hit(O, d, c, xl, xr, zb, zf):
    big = 1e9
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[..., 0] > 1e-6, (xr - O[0]) / d[..., 0], np.where(d[..., 0] < -1e-6, (xl - O[0]) / d[..., 0], big))
        ty = np.where(d[..., 1] > 1e-6, (c - O[1]) / d[..., 1], np.where(d[..., 1] < -1e-6, (-H_CAM - O[1]) / d[..., 1], big))
        tz = np.where(d[..., 2] > 1e-6, (zf - O[2]) / d[..., 2], np.where(d[..., 2] < -1e-6, (zb - O[2]) / d[..., 2], big))
    return np.minimum(np.minimum(tx, ty), tz)

def sample(img, lon, lat):
    h, w = img.shape
    x = (lon / (2 * np.pi) + 0.5) * w - 0.5
    y = (0.5 - lat / np.pi) * h - 0.5
    x0 = np.floor(x).astype(int); fx = x - x0
    y0 = np.floor(y).astype(int); fy = y - y0
    x1 = (x0 + 1) % w; x0 %= w
    y1 = np.clip(y0 + 1, 0, h - 1); y0 = np.clip(y0, 0, h - 1)
    return (img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy) + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy

def warp_A_to_B(A, p):
    bx, bz, psi, c, xl, xr, zb, zf = p
    dA = rot_yaw(DIR, psi)
    O = (bx, 0.0, bz)
    t = box_hit(O, dA, c, xl, xr, zb, zf)
    P = np.stack([O[0] + t * dA[..., 0], O[1] + t * dA[..., 1], O[2] + t * dA[..., 2]], -1)
    n = np.linalg.norm(P, axis=-1) + 1e-9
    return sample(A, np.arctan2(P[..., 0], P[..., 2]), np.arcsin(np.clip(P[..., 1] / n, -1, 1)))

LO = np.array([-14, -14, -180, 2.0, -25, 1.0, -40, 3.0])
HI = np.array([14, 14, 180, 9.0, -1.0, 25, -1.0, 60.0])

def valid(p):
    bx, bz, psi, c, xl, xr, zb, zf = p
    return xl + 0.5 < min(0, bx) and xr - 0.5 > max(0, bx) and zb + 0.5 < min(0, bz) and zf - 0.5 > max(0, bz)

def cost(p, hpA, hpB):
    p = np.clip(p, LO, HI)
    if not valid(p): return 1.0
    w = warp_A_to_B(hpA, p)
    m = MASK
    a, b = w[m], hpB[m]
    a = a - a.mean(); b = b - b.mean()
    ncc = (a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9)
    return -ncc

PSIS = (-20, 0, 20)

def optimise(A, B, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    hpA, hpB = highpass(A), highpass(B)
    starts = []
    for r in ((2, 4, 7) if len(PSIS) < 5 else (3, 6, 9)):
        for ang in range(0, 360, 30):
            for psi in PSIS:
                starts.append(np.array([r * math.sin(math.radians(ang)), r * math.cos(math.radians(ang)), psi, 3.8, -7, 7, -12, 20.0]))
    scored = sorted(((cost(s, hpA, hpB), i) for i, s in enumerate(starts)), key=lambda x: x[0])
    base = cost(np.array([0, 0, 0, 3.8, -7, 7, -12, 20.0]), hpA, hpB)
    if verbose: print("no-move baseline cost", round(base, 3), "best start", round(scored[0][0], 3))
    results = []
    for sc, i in scored[:3]:
        best, bc = starts[i].copy(), sc
        sig = np.array([0.8, 0.8, 6, 0.4, 2, 2, 3, 4.0])
        for it in range(180):
            cand = np.clip(best + rng.normal(size=8) * sig, LO, HI)
            cc = cost(cand, hpA, hpB)
            if cc < bc: best, bc = cand, cc
            if it % 45 == 44: sig *= 0.6
        results.append((bc, best))
        if verbose: print(" refine ->", round(bc, 3), np.round(best, 2))
    results.sort(key=lambda x: x[0])
    return results[0][1], results[0][0], base

if __name__ == "__main__":
    a, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
    A, B = load(a), load(b)
    if len(sys.argv) > 4 and sys.argv[4] == "wide": globals()["PSIS"] = tuple(range(-180, 180, 30))
    p, c, base = optimise(A, B)
    json.dump({"params": [float(x) for x in p], "cost": float(c), "baseline": float(base)}, open(out, "w"))
    print("BEST", np.round(p, 2), "cost", round(c, 3), "baseline", round(base, 3))
