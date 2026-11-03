import json, math, numpy as np, sys
from PIL import Image
import importlib.util
spec = importlib.util.spec_from_file_location("est", "estimate.py"); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
W_, H_ = 960, 540
def loadc(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)/255
def samp(img, lon, lat):
    return np.stack([E.sample(img[...,c], lon, lat) for c in range(3)], -1)
def view_dirs(yaw, pitch, fov):
    ty = math.tan(math.radians(fov)/2); tx = ty*W_/H_
    u, v = np.meshgrid((np.arange(W_)+.5)/W_*2-1, 1-(np.arange(H_)+.5)/H_*2)
    d = np.stack([u*tx, v*ty, np.ones_like(u)], -1); d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p = math.radians(pitch); c, s = math.cos(p), math.sin(p)
    d = np.stack([d[...,0], c*d[...,1]+s*d[...,2], -s*d[...,1]+c*d[...,2]], -1)
    return E.rot_yaw(d, yaw)
def lonlat(P):
    n = np.linalg.norm(P, axis=-1)+1e-9
    return np.arctan2(P[...,0], P[...,2]), np.arcsin(np.clip(P[...,1]/n, -1, 1))
def frame(A, B, prm, t, yaw, pitch, fov):
    bx, bz, psi, c, xl, xr, zb, zf = prm
    d = view_dirs(yaw, pitch, fov); C = np.array([bx*t, 0, bz*t])
    tt = E.box_hit(C, d, c, xl, xr, zb, zf)
    P = C + tt[...,None]*d
    a = samp(A, *lonlat(P))
    Pb = np.array([bx, 0, bz]); Q = E.rot_yaw(P - Pb, -psi)
    b = samp(B, *lonlat(Q))
    k = t*t*(3-2*t)
    return a*(1-k)+b*k, a, b
if __name__ == "__main__":
    pa, pb, js, out = sys.argv[1:5]
    A, B = loadc(pa), loadc(pb); prm = json.load(open(js))["params"]
    rows = []
    for t in (0.35, 0.6):
        # naive crossfade: same view directions, no movement
        d = view_dirs(0, -2, 50); la, lb = lonlat(d), lonlat(E.rot_yaw(d, -prm[2]))
        k = t*t*(3-2*t); naive = samp(A,*la)*(1-k) + samp(B,*la)*k
        geo, _, _ = frame(A, B, prm, t, 0, -2, 50)
        rows.append(np.concatenate([naive, geo], 1))
    im = np.concatenate(rows, 0)
    Image.fromarray((im.clip(0,1)*255).astype(np.uint8)).resize((1440, 810)).save(out)
