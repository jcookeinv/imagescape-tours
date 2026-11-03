"""Render the 006 > 007 > 025 > 026 tour video from the full-resolution originals, with the tour viewer's interface
(title card, buttons, scene chips, hotspot rings, mouse click) drawn into the frames.

  python3 tools/render_tour_video.py out.mp4 [--w 1920 --h 1080 --fps 60 --workers 4] [--stills t1,t2 ...] [--src web]

All camera moves stay at horizon level. Moves between scenes travel along the measured line between the two photo
positions (see tools/estimate_all.py); poses are (yaw, pitch, hfov) in degrees.
"""
import sys, os, math, json, subprocess, argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
ap.add_argument("--fps", type=int, default=60); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--from", dest="t_from", type=float, default=0.0); ap.add_argument("--to", dest="t_to", type=float, default=None); ap.add_argument("--stills", default=""); ap.add_argument("--src", default="full"); ap.add_argument("--no-ui", action="store_true")
ARGS = ap.parse_args()
W, H, FPS = ARGS.w, ARGS.h, ARGS.fps
H_CAM = 1.6
D2R = math.pi / 180
S_UI = W / 1280.0

TOUR = json.loads(open(os.path.join(ROOT, "docs", "scenes.js")).read().split("=", 1)[1].strip().rstrip(";"))
IDS = [x["id"] for x in TOUR["scenes"]]
# names for this video: scenes are numbered by position from 01 (the entrance photo), distinctive places are named
NAMED = {"005": "Coffee Shop Entrance", "010": "Sofa Lounge", "017": "Coffee Counter", "021": "Black Sofa Area",
         "025": "Till Area", "026": "Shop Floor", "027": "Homeware Shelves", "028": "Christmas Gifts", "029": "Clothing Rails",
         "030": "Flowers & Clothing", "031": "Gift Cabinets", "032": "Farmyard", "033": "Courtyard", "034": "Paddock",
         "035": "Farm Track", "036": "Car Park"}
def title_of(i): return NAMED.get(i, "Scene %02d" % (IDS.index(i) + 1))
VIDEO_EXTRA = {"008": [{"to": "025", "yaw": 80, "pitch": -14, "label": "Gift Shop"}],
               "026": [{"to": "025", "yaw": 71, "pitch": -9, "label": "Till Area"}]}
def link(a, b):
    l = TOUR["links"].get(a + ">" + b); return l if l and l["ok"] else None
def hotspots_of(i):
    k = IDS.index(i); out = []
    for nb in (IDS[k - 1] if k > 0 else None, IDS[k + 1] if k + 1 < len(IDS) else None):
        l = link(i, nb) if nb else None
        if l: out.append({"to": nb, "yaw": math.degrees(math.atan2(l["p"][0], l["p"][1])), "pitch": -9, "label": title_of(nb)})
    for e in VIDEO_EXTRA.get(i, []): out.append(dict(e))
    return out

G67 = TOUR["links"]["006>007"]["p"]
G78_OLD = TOUR["links"]["007>008"]["p"]
# 007 > 008 re-fitted so the wall wreaths and the big ring line up through the move (fit on those lights, see tools/fit_rings.py)
G78 = [-0.1024, 3.1791, 4.3208, 3.9067, -10.3655, 4.1125, -21.3168, 21.3357]
G2526 = TOUR["links"]["025>026"]["p"]
HEAD67 = math.degrees(math.atan2(G67[0], G67[1]))
HEAD78 = math.degrees(math.atan2(G78_OLD[0], G78_OLD[1]))      # unchanged: the camera still faces the same way
HEAD2526 = math.degrees(math.atan2(G2526[0], G2526[1]))
y7 = HEAD67 - G67[2]
y8 = HEAD78 - G78_OLD[2]                                       # unchanged: lands on the same view of 008
Y8_END = y8 + G78[2]                                           # the pose at the end of the move that lands on y8 with the new rotation
y26 = HEAD2526 - G2526[2]

# ---- timeline (seconds): 006 > 007 > 008 > (into the shop) 025 > 026 ----------------------------------------------
T_TR1 = (19.6, 24.1)           # 006 -> 007
T_TR2 = (33.9, 38.4)           # 007 -> 008
T_BLUR = (47.9, 49.3)          # 008 -> 025 (zoom dissolve, as in the viewer)
T_TR3 = (62.9, 67.4)           # 025 -> 026
TRAVELS = [(T_TR1, "006", "007", G67), (T_TR2, "007", "008", G78), (T_TR3, "025", "026", G2526)]
TOTAL = 101.2
FADE_IN, FADE_OUT = 0.8, 1.5
SEG = {
    "006": [(0, -66, -2, 76), (4.8, -56, -2, 70), (9.6, -36, -2, 56), (12, -30, -2, 52), (13.6, -28, -2, 52),
            (16, -10, -3, 60), (18, HEAD67, -3, 64), (19.6, HEAD67, -3, 64), (24.1, HEAD67, -3, 64)],
    # look across to the stall on the right, linger on it, then come back to the Scene 04 ring
    "007": [(24.1, y7, -3, 64), (24.7, y7, -3, 64), (27.5, 44, -3, 56), (30.1, 52, -3, 48), (33.4, HEAD78, -3, 64),
            (33.9, HEAD78, -3, 64), (38.4, Y8_END, -3, 64)],
    "008": [(38.4, y8, -3, 64), (39.0, y8, -3, 64), (46.6, 80, -11, 56), (47.9, 80, -11, 54), (49.3, 80, -11, 52)],
    "025": [(47.9, -148, -3, 58), (51.5, -140, -3, 56), (55.5, -130, -3, 56), (59.5, -122, -3, 56), (62.9, -110, -3, 58), (67.4, HEAD2526, -3, 62)],
    "026": [(67.4, y26, -3, 62), (69.8, y26, -3, 62), (77.8, y26 + 70, -3, 62), (85.8, y26 + 140, -3, 61), (93.8, y26 + 210, -3, 60),
            (99.4, y26 + 258, -3, 60), (101.2, y26 + 260, -3, 60)],
}
# mouse clicks: (cursor appears, click, scene, target scene)
CLICKS = [(16.9, 19.1, "006", "007"), (31.3, 33.6, "007", "008"), (45.3, 47.6, "008", "025"), (60.3, 62.6, "025", "026")]
SCENE_AT = [(0, "006"), (T_TR1[1], "007"), (T_TR2[1], "008"), (T_BLUR[1], "025"), (T_TR3[1], "026")]
HINT = (1.5, 6.5)

def pchip(ts, vs):
    ts = np.asarray(ts, float); vs = np.asarray(vs, float); n = len(ts)
    h = np.diff(ts); d = np.diff(vs) / h; m = np.zeros(n)
    for k in range(1, n - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    def f(t):
        t = min(max(t, ts[0]), ts[-1]); k = min(np.searchsorted(ts, t, side="right") - 1, n - 2)
        s = (t - ts[k]) / h[k]; s2, s3 = s * s, s * s * s
        return (2 * s3 - 3 * s2 + 1) * vs[k] + (s3 - 2 * s2 + s) * h[k] * m[k] + (-2 * s3 + 3 * s2) * vs[k + 1] + (s3 - s2) * h[k] * m[k + 1]
    return f

POSE = {}
for name, keys in SEG.items():
    fs = [pchip([k[0] for k in keys], [k[c] for k in keys]) for c in (1, 2, 3)]
    POSE[name] = lambda t, fs=fs: tuple(float(f(t)) for f in fs)

def sm(a, b, x):
    x = min(1.0, max(0.0, (x - a) / (b - a))); return x * x * x * (x * (6 * x - 15) + 10)

# measured gain that makes a scene continue the exposure of the previous one (median ratio over the shared view)
EXPO = {"026": np.array([1.213, 1.269, 1.28], np.float32)}
def expo(img, name):
    g = EXPO.get(name)
    if g is None or img is None: return img
    y = img * g
    hi = 225.0                                   # soft shoulder so lifted highlights roll off instead of clipping
    over = y > hi
    return np.where(over, hi + (255 - hi) * (1 - np.exp(-(y - hi) / (255 - hi))), y).astype(np.float32)

# ---- panorama rendering ----------------------------------------------------------------------------------------
SRC = {}
def src(num):
    if num not in SRC:
        p = os.path.join(ROOT, "ImageScape_360_%s.jpg" % num) if ARGS.src == "full" else os.path.join(ROOT, "docs/img/%s.jpg" % num)
        SRC[num] = np.ascontiguousarray(np.asarray(Image.open(p).convert("RGB")))
    return SRC[num]

def rot_yaw(d, deg):
    c, s = math.cos(deg * D2R), math.sin(deg * D2R)
    return np.stack([c * d[..., 0] + s * d[..., 2], d[..., 1], -s * d[..., 0] + c * d[..., 2]], -1)

R0, R1, STRIP = 0, H, 60
U = (((np.arange(W, dtype=np.float32) + .5) / W * 2 - 1)[None, :]).repeat(H, 0)
V = (((1 - (np.arange(H, dtype=np.float32) + .5) / H * 2))[:, None]).repeat(W, 1)

def view_dirs(yaw, pitch, hfov):
    tx = math.tan(hfov * D2R / 2); ty = tx * H / W
    u, v = U[R0:R1], V[R0:R1]
    d = np.stack([u * tx, v * ty, np.ones_like(u)], -1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    p = pitch * D2R; c, s = math.cos(p), math.sin(p)
    d = np.stack([d[..., 0], c * d[..., 1] + s * d[..., 2], -s * d[..., 1] + c * d[..., 2]], -1)
    return rot_yaw(d, yaw).astype(np.float32)

def sample(img, d):
    h, w, _ = img.shape
    n = np.linalg.norm(d, axis=-1) + 1e-9
    lon = np.arctan2(d[..., 0], d[..., 2]); lat = np.arcsin(np.clip(d[..., 1] / n, -1, 1))
    x = (lon / (2 * np.pi) + 0.5) * w - 0.5; y = (0.5 - lat / np.pi) * h - 0.5
    x0 = np.floor(x).astype(np.int32); fx = (x - x0)[..., None]
    y0 = np.floor(y).astype(np.int32); fy = (y - y0)[..., None]
    x1 = (x0 + 1) % w; x0 %= w
    y1 = np.clip(y0 + 1, 0, h - 1); y0 = np.clip(y0, 0, h - 1)
    flat = img.reshape(-1, 3)
    a = flat[y0 * w + x0].astype(np.float32) * (1 - fx) + flat[y0 * w + x1].astype(np.float32) * fx
    b = flat[y1 * w + x0].astype(np.float32) * (1 - fx) + flat[y1 * w + x1].astype(np.float32) * fx
    return a * (1 - fy) + b * fy

def box_hit(O, d, c, xl, xr, zb, zf):
    big = np.float32(1e9)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(d[..., 0] > 1e-6, (xr - O[0]) / d[..., 0], np.where(d[..., 0] < -1e-6, (xl - O[0]) / d[..., 0], big))
        ty = np.where(d[..., 1] > 1e-6, (c - O[1]) / d[..., 1], np.where(d[..., 1] < -1e-6, (-H_CAM - O[1]) / d[..., 1], big))
        tz = np.where(d[..., 2] > 1e-6, (zf - O[2]) / d[..., 2], np.where(d[..., 2] < -1e-6, (zb - O[2]) / d[..., 2], big))
    return np.minimum(np.minimum(tx, ty), tz)

TILT = {("007", "008"): (0.172, 1.145)}   # (A, B) -> (pitch, roll) in degrees: B photo's slight tilt relative to A
def tilt(v, pitch, roll):
    cp, sp, cr, sr = math.cos(pitch * D2R), math.sin(pitch * D2R), math.cos(roll * D2R), math.sin(roll * D2R)
    v = np.stack([v[..., 0], cp * v[..., 1] + sp * v[..., 2], -sp * v[..., 1] + cp * v[..., 2]], -1)
    return np.stack([cr * v[..., 0] - sr * v[..., 1], sr * v[..., 0] + cr * v[..., 1], v[..., 2]], -1)

def travel_ab(numA, numB, prm, pose, prog, kmix):
    """the two photos seen from the moving camera: (view from A, view from B or None)"""
    bx, bz, psi, c, xl, xr, zb, zf = prm
    d = view_dirs(*pose)
    C = np.array([bx * prog, 0, bz * prog], np.float32)
    P = C + box_hit(C, d, c, xl, xr, zb, zf)[..., None] * d
    a = sample(src(numA), P) if kmix < 0.998 else None
    b = None
    if kmix > 0.002:
        q = rot_yaw(P - np.array([bx, 0, bz], np.float32), -psi).astype(np.float32)
        if (numA, numB) in TILT:                       # small tilt between the photos, eased out so the move ends exactly on photo B
            w_ = 1.0 - sm(0.64, 1.0, prog); q = tilt(q, TILT[(numA, numB)][0] * w_, TILT[(numA, numB)][1] * w_).astype(np.float32)
        b = expo(sample(src(numB), q), numB)
    return a, b

def scene_view(name, t, fov_scale=1.0):
    y, p, f = POSE[name](t)
    return expo(sample(src(name), view_dirs(y, p, f * fov_scale)), name)

def plan(t):
    """what to draw at time t: ('geo', A, B, params, prog, kmix, blur) | ('zoom', e) | ('scene', name)"""
    for (t0, t1), A, B, g in TRAVELS:
        if t0 <= t <= t1:
            u = (t - t0) / (t1 - t0); prog = sm(0, 1, u)
            lo, hi = (0.36, 0.64) if (A, B) == ("007", "008") else (0.42, 0.58)
            return ("geo", A, B, g, prog, sm(lo, hi, prog), 0.035 * math.sin(math.pi * u) ** 1.5)
    if T_BLUR[0] <= t <= T_BLUR[1]:
        return ("zoom", sm(0, 1, (t - T_BLUR[0]) / (T_BLUR[1] - T_BLUR[0])))
    name = SCENE_AT[0][1]
    for ts, sid in SCENE_AT:
        if t >= ts: name = sid
    return ("scene", name)

def strip_views(t, pl):
    if pl[0] == "geo":
        _, A, B, g, prog, kmix, _ = pl
        return travel_ab(A, B, g, POSE[A](t), prog, kmix)
    if pl[0] == "zoom":
        e = pl[1]; S = 1.5; z = 1 + (S - 1) * e; k = sm(0.44, 0.56, e)
        return (scene_view("008", t, 1 / z) if k < 0.998 else None), (scene_view("025", t, S / z) if k > 0.002 else None)
    return scene_view(pl[1], t), None

def render_pano(t):
    global R0, R1
    pl = plan(t); A, B = [], []
    for r in range(0, H, STRIP):
        R0, R1 = r, min(H, r + STRIP); a, b = strip_views(t, pl); A.append(a); B.append(b)
    R0, R1 = 0, H
    a = np.concatenate(A, 0) if A[0] is not None else None
    b = np.concatenate(B, 0) if B[0] is not None else None
    if pl[0] == "scene": return a
    k = pl[5] if pl[0] == "geo" else sm(0.44, 0.56, pl[1])
    if pl[0] == "geo" and (pl[1], pl[2]) == ("007", "008") and a is not None and b is not None:
        a, b = ring_morph(a, b, t, k, pl[4])
    img = a if b is None else b if a is None else match_blend(a, b, k)
    amount = pl[6] if pl[0] == "geo" else 0.12 * math.sin(math.pi * pl[1]) ** 2
    return radial_blur(img, amount, taps=11 if pl[0] == "geo" else 15)

# ---- blending helpers -------------------------------------------------------------------------------------------
def remap(img, xs, ys):
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int32); y0 = np.floor(ys).astype(np.int32)
    fx = (xs - x0)[..., None]; fy = (ys - y0)[..., None]
    x0c = np.clip(x0, 0, w - 1); x1c = np.clip(x0 + 1, 0, w - 1); y0c = np.clip(y0, 0, h - 1); y1c = np.clip(y0 + 1, 0, h - 1)
    a = img[y0c, x0c] * (1 - fx) + img[y0c, x1c] * fx; b = img[y1c, x0c] * (1 - fx) + img[y1c, x1c] * fx
    return a * (1 - fy) + b * fy

def resize(a, h2, w2):
    h, w = a.shape[:2]
    ys, xs = np.meshgrid((np.arange(h2) + .5) * h / h2 - .5, (np.arange(w2) + .5) * w / w2 - .5, indexing="ij")
    return remap(a if a.ndim == 3 else a[..., None], xs.astype(np.float32), ys.astype(np.float32)).reshape((h2, w2) + a.shape[2:])

def match_blend(a, b, k):
    """cross-fade a -> b with the overall brightness/colour interpolated smoothly, so exposure differences do not pop"""
    ma = a.reshape(-1, 3).mean(0) + 1.0; mb = b.reshape(-1, 3).mean(0) + 1.0
    mt = ma ** (1 - k) * mb ** k
    return a * (mt / ma) * (1 - k) + b * (mt / mb) * k

def radial_blur(img, amount, centre=(0.5, 0.5), taps=9):
    if amount < 0.002: return img
    h, w = img.shape[:2]; ys, xs = np.mgrid[0:h, 0:w].astype(np.float32); cx, cy = centre[0] * w, centre[1] * h
    acc = np.zeros_like(img)
    for f in np.linspace(-0.5, 0.5, taps): acc += remap(img, cx + (xs - cx) * (1 + amount * f), cy + (ys - cy) * (1 + amount * f))
    return acc / taps


# ---- keep the circular lights together through 007 > 008 ----------------------------------------------------------
# One room box cannot place lights at different depths exactly. So each light is located once in each photo (as a 3D point),
# and during the move it is slid from where photo A puts it to where photo B puts it, with a small local warp around it.
RING_FILE = os.path.join(ROOT, "tools", "ring_points_007_008.json")
RING_START = [(237, 237), (575, 236), (1085, 108)]            # left wreath, right wreath (both on the far wall), large ring on the right wall: first view, 1280 wide
RING_POINTS = json.load(open(RING_FILE)) if os.path.exists(RING_FILE) else None

def _look(P):
    """(yaw, pitch) in degrees of a vector"""
    n = np.linalg.norm(P); return math.degrees(math.atan2(P[0], P[2])), math.degrees(math.asin(P[1] / n))

def _centroid(g, cx, cy, rad, thr_frac=0.86):
    x0, x1, y0, y1 = int(max(0, cx - rad)), int(min(W, cx + rad)), int(max(0, cy - rad)), int(min(H, cy + rad))
    sub = g[y0:y1, x0:x1]; ys, xs = np.nonzero(sub > thr_frac * sub.max())
    return (xs.mean() + x0, ys.mean() + y0)

def _ray(x, y, pose):
    """world ray through pixel (x, y) of a view with this pose"""
    tx = math.tan(pose[2] * D2R / 2); ty = tx * H / W
    d = np.array([((x / W) * 2 - 1) * tx, (1 - (y / H) * 2) * ty, 1.0]); d /= np.linalg.norm(d)
    p = pose[1] * D2R; c, s_ = math.cos(p), math.sin(p); d = np.array([d[0], c * d[1] + s_ * d[2], -s_ * d[1] + c * d[2]])
    y_ = pose[0] * D2R; c, s_ = math.cos(y_), math.sin(y_); return np.array([c * d[0] + s_ * d[2], d[1], -s_ * d[0] + c * d[2]])

def _radius(g, cx, cy, rad, thr_frac=0.86):
    x0, x1, y0, y1 = int(max(0, cx - rad)), int(min(W, cx + rad)), int(max(0, cy - rad)), int(min(H, cy + rad))
    sub = g[y0:y1, x0:x1]; ys, xs = np.nonzero(sub > thr_frac * sub.max())
    return max(xs.max() - xs.min(), ys.max() - ys.min()) / 2.0

WIDE = 100.0                                                       # measure through a wide view so no light is cut off by the frame edge
def build_ring_points():
    global R0, R1, RING_POINTS
    R0, R1 = 0, H; sc = W / 1280.0; bx, bz, psi, c, xl, xr, zb, zf = G78
    t0, t1 = T_TR2; pose0, pose1 = POSE["007"](t0), POSE["007"](t1)
    w0, w1 = (pose0[0], pose0[1], WIDE), (pose1[0], pose1[1], WIDE)
    k = math.tan(WIDE * D2R / 2) / math.tan(pose0[2] * D2R / 2)    # how much smaller things are in the wide view
    a0, _ = travel_ab("007", "008", G78, w0, 0.0, 0.0); ga0 = a0.mean(2)
    _, b1 = travel_ab("007", "008", G78, w1, 1.0, 1.0); gb1 = b1.mean(2)
    PA, PB, RA, RB = [], [], [], []
    Z = np.zeros(3, np.float32); Pb = np.array([bx, 0, bz])
    for (sx, sy) in RING_START:
        d0 = _ray(sx * sc, sy * sc, pose0); px = project(*_look(d0), w0)       # where that light sits in the wide view
        cxa, cya = _centroid(ga0, px[0], px[1], 62 * sc / k); d = _ray(cxa, cya, w0)
        PA.append((d * box_hit(Z, d[None, None, :].astype(np.float32), c, xl, xr, zb, zf)[0, 0]).tolist()); RA.append(_radius(ga0, cxa, cya, 62 * sc / k) * k / sc)
    for P in PA:                                                       # find the same light in photo B near where A's 3D point lands
        px = project(*_look(np.array(P) - Pb), w1)
        cxb, cyb = _centroid(gb1, px[0], px[1], 70 * sc / k); d = _ray(cxb, cyb, w1)
        PB.append((Pb + d * box_hit(Pb.astype(np.float32), d[None, None, :].astype(np.float32), c, xl, xr, zb, zf)[0, 0]).tolist()); RB.append(_radius(gb1, cxb, cyb, 70 * sc / k) * k / sc)
    RING_POINTS = {"PA": PA, "PB": PB, "rA": RA, "rB": RB}; json.dump(RING_POINTS, open(RING_FILE, "w")); return RING_POINTS

def ring_morph(a, b, t, k, prog):
    """glide each light from its place and size in view A to its place and size in view B (a at k=0, b at k=1 untouched)"""
    if RING_POINTS is None: return a, b
    h, w = a.shape[:2]; ys, xs = np.mgrid[0:h, 0:w].astype(np.float32); sc = w / 1280.0
    pose = POSE["007"](t); Pb = np.array([G78[0], 0, G78[1]]); C = Pb * prog
    SAx, SAy = xs.copy(), ys.copy(); SBx, SBy = xs.copy(), ys.copy()
    for PA, PB, rA0, rB1 in zip(RING_POINTS["PA"], RING_POINTS["PB"], RING_POINTS["rA"], RING_POINTS["rB"]):
        PA, PB = np.array(PA), np.array(PB)
        ca = project(*_look(PA - C), pose); cb = project(*_look(PB - C), pose)
        if ca is None or cb is None: continue
        rA = rA0 * np.linalg.norm(PA) / np.linalg.norm(PA - C); rB = rB1 * np.linalg.norm(PB - Pb) / np.linalg.norm(PB - C)
        s_ = float(np.clip(rB / rA, 0.7, 1.4)); fA = 1 + k * (s_ - 1); fB = fA / s_
        cm = (ca[0] + k * (cb[0] - ca[0]), ca[1] + k * (cb[1] - ca[1])); rm = rA * sc * fA
        g = np.exp(-((xs - cm[0]) ** 2 + (ys - cm[1]) ** 2) / (2 * (1.7 * rm) ** 2))
        SAx += g * ((ca[0] + (xs - cm[0]) / fA) - xs); SAy += g * ((ca[1] + (ys - cm[1]) / fA) - ys)
        SBx += g * ((cb[0] + (xs - cm[0]) / fB) - xs); SBy += g * ((cb[1] + (ys - cm[1]) / fB) - ys)
    return remap(a, SAx, SAy), remap(b, SBx, SBy)

# ---- interface overlay -----------------------------------------------------------------------------------------
FONT_DIR = "/usr/share/fonts/opentype/inter/"
def font(weight, px):
    try: return ImageFont.truetype(FONT_DIR + "Inter-%s.otf" % weight, int(round(px)))
    except Exception: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(round(px)))
SS = 3                                                  # supersampling for crisp small shapes
PANEL = (16, 20, 18, 168); LINE = (255, 255, 255, 46); TEXT = (243, 239, 230, 255); MUTED = (194, 189, 176, 255); LAMP = (242, 180, 92, 255); LAMP_INK = (29, 20, 7, 255)

def rr(d, box, r, fill, outline=None):
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=SS if outline else 0)

def text_w(f, s): return f.getlength(s)

def pill(w_ref, h_ref, fill=PANEL, outline=LINE, r=10):
    im = Image.new("RGBA", (int(w_ref * S_UI * SS), int(h_ref * S_UI * SS)), (0, 0, 0, 0))
    rr(ImageDraw.Draw(im), (0, 0, im.width - 1, im.height - 1), r * S_UI * SS, fill, outline)
    return im

def down(im): return im.resize((max(1, im.width // SS), max(1, im.height // SS)), Image.LANCZOS)

def build_static(scene_id):
    """title card + top-right buttons for one scene, as a full-frame RGBA"""
    full = Image.new("RGBA", (W, H), (0, 0, 0, 0)); s = S_UI * SS
    # title card
    f1, f2 = font("Medium", 11 * s), font("Bold", 18 * s)
    n = "SCENE %02d OF %d" % (IDS.index(scene_id) + 1, len(IDS))
    tw = max(text_w(f1, n) * 1.12, text_w(f2, title_of(scene_id))) / s
    card = pill(tw + 28, 54); d = ImageDraw.Draw(card)
    x = 14 * s; sp = 0
    for ch in n:
        d.text((x + sp, 10 * s), ch, font=f1, fill=MUTED); sp += text_w(f1, ch) * 1.12
    d.text((14 * s, 24 * s), title_of(scene_id), font=f2, fill=TEXT)
    full.alpha_composite(down(card), (int(16 * S_UI), int(14 * S_UI)))
    # buttons, right-aligned
    fb = font("Medium", 14 * s); gap = 8; x_right = 1280 - 16
    btns = [("fs", "", 40, False), ("rec", "Record tour", None, False), ("walk", "Walk the route", None, False), ("play", "Play guided tour", None, True)]
    for kind, label, fixed, primary in btns:
        w = fixed if fixed else (14 + 22 + text_w(fb, label) / s + 14)
        b = pill(w, 40, fill=LAMP if primary else PANEL, outline=None if primary else LINE); d = ImageDraw.Draw(b); ink = LAMP_INK if primary else TEXT
        cy = 20 * s
        if kind == "fs":
            for (ax, ay, bx_, by_, cx_, cy_) in [(10, 14, 10, 10, 14, 10), (26, 10, 30, 10, 30, 14), (30, 26, 30, 30, 26, 30), (14, 30, 10, 30, 10, 26)]:
                d.line([(ax * s, ay * s), (bx_ * s, by_ * s), (cx_ * s, cy_ * s)], fill=ink, width=int(1.6 * s))
        else:
            ix = 14 * s
            if kind == "play": d.polygon([(ix, cy - 7 * s), (ix, cy + 7 * s), (ix + 11 * s, cy)], fill=ink)
            if kind == "walk":
                d.line([(ix, cy), (ix + 10 * s, cy)], fill=ink, width=int(1.8 * s)); d.line([(ix + 6 * s, cy - 4 * s), (ix + 10 * s, cy), (ix + 6 * s, cy + 4 * s)], fill=ink, width=int(1.8 * s))
            if kind == "rec": d.ellipse((ix, cy - 5 * s, ix + 10 * s, cy + 5 * s), fill=(255, 90, 95, 255))
            d.text((ix + 22 * s, cy), label, font=fb, fill=ink, anchor="lm")
        im = down(b); x_right -= w
        full.alpha_composite(im, (int(x_right * S_UI), int(14 * S_UI))); x_right -= gap
    return np.asarray(full).astype(np.float32)

CHIP_H = 46; CHIP_GAP = 8
CHIPS, CHIPS_W = None, None
CHIP_CACHE = {}
def chip_strip(cur_id):
    """full-width strip with the current chip highlighted (cached per scene)"""
    if cur_id in CHIP_CACHE: return CHIP_CACHE[cur_id]
    s = S_UI * SS; f1, f2 = font("Medium", 11 * s), font("Medium", 14 * s)
    items, x = [], 0
    for i, sid in enumerate(IDS):
        t1 = "%02d / %d" % (i + 1, len(IDS)); t2 = title_of(sid)
        w = max(text_w(f1, t1), text_w(f2, t2)) / s + 24; items.append((sid, t1, t2, x, w)); x += w + CHIP_GAP
    strip = Image.new("RGBA", (int(x * S_UI) + 4, int((CHIP_H + 4) * S_UI)), (0, 0, 0, 0)); xs = {}
    for sid, t1, t2, cx, w in items:
        on = sid == cur_id
        im = pill(w, CHIP_H, fill=PANEL, outline=LAMP if on else LINE, r=10); d = ImageDraw.Draw(im)
        d.text((12 * s, 12 * s), t1, font=f1, fill=LAMP if on else MUTED, anchor="lm"); d.text((12 * s, 31 * s), t2, font=f2, fill=TEXT, anchor="lm")
        strip.alpha_composite(down(im), (int(cx * S_UI), int(2 * S_UI))); xs[sid] = (cx, w)
    CHIP_CACHE[cur_id] = (np.asarray(strip).astype(np.float32), xs, x * S_UI)
    return CHIP_CACHE[cur_id]

def blend(img, rgba, x, y, mul=1.0):
    """alpha-blend float RGBA patch (h,w,4) onto img at (x,y), clipped"""
    h, w = rgba.shape[:2]; x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    p = rgba[y0 - y:y1 - y, x0 - x:x1 - x]; a = (p[..., 3:4] / 255.0) * mul
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + p[..., :3] * a

STATIC = {}
def scene_at(t):
    cur = SCENE_AT[0][1]
    for ts, sid in SCENE_AT:
        if t >= ts: cur = sid
    return cur

def project(yaw, pitch, hs):
    """screen position (px) of a world direction for the camera pose (yaw, pitch, hfov); None if behind"""
    ty0 = math.tan(hs[2] * D2R / 2); tx = ty0; ty = tx * H / W
    cy, sy, cp, sp = math.cos(hs[0] * D2R), math.sin(hs[0] * D2R), math.cos(hs[1] * D2R), math.sin(hs[1] * D2R)
    p, y = pitch * D2R, yaw * D2R
    x0, y0, z0 = math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)
    x1, z1 = cy * x0 - sy * z0, sy * x0 + cy * z0
    y2, z2 = cp * y0 - sp * z1, sp * y0 + cp * z1
    if z2 <= 0.05: return None
    return ((x1 / z2 / tx * 0.5 + 0.5) * W, (0.5 - y2 / z2 / ty * 0.5) * H)

HS_CACHE = {}
def hotspot_patch(label, hover, pulse):
    """RGBA float patch + ring-centre offset. hover = 0..1 (amber fill), pulse = 0..1 phase"""
    key = (label, round(hover, 2), round(pulse, 2))
    if key in HS_CACHE: return HS_CACHE[key]
    s = S_UI * SS; f = font("Medium", 13 * s)
    lw = text_w(f, label) / s + 20; R = 23; pad = 14
    w_ref = pad + R * 2 + 10 + lw + 4; h_ref = R * 2 + pad * 2
    im = Image.new("RGBA", (int(w_ref * s), int(h_ref * s)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    cx, cy = (pad + R) * s, (pad + R) * s
    # pulse ring
    k = pulse; scale = 0.8 + 0.55 * k; alpha = int(255 * 0.9 * (1 - k)); rr_ = (R + 8) * scale * s
    d.ellipse((cx - rr_, cy - rr_, cx + rr_, cy + rr_), outline=LAMP[:3] + (alpha,), width=int(2 * s))
    # ring
    rs = 1 + 0.08 * hover; r = R * rs * s
    fill = tuple(int(a * (1 - hover) + b * hover) for a, b in zip((16, 20, 18, 160), LAMP))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=TEXT if hover < 0.5 else LAMP, width=int(2 * s))
    ink = LAMP_INK if hover > 0.5 else TEXT; sc = rs * s
    d.line([(cx - 6 * sc, cy + 2 * sc), (cx, cy - 4 * sc), (cx + 6 * sc, cy + 2 * sc)], fill=ink, width=int(2.4 * sc), joint="curve")
    # label
    lx = cx + (R + 10) * s
    rr(d, (lx, cy - 15 * s, lx + lw * s, cy + 15 * s), 8 * s, PANEL, LINE)
    d.text((lx + 10 * s, cy), label, font=f, fill=TEXT, anchor="lm")
    out = (np.asarray(down(im)).astype(np.float32), (pad + R) * S_UI, (pad + R) * S_UI)
    if len(HS_CACHE) > 400: HS_CACHE.clear()
    HS_CACHE[key] = out
    return out

def cursor_patch(press):
    s = S_UI * SS * (0.86 if press else 1.0)
    im = Image.new("RGBA", (int(26 * s), int(32 * s)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    pts = [(2, 2), (2, 23), (7, 18.5), (10.5, 26.5), (14, 25), (10.5, 17), (17, 17)]
    pts = [(x * s, y * s) for x, y in pts]
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255)); d.line(pts + [pts[0]], fill=(20, 20, 20, 255), width=int(1.4 * s))
    return np.asarray(down(im)).astype(np.float32)

def ripple_patch(k):
    """expanding click ripple, k = 0..1"""
    R = 40 * S_UI; im = Image.new("RGBA", (int(2 * R * SS), int(2 * R * SS)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    r = (6 + 30 * k) * S_UI * SS; a = int(150 * (1 - k)); c = R * SS
    d.ellipse((c - r, c - r, c + r, c + r), outline=(255, 255, 255, a), width=int(2.2 * S_UI * SS))
    return np.asarray(down(im)).astype(np.float32), R

MOUSE_START = [(0.74, 0.70), (0.30, 0.68), (0.70, 0.30), (0.26, 0.34)]
def minjerk(u):
    u = min(1.0, max(0.0, u)); return 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5

def human_cursor(t, click, tgt, idx):
    """curved approach, a soft overshoot and correction, a pause; returns (x, y, alpha, pressed, ripple_k or None)"""
    t0, t1 = click[0], click[1]; t_arr = t1 - 0.55
    P0 = np.array([MOUSE_START[idx][0] * W, MOUSE_START[idx][1] * H]); T = np.array(tgt, float)
    v = T - P0; dist = np.linalg.norm(v) + 1e-6; dirv = v / dist; perp = np.array([-dirv[1], dirv[0]]) * (1 if idx % 2 == 0 else -1)
    T1 = T + dirv * 7 * S_UI                                      # overshoots a touch
    ctrl = P0 + v * 0.45 + perp * 0.17 * dist
    u = minjerk((t - t0) / (t_arr - t0))
    pos = (1 - u) ** 2 * P0 + 2 * (1 - u) * u * ctrl + u ** 2 * T1
    if t > t_arr: pos = T1 + (T - T1) * minjerk((t - t_arr) / 0.30)
    pressed = t1 <= t <= t1 + 0.14
    ripple = (t - t1) / 0.45 if t1 <= t <= t1 + 0.45 else None
    return pos[0], pos[1], sm(t0, t0 + 0.35, t), pressed, ripple

def ease(k): return 4 * k ** 3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2

def in_transition(t):
    return any(t0 <= t <= t1 for (t0, t1), *_ in TRAVELS) or T_BLUR[0] <= t <= T_BLUR[1]

def draw_ui(img, t):
    sid = scene_at(t)
    if sid not in STATIC: STATIC[sid] = build_static(sid)
    blend(img, STATIC[sid], 0, 0)
    # chip strip, centred on the current chip, scrolling smoothly when the scene changes
    strip, xs, total = chip_strip(sid)
    cx, w = xs[sid]; centre = (cx + w / 2) * S_UI
    prev_t = max([ts for ts, _ in SCENE_AT if ts <= t])
    if prev_t > 0:
        pid = SCENE_AT[[ts for ts, _ in SCENE_AT].index(prev_t) - 1][1]; pcx, pw = xs[pid]
        centre = (pcx + pw / 2) * S_UI + ((cx + w / 2) * S_UI - (pcx + pw / 2) * S_UI) * ease(min(1, (t - prev_t) / 0.7))
    left = int(round(min(max(centre - W / 2, 0), max(0, strip.shape[1] - W))))
    crop = strip[:, left:left + W]; blend(img, crop, 0, H - crop.shape[0] - 14 * S_UI)
    # hint pill
    if HINT[0] <= t <= HINT[1] + 0.8:
        a = sm(HINT[0], HINT[0] + 0.5, t) * (1 - sm(HINT[1], HINT[1] + 0.8, t))
        s = S_UI * SS; f = font("Medium", 13 * s); txt = "Drag to look around. Scroll or pinch to zoom. Tap a ring to move."
        hp = pill(text_w(f, txt) / s + 28, 34, r=17); ImageDraw.Draw(hp).text((14 * s, 17 * s), txt, font=f, fill=MUTED, anchor="lm")
        im = np.asarray(down(hp)).astype(np.float32); blend(img, im, (W - im.shape[1]) / 2, H - (CHIP_H + 14 + 10 + 34) * S_UI, a)
    if in_transition(t): return
    # hotspots for this scene
    y_, p_, f_ = POSE[sid](t); pose = (y_, p_, f_)
    click = next((c for c in CLICKS if c[2] == sid and c[0] <= t <= c[1] + 0.35), None)
    for h in hotspots_of(sid):
        pos = project(h["yaw"], h["pitch"], pose)
        if pos is None or not (-120 < pos[0] < W + 120 and -60 < pos[1] < H + 60): continue
        hover = 0.0; is_target = bool(click and click[3] == h["to"])
        if is_target: hover = sm(click[1] - 0.55, click[1] - 0.3, t)         # lights up as the pointer arrives
        patch, ox, oy = hotspot_patch(h["label"], hover, (t % 2.4) / 2.4)
        blend(img, patch, pos[0] - ox, pos[1] - oy)
        if is_target:
            cx_, cy_, ca, pressed, rip = human_cursor(t, click, pos, CLICKS.index(click))
            if rip is not None:
                rp, R = ripple_patch(rip); blend(img, rp, cx_ - R, cy_ - R)
            blend(img, cursor_patch(pressed), cx_ - 2 * S_UI, cy_ - 2 * S_UI, ca)

def frame_at(t):
    img = render_pano(t)
    if not ARGS.no_ui: draw_ui(img, t)
    img = img * (sm(0, FADE_IN, t) * sm(0, FADE_OUT, TOTAL - t))
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)

def job(i): return frame_at(i / FPS).tobytes()

if __name__ == "__main__":
    if RING_POINTS is None and (ARGS.stills or ARGS.t_from < T_TR2[1]) and (ARGS.t_to is None or ARGS.t_to > T_TR2[0]): build_ring_points()
    if ARGS.stills:
        for s_ in ARGS.stills.split(","): Image.fromarray(frame_at(float(s_))).save(ARGS.out + "_%06.2f.png" % float(s_))
        sys.exit()
    f0 = int(round(ARGS.t_from * FPS)); f1 = int(round((ARGS.t_to if ARGS.t_to is not None else TOTAL) * FPS)); n = f1 - f0
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
                           "-i", "-", "-vf", "unsharp=5:5:0.4:5:5:0.0", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", ARGS.out], stdin=subprocess.PIPE)
    with Pool(ARGS.workers) as pool:
        for k, raw in enumerate(pool.imap(job, range(f0, f1), chunksize=2)):
            ff.stdin.write(raw)
            if k % 120 == 0: print(k, "/", n, flush=True)
    ff.stdin.close(); ff.wait(); print("done")
