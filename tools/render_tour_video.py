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
ap.add_argument("--stills", default=""); ap.add_argument("--src", default="full"); ap.add_argument("--no-ui", action="store_true")
ARGS = ap.parse_args()
W, H, FPS = ARGS.w, ARGS.h, ARGS.fps
H_CAM = 1.6
D2R = math.pi / 180
S_UI = W / 1280.0

TOUR = json.loads(open(os.path.join(ROOT, "docs", "scenes.js")).read().split("=", 1)[1].strip().rstrip(";"))
IDS = [s["id"] for s in TOUR["scenes"]]
def title_of(i): return TOUR["titles"].get(i, "Scene " + i)
def link(a, b):
    l = TOUR["links"].get(a + ">" + b); return l if l and l["ok"] else None
def hotspots_of(i):
    k = IDS.index(i); out = []
    extras = TOUR["extra"].get(i, [])
    for nb in (IDS[k - 1] if k > 0 else None, IDS[k + 1] if k + 1 < len(IDS) else None):
        if nb is None: continue
        l = link(i, nb)
        if l and not any(e["to"] == nb for e in extras):
            out.append({"to": nb, "yaw": math.degrees(math.atan2(l["p"][0], l["p"][1])), "pitch": -9, "label": title_of(nb)})
    for e in extras: out.append({"to": e["to"], "yaw": e["yaw"], "pitch": e["pitch"], "label": e.get("label") or title_of(e["to"])})
    return out

G67 = TOUR["links"]["006>007"]["p"]
G2526 = TOUR["links"]["025>026"]["p"]
HEAD67 = math.degrees(math.atan2(G67[0], G67[1]))
HEAD2526 = math.degrees(math.atan2(G2526[0], G2526[1])) + 360          # unwrapped so the sweep to it is clockwise
y7 = HEAD67 - G67[2]
y26 = HEAD2526 - G2526[2]

# ---- timeline (seconds) ---------------------------------------------------------------------------------------
T_TRAVEL1 = (24.5, 33.5)       # 006 -> 007
T_BLUR = (47.3, 49.5)          # 007 -> 025 (zoom dissolve, as in the viewer)
T_TRAVEL2 = (88.6, 97.6)       # 025 -> 026
TOTAL = 139.5
FADE_IN, FADE_OUT = 0.8, 1.5
SEG = {
    "006": [(0, -66, -2, 76), (6, -56, -2, 70), (12, -36, -2, 56), (15, -30, -2, 52), (17, -28, -2, 52),
            (20, -10, -3, 60), (22.5, HEAD67, -3, 64), (24.5, HEAD67, -3, 64), (33.5, HEAD67, -3, 64)],
    "007": [(33.5, y7, -3, 64), (36, y7, -3, 64), (42, 35, -3, 58), (47.3, 70, -4, 54), (49.5, 74, -4, 52)],
    "025": [(47.3, 8, -4, 60), (51.5, 8, -3, 58), (57, 40, -3, 60), (65, 100, -3, 62), (73, 160, -3, 62),
            (80, 215, -3, 62), (85, HEAD2526, -3, 62), (88.6, HEAD2526, -3, 62), (97.6, HEAD2526, -3, 62)],
    "026": [(97.6, y26, -3, 62), (100, y26, -3, 62), (110, y26 + 70, -3, 62), (120, y26 + 140, -3, 61), (130, y26 + 210, -3, 60),
            (137, y26 + 258, -3, 60), (139.5, y26 + 260, -3, 60)],
}
# mouse clicks: (cursor appears, click, scene, target scene)
CLICKS = [(21.8, 24.1, "006", "007"), (44.6, 47.0, "007", "025"), (86.0, 88.3, "025", "026")]
SCENE_AT = [(0, "006"), (T_TRAVEL1[1], "007"), (T_BLUR[1], "025"), (T_TRAVEL2[1], "026")]
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

def travel(numA, numB, prm, pose, prog, kmix):
    bx, bz, psi, c, xl, xr, zb, zf = prm
    d = view_dirs(*pose)
    C = np.array([bx * prog, 0, bz * prog], np.float32)
    P = C + box_hit(C, d, c, xl, xr, zb, zf)[..., None] * d
    a = sample(src(numA), P)
    if kmix > 0.002:
        q = rot_yaw(P - np.array([bx, 0, bz], np.float32), -psi).astype(np.float32)
        a = a * (1 - kmix) + sample(src(numB), q) * kmix
    return a

def scene_view(name, t, fov_scale=1.0):
    y, p, f = POSE[name](t)
    return sample(src(name), view_dirs(y, p, f * fov_scale))

def frame_strip(t):
    if T_TRAVEL1[0] <= t <= T_TRAVEL1[1]:
        prog = sm(0, 1, (t - T_TRAVEL1[0]) / (T_TRAVEL1[1] - T_TRAVEL1[0]))
        return travel("006", "007", G67, POSE["006"](t), prog, sm(0.4, 0.6, prog))
    if T_TRAVEL2[0] <= t <= T_TRAVEL2[1]:
        prog = sm(0, 1, (t - T_TRAVEL2[0]) / (T_TRAVEL2[1] - T_TRAVEL2[0]))
        return travel("025", "026", G2526, POSE["025"](t), prog, sm(0.4, 0.6, prog))
    if T_BLUR[0] <= t <= T_BLUR[1]:                      # zoom dissolve, as the viewer does for distant scenes
        e = sm(0, 1, (t - T_BLUR[0]) / (T_BLUR[1] - T_BLUR[0])); S = 1.35; z = 1 + (S - 1) * e
        return scene_view("007", t, 1 / z) * (1 - sm(0.32, 0.68, e)) + scene_view("025", t, S / z) * sm(0.32, 0.68, e)
    name = "006" if t < T_TRAVEL1[0] else "007" if t < T_BLUR[0] else "025" if t < T_TRAVEL2[0] else "026"
    return scene_view(name, t)

def render_pano(t):
    global R0, R1
    out = []
    for r in range(0, H, STRIP):
        R0, R1 = r, min(H, r + STRIP); out.append(frame_strip(t))
    R0, R1 = 0, H
    return np.concatenate(out, 0)

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
    n = "SCENE %d OF %d" % (IDS.index(scene_id) + 1, len(IDS))
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
        t1 = "%d / %d" % (i + 1, len(IDS)); t2 = title_of(sid)
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
    s = S_UI * SS * (0.88 if press else 1.0)
    im = Image.new("RGBA", (int(26 * s), int(32 * s)), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    pts = [(2, 2), (2, 23), (7, 18.5), (10.5, 26.5), (14, 25), (10.5, 17), (17, 17)]
    pts = [(x * s, y * s) for x, y in pts]
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(20, 20, 20, 255)); d.line(pts + [pts[0]], fill=(20, 20, 20, 255), width=int(1.4 * s))
    return np.asarray(down(im)).astype(np.float32)

def ease(k): return 4 * k ** 3 if k < 0.5 else 1 - (-2 * k + 2) ** 3 / 2

def in_transition(t):
    return T_TRAVEL1[0] <= t <= T_TRAVEL1[1] or T_BLUR[0] <= t <= T_BLUR[1] or T_TRAVEL2[0] <= t <= T_TRAVEL2[1]

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
        hover = 0.0
        if click and click[3] == h["to"]: hover = sm(click[1] - 0.6, click[1] - 0.2, t)
        patch, ox, oy = hotspot_patch(h["label"], hover, (t % 2.4) / 2.4)
        blend(img, patch, pos[0] - ox, pos[1] - oy)
        if click and click[3] == h["to"]:
            k = ease(min(1, (t - click[0]) / (click[1] - click[0] - 0.3)))
            sx0, sy0 = W * 0.70, H * 0.62
            cxp, cyp = sx0 + (pos[0] - sx0) * k, sy0 + (pos[1] - sy0) * k
            press = click[1] - 0.05 <= t <= click[1] + 0.25
            blend(img, cursor_patch(press), cxp - 2 * S_UI, cyp - 2 * S_UI)

def frame_at(t):
    img = render_pano(t)
    if not ARGS.no_ui: draw_ui(img, t)
    img = img * (sm(0, FADE_IN, t) * sm(0, FADE_OUT, TOTAL - t))
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)

def job(i): return frame_at(i / FPS).tobytes()

if __name__ == "__main__":
    if ARGS.stills:
        for s_ in ARGS.stills.split(","): Image.fromarray(frame_at(float(s_))).save(ARGS.out + "_%06.2f.png" % float(s_))
        sys.exit()
    n = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
                           "-i", "-", "-vf", "unsharp=5:5:0.4:5:5:0.0", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", ARGS.out], stdin=subprocess.PIPE)
    with Pool(ARGS.workers) as pool:
        for k, raw in enumerate(pool.imap(job, range(n), chunksize=2)):
            ff.stdin.write(raw)
            if k % 120 == 0: print(k, "/", n, flush=True)
    ff.stdin.close(); ff.wait(); print("done")
