"""Render the 006 > 007 and 025 > 026 tour video straight from the full-resolution equirect originals.

  python3 tools/render_video.py out.mp4 [--w 3840 --h 2160 --fps 30 --workers 4] [--stills t1,t2,...]

Camera poses are (yaw, pitch, hfov) in degrees, smoothly interpolated (monotone cubic) between keyframes.
A "travel" moves the camera along the measured line between two photo positions, reprojecting both photos onto an
estimated room box, so far walls and the ceiling stay locked while the nearest furniture dissolves.
"""
import sys, os, math, json, subprocess, argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--w", type=int, default=3840); ap.add_argument("--h", type=int, default=2160)
ap.add_argument("--fps", type=int, default=30); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--no-ring", action="store_true"); ap.add_argument("--stills", default=""); ap.add_argument("--src", default="full")
ARGS = ap.parse_args() if __name__ == "__main__" else ap.parse_args(["x"])
W, H, FPS = ARGS.w, ARGS.h, ARGS.fps
H_CAM = 1.6
D2R = math.pi / 180

# measured camera moves (refined at 1024 px): [bx, bz, psi, ceiling, xl, xr, zb, zf]
G67 = [0.297, 3.547, 1.845, 3.659, -16.0, 5.63, -18.699, 34.208]
G2526 = [-3.465, -1.05, 1.129, 4.453, -4.931, 5.194, -19.82, 6.128]
HEAD67 = math.degrees(math.atan2(G67[0], G67[1]))        # bearing of 007 seen from 006
HEAD2526 = math.degrees(math.atan2(G2526[0], G2526[1]))  # bearing of 026 seen from 025

# ---- timeline (seconds) --------------------------------------------------------------------------------------
T_TRAVEL1 = (22.5, 28.5)      # 006 -> 007
T_MIX = (37.0, 38.5)          # 007 -> 025 dissolve
T_TRAVEL2 = (52.5, 58.5)      # 025 -> 026
TOTAL = 76.0
FADE_IN, FADE_OUT = 0.8, 1.2
y7 = HEAD67 - G67[2]          # yaw after arriving in 007 (same world direction)
y26 = HEAD2526 - G2526[2]
SEG = {
    # scene : (file number, keyframes (t, yaw, pitch, hfov))
    "006": ("006", [(0, -42, -3, 78), (5.5, -20, -2, 70), (11.5, 2, 0, 36), (14, 3.5, 0, 34), (19, HEAD67, -11, 70),
                     (22.5, HEAD67, -16, 50), (25.5, HEAD67, -9, 56), (27.5, HEAD67, -3, 62), (28.5, HEAD67, -3, 62)]),
    "007": ("007", [(28.5, y7, -3, 62), (30, y7, -3, 62), (33.5, y7 - 10, -2, 58), (37.0, 70, -4, 54), (38.5, 74, -4, 50)]),
    "025": ("025", [(37.0, 4, -4, 56), (43, -20, -3, 62), (49, HEAD2526 + 5, -8, 68), (52.5, HEAD2526, -16, 50),
                     (55.5, HEAD2526, -9, 56), (57.5, HEAD2526, -5, 64), (58.5, HEAD2526, -5, 64)]),
    "026": ("026", [(58.5, y26, -5, 64), (60, y26, -5, 64), (65, y26 + 34, -4, 60), (70, y26 + 82, -4, 58), (76, y26 + 128, -4, 58)]),
}

def pchip(ts, vs):
    ts = np.asarray(ts, float); vs = np.asarray(vs, float); n = len(ts)
    h = np.diff(ts); d = np.diff(vs) / h; m = np.zeros(n)
    for k in range(1, n - 1):
        if d[k - 1] * d[k] > 0:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    m[0] = m[-1] = 0.0                                   # come to rest at the ends
    def f(t):
        t = min(max(t, ts[0]), ts[-1]); k = min(np.searchsorted(ts, t, side="right") - 1, n - 2)
        s = (t - ts[k]) / h[k]; s2, s3 = s * s, s * s * s
        return (2 * s3 - 3 * s2 + 1) * vs[k] + (s3 - 2 * s2 + s) * h[k] * m[k] + (-2 * s3 + 3 * s2) * vs[k + 1] + (s3 - s2) * h[k] * m[k + 1]
    return f

POSE = {}
for name, (_, keys) in SEG.items():
    fs = [pchip([k[0] for k in keys], [k[c] for k in keys]) for c in (1, 2, 3)]
    POSE[name] = (keys[0][0], keys[-1][0], lambda t, fs=fs: tuple(f(t) for f in fs))

def sm(a, b, x):
    x = min(1.0, max(0.0, (x - a) / (b - a))); return x * x * x * (x * (6 * x - 15) + 10)

# ---- rendering -----------------------------------------------------------------------------------------------
SRC = {}
def src(num):
    if num not in SRC:
        im = Image.open(os.path.join(ROOT, "ImageScape_360_%s.jpg" % num) if ARGS.src == "full" else os.path.join(ROOT, "docs/img/%s.jpg" % num)).convert("RGB")
        SRC[num] = np.ascontiguousarray(np.asarray(im))
    return SRC[num]

def rot_yaw(d, deg):
    c, s = math.cos(deg * D2R), math.sin(deg * D2R)
    return np.stack([c * d[..., 0] + s * d[..., 2], d[..., 1], -s * d[..., 0] + c * d[..., 2]], -1)

R0, R1 = 0, H
STRIP = 72
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

def travel(numA, numB, prm, pose, prog, kmix, ring_a=0.0):
    bx, bz, psi, c, xl, xr, zb, zf = prm
    d = view_dirs(*pose)
    C = np.array([bx * prog, 0, bz * prog], np.float32)
    t = box_hit(C, d, c, xl, xr, zb, zf)
    P = C + t[..., None] * d
    a = sample(src(numA), P)
    if kmix > 0.002:
        q = rot_yaw(P - np.array([bx, 0, bz], np.float32), -psi).astype(np.float32)
        a = a * (1 - kmix) + sample(src(numB), q) * kmix
    return ring(a, d, C, (bx, bz), ring_a)

def ring(img, d, C, target, alpha):
    """soft white ring + faint disc on the floor around `target` (x, z), seen from camera C along rays d"""
    if alpha <= 0.002 or ARGS.no_ring: return img
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-H_CAM - C[1]) / d[..., 1]
    ok = (d[..., 1] < -0.02) & (t > 0)
    px = C[0] + np.where(ok, t, 0) * d[..., 0]; pz = C[2] + np.where(ok, t, 0) * d[..., 2]
    r = np.hypot(px - target[0], pz - target[1])
    edge = np.clip(1 - np.abs(r - 0.55) / 0.05, 0, 1) ** 1.5 * 0.95        # ring
    disc = np.clip((0.55 - r) / 0.12, 0, 1) * 0.10                         # faint fill
    a = (np.maximum(edge, disc) * ok * alpha)[..., None].astype(np.float32)
    return img * (1 - a) + 255.0 * a

def ring_alpha(t, t_show, t_travel, prog=0.0):
    return sm(t_show, t_show + 2.0, t) * (1 - sm(0.0, 0.45, prog))

def scene_view(name, t):
    return sample(src(SEG[name][0]), view_dirs(*POSE[name][2](t)))

def frame_strip(t):
    if T_TRAVEL1[0] <= t <= T_TRAVEL1[1]:
        u = (t - T_TRAVEL1[0]) / (T_TRAVEL1[1] - T_TRAVEL1[0]); prog = sm(0, 1, u)
        img = travel("006", "007", G67, POSE["006"][2](t), prog, sm(0.36, 0.64, prog), ring_alpha(t, 19.5, 22.5, prog))
    elif T_TRAVEL2[0] <= t <= T_TRAVEL2[1]:
        u = (t - T_TRAVEL2[0]) / (T_TRAVEL2[1] - T_TRAVEL2[0]); prog = sm(0, 1, u)
        img = travel("025", "026", G2526, POSE["025"][2](t), prog, sm(0.36, 0.64, prog), ring_alpha(t, 49.5, 52.5, prog))
    elif T_MIX[0] <= t <= T_MIX[1]:
        k = sm(0, 1, (t - T_MIX[0]) / (T_MIX[1] - T_MIX[0]))
        img = scene_view("007", t) * (1 - k) + scene_view("025", t) * k
    else:
        name = "006" if t < T_TRAVEL1[0] else "007" if t < T_MIX[0] else "025" if t < T_TRAVEL2[0] else "026"
        img = scene_view(name, t)
        if name == "006" and t > 19.5:
            img = ring(img, view_dirs(*POSE["006"][2](t)), (0, 0, 0), (G67[0], G67[1]), ring_alpha(t, 19.5, 22.5))
        if name == "025" and t > 49.5:
            img = ring(img, view_dirs(*POSE["025"][2](t)), (0, 0, 0), (G2526[0], G2526[1]), ring_alpha(t, 49.5, 52.5))
    f = sm(0, FADE_IN, t) * sm(0, FADE_OUT, TOTAL - t)
    return np.clip(img * f + 0.5, 0, 255).astype(np.uint8)

def frame_at(t):
    global R0, R1
    out = []
    for r in range(0, H, STRIP):
        R0, R1 = r, min(H, r + STRIP)
        out.append(frame_strip(t))
    R0, R1 = 0, H
    return np.concatenate(out, 0)

def job(i):
    return frame_at(i / FPS).tobytes()

if __name__ == "__main__":
    if ARGS.stills:
        for s in ARGS.stills.split(","):
            Image.fromarray(frame_at(float(s))).save(ARGS.out + "_%05.1f.png" % float(s))
        sys.exit()
    n = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
                           "-i", "-", "-vf", "unsharp=5:5:0.5:5:5:0.0", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", ARGS.out], stdin=subprocess.PIPE)
    with Pool(ARGS.workers) as pool:
        for k, raw in enumerate(pool.imap(job, range(n), chunksize=2)):
            ff.stdin.write(raw)
            if k % 60 == 0: print(k, "/", n, flush=True)
    ff.stdin.close(); ff.wait(); print("done")
