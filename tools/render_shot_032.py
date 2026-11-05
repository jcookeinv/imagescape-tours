"""10 s shot from photo 032: the metal gate on the right of the yard, then a pan across to the double doors of the timber building,
zooming into the middle of the doors.  Rendered straight from the 12K original.

  python3 tools/render_shot_032.py out.mp4 [--w 1920 --h 1080 --fps 60 --workers 4] [--stills t1,t2,...] [--no-repair]

The lens-stitch band around yaw +90 (where the gate sits) is smeared by the two-lens blend, so that band of the panorama is repaired
before rendering (Richardson-Lucy deconvolution, feathered in; see tools/deblur_patch.py).
"""
import sys, os, math, subprocess, argparse
import numpy as np
from multiprocessing import Pool
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--w", type=int, default=1920); ap.add_argument("--h", type=int, default=1080)
ap.add_argument("--fps", type=int, default=60); ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--stills", default=""); ap.add_argument("--no-repair", action="store_true"); ap.add_argument("--duration", type=float, default=10.0)
A = ap.parse_args()

# reuse the tour renderer's panorama maths (view directions, sampling, smooth interpolation)
sys.argv = ["render_tour_video.py", "x", "--w", str(A.w), "--h", str(A.h), "--fps", str(A.fps), "--src", "full"]
import importlib.util
spec = importlib.util.spec_from_file_location("rv", os.path.join(HERE, "render_tour_video.py")); rv = importlib.util.module_from_spec(spec); rv.__name__ = "rv"; spec.loader.exec_module(rv)
sys.path.insert(0, HERE)
import deblur_patch as DB
Image.MAX_IMAGE_PIXELS = None
W, H, FPS, TOTAL = A.w, A.h, A.fps, A.duration

# ---- the photo, with the seam band repaired ---------------------------------------------------------------------------
def load_photo():
    a = np.array(Image.open(os.path.join(ROOT, "ImageScape_360_032.jpg")).convert("RGB"))   # writable copy
    if A.no_repair: return a
    h, w, _ = a.shape; ppd = w / 360.0
    x0, x1 = int(round((78 / 360 + 0.5) * w)), int(round((106 / 360 + 0.5) * w))
    y0, y1 = int(round((0.5 - 16 / 180) * h)), int(round((0.5 + 12 / 180) * h))
    patch = a[y0:y1, x0:x1]
    fixed = DB.deblur_rgb(patch, sigma=1.6, iters=14, damp=0.12, strength=0.85)
    # feather: ramp over 5 degrees at the left/right edges and 3 degrees at top/bottom, so the repair fades into the original
    xs = np.arange(x1 - x0) / ppd; ys = np.arange(y1 - y0) / ppd
    wx = np.clip(np.minimum(xs / 5.0, (xs[-1] - xs) / 5.0), 0, 1); wy = np.clip(np.minimum(ys / 3.0, (ys[-1] - ys) / 3.0), 0, 1)
    wt = (wx[None, :] * wy[:, None]); wt = (wt * wt * (3 - 2 * wt))[..., None].astype(np.float32)
    a[y0:y1, x0:x1] = np.clip(patch.astype(np.float32) * (1 - wt) + fixed.astype(np.float32) * wt, 0, 255).astype(np.uint8)
    return a

rv.SRC["032"] = load_photo()

# ---- the camera ------------------------------------------------------------------------------------------------------------
DOOR_YAW = 182.5            # the middle of the double doors (yaw -177.5, unwrapped so the pan keeps turning the same way)
KEYS = [  # (t, yaw, pitch, horizontal fov)
    (0.0, 80.0, 1.0, 58.0),         # the gate sits on the right of the frame
    (1.6, 86.0, 0.5, 54.0),
    (3.0, 93.0, 0.0, 46.0),         # settled on the gate
    (3.5, 94.0, 0.0, 46.0),
    (5.0, 126.0, -0.5, 56.0),       # sweeping across the barn side and the picnic tables
    (6.4, 160.0, -2.0, 46.0),
    (8.0, DOOR_YAW, -2.8, 34.0),
    (10.0, DOOR_YAW, -2.5, 29.0),   # push in on the middle of the two doors
]
def interp(c): return rv.pchip([k[0] for k in KEYS], [k[c] for k in KEYS])
FY, FP, FF = interp(1), interp(2), interp(3)
def pose(t): return (float(FY(t)), float(FP(t)), float(FF(t)))

def frame_at(t):
    out = []
    for r in range(0, H, 60):
        rv.R0, rv.R1 = r, min(H, r + 60)
        out.append(rv.sample(rv.SRC["032"], rv.view_dirs(*pose(t))))
    img = np.concatenate(out, 0)
    f = rv.sm(0, 0.5, t) * rv.sm(0, 0.7, TOTAL - t)
    return np.clip(img * f + 0.5, 0, 255).astype(np.uint8)

def job(i): return frame_at(i / FPS).tobytes()

if __name__ == "__main__":
    if A.stills:
        for s in A.stills.split(","): Image.fromarray(frame_at(float(s))).save(A.out + "_%05.2f.png" % float(s))
        sys.exit()
    n = int(TOTAL * FPS)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS),
                           "-i", "-", "-vf", "unsharp=5:5:0.4:5:5:0.0", "-c:v", "libx264", "-preset", "medium", "-crf", "15",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", A.out], stdin=subprocess.PIPE)
    with Pool(A.workers) as pool:
        for k, raw in enumerate(pool.imap(job, range(n), chunksize=2)):
            ff.stdin.write(raw)
            if k % 60 == 0: print(k, "/", n, flush=True)
    ff.stdin.close(); ff.wait(); print("done")
