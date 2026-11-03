"""Turn the full-resolution equirectangular photos into web versions.

For every ImageScape_360_NNN.jpg in the repo root this writes
  docs/img/NNN.jpg          8192 x 4096, progressive JPEG (sharp at full-screen size)
  docs/img/NNN_preview.jpg  2048 x 1024 (loads instantly, shown until the big one arrives)
Run:  python3 tools/build_scenes.py [workers]
"""
import glob, os, re, sys
from multiprocessing import Pool
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")

def build(path):
    num = re.search(r"_(\d+)\.jpg$", path).group(1)
    big, small = os.path.join(OUT, num + ".jpg"), os.path.join(OUT, num + "_preview.jpg")
    if os.path.exists(big) and os.path.exists(small):
        return num, "skip"
    im = Image.open(path).convert("RGB")
    im.resize((8192, 4096), Image.LANCZOS).save(big, quality=84, optimize=True, progressive=True, subsampling="4:2:0")
    im.resize((2048, 1024), Image.LANCZOS).save(small, quality=82, optimize=True, progressive=True)
    return num, "%.1f MB" % (os.path.getsize(big) / 1048576)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(ROOT, "ImageScape_360_*.jpg")))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 3) as p:
        for num, r in p.imap_unordered(build, files):
            print(num, r, flush=True)
