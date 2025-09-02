# dm_decode_fast.py
import sys, os, time, glob
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pylibdmtx.pylibdmtx import decode as dmtx_decode

def variants(gray):
    # small white quiet zone
    g = cv2.copyMakeBorder(gray, 6,6,6,6, cv2.BORDER_CONSTANT, value=255)
    # keep it lean: raw + Otsu at x3/x4
    for scale in (3, 4):
        up = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        yield f"x{scale}_raw", up
        _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        yield f"x{scale}_otsu", otsu

def attempt(arr, inv, rotk, shrink, gap, timeout=200):
    v = cv2.bitwise_not(arr) if inv else arr
    r = np.rot90(v, rotk)                 # 0 or 1 (=> 0°/90°)
    # pass NumPy directly; keep param set tiny and fast
    try:
        res = dmtx_decode(r, timeout=timeout, max_count=1, shrink=shrink, gap_size=gap)
    except TypeError:  # older wheels don't support gap_size
        res = dmtx_decode(r, timeout=timeout, max_count=1, shrink=shrink)
    if res:
        return [res[0].data.decode("utf-8", "ignore")]
    return None

def decode_fast(path, time_budget_s=8, workers=6):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[ERR] cannot open {path}")
        return
    print(f"[{os.path.basename(path)}] {gray.shape[1]}x{gray.shape[0]}")

    start = time.monotonic()
    # progressive stages: cheap first, then slightly broader
    stages = [
        dict(inv=[False], rotk=[0,1], shrink=[2], gap=[2], timeout=150),
        dict(inv=[True],  rotk=[0,1], shrink=[2], gap=[2], timeout=150),
        dict(inv=[False,True], rotk=[0,1], shrink=[3,2], gap=[1,2], timeout=250),
    ]

    for name, view in variants(gray):
        for s, spec in enumerate(stages, 1):
            jobs = []
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for inv in spec["inv"]:
                    for rotk in spec["rotk"]:
                        for shrink in spec["shrink"]:
                            for gap in spec["gap"]:
                                jobs.append(ex.submit(
                                    attempt, view, inv, rotk, shrink, gap, spec["timeout"]
                                ))
                for fut in as_completed(jobs):
                    out = fut.result()
                    if out:
                        print(f"  ✓ {name} -> {out}  (stage{s})")
                        return
                    if time.monotonic() - start > time_budget_s:
                        print("  .. time budget hit")
                        return
        print(f"  .. {name} (no hit)")

    print("  -> No codes decoded.")

def collect(args):
    out=[]
    for a in args:
        if os.path.isdir(a): out += glob.glob(os.path.join(a,"*.*"))
        else:                out += glob.glob(a)
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return [p for p in out if os.path.splitext(p)[1].lower() in exts]

if __name__ == "__main__":
    imgs = collect(sys.argv[1:])
    if not imgs:
        print("usage: python -u dm_decode_fast.py <crop1.png> [crop2.png ...]")
        sys.exit(1)
    for p in imgs:
        decode_fast(p, time_budget_s=8, workers=6)

