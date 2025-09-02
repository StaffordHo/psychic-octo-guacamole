# dm_autofind.py
# Auto-locate, rectify, and decode Data Matrix from full images (no manual crop).
# Fast defaults; flags let you trade speed vs robustness.
#
# Usage:
#   python -u dm_autofind.py IMG1 [IMG2 ...]
#   python -u dm_autofind.py --time 10 --workers 8 --debug warps 1.png 2.png
#
# Dependencies: pip install opencv-python pillow pylibdmtx numpy

import sys, os, glob, time, math
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pylibdmtx.pylibdmtx import decode as dmtx_decode

# ---------------- utilities ----------------

def imread_gray(path:str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Cannot open {path}")
    return g

def add_quiet_zone(img: np.ndarray, px:int=6, val:int=255) -> np.ndarray:
    return cv2.copyMakeBorder(img, px,px,px,px, cv2.BORDER_CONSTANT, value=val)

def order_quad(pts: np.ndarray) -> np.ndarray:
    """Return points as tl,tr,br,bl."""
    pts = np.array(pts, dtype=np.float32)
    idx = np.argsort(pts[:,1])      # by y
    top = pts[idx[:2]]
    bot = pts[idx[2:]]
    tl, tr = top[np.argsort(top[:,0])]
    bl, br = bot[np.argsort(bot[:,0])]
    return np.array([tl,tr,br,bl], np.float32)

def warp_quad(gray: np.ndarray, quad: np.ndarray, out_size:int=164) -> np.ndarray:
    quad = order_quad(quad)
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(gray, M, (out_size,out_size))

def decode_numpy(arr: np.ndarray):
    """
    Lean, fast libdmtx sweep:
      polarity (normal/invert) × rotation (0°,90°) × shrink {2,3} × gap {2,1}
    Early-exits on first hit. Returns list[str] or None.
    """
    for inv in (False, True):
        v = cv2.bitwise_not(arr) if inv else arr
        for rotk in (0,1):  # 0°, 90°
            r = np.rot90(v, rotk)
            for shrink in (2,3):
                for gap in (2,1):
                    try:
                        res = dmtx_decode(r, timeout=220, max_count=1, shrink=shrink, gap_size=gap)
                    except TypeError:
                        res = dmtx_decode(r, timeout=220, max_count=1, shrink=shrink)
                    if res:
                        return [res[0].data.decode("utf-8","ignore")]
    return None

# ---------------- preprocessing views for locator ----------------

def make_locator_views(gray: np.ndarray):
    """
    Build a few complementary binary/edge views to locate square-ish DM candidates.
    Keeps it light for speed.
    """
    views = []
    # CLAHE for local contrast (good on shiny metal), mild denoise, dot-peen pop
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 3)
    g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))

    # 1) Otsu binarization
    _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    views.append(b1)

    # 2) Adaptive Gaussian
    b2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    views.append(b2)

    # 3) Edge magnitude (helps when intensity varies)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    _, b3 = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    views.append(b3)

    return views

# ---------------- candidate locator ----------------

def quad_candidates(bin_img: np.ndarray, size_px_range, aspect_tol=1.25, min_area=200, max_candidates=10):
    """
    Extract square-ish 4-point candidates from a binary image using contours.
    Returns list of 4-point boxes (float32).
    """
    quads = []
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    bw = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    contours,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    smin, smax = size_px_range
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        (cx,cy),(w,h),ang = rect
        if min(w,h) < 12:
            continue
        # filter by expected size (longer side)
        long = max(w,h)
        if not (smin <= long <= smax):
            continue
        # filter by squareness
        ar = long / max(1.0, min(w,h))
        if ar > aspect_tol:
            continue
        # polygonal approximation to check "rect-like"
        eps = 0.03 * cv2.arcLength(cnt, True)
        poly = cv2.approxPolyDP(cnt, eps, True)
        if len(poly) < 4:
            continue
        box = cv2.boxPoints(rect).astype(np.float32)
        quads.append(box)

    # prefer larger, more likely squares
    quads.sort(key=lambda q: -cv2.contourArea(q.astype(np.int32)))
    return quads[:max_candidates]

def merge_by_center(quads, min_dist=12.0):
    """Merge near-duplicate quads by center distance."""
    keep = []
    centers = []
    for q in quads:
        c = q.mean(axis=0)
        if all(np.hypot(c[0]-cc[0], c[1]-cc[1]) > min_dist for cc in centers):
            keep.append(q)
            centers.append(c)
    return keep

def locate_quads(gray: np.ndarray,
                 bias_topcenter: bool = True,
                 expected_frac: float | None = 0.12,
                 max_candidates: int = 12):
    """
    Locate candidate DMs as quads. If expected_frac set (≈ symbol width as fraction of image width),
    constrain candidates to that size ±40% for speed/precision.
    """
    H, W = gray.shape
    roi = gray
    x_off = y_off = 0

    if bias_topcenter:
        # Your photos: codes live in the top center band; search there first
        x0, x1 = int(0.25*W), int(0.75*W)
        y0, y1 = int(0.08*H), int(0.50*H)
        roi = gray[y0:y1, x0:x1]
        x_off, y_off = x0, y0

    # expected symbol width in pixels (if known); else fall back to a broad band
    if expected_frac:
        exp_px = int(expected_frac * W)
        smin, smax = int(exp_px*0.6), int(exp_px*1.4)
    else:
        smin, smax = int(0.06*W), int(0.30*W)  # broad default

    quads = []
    for v in make_locator_views(roi):
        cand = quad_candidates(v, (smin, smax), max_candidates=max_candidates)
        # shift back to full image coords
        for q in cand:
            q[:,0] += x_off
            q[:,1] += y_off
        quads.extend(cand)

    quads = merge_by_center(quads, min_dist=14.0)
    return quads[:max_candidates]

# ---------------- per-candidate decode ----------------

def decode_candidate(gray: np.ndarray, quad: np.ndarray, debug_dir: str | None, idx:int):
    warped = warp_quad(gray, quad, out_size=164)      # perspective-rectified
    z = add_quiet_zone(warped, px=6, val=255)         # add white quiet zone

    # Try raw first (fast), then a light adaptive view
    out = decode_numpy(z)
    if out:
        return out

    thr = cv2.adaptiveThreshold(z, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    out = decode_numpy(thr)
    if out:
        return out

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"cand_{idx}_warp.png"), z)
        cv2.imwrite(os.path.join(debug_dir, f"cand_{idx}_thr.png"), thr)
    return None

# ---------------- whole image pipeline ----------------

def decode_image(path: str, time_budget_s:int=8, workers:int=6,
                 bias_topcenter: bool=True, expected_frac: float | None = 0.12,
                 debug_dir: str | None=None, want_all: bool=False):
    gray = imread_gray(path)
    H, W = gray.shape
    print(f"[{os.path.basename(path)}] {W}x{H}")

    t0 = time.monotonic()
    results = []

    # 0) quick whole-image attempt (very cheap)
    quick = decode_numpy(add_quiet_zone(cv2.resize(gray, None, fx=2, fy=2,
                                                   interpolation=cv2.INTER_NEAREST)))
    if quick:
        print("  ✓ whole-x2 ->", quick)
        if not want_all: return quick
        results.extend(quick)

    # 1) locate candidates
    quads = locate_quads(gray, bias_topcenter=bias_topcenter,
                         expected_frac=expected_frac, max_candidates=10)
    print(f"  candidates: {len(quads)}")

    # 2) decode candidates in parallel (fast)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(decode_candidate, gray, q, debug_dir, i) for i,q in enumerate(quads)]
        for f in as_completed(futs):
            out = f.result()
            if out:
                for s in out:
                    if s not in results:
                        results.append(s)
                if not want_all:
                    print("  ✓ candidate ->", out)
                    return results
            if time.monotonic() - t0 > time_budget_s:
                print("  .. time budget hit")
                return results if results else None

    # 3) if nothing yet, tiny fallback grid inside top-center (still fast)
    if not results:
        x0,y0,x1,y1 = (int(0.25*W), int(0.08*H), int(0.75*W), int(0.50*H)) if bias_topcenter else (0,0,W,H)
        band = gray[y0:y1, x0:x1]
        for s in (56, 64, 72, 80):
            for dx in (-20,0,20):
                for dy in (-20,0,20):
                    cx, cy = band.shape[1]//2 + dx, band.shape[0]//2 + dy
                    xL, xR = max(0, cx-s//2), min(band.shape[1], cx+s//2)
                    yT, yB = max(0, cy-s//2), min(band.shape[0], cy+s//2)
                    roi = band[yT:yB, xL:xR]
                    z = add_quiet_zone(cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
                    out = decode_numpy(z)
                    if out:
                        print("  ✓ fallback-grid ->", out)
                        if not want_all:
                            return out
                        for s in out:
                            if s not in results:
                                results.append(s)
                    if time.monotonic() - t0 > time_budget_s:
                        print("  .. time budget hit"); return results if results else None

    if results:
        print("  ✓ FINAL ->", results)
        return results
    else:
        print("  -> No codes decoded.")
        return None

# ---------------- CLI ----------------

def collect_inputs(args):
    out=[]
    for a in args:
        if os.path.isdir(a): out += glob.glob(os.path.join(a,"*.*"))
        else:                out += glob.glob(a)
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return [p for p in out if os.path.splitext(p)[1].lower() in exts]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Auto-locate and decode Data Matrix from full images (libdmtx).")
    ap.add_argument("inputs", nargs="+", help="images or folders")
    ap.add_argument("--time", type=int, default=8, help="time budget per image (seconds)")
    ap.add_argument("--workers", type=int, default=6, help="thread workers for candidates")
    ap.add_argument("--no-topcenter-bias", action="store_true", help="search full frame (slower) instead of top-center band")
    ap.add_argument("--expected-frac", type=float, default=0.12,
                    help="expected symbol width as fraction of image width (e.g., 0.12 ≈ 12%%). Set 0 to disable.")
    ap.add_argument("--debug", help="folder to dump warped candidates")
    ap.add_argument("--all", action="store_true", help="return all codes instead of stopping at first")
    args = ap.parse_args()

    imgs = collect_inputs(args.inputs)
    if not imgs:
        print("No images found."); sys.exit(1)

    for p in imgs:
        decode_image(
            p,
            time_budget_s=args.time,
            workers=args.workers,
            bias_topcenter=not args.no_topcenter_bias,
            expected_frac=(None if args.expected_frac and args.expected_frac <= 0 else args.expected_frac),
            debug_dir=args.debug,
            want_all=args.all
        )
