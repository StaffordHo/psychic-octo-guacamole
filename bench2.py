import sys, os, glob, time
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import zxingcpp
from pylibdmtx.pylibdmtx import decode as dmtx_decode

# ---------------- utils ----------------
def imread_gray(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None: raise RuntimeError(f"Cannot open {p}")
    return g

def add_qz(img, px=6, val=255):
    return cv2.copyMakeBorder(img, px,px,px,px, cv2.BORDER_CONSTANT, value=val)

def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    idx = np.argsort(pts[:,1]); top, bot = pts[idx[:2]], pts[idx[2:]]
    tl, tr = top[np.argsort(top[:,0])]; bl, br = bot[np.argsort(bot[:,0])]
    return np.array([tl,tr,br,bl], np.float32)

def warp_quad(gray, quad, out=168):
    M = cv2.getPerspectiveTransform(order_quad(quad),
                                    np.array([[0,0],[out-1,0],[out-1,out-1],[0,out-1]],np.float32))
    return cv2.warpPerspective(gray, M, (out,out))

# ---------------- decoders ----------------
def decode_zxing(arr):
    try:
        res = zxingcpp.read_barcodes(arr, formats=zxingcpp.BarcodeFormat.DataMatrix)
    except TypeError:
        res = zxingcpp.read_barcodes(arr)
    return [r.text for r in res] if res else []

def decode_libdmtx(arr):
    for inv in (False, True):
        v = cv2.bitwise_not(arr) if inv else arr
        for rotk in (0,1):  # 0°/90°
            r = np.rot90(v, rotk)
            for shrink in (2,3):
                for gap in (2,1):
                    try:
                        out = dmtx_decode(r, timeout=220, max_count=1, shrink=shrink, gap_size=gap)
                    except TypeError:
                        out = dmtx_decode(r, timeout=220, max_count=1, shrink=shrink)
                    if out:
                        return [out[0].data.decode("utf-8","ignore")]
    return []

def decode_candidate(warped_square):
    z = add_qz(warped_square, 6, 255)
    hit = decode_zxing(z)
    if hit: return hit
    thr = cv2.adaptiveThreshold(z,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    hit = decode_zxing(thr)
    if hit: return hit
    hit = decode_libdmtx(z)
    if hit: return hit
    return decode_libdmtx(thr)

# ---------------- classical locator ----------------
def locator_views(gray):
    views=[]
    clahe = cv2.createCLAHE(2.0, (8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 3)
    g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))
    _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    views.append(b1)
    b2 = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    views.append(b2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1,0,3); gy = cv2.Sobel(g, cv2.CV_32F, 0,1,3)
    mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    _, b3 = cv2.threshold(mag, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    views.append(b3)
    return views

def quad_candidates(bin_img, size_px_range, aspect_tol=1.35, min_area=150, maxc=10):
    quads=[]
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    bw = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, 1)
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    smin, smax = size_px_range
    for c in cnts:
        if cv2.contourArea(c) < min_area: continue
        rect = cv2.minAreaRect(c)
        (cx,cy),(w,h),ang = rect
        long, short = max(w,h), max(1.0, min(w,h))
        if long < 14 or not (smin <= long <= smax): continue
        if long/short > aspect_tol: continue
        box = cv2.boxPoints(rect).astype(np.float32)
        quads.append(box)
    quads.sort(key=lambda q: -cv2.contourArea(q.astype(np.int32)))
    return quads[:maxc]

def merge_by_center(quads, min_dist=14.0):
    keep, centers = [], []
    for q in quads:
        c = q.mean(axis=0)
        if all(np.hypot(c[0]-cc[0], c[1]-cc[1]) > min_dist for cc in centers):
            keep.append(q); centers.append(c)
    return keep

def find_quads(gray, expected_frac=0.12, topcenter_bias=True, maxc=10):
    H,W = gray.shape
    roi = gray; xoff=yoff=0
    if topcenter_bias:
        x0,x1 = int(0.20*W), int(0.80*W)   # slightly wider band
        y0,y1 = int(0.06*H), int(0.55*H)
        roi = gray[y0:y1, x0:x1]; xoff,yoff = x0,y0

    if expected_frac and expected_frac > 0:
        exp = int(expected_frac * W)
        smin,smax = int(exp*0.5), int(exp*1.6)  # widened vs previous
    else:
        smin,smax = int(0.05*W), int(0.35*W)

    quads=[]
    for v in locator_views(roi):
        cand = quad_candidates(v, (smin,smax), maxc=maxc)
        for q in cand:
            q[:,0]+=xoff; q[:,1]+=yoff
            quads.append(q)
    return merge_by_center(quads, 14.0)[:maxc]

# ---------------- brute-force tile scan (fallback) ----------------
def tile_variants(gray):
    """Generate threshold variants of a view."""
    yield "raw", gray
    _, o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    yield "otsu", o
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    thr = cv2.morphologyEx(cv2.morphologyEx(thr, cv2.MORPH_OPEN, k3), cv2.MORPH_CLOSE, k3)
    yield "gauss31_C5", thr

def brute_force_scan(gray, time_budget_s=6, workers=8, topcenter_bias=True, expected_frac=None):
    """Slide square tiles over the image at a few scales and try ZXing/libdmtx directly."""
    H,W = gray.shape
    t0 = time.monotonic()
    # region of interest
    if topcenter_bias:
        x0,x1 = int(0.15*W), int(0.85*W)
        y0,y1 = int(0.05*H), int(0.60*H)
        base = gray[y0:y1, x0:x1]
    else:
        x0,y0=0,0; x1,y1=W,H; base = gray

    # expected size range → tile sizes (px)
    if expected_frac and expected_frac > 0:
        exp = int(expected_frac * W)
        sizes = [int(s) for s in (0.6*exp, 0.85*exp, 1.1*exp, 1.35*exp) if s>=28]
    else:
        sizes = [int(W*f) for f in (0.08, 0.11, 0.14, 0.18) if int(W*f)>=28]

    # build tiles (center-dense stride)
    jobs=[]
    for S in sizes:
        stride = max(12, int(S*0.45))
        for y in range(0, max(1, base.shape[0]-S+1), stride):
            for x in range(0, max(1, base.shape[1]-S+1), stride):
                tile = base[y:y+S, x:x+S]
                if tile.shape[0] < 24 or tile.shape[1] < 24: continue
                # upscale 3x for small modules
                up = cv2.resize(tile, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
                for name, v in tile_variants(up):
                    jobs.append(v)
        if time.monotonic()-t0 > time_budget_s*0.25:
            break  # don't explode tiles on huge images

    # decode tiles in parallel, early exit
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(lambda a: (decode_zxing(a) or decode_libdmtx(a)), v) for v in jobs]
        for f in as_completed(futs):
            res = f.result()
            if res:
                return res
            if time.monotonic()-t0 > time_budget_s:
                return None
    return None

# ---------------- main pipeline ----------------
def decode_image(path, time_budget_s=8, workers=8, expected_frac=0.12, topcenter_bias=True, debug_dir=None):
    gray = imread_gray(path); H,W = gray.shape
    print(f"[{os.path.basename(path)}] {W}x{H}")
    t0 = time.monotonic()

    # Already-tight crop? Go straight to direct attempts.
    if max(H,W) <= 220:
        z = add_qz(cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        hit = decode_zxing(z) or decode_libdmtx(z)
        if hit: print("  ✓ crop-direct ->", hit); return
        # try threshold variants
        for _, v in tile_variants(z):
            hit = decode_zxing(v) or decode_libdmtx(v)
            if hit: print("  ✓ crop-direct thr ->", hit); return
        print("  -> No codes decoded."); return

    # quick whole-image attempt (fast)
    quick = decode_zxing(add_qz(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)))
    if quick:
        print("  ✓ whole-x2 (ZXing) ->", quick); return

    # locator
    quads = find_quads(gray, expected_frac=expected_frac, topcenter_bias=topcenter_bias, maxc=10)
    print(f"  candidates: {len(quads)}")

    if quads:
        if debug_dir: os.makedirs(debug_dir, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs=[]
            for i,q in enumerate(quads):
                warped = warp_quad(gray, q, out=168)
                if debug_dir:
                    cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(path)}_cand{i}_warp.png"), warped)
                futs.append(ex.submit(decode_candidate, warped))
            for f in as_completed(futs):
                out = f.result()
                if out:
                    print("  ✓ candidate ->", out); return
                if time.monotonic()-t0 > time_budget_s:
                    print("  .. time budget hit"); return

    # brute-force fallback when locator misses
    bf = brute_force_scan(gray, time_budget_s=max(3, int(time_budget_s*0.8)),
                          workers=workers, topcenter_bias=topcenter_bias,
                          expected_frac=expected_frac)
    if bf:
        print("  ✓ brute-force ->", bf); return

    print("  -> No codes decoded.")

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
    ap = argparse.ArgumentParser(description="Auto-locate → rectify → decode DM. Locator + brute-force fallback. No crop needed.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--time", type=int, default=8, help="time budget per image (s)")
    ap.add_argument("--workers", type=int, default=8, help="threads for candidates/tiles")
    ap.add_argument("--no-topcenter-bias", action="store_true", help="search full frame (slower)")
    ap.add_argument("--expected-frac", type=float, default=0.12, help="symbol width as fraction of image width; set 0 to disable")
    ap.add_argument("--debug", help="folder to save warped candidates")
    args = ap.parse_args()

    imgs = collect_inputs(args.inputs)
    if not imgs:
        print("No images found."); sys.exit(1)

    for p in imgs:
        decode_image(
            p,
            time_budget_s=args.time,
            workers=args.workers,
            expected_frac=(None if args.expected_frac<=0 else args.expected_frac),
            topcenter_bias=(not args.no_topcenter_bias),
            debug_dir=args.debug
        )
