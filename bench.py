# dm_pipeline.py
# Auto-locate → rectify → decode Data Matrix from FULL images (no manual crop).
# ZXing-C++ first (fast), libdmtx fallback. Optional YOLOv8 detector.
#
# Usage:
#   python -u dm_pipeline.py IMG1 [IMG2 ...]
#   python -u dm_pipeline.py IMG --yolo-weights best.pt --time 10 --workers 8 --debug warps
#
# Deps: pip install zxing-cpp pylibdmtx opencv-python pillow
# Optional: pip install ultralytics  (only if you use --yolo-weights)

import sys, os, glob, time, math
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import zxingcpp
from pylibdmtx.pylibdmtx import decode as dmtx_decode

# ---------------- utilities ----------------

def imread_gray(path:str) -> np.ndarray:
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"Cannot open {path}")
    return g

def add_qz(img: np.ndarray, px:int=6, val:int=255) -> np.ndarray:
    return cv2.copyMakeBorder(img, px,px,px,px, cv2.BORDER_CONSTANT, value=val)

def order_quad(pts: np.ndarray) -> np.ndarray:
    """Return points ordered tl,tr,br,bl for perspective warp."""
    pts = np.array(pts, dtype=np.float32)
    idx = np.argsort(pts[:,1])      # by y
    top = pts[idx[:2]]; bot = pts[idx[2:]]
    tl, tr = top[np.argsort(top[:,0])]
    bl, br = bot[np.argsort(bot[:,0])]
    return np.array([tl,tr,br,bl], np.float32)

def warp_quad(gray: np.ndarray, quad: np.ndarray, out_size:int=168) -> np.ndarray:
    quad = order_quad(quad)
    dst = np.array([[0,0],[out_size-1,0],[out_size-1,out_size-1],[0,out_size-1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(gray, M, (out_size,out_size))

# ---------------- decoders ----------------

def decode_zxing(arr: np.ndarray):
    """ZXing-C++ first (fast). Works across wrapper versions."""
    try:
        res = zxingcpp.read_barcodes(arr, formats=zxingcpp.BarcodeFormat.DataMatrix)
    except TypeError:
        res = zxingcpp.read_barcodes(arr)
    return [r.text for r in res] if res else []

def decode_libdmtx(arr: np.ndarray):
    """Small, fast sweep for libdmtx."""
    for inv in (False, True):
        v = cv2.bitwise_not(arr) if inv else arr
        for rotk in (0,1):         # 0°, 90° (180/270 redundant for DM)
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

def decode_candidate(warped_square: np.ndarray):
    """Try raw + one adaptive view, ZXing first then libdmtx."""
    z = add_qz(warped_square, 6, 255)
    hit = decode_zxing(z)
    if hit: return hit
    thr = cv2.adaptiveThreshold(z,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    hit = decode_zxing(thr)
    if hit: return hit
    hit = decode_libdmtx(z)
    if hit: return hit
    return decode_libdmtx(thr)

# ---------------- classical locator (fast) ----------------

def locator_views(gray: np.ndarray):
    """A few complementary views to find square-ish candidates quickly."""
    views = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 3)
    # dot-peen: pop dots
    g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))
    # 1) Otsu
    _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    views.append(b1)
    # 2) Adaptive Gaussian
    b2 = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    views.append(b2)
    # 3) Edge magnitude
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.magnitude(gx, gy))
    _, b3 = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    views.append(b3)
    return views

def quad_candidates(bin_img: np.ndarray, size_px_range, aspect_tol=1.25, min_area=200, maxc=10):
    """Extract square-ish 4-pt candidates from a binary image."""
    quads = []
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    bw = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, k, iterations=1)
    contours,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    smin, smax = size_px_range
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area: continue
        rect = cv2.minAreaRect(cnt)
        (cx,cy),(w,h),ang = rect
        long = max(w,h); short = max(1.0,min(w,h))
        if long < 16 or not (smin <= long <= smax): continue
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

def find_quads(gray: np.ndarray, expected_frac: float | None = 0.12,
               topcenter_bias: bool=True, max_candidates:int=10):
    """Find candidate quads; bias to top-center band for your layout."""
    H,W = gray.shape
    roi = gray; xoff=yoff=0
    if topcenter_bias:
        x0,x1 = int(0.25*W), int(0.75*W)
        y0,y1 = int(0.08*H), int(0.50*H)
        roi = gray[y0:y1, x0:x1]; xoff,yoff = x0,y0

    if expected_frac and expected_frac > 0:
        exp = int(expected_frac * W)
        smin,smax = int(exp*0.6), int(exp*1.4)
    else:
        smin,smax = int(0.06*W), int(0.30*W)

    quads=[]
    for v in locator_views(roi):
        cand = quad_candidates(v, (smin,smax), maxc=max_candidates)
        for q in cand:
            q[:,0]+=xoff; q[:,1]+=yoff
            quads.append(q)
    return merge_by_center(quads, min_dist=14.0)[:max_candidates]

# ---------------- optional YOLOv8 detector ----------------

def detect_yolo_boxes_bgr(bgr: np.ndarray, weights_path: str, conf: float=0.25):
    """Return axis-aligned boxes [(x1,y1,x2,y2), ...] via YOLOv8 if available."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[YOLO] ultralytics not installed; skipping detector.")
        return []
    model = YOLO(weights_path)
    res = model.predict(bgr, conf=conf, verbose=False)
    boxes = []
    for r in res:
        if r.boxes is None: continue
        for b in r.boxes.xyxy.cpu().numpy().astype(int):
            x1,y1,x2,y2 = b[:4]
            boxes.append([x1,y1,x2,y2])
    return boxes

def boxes_to_quads(boxes):
    """Convert axis-aligned boxes to 4-pt quads (tl,tr,br,bl)."""
    quads=[]
    for x1,y1,x2,y2 in boxes:
        quads.append(np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], np.float32))
    return quads

# ---------------- pipeline ----------------

def decode_image(path: str, time_budget_s:int=8, workers:int=8,
                 expected_frac: float | None = 0.12, topcenter_bias: bool=True,
                 debug_dir: str | None=None, yolo_weights: str | None=None,
                 want_all: bool=False):
    gray = imread_gray(path); H,W = gray.shape
    print(f"[{os.path.basename(path)}] {W}x{H}")
    t0 = time.monotonic()
    results = []

    # 0) very quick whole-image ZXing try (helps on clean labels)
    hit = decode_zxing(add_qz(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)))
    if hit:
        print("  ✓ whole-x2 (ZXing) ->", hit)
        if not want_all: return hit
        results.extend(hit)

    # 1) candidates from classical locator
    quads = find_quads(gray, expected_frac=expected_frac, topcenter_bias=topcenter_bias, max_candidates=8)

    # 2) optional YOLO proposals (treated as quads from their boxes)
    if yolo_weights:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        yboxes = detect_yolo_boxes_bgr(bgr, yolo_weights, conf=0.25)
        quads += boxes_to_quads(yboxes)

    # dedupe, prefer larger
    quads = merge_by_center(sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.int32))), min_dist=14.0)

    print(f"  candidates: {len(quads)}")
    if debug_dir: os.makedirs(debug_dir, exist_ok=True)

    # 3) decode candidates in parallel
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i,q in enumerate(quads):
            warped = warp_quad(gray, q, out_size=168)
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(path)}_cand{i}_warp.png"), warped)
            futs.append(ex.submit(decode_candidate, warped))
        for f in as_completed(futs):
            out = f.result()
            if out:
                for s in out:
                    if s not in results:
                        results.append(s)
                if not want_all:
                    print("  ✓ candidate ->", out); return results
            if time.monotonic() - t0 > time_budget_s:
                print("  .. time budget hit")
                return results if results else None

    # 4) tiny fallback: scan left/right halves of the top-center band
    if not results:
        x0,y0,x1,y1 = (int(0.25*W), int(0.08*H), int(0.75*W), int(0.50*H)) if topcenter_bias else (0,0,W,H)
        band = gray[y0:y1, x0:x1]
        halves = [band[:, :band.shape[1]//2], band[:, band.shape[1]//2:]]
        for h in halves:
            roi = add_qz(cv2.resize(h, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
            out = decode_zxing(roi) or decode_libdmtx(roi)
            if out:
                print("  ✓ fallback-half ->", out)
                if not want_all: return out
                for s in out:
                    if s not in results: results.append(s)
            if time.monotonic() - t0 > time_budget_s:
                print("  .. time budget hit")
                return results if results else None

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
    ap = argparse.ArgumentParser(description="Auto-locate → rectify → decode Data Matrix (ZXing first, libdmtx fallback). No manual crop needed.")
    ap.add_argument("inputs", nargs="+", help="images or folders")
    ap.add_argument("--time", type=int, default=8, help="time budget per image (seconds)")
    ap.add_argument("--workers", type=int, default=8, help="threads for decoding candidates")
    ap.add_argument("--no-topcenter-bias", action="store_true", help="search whole frame (slower)")
    ap.add_argument("--expected-frac", type=float, default=0.12, help="symbol width as fraction of image width; set 0 to disable")
    ap.add_argument("--debug", help="folder to save warped candidates")
    ap.add_argument("--yolo-weights", help="optional YOLOv8 weights to propose ROIs")
    ap.add_argument("--all", action="store_true", help="return all reads (don’t stop at first)")
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
            debug_dir=args.debug,
            yolo_weights=args.yolo_weights,
            want_all=args.all
        )
