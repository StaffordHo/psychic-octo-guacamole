# dm_autofind_ensemble.py
# Full-image auto-locate + perspective warp + decode (ZXing first, libdmtx fallback)
# pip install zxing-cpp pylibdmtx opencv-python pillow

import sys, os, glob, time
import cv2, numpy as np
import zxingcpp
from pylibdmtx.pylibdmtx import decode as dmtx_decode
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- small utils ----
def imread_gray(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None: raise RuntimeError(f"Cannot open {p}")
    return g

def add_qz(img, px=6, val=255):
    return cv2.copyMakeBorder(img, px,px,px,px, cv2.BORDER_CONSTANT, value=val)

def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    idx = np.argsort(pts[:,1])  # by y
    top, bot = pts[idx[:2]], pts[idx[2:]]
    tl, tr = top[np.argsort(top[:,0])]
    bl, br = bot[np.argsort(bot[:,0])]
    return np.array([tl,tr,br,bl], np.float32)

def warp_quad(gray, quad, out=168):
    M = cv2.getPerspectiveTransform(order_quad(quad),
                                    np.array([[0,0],[out-1,0],[out-1,out-1],[0,out-1]],np.float32))
    return cv2.warpPerspective(gray, M, (out,out))

# ---- decoders ----
def decode_zxing(arr):
    # Works across wrapper versions: prefer format hint, fall back if unsupported
    try:
        res = zxingcpp.read_barcodes(arr, formats=zxingcpp.BarcodeFormat.DataMatrix)
    except TypeError:
        res = zxingcpp.read_barcodes(arr)
    return [r.text for r in res] if res else []

def decode_libdmtx(arr):
    for inv in (False, True):
        v = cv2.bitwise_not(arr) if inv else arr
        for rotk in (0,1):  # 0°, 90°
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

def decode_candidate(warped):
    z = add_qz(warped, 6, 255)
    hit = decode_zxing(z)
    if hit: return hit
    # one adaptive view helps dot-peen
    thr = cv2.adaptiveThreshold(z,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    hit = decode_zxing(thr)
    if hit: return hit
    hit = decode_libdmtx(z)
    if hit: return hit
    return decode_libdmtx(thr)

# ---- locator (fast, tuned for your pictures) ----
def locator_views(gray):
    # CLAHE + mild denoise + tophat to pop dots, then 2 binarizations + edge view
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g = clahe.apply(gray)
    g = cv2.medianBlur(g, 3)
    g = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))
    _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    b2 = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1,0,3); gy = cv2.Sobel(g, cv2.CV_32F, 0,1,3)
    mag = cv2.convertScaleAbs(cv2.magnitude(gx,gy))
    _, b3 = cv2.threshold(mag, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return [b1,b2,b3]

def find_quads(gray, expected_frac=0.12, topcenter_bias=True, maxc=10):
    H,W = gray.shape
    roi = gray; xoff=yoff=0
    # Bias search to the top-center band (your layout); disable with flag if needed
    if topcenter_bias:
        x0,x1 = int(0.25*W), int(0.75*W)
        y0,y1 = int(0.08*H), int(0.50*H)
        roi = gray[y0:y1, x0:x1]; xoff,yoff = x0,y0
    # expected symbol width in px (≈12% of image width on your samples), allow ±40%
    exp = int(expected_frac*W) if expected_frac else None
    smin,smax = (int(exp*0.6), int(exp*1.4)) if exp else (int(0.06*W), int(0.30*W))

    quads=[]
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    for v in locator_views(roi):
        bw = cv2.morphologyEx(v, cv2.MORPH_CLOSE, k, 1)
        cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            rect = cv2.minAreaRect(c)
            (cx,cy),(w,h),ang = rect
            long = max(w,h); short = max(1.0,min(w,h))
            if long < 16 or not (smin <= long <= smax): continue
            if long/short > 1.25: continue  # near-square only
            box = cv2.boxPoints(rect).astype(np.float32)
            box[:,0]+=xoff; box[:,1]+=yoff
            quads.append(box)

    # de-dup by center distance, prefer larger first
    keep=[]; centers=[]
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.int32))):
        c = q.mean(axis=0)
        if all(np.hypot(c[0]-cc[0], c[1]-cc[1])>14 for cc in centers):
            keep.append(q); centers.append(c)
        if len(keep)>=maxc: break
    return keep

# ---- full-image pipeline ----
def decode_full(path, time_budget_s=8, workers=8, expected_frac=0.12, topcenter_bias=True, debug_dir=None):
    g = imread_gray(path); H,W = g.shape
    print(f"[{os.path.basename(path)}] {W}x{H}")
    t0 = time.monotonic()

    # very quick whole-image ZXing attempt (helps on clean labels)
    hit = decode_zxing(add_qz(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)))
    if hit:
        print("  ✓ whole-x2 (ZXing) ->", hit); return

    quads = find_quads(g, expected_frac=expected_frac, topcenter_bias=topcenter_bias, maxc=8)
    print(f"  candidates: {len(quads)}")

    if debug_dir: os.makedirs(debug_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i,q in enumerate(quads):
            warped = warp_quad(g, q, out=168)
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f"{os.path.basename(path)}_cand{i}_warp.png"), warped)
            futs.append(ex.submit(decode_candidate, warped))
        for f in as_completed(futs):
            out = f.result()
            if out:
                print("  ✓ candidate ->", out); return
            if time.monotonic()-t0 > time_budget_s:
                print("  .. time budget hit"); return

    # tiny fallback: left/right halves of the band (still fast)
    x0,y0,x1,y1 = (int(0.25*W), int(0.08*H), int(0.75*W), int(0.50*H)) if topcenter_bias else (0,0,W,H)
    band = g[y0:y1, x0:x1]
    halves = [band[:, :band.shape[1]//2], band[:, band.shape[1]//2:]]
    for h in halves:
        roi = add_qz(cv2.resize(h, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
        out = decode_zxing(roi) or decode_libdmtx(roi)
        if out:
            print("  ✓ fallback-half ->", out); return
        if time.monotonic()-t0 > time_budget_s:
            print("  .. time budget hit"); return

    print("  -> No codes decoded.")

# ---- CLI ----
def collect(args):
    paths=[]
    for a in args:
        if os.path.isdir(a): paths += glob.glob(os.path.join(a,"*.*"))
        else:                paths += glob.glob(a)
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return [p for p in paths if os.path.splitext(p)[1].lower() in exts]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Auto-find Data Matrix (ZXing first, libdmtx fallback). No manual crop needed.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--time", type=int, default=8, help="time budget per image (s)")
    ap.add_argument("--workers", type=int, default=8, help="candidate threads")
    ap.add_argument("--no-topcenter-bias", action="store_true", help="search whole frame (slower)")
    ap.add_argument("--expected-frac", type=float, default=0.12, help="symbol width as fraction of image width; set 0 to disable")
    ap.add_argument("--debug", help="folder to save warped candidates")
    args = ap.parse_args()

    for p in collect(args.inputs):
        decode_full(
            p,
            time_budget_s=args.time,
            workers=args.workers,
            expected_frac=(None if args.expected_frac<=0 else args.expected_frac),
            topcenter_bias=(not args.no_topcenter_bias),
            debug_dir=args.debug
        )
