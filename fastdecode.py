# decode_fast.py
from pylibdmtx.pylibdmtx import decode as dmtx_decode
from PIL import Image
import cv2, numpy as np, os, sys, glob

def pil(a): return Image.fromarray(a)

def log(msg): print(msg, flush=True)

def rotations_and_polarities(img_bin, timeout=400):
    """Try 0/90/180/270 and both polarities with a short timeout."""
    outs = []
    for inv in (False, True):
        v = cv2.bitwise_not(img_bin) if inv else img_bin
        for k in range(4):
            rot = np.rot90(v, k)
            res = dmtx_decode(pil(rot), timeout=timeout, max_count=4)
            if res:
                outs.extend(r.data.decode("utf-8","ignore") for r in res)
    return outs

def quick_variants(gray):
    """Small, fast set of variants (raw + a couple binarizations)."""
    eq = cv2.equalizeHist(gray)
    yield ("raw", eq)

    # Adaptive Gaussian (one decent setting)
    thr = cv2.adaptiveThreshold(eq, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 5)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k3)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k3)
    yield ("adaptive31_C5", thr)

    # Otsu fallback
    _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    yield ("otsu", otsu)

def forced_top_center_rois(gray):
    """Your images: two codes at the top–center; crop that region and split L/R."""
    H, W = gray.shape
    x0, x1 = int(0.20*W), int(0.80*W)
    y0, y1 = int(0.05*H), int(0.55*H)
    plate = gray[y0:y1, x0:x1]
    mid = (x1 - x0) // 2
    left  = plate[:, :mid]
    right = plate[:, mid:]
    pad = 8
    left  = cv2.copyMakeBorder(left,  pad,pad,pad,pad,  cv2.BORDER_REPLICATE)
    right = cv2.copyMakeBorder(right, pad,pad,pad,pad, cv2.BORDER_REPLICATE)
    return [("plate", plate), ("left", left), ("right", right)]

def decode_roi(tag, roi):
    """Upscale quickly and try a few variants."""
    for scale in (2, 3):  # keep small for speed
        up = cv2.resize(roi, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
        for name, var in quick_variants(up):
            log(f"  [{tag}] scale{scale} {name} …")
            outs = rotations_and_polarities(var, timeout=400)
            if outs:
                return outs
    return []

def decode_one(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        log(f"[{path}] cannot open image"); return []

    H, W = gray.shape
    log(f"[{os.path.basename(path)}] loaded {W}x{H}")

    # Try whole image first (quick)
    out = decode_roi("whole", gray)
    if out:
        log(f"[{os.path.basename(path)}] -> {out}")
        return out

    # Then the top-center plate & left/right halves (your layout)
    for tag, roi in forced_top_center_rois(gray):
        out = decode_roi(tag, roi)
        if out:
            log(f"[{os.path.basename(path)}] {tag} -> {out}")
            return out

    # Last resort: simple left/right split of the full frame
    mid = W // 2
    for tag, roi in (("Lhalf", gray[:, :mid]), ("Rhalf", gray[:, mid:])):
        out = decode_roi(tag, roi)
        if out:
            log(f"[{os.path.basename(path)}] {tag} -> {out}")
            return out

    log(f"[{os.path.basename(path)}] No codes decoded.")
    return []

def collect_images(args):
    files=[]
    for a in args:
        if os.path.isdir(a): files += glob.glob(os.path.join(a, "*.*"))
        else:                files += glob.glob(a)
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return [f for f in files if os.path.splitext(f)[1].lower() in exts]

if __name__ == "__main__":
    imgs = collect_images(sys.argv[1:])
    if not imgs:
        print("usage: python -u decode_fast.py <image(s) or folder or *.png>", flush=True)
        sys.exit(1)
    for p in imgs:
        decode_one(p)
