#!/usr/bin/env python3
# dmdecode_thresh.py
# Data Matrix decoding for dot-peen marks using libdmtx (pylibdmtx) + threshold sweeps.

from pylibdmtx.pylibdmtx import decode as dmtx_decode
from PIL import Image
import cv2, numpy as np, os, sys, glob, argparse
from datetime import datetime

# -------------------- utils --------------------

def pil(a: np.ndarray) -> Image.Image:
    return Image.fromarray(a)

def log(s: str, verbose: bool):
    if verbose:
        print(s, flush=True)

def uniq(seq):
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def save_if(path, img, dump_dir):
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        cv2.imwrite(os.path.join(dump_dir, path), img)

# ---------------- threshold + decode ----------------

def threshold_variants(gray: np.ndarray):
    """Yield (name, image) thresholded variants (plus equalized 'raw')."""
    eq = cv2.equalizeHist(gray)
    yield ("raw_eq", eq)

    # Otsu
    _, otsu = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield ("otsu", otsu)

    # Adaptive (Gaussian + Mean), sweep some block sizes & C offsets
    for block in (21, 31, 41):
        for C in (3, 5, 7):
            thr_g = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, block, C)
            thr_m = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, block, C)
            # light morphology to connect dots
            k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            thr_g = cv2.morphologyEx(cv2.morphologyEx(thr_g, cv2.MORPH_OPEN, k3),
                                     cv2.MORPH_CLOSE, k3)
            thr_m = cv2.morphologyEx(cv2.morphologyEx(thr_m, cv2.MORPH_OPEN, k3),
                                     cv2.MORPH_CLOSE, k3)
            yield (f"gauss_b{block}_C{C}", thr_g)
            yield (f"mean_b{block}_C{C}", thr_m)

def try_decode_variant(img: np.ndarray, verbose=False, timeout=1200, max_count=8):
    """Try both polarities and 0/90/180/270 rotations for one variant."""
    outs = []
    for inv in (False, True):
        v = cv2.bitwise_not(img) if inv else img
        for k in range(4):
            rot = np.rot90(v, k)
            res = dmtx_decode(pil(rot), timeout=timeout, max_count=max_count)
            if res:
                outs.extend(r.data.decode("utf-8", "ignore") for r in res)
    return uniq(outs)

def decode_roi(tag, roi_gray, dump_dir=None, verbose=False):
    """Upscale + threshold sweep + decode. Returns list of decoded strings."""
    results = []
    # Mild denoise & dot enhancement
    base = cv2.equalizeHist(roi_gray)
    base = cv2.medianBlur(base, 3)
    base = cv2.morphologyEx(base, cv2.MORPH_TOPHAT,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9,9)))

    for scale in (2, 3, 4):
        up = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for name, var in threshold_variants(up):
            save_if(f"{tag}_x{scale}_{name}.png", var, dump_dir)
            log(f"  [{tag}] x{scale} {name} …", verbose)
            outs = try_decode_variant(var, verbose=verbose)
            if outs:
                return outs
    return results

# -------------- ROI discovery --------------

def find_plate_bbox(gray: np.ndarray):
    """Find bright plate region by thresholding and largest contour."""
    h, w = gray.shape
    eq = cv2.equalizeHist(gray)
    _, bw = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, ww, hh = cv2.boundingRect(c)
    # sanity clamp
    if ww * hh < 0.05 * w * h:
        return None
    return (x, y, x + ww, y + hh)

def default_plate_splits(gray: np.ndarray):
    """Split top-center region into left/right (works for your samples)."""
    H, W = gray.shape
    x0, x1 = int(0.20*W), int(0.80*W)
    y0, y1 = int(0.05*H), int(0.55*H)
    plate = (x0, y0, x1, y1)
    mid = (x0 + x1) // 2
    pad_x = int(0.04 * (x1 - x0))
    pad_y = int(0.05 * (y1 - y0))
    left  = (x0 + pad_x, y0 + pad_y, mid - pad_x, y1 - pad_y)
    right = (mid + pad_x, y0 + pad_y, x1 - pad_x, y1 - pad_y)
    return plate, left, right

def get_rois(gray: np.ndarray):
    """Return dict of named ROIs: 'left', 'right', 'plate'."""
    bbox = find_plate_bbox(gray)
    if bbox:
        x0, y0, x1, y1 = bbox
        W, H = x1 - x0, y1 - y0
        yt, yb = y0, y0 + int(0.55 * H)
        xm = x0 + W // 2
        pad_x = int(0.04 * W)
        pad_y = int(0.05 * H)
        rois = {
            "plate": (x0, y0, x1, y1),
            "left":  (x0 + pad_x, yt + pad_y, xm - pad_x, yb - pad_y),
            "right": (xm + pad_x, yt + pad_y, x1 - pad_x, yb - pad_y),
        }
    else:
        plate, left, right = default_plate_splits(gray)
        rois = {"plate": plate, "left": left, "right": right}
    return rois

# -------------- manual crop (fixed UX) --------------

def manual_rois(image_bgr):
    """
    Repeated single-ROI selection to avoid confusion with selectROIs.
    - Draw a rectangle.
    - Press ENTER/SPACE to accept.
    - Press ESC to finish (no more ROIs).
    """
    rois = []
    clone = image_bgr.copy()
    while True:
        box = cv2.selectROI("Draw ROI (Enter=accept, Esc=finish)", clone, showCrosshair=True, fromCenter=False)
        # box is (x,y,w,h); if Esc pressed immediately, it's (0,0,0,0)
        if box[2] == 0 or box[3] == 0:
            cv2.destroyWindow("Draw ROI (Enter=accept, Esc=finish)")
            break
        rois.append(box)
        # Draw the accepted ROI onto the clone for visual confirmation
        x, y, w, h = map(int, box)
        cv2.rectangle(clone, (x, y), (x+w, y+h), (0,255,0), 2)
    return rois

# -------------- main per-image pipeline --------------

def process_image(path, dump=None, manual=False, verbose=False):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[WARN] cannot open {path}")
        return []

    h, w = gray.shape
    print(f"[{os.path.basename(path)}] loaded {w}x{h}")

    crops = []

    if manual:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        print("Draw one ROI per code. Press ENTER to accept each. Press ESC when done.")
        boxes = manual_rois(bgr)
        for i, (x, y, ww, hh) in enumerate(boxes, 1):
            c = gray[y:y+hh, x:x+ww]
            c = cv2.copyMakeBorder(c, 8,8,8,8, cv2.BORDER_REPLICATE)  # tiny quiet zone
            crops.append((f"roi{i}", c))
    else:
        rois = get_rois(gray)
        for name, (x0,y0,x1,y1) in rois.items():
            c = gray[y0:y1, x0:x1]
            crops.append((name, c))

    # Save crops if dumping
    if dump:
        base = os.path.splitext(os.path.basename(path))[0]
        for tag, c in crops:
            save_if(f"{base}_{tag}.png", c, dump)

    # Try left/right first, then plate, then whole
    order = ["left", "right", "plate"]
    crops_dict = {tag: c for tag, c in crops}
    results = []

    for tag in order:
        if tag in crops_dict:
            out = decode_roi(tag, crops_dict[tag], dump_dir=dump, verbose=verbose)
            if out:
                print(f"  -> {tag} decoded: {out}")
                results += out

    # whole image last
    out = decode_roi("whole", gray, dump_dir=dump, verbose=verbose)
    if out:
        print(f"  -> whole decoded: {out}")
        results += out

    final = uniq(results)
    if final:
        print(f"[{os.path.basename(path)}] FINAL -> {final}")
    else:
        print(f"[{os.path.basename(path)}] No codes decoded.")
    return final

# ---------------- CLI ----------------

def collect_images(inputs):
    files=[]
    for a in inputs:
        if os.path.isdir(a): files += glob.glob(os.path.join(a, "*.*"))
        else:                files += glob.glob(a)
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return [f for f in files if os.path.splitext(f)[1].lower() in exts]

def main():
    ap = argparse.ArgumentParser(description="Decode Data Matrix (dot-peen) with libdmtx + thresholding.")
    ap.add_argument("inputs", nargs="+", help="image files, folders, or globs")
    ap.add_argument("--dump", help="directory to save crops & threshold variants")
    ap.add_argument("--manual", action="store_true", help="manually draw ROIs (Enter=accept, Esc=finish)")
    ap.add_argument("--verbose", action="store_true", help="print progress for each variant")
    args = ap.parse_args()

    imgs = collect_images(args.inputs)
    if not imgs:
        print("No images found.")
        sys.exit(1)

    for p in imgs:
        process_image(p, dump=args.dump, manual=args.manual, verbose=args.verbose)

if __name__ == "__main__":
    main()
