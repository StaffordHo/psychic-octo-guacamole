# dm_manual_ultra.py
# Draw one or more ROIs, then libdmtx decodes them with tiny, safe sweeps.
import sys, os, cv2, numpy as np
from PIL import Image
from pylibdmtx.pylibdmtx import decode as dmtx_decode

def pil(a):  # ensure contiguous uint8
    return Image.fromarray(np.ascontiguousarray(a.astype("uint8")))

def decode_direct(arr):
    """No filtering. Try both polarities, 4 rotations, small shrink sweep, and a couple scales."""
    results = []
    # add a *white* quiet zone (not replicated pixels)
    arr = cv2.copyMakeBorder(arr, 6,6,6,6, cv2.BORDER_CONSTANT, value=255)
    for scale in (1, 2, 3):  # nearest-neighbor upscale for tiny modules
        up = arr if scale == 1 else cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        for inv in (False, True):
            v = cv2.bitwise_not(up) if inv else up
            for rotk in (0,1,2,3):  # 0/90/180/270
                r = np.rot90(v, rotk)
                pimg = pil(r)
                # small libdmtx param set; avoid exotic kwargs for compatibility
                for shrink in (1, 2, 3, 4):
                    for gap in (1, 2):
                        try:
                            res = dmtx_decode(pimg, timeout=1200, max_count=8, shrink=shrink, gap_size=gap)
                        except TypeError:
                            res = dmtx_decode(pimg, timeout=1200, max_count=8, shrink=shrink)
                        if res:
                            results.extend(x.data.decode("utf-8","ignore") for x in res)
                            # return early on first hit
                            return list(dict.fromkeys(results))
    return []

def select_rois(bgr):
    print("Draw a tight box around each code (Enter=accept). Press ESC when done.", flush=True)
    rois = []
    preview = bgr.copy()
    while True:
        x,y,w,h = map(int, cv2.selectROI("ROI", preview, showCrosshair=True, fromCenter=False))
        if w==0 or h==0:  # ESC or window closed
            cv2.destroyWindow("ROI")
            break
        rois.append((x,y,w,h))
        cv2.rectangle(preview,(x,y),(x+w,y+h),(0,255,0),2)
    return rois

def main(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Cannot open: {path}"); return
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    print(f"[{os.path.basename(path)}] {gray.shape[1]}x{gray.shape[0]}", flush=True)

    boxes = select_rois(bgr)
    if not boxes:
        # If you didn’t draw anything, try the whole image once.
        print("No ROI drawn; trying whole image once…")
        out = decode_direct(gray)
        print("Decoded ->", out if out else "[]")
        return

    any_hits = []
    for i,(x,y,w,h) in enumerate(boxes, 1):
        roi = gray[y:y+h, x:x+w]
        print(f"Decoding ROI #{i} (size {w}x{h}) …", flush=True)
        out = decode_direct(roi)
        print("  ->", out if out else "no read")
        any_hits += out
    if any_hits:
        print("FINAL ->", list(dict.fromkeys(any_hits)))
    else:
        print("No codes decoded.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -u dm_manual_ultra.py <image>")
        sys.exit(1)
    main(sys.argv[1])
