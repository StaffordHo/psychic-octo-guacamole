import sys, time, math, threading
from typing import List, Tuple
import numpy as np
import cv2

# Optional ZXing; script still runs with libdmtx only
try:
    import zxingcpp
    ZX_OK = True
except Exception:
    ZX_OK = False

from pylibdmtx.pylibdmtx import decode as dmtx_decode

# ---------------- utilities ----------------
def add_qz(img, px=10, val=255):
    return cv2.copyMakeBorder(img, px,px,px,px, cv2.BORDER_CONSTANT, value=val)

def order_quad(pts):
    pts = np.array(pts, np.float32)
    idx = np.argsort(pts[:,1]); top, bot = pts[idx[:2]], pts[idx[2:]]
    tl, tr = top[np.argsort(top[:,0])]; bl, br = bot[np.argsort(bot[:,0])]
    return np.array([tl,tr,br,bl], np.float32)

def warp_quad(gray, quad, out=288):
    M = cv2.getPerspectiveTransform(order_quad(quad),
                                    np.array([[0,0],[out-1,0],[out-1,out-1],[0,out-1]],np.float32))
    return cv2.warpPerspective(gray, M, (out,out))

def dpm_enhance(img):
    cla = cv2.createCLAHE(2.0, (8,8)).apply(img)
    den = cv2.medianBlur(cla, 3)
    se9 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    th = cv2.morphologyEx(den, cv2.MORPH_TOPHAT, se9)
    bh = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, se9)
    mix = cv2.addWeighted(th, 0.6, bh, 0.6, 0)
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    return cv2.morphologyEx(mix, cv2.MORPH_CLOSE, se3, iterations=1)

# ---------------- quick rankers ----------------
def border_black_ratio(img, band=12):
    _, bw = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    strips = [bw[:band,:], bw[-band:,:], bw[:,:band], bw[:,-band:]]
    return [(s < 128).mean() for s in strips]

def l_finder_score(img):
    r = border_black_ratio(img, 12)
    best = -1.0
    for k in range(4):
        rr = r[k:]+r[:k]
        s = sum(sorted(rr, reverse=True)[:2]) - sum(sorted(rr)[:2])
        best = max(best, s)
    return float(best)

def focus_score(img):
    gy = cv2.Sobel(img, cv2.CV_32F, 0,1,ksize=3)
    gx = cv2.Sobel(img, cv2.CV_32F, 1,0,ksize=3)
    return float(cv2.mean(cv2.magnitude(gx,gy))[0])

# ---------------- decoders ----------------
def decode_zxing(arr):
    if not ZX_OK: return []
    try:
        res = zxingcpp.read_barcodes(arr, formats=zxingcpp.BarcodeFormat.DataMatrix, try_harder=True)
    except TypeError:
        res = zxingcpp.read_barcodes(arr)
    return [r.text for r in res] if res else []

def decode_libdmtx(arr, timeout=200):
    for inv in (False, True):
        v = cv2.bitwise_not(arr) if inv else arr
        for rotk in (0,1,2,3):
            r = np.rot90(v, rotk)
            for shrink in (2,3):
                try:
                    out = dmtx_decode(r, timeout=timeout, max_count=1, shrink=shrink, gap_size=2)
                except TypeError:
                    out = dmtx_decode(r, timeout=timeout, max_count=1, shrink=shrink)
                if out:
                    return [out[0].data.decode("utf-8","ignore")]
    return []

def try_decode(square):
    base = square
    inv = cv2.bitwise_not(base)
    thrA = cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    _, thrO = cv2.threshold(base,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    views = [
        add_qz(base, 12, 255), add_qz(base, 12, 0),
        add_qz(inv, 12, 255),
        add_qz(thrA, 12, 255), add_qz(thrO, 12, 255)
    ]
    for v in views:
        hit = decode_zxing(v)
        if hit: return hit
    for v in views[:3]:
        hit = decode_libdmtx(v, timeout=230)
        if hit: return hit
    return []

def heavy_decode(square, max_ms=1200):
    start = time.monotonic()
    def left(): return max_ms/1000.0 - (time.monotonic() - start)

    for base in (square, dpm_enhance(square), cv2.resize(square,None,fx=1.35,fy=1.35,interpolation=cv2.INTER_NEAREST)):
        if left() <= 0: return []
        variants = [base]
        variants.append(cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5))
        _, o = cv2.threshold(base,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU); variants.append(o)

        for v in variants:
            if left() <= 0: return []
            for qz in (255, 0):
                z = add_qz(v, 12, qz)
                hit = decode_zxing(z)
                if hit: return hit
        for v in variants[:2]:
            if left() <= 0: return []
            for qz in (255, 0):
                z = add_qz(v, 12, qz)
                for inv in (False, True):
                    if left() <= 0: return []
                    vv = cv2.bitwise_not(z) if inv else z
                    for rotk in (0,1,2,3):
                        if left() <= 0: return []
                        rr = np.rot90(vv, rotk)
                        for shrink in (2,3,4):
                            if left() <= 0: return []
                            try:
                                out = dmtx_decode(rr, timeout=140, max_count=1, shrink=shrink, gap_size=2)
                            except TypeError:
                                out = dmtx_decode(rr, timeout=140, max_count=1, shrink=shrink)
                            if out:
                                return [out[0].data.decode("utf-8","ignore")]
    return []

# ---------------- locator (L-finder + contour/MSER) ----------------
def lshape_candidates(gray, expected_frac=None, maxc=24):
    H,W = gray.shape
    cla = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    den = cv2.medianBlur(cla, 3)
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    views = []
    for v in (cv2.morphologyEx(den, cv2.MORPH_TOPHAT, se),
              cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, se)):
        edges = cv2.Canny(v, 40, 120); views.append(edges)

    quads=[]
    for edges in views:
        segs = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                               minLineLength=max(18, int(0.05*W)), maxLineGap=8)
        if segs is None: continue
        lines=[]
        for x1,y1,x2,y2 in segs[:,0]:
            L = math.hypot(x2-x1, y2-y1)
            if L < 18: continue
            ang = abs((math.degrees(math.atan2(y2-y1, x2-x1)) % 180) - 90)
            if min(ang, 90-ang) > 20: continue
            lines.append((x1,y1,x2,y2,L))
        n=len(lines)
        for i in range(n):
            x1,y1,x2,y2,L1 = lines[i]
            for j in range(i+1,n):
                u1,v1,u2,v2,L2 = lines[j]
                a1 = math.atan2(y2-y1, x2-x1); a2 = math.atan2(v2-v1, u2-u1)
                if abs(abs(a1-a2) - np.pi/2) > np.deg2rad(18): continue
                A = np.array([[x1,y1],[x2,y2]],np.float32)
                B = np.array([[u1,v1],[u2,v2]],np.float32)
                d = ((A[:,None,:]-B[None,:,:])**2).sum(-1)
                ai,bj = np.unravel_index(np.argmin(d), d.shape)
                corner = (A[ai]+B[bj])/2.0
                vA = (A[1-ai]-A[ai]); vB = (B[1-bj]-B[bj])
                sA, sB = np.linalg.norm(vA), np.linalg.norm(vB)
                long = max(sA,sB)
                if long < 22: continue
                if expected_frac and expected_frac>0:
                    exp = expected_frac*W
                    if not (0.5*exp <= long <= 1.8*exp): continue
                p0 = corner; p1 = corner + vA; p3 = corner + vB; p2 = p1 + vB
                quad = np.array([p0,p1,p2,p3],np.float32)
                quads.append(quad)

    keep, centers = [], []
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.int32))):
        c = q.mean(axis=0)
        if all(np.hypot(c[0]-cc[0], c[1]-cc[1])>12 for cc in centers):
            keep.append(q); centers.append(c)
        if len(keep) >= maxc: break
    return keep

def contour_mser_candidates(gray, expected_frac=None, maxc=24):
    H,W = gray.shape
    cla = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    g = cv2.medianBlur(cla, 3)
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    _, b1 = cv2.threshold(th,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    bw = cv2.morphologyEx(b1, cv2.MORPH_CLOSE, k3, 1)
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if expected_frac and expected_frac>0:
        smin, smax = int(0.5*expected_frac*W), int(1.7*expected_frac*W)
    else:
        smin, smax = int(0.05*W), int(0.38*W)

    quads=[]
    for c in cnts:
        if cv2.contourArea(c) < 120: continue
        rect = cv2.minAreaRect(c)
        (cx,cy),(w,h),_ = rect
        long, short = max(w,h), max(1.0,min(w,h))
        if long < 18 or long> smax or long < smin: continue
        if long/short > 1.4: continue
        box = cv2.boxPoints(rect).astype(np.float32)
        quads.append(box)

    # MSER (dot clusters)
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=60, _max_area=int(1e5))
    except TypeError:
        try: mser = cv2.MSER_create(5,60,int(1e5))
        except TypeError: mser = cv2.MSER_create()
    regions,_ = mser.detectRegions(cla)
    for r in regions:
        x,y,w,h = cv2.boundingRect(r.reshape(-1,1,2))
        long, short = max(w,h), min(w,h)
        if not (smin <= long <= smax): continue
        if long/max(1,short) > 1.45: continue
        q = np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.float32)
        quads.append(q)

    keep, centers = [], []
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.int32))):
        c = q.mean(axis=0)
        if all(np.hypot(c[0]-cc[0], c[1]-cc[1])>12 for cc in centers):
            keep.append(q); centers.append(c)
        if len(keep) >= maxc: break
    return keep

# ---------------- ROI decoder ----------------
def decode_roi(gray_roi, maxc=6, heavy_ms=1200, return_all=False):
    H,W = gray_roi.shape
    quads = []
    quads += lshape_candidates(gray_roi, expected_frac=None, maxc=18)
    quads += contour_mser_candidates(gray_roi, expected_frac=None, maxc=18)

    # rank
    ranked=[]
    for q in quads:
        w = warp_quad(gray_roi, q, out=288)
        sL = l_finder_score(w); sF = focus_score(w)
        ranked.append((0.9*sL + 0.1*(sF/100.0), sL, sF, q, w))
    ranked.sort(key=lambda t: -t[0])
    ranked = ranked[:maxc]

    results=[]
    if ranked:
        # heavy on top-1
        hit = heavy_decode(ranked[0][4], max_ms=heavy_ms)
        if hit:
            results += hit
            if not return_all: return results, ranked
        # quick on rest
        for _,_,_,_,w in ranked[1:]:
            out = try_decode(w)
            if out:
                results += out
                if not return_all: return results, ranked
    return results if results else [], ranked

# ---------------- interactive watch ----------------
def main(img_path):
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if src is None:
        print(f"Cannot open {img_path}")
        return

    H,W = src.shape
    win = "DM Watch (drag to set ROI; q=quit, r=reset, a=all)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # shared state
    state = {
        "roi": (0,0,W,H),
        "drag": False,
        "start": (0,0),
        "result": [],
        "ranked": [],
        "version": 0,
        "all": False
    }
    lock = threading.Lock()

    def on_mouse(event, x, y, flags, param):
        nonlocal state
        if event == cv2.EVENT_LBUTTONDOWN:
            with lock:
                state["drag"] = True; state["start"] = (x,y)
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            sx,sy = state["start"]; x0,y0 = min(sx,x), min(sy,y); x1,y1 = max(sx,x), max(sy,y)
            x0 = max(0, min(x0, W-2)); y0 = max(0, min(y0, H-2))
            x1 = max(x0+2, min(x1, W-1)); y1 = max(y0+2, min(y1, H-1))
            with lock:
                state["roi"] = (x0,y0,x1,y1)
        elif event == cv2.EVENT_LBUTTONUP:
            sx,sy = state["start"]; x0,y0 = min(sx,x), min(sy,y); x1,y1 = max(sx,x), max(sy,y)
            x0 = max(0, min(x0, W-2)); y0 = max(0, min(y0, H-2))
            x1 = max(x0+2, min(x1, W-1)); y1 = max(y0+2, min(y1, H-1))
            with lock:
                state["drag"] = False
                state["roi"] = (x0,y0,x1,y1)
                state["version"] += 1  # trigger decode

    cv2.setMouseCallback(win, on_mouse)

    # worker thread: decode when ROI changes or every 0.5 s
    stop = threading.Event()
    def worker():
        last_v = -1
        last_t = 0
        while not stop.is_set():
            with lock:
                v = state["version"]
                x0,y0,x1,y1 = state["roi"]
                gray = src[y0:y1, x0:x1].copy()
                want_all = state["all"]
            if v != last_v or (time.time()-last_t) > 0.5:
                last_v = v; last_t = time.time()
                if gray.shape[0] >= 24 and gray.shape[1] >= 24:
                    out, ranked = decode_roi(gray, maxc=6, heavy_ms=1200, return_all=want_all)
                else:
                    out, ranked = [], []
                with lock:
                    state["result"] = out
                    state["ranked"] = ranked
            time.sleep(0.03)
    th = threading.Thread(target=worker, daemon=True)
    th.start()

    try:
        while True:
            with lock:
                x0,y0,x1,y1 = state["roi"]
                res = state["result"]
                ranked = state["ranked"]
                want_all = state["all"]

            vis = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
            color = (0,200,0) if res else (0,0,255)
            cv2.rectangle(vis, (x0,y0), (x1,y1), color, 2)

            # draw top candidate corners (for feedback)
            if ranked:
                q = ranked[0][3].copy()
                q[:,0]+=x0; q[:,1]+=y0
                q = q.astype(int)
                cv2.polylines(vis, [q], True, (255,200,0), 2)

            if res:
                txt = " | ".join(res if want_all else res[:1])
                cv2.putText(vis, txt, (max(5,x0+3), max(20,y0-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2, cv2.LINE_AA)

            cv2.imshow(win, vis)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                with lock:
                    state["roi"] = (0,0,W,H); state["version"] += 1
            elif k == ord('a'):
                with lock:
                    state["all"] = not state["all"]; state["version"] += 1
    finally:
        stop.set(); th.join(timeout=0.5)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dm_watch.py <image>")
        sys.exit(1)
    main(sys.argv[1])
