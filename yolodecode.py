import argparse, os, sys, time, math
import numpy as np
import cv2

# ---------- Optional ZXing ----------
try:
    import zxingcpp
    ZX_OK = True
except Exception:
    ZX_OK = False

from pylibdmtx.pylibdmtx import decode as dmtx_decode

# ---------------- utils ----------------
def ensure_gray(img):
    if img is None: return None
    if img.ndim == 2:
        g = img
    elif img.ndim == 3:
        c = img.shape[2]
        if   c == 1: g = img[:,:,0]
        elif c == 3: g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif c == 4: g = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else: raise ValueError(f"Unsupported channels: {img.shape}")
    else:
        raise ValueError(f"Unsupported ndim: {img.ndim}")
    return g.astype(np.uint8) if g.dtype != np.uint8 else g

def add_qz(img, px=10, val=255):
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=val)

def dpm_enhance(img):
    # Dot-peen friendly local contrast + small morphology
    cla = cv2.createCLAHE(2.0,(8,8)).apply(img)
    den = cv2.medianBlur(cla, 3)
    se9 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    th = cv2.morphologyEx(den, cv2.MORPH_TOPHAT, se9)
    bh = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, se9)
    mix = cv2.addWeighted(th, 0.6, bh, 0.6, 0)
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    return cv2.morphologyEx(mix, cv2.MORPH_CLOSE, se3, 1)

def order_quad(pts):
    pts = np.array(pts, np.float32)
    idx = np.argsort(pts[:,1]); top, bot = pts[idx[:2]], pts[idx[2:]]
    tl, tr = top[np.argsort(top[:,0])]; bl, br = bot[np.argsort(bot[:,0])]
    return np.array([tl,tr,br,bl], np.float32)

def warp_quad(gray, quad, out=288):
    M = cv2.getPerspectiveTransform(order_quad(quad),
        np.array([[0,0],[out-1,0],[out-1,out-1],[0,out-1]],np.float32))
    return cv2.warpPerspective(gray, M, (out,out))

# ---------------- rankers ----------------
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
    gx = cv2.Sobel(img, cv2.CV_32F, 1,0,3); gy = cv2.Sobel(img, cv2.CV_32F, 0,1,3)
    return float(cv2.mean(cv2.magnitude(gx,gy))[0])

# ---------------- decoders ----------------
def decode_zxing_dm(gray):
    if not ZX_OK: return []
    try:
        res = zxingcpp.read_barcodes(gray, formats=zxingcpp.BarcodeFormat.DataMatrix, try_harder=True)
    except TypeError:
        res = zxingcpp.read_barcodes(gray)
    return [r.text for r in res] if res else []

def decode_libdmtx_views(gray, timeout=230):
    for inv in (False, True):
        v = cv2.bitwise_not(gray) if inv else gray
        for k in (0,1,2,3):
            r = np.rot90(v, k)
            for shrink in (2,3):
                try:
                    out = dmtx_decode(r, timeout=timeout, max_count=1, shrink=shrink, gap_size=2)
                except TypeError:
                    out = dmtx_decode(r, timeout=timeout, max_count=1, shrink=shrink)
                if out:
                    return [out[0].data.decode("utf-8","ignore")]
    return []

def libdmtx_param_sweep(square_gray, time_ms=220):
    base = add_qz(square_gray, px=12, val=255)
    edges_px = base.shape[0]
    min_edge_opts = [None, int(edges_px*0.30)]
    max_edge_opts = [None, int(edges_px*1.3)]
    for shrink in (1,2,3):
        for gap in (1,2):
            for thr in (12,20,30,50):
                for me in min_edge_opts:
                    for Ma in max_edge_opts:
                        kwargs = dict(timeout=time_ms, max_count=1, shrink=shrink, gap_size=gap, threshold=thr)
                        if me is not None: kwargs["min_edge"] = me
                        if Ma is not None: kwargs["max_edge"] = Ma
                        for inv in (False, True):
                            patch = (255-base) if inv else base
                            out = dmtx_decode(patch, **kwargs)
                            if out:
                                return [out[0].data.decode("utf-8","ignore")]
    return []

def try_views(square):
    base = square
    inv  = cv2.bitwise_not(base)
    thrA = cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    _, thrO = cv2.threshold(base,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    views = [add_qz(base,12,255), add_qz(base,12,0), add_qz(inv,12,255),
             add_qz(thrA,12,255), add_qz(thrO,12,255)]
    for v in views:
        hit = decode_zxing_dm(v)
        if hit: return hit
    for v in views[:3]:
        hit = decode_libdmtx_views(v, timeout=230)
        if hit: return hit
    # final: param sweep on original
    hit = libdmtx_param_sweep(base, time_ms=220)
    return hit if hit else []

# ---------------- classical locator (inside ROI/image) ----------------
def lshape_candidates(gray, expected_frac=None, maxc=24):
    H, W = gray.shape
    cla = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    den = cv2.medianBlur(cla, 3)
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    edgesA = cv2.Canny(cv2.morphologyEx(den, cv2.MORPH_TOPHAT, se), 30, 100)
    edgesB = cv2.Canny(cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, se), 30, 100)

    quads=[]
    for edges in (edgesA, edgesB):
        segs = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30,
                               minLineLength=max(12, int(0.04*W)), maxLineGap=10)
        if segs is None: continue
        lines=[]
        for x1,y1,x2,y2 in segs[:,0]:
            L = math.hypot(x2-x1, y2-y1)
            if L < 12: continue
            ang = abs((math.degrees(math.atan2(y2-y1, x2-x1)) % 180) - 90)
            if min(ang, 90-ang) > 28: continue  # looser
            lines.append((x1,y1,x2,y2,L))

        n=len(lines)
        for i in range(n):
            x1,y1,x2,y2,L1 = lines[i]
            for j in range(i+1,n):
                u1,v1,u2,v2,L2 = lines[j]
                a1 = math.atan2(y2-y1, x2-x1); a2 = math.atan2(v2-v1, u2-u1)
                if abs(abs(a1-a2) - np.pi/2) > np.deg2rad(22): continue  # looser
                A = np.array([[x1,y1],[x2,y2]],np.float32)
                B = np.array([[u1,v1],[u2,v2]],np.float32)
                d = ((A[:,None,:]-B[None,:,:])**2).sum(-1)
                ai,bj = np.unravel_index(np.argmin(d), d.shape)
                corner = (A[ai]+B[bj])/2.0
                vA = (A[1-ai]-A[ai]); vB = (B[1-bj]-B[bj])
                long = max(np.linalg.norm(vA), np.linalg.norm(vB))
                if long < 16: continue
                if expected_frac and expected_frac>0:
                    exp = expected_frac*W
                    if not (0.6*exp <= long <= 1.7*exp):  # size hint
                        continue
                p0 = corner; p1 = corner + vA; p3 = corner + vB; p2 = p1 + vB
                quad = np.array([p0,p1,p2,p3],np.float32)
                w = np.linalg.norm(quad[1]-quad[0]); h = np.linalg.norm(quad[3]-quad[0])
                if max(w,h)/max(1.0,min(w,h)) > 1.75 or w*h < 420:
                    continue
                quads.append(quad)

    keep, centers = [], []
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.int32))):
        c = q.mean(axis=0)
        if all(np.hypot(c[0]-cc[0], c[1]-cc[1])>10 for cc in centers):
            keep.append(q); centers.append(c)
        if len(keep) >= maxc: break
    return keep

def contour_mser_candidates(gray, expected_frac=None, maxc=24):
    H, W = gray.shape
    cla = cv2.createCLAHE(2.0, (8,8)).apply(gray)
    g = cv2.medianBlur(cla, 3)
    se9 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se9)
    _, b1 = cv2.threshold(th,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(b1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if expected_frac and expected_frac>0:
        smin, smax = int(0.55*expected_frac*W), int(1.8*expected_frac*W)
    else:
        smin, smax = int(0.04*W), int(0.45*W)

    quads=[]
    for c in cnts:
        if cv2.contourArea(c) < 64: continue
        rect = cv2.minAreaRect(c)
        (cx,cy),(w,h),_ = rect
        long, short = max(w,h), max(1.0,min(w,h))
        if long < smin or long > smax: continue
        if long/short > 1.6: continue
        box = cv2.boxPoints(rect).astype(np.float32)
        quads.append(box)

    # MSER (dot clusters)
    try:
        mser = cv2.MSER_create(_delta=5, _min_area=40, _max_area=int(1e5))
    except TypeError:
        try: mser = cv2.MSER_create(5,40,int(1e5))
        except TypeError: mser = cv2.MSER_create()
    regions,_ = mser.detectRegions(cla)
    for r in regions:
        x,y,w,h = cv2.boundingRect(r.reshape(-1,1,2))
        long, short = max(w,h), min(w,h)
        if not (smin <= long <= smax): continue
        if long/max(1,short) > 1.6: continue
        quads.append(np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], np.float32))

    keep, centers = [], []
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.int32))):
        c = q.mean(axis=0)
        if all(np.hypot(c[0]-cc[0], c[1]-cc[1])>10 for cc in centers):
            keep.append(q); centers.append(c)
        if len(keep) >= maxc: break
    return keep

# ---------------- pipeline ----------------
def decode_square(square, time_budget_ms=1000, save_to=None, idx=0):
    # quick attempts
    hit = try_views(square)
    if hit: 
        if save_to is not None:
            cv2.imwrite(os.path.join(save_to, f"ok_{idx:02d}.png"), square)
        return hit
    # short heavy tries (scale up)
    for fx in (1.25, 1.5):
        gg = cv2.resize(square, None, fx=fx, fy=fx, interpolation=cv2.INTER_NEAREST)
        hit = try_views(gg)
        if hit:
            if save_to is not None:
                cv2.imwrite(os.path.join(save_to, f"ok_{idx:02d}.png"), gg)
            return hit
    # libdmtx param sweep (expensive but last resort)
    hit = libdmtx_param_sweep(square, time_ms=220)
    if hit and save_to is not None:
        cv2.imwrite(os.path.join(save_to, f"ok_{idx:02d}.png"), square)
    return hit if hit else []

def process_with_locator(gray, roi=None, dpm=False, topk=6, time_budget_s=16,
                         save_dir=None, expected_frac=None, debug_dir=None, dbg_name="overlay.png"):
    H,W = gray.shape
    x0,y0,x1,y1 = (0,0,W,H) if roi is None else roi
    crop = gray[y0:y1, x0:x1]
    if dpm:
        crop = dpm_enhance(crop)

    # propose quads
    quads = []
    quads += lshape_candidates(crop, expected_frac=expected_frac, maxc=32)
    quads += contour_mser_candidates(crop, expected_frac=expected_frac, maxc=32)

    ranked=[]
    for q in quads:
        w = warp_quad(crop, q, out=288)
        sL = l_finder_score(w); sF = focus_score(w)
        ranked.append((0.9*sL + 0.1*(sF/100.0), q, w, sL, sF))
    ranked.sort(key=lambda t: -t[0])
    ranked = ranked[:topk]

    # save cand warps no matter what
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for i,(_,_,w,_,_) in enumerate(ranked):
            cv2.imwrite(os.path.join(save_dir, f"cand_{i:02d}.png"), w)

    # debug overlay (on full image, shifted by ROI)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        vis = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)
        for i,(score,q,_,sL,sF) in enumerate(ranked):
            qq = q.copy(); qq[:,0]+=x0; qq[:,1]+=y0
            qq = qq.astype(int)
            cv2.polylines(vis, [qq], True, (0,255,255), 2)
            cx,cy = int(qq[:,0].mean()), int(qq[:,1].mean())
            cv2.putText(vis, f"{i}:{score:.2f}", (cx-15, cy-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (x0,y0), (x1,y1), (0,200,0), 2)
        cv2.imwrite(os.path.join(debug_dir, dbg_name), vis)

    # decode best-first within a time budget
    t0 = time.monotonic()
    def left(): return time_budget_s - (time.monotonic()-t0)
    hits=[]
    for i,(score,q,w,_,_) in enumerate(ranked):
        if left() <= 0: break
        out = decode_square(w, save_to=save_dir, idx=i)
        if out:
            hits += out
            break  # stop at first success (flip to continue to try all)
    return hits

def smart_decode_whole(gray, dpm=False):
    g = dpm_enhance(gray) if dpm else gray
    s = 360 / max(g.shape)
    if s < 1.0:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    return try_views(g)

# ---------------- main driver ----------------
def run(weights, imgs, conf=0.25, dpm=False, resize=320, locate=False, roi=None,
        save_dir=None, time_budget_s=16, exp_frac=0.0, topk=6, debug_dir=None):
    model = None
    if weights:
        try:
            from ultralytics import YOLO
            model = YOLO(weights)
        except Exception as e:
            print(f"(warning) could not load YOLO weights: {e}")
            model = None

    for p in imgs:
        img0 = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img0 is None:
            print(f"[{p}] cannot open"); continue
        img = ensure_gray(img0)
        H,W = img.shape
        print(f"[{p}] {W}x{H}")

        # YOLO path
        if model is not None:
            r = model.predict(source=img, imgsz=1024, conf=conf, verbose=False)[0]
            if len(r.boxes)==0:
                print("  (no detections)"); continue
            hits=[]
            for i, b in enumerate(r.boxes):
                x1,y1,x2,y2 = [int(v) for v in b.xyxy[0].tolist()]
                pad = int(0.06*max(x2-x1, y2-y1))
                x0 = max(0, x1-pad); y0 = max(0, y1-pad)
                x3 = min(W-1, x2+pad); y3 = min(H-1, y2+pad)
                crop = img[y0:y3, x0:x3]
                g = dpm_enhance(crop) if dpm else crop
                s = max(1, int(resize / max(g.shape)))
                g = cv2.resize(g, (g.shape[1]*s, g.shape[0]*s), interpolation=cv2.INTER_NEAREST)
                out = try_views(g)
                if not out:
                    # as backup, run classical locator inside the detected box
                    name = os.path.splitext(os.path.basename(p))[0]
                    out = process_with_locator(img, roi=(x0,y0,x3,y3), dpm=dpm, topk=topk,
                                               time_budget_s=time_budget_s, save_dir=save_dir,
                                               expected_frac=exp_frac, debug_dir=debug_dir,
                                               dbg_name=f"{name}_overlay.png")
                if out: hits += out; print(f"  box#{i} -> {out}")
                else:   print(f"  box#{i} -> no read")
            print(("FINAL -> "+str(list(dict.fromkeys(hits)))) if hits else "  -> No codes decoded.")
            continue

        # Classical path (no weights)
        if locate or roi is not None:
            name = os.path.splitext(os.path.basename(p))[0]
            hits = process_with_locator(img, roi=roi, dpm=dpm, topk=topk,
                                        time_budget_s=time_budget_s, save_dir=save_dir,
                                        expected_frac=exp_frac, debug_dir=debug_dir,
                                        dbg_name=f"{name}_overlay.png")
            print(("FINAL -> "+str(hits)) if hits else "  -> No codes decoded.")
        else:
            out = smart_decode_whole(img, dpm=dpm)
            print(("FINAL -> "+str(out)) if out else "  -> No codes decoded.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None, help="YOLO weights for 'datamatrix' detection (.pt)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dpm", action="store_true", help="Apply dot-peen enhancer before locate/decode")
    ap.add_argument("--resize", type=int, default=320)
    ap.add_argument("--locate", action="store_true", help="Use classical locator when no weights are provided")
    ap.add_argument("--roi", nargs=4, type=int, help="Restrict search to ROI: x y w h")
    ap.add_argument("--exp-frac", type=float, default=0.0, help="Expected code side as fraction of ROI width (e.g., 0.35)")
    ap.add_argument("--topk", type=int, default=6, help="How many top candidates to try decoding")
    ap.add_argument("--save-warps", default=None, help="Directory to save candidate/ok warps")
    ap.add_argument("--save-debug", default=None, help="Directory to save overlay images with candidate boxes")
    ap.add_argument("--time", type=int, default=16, help="Time budget per image (seconds)")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    roi_tuple = None
    if args.roi and len(args.roi)==4:
        x,y,w,h = args.roi
        roi_tuple = (x,y,x+w,y+h)

    run(args.weights, args.images, conf=args.conf, dpm=args.dpm, resize=args.resize,
        locate=args.locate, roi=roi_tuple, save_dir=args.save_warps, time_budget_s=args.time,
        exp_frac=args.exp_frac, topk=args.topk, debug_dir=args.save_debug)
