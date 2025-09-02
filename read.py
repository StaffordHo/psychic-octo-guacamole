import sys, time, math, threading, os, statistics, re
import cv2, numpy as np
from collections import deque
from PIL import Image

# ---- decoders ----
try:
    import zxingcpp
    ZX_OK = True
except Exception:
    ZX_OK = False

from pylibdmtx.pylibdmtx import decode as dmtx_decode, encode as dmtx_encode

WIN = "DM ROI Watch (strict)"

CTRL_TOKENS = {0x1D:"<GS>",0x1E:"<RS>",0x1F:"<US>",0x1C:"<FS>",0x04:"<EOT>"}
def sanitize(s:str)->str:
    out=[]
    for ch in s:
        oc=ord(ch)
        out.append(CTRL_TOKENS.get(oc, ch if oc>=32 else ""))
    return "".join(out)

def alnum_only(s:str)->str:
    return "".join(ch for ch in s if ch.isalnum())

def imread_gray(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None: raise SystemExit(f"Cannot open {p}")
    return g

def add_qz(img, px=10, val=255):
    return cv2.copyMakeBorder(img, px,px,px,px, cv2.BORDER_CONSTANT, value=val)

def dpm_enhance(img):
    cla = cv2.createCLAHE(2.0,(8,8)).apply(img)
    den = cv2.medianBlur(cla, 3)
    se9 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    th = cv2.morphologyEx(den, cv2.MORPH_TOPHAT, se9)
    bh = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, se9)
    mix = cv2.addWeighted(th, 0.6, bh, 0.6, 0)
    return cv2.morphologyEx(mix, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), 1)

def order_quad(pts):
    pts = np.array(pts, np.float32)
    idx = np.argsort(pts[:,1]); top, bot = pts[idx[:2]], pts[idx[2:]]
    tl, tr = top[np.argsort(top[:,0])]; bl, br = bot[np.argsort(bot[:,0])]
    return np.array([tl,tr,br,bl], np.float32)

def warp_quad(gray, quad, out=288):
    M = cv2.getPerspectiveTransform(order_quad(quad),
        np.array([[0,0],[out-1,0],[out-1,out-1],[0,out-1]],np.float32))
    return cv2.warpPerspective(gray, M, (out,out))

# ---------- DM-only decode variants ----------
def decode_zxing(arr):
    if not ZX_OK: return []
    try:
        res = zxingcpp.read_barcodes(arr, formats=zxingcpp.BarcodeFormat.DataMatrix, try_harder=True)
    except TypeError:
        res = zxingcpp.read_barcodes(arr)
    return [r.text for r in res] if res else []

def decode_libdmtx(arr, timeout=220):
    for inv in (False, True):
        v = cv2.bitwise_not(arr) if inv else arr
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

def quick_views(square):
    base = square
    inv  = cv2.bitwise_not(base)
    thrA = cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    _, thrO = cv2.threshold(base,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return [add_qz(base,12,255), add_qz(inv,12,255), add_qz(thrA,12,255), add_qz(thrO,12,255)]

# ---------- STRICT VALIDATION ----------
def reencode_bitmap(payload:str):
    """Encode with libdmtx and return a binary image (255=white, 0=black)."""
    enc = dmtx_encode(payload.encode("utf-8"))
    arr = np.frombuffer(enc.pixels, dtype=np.uint8).reshape((enc.height, enc.width, 3))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw

def validate_by_reencode(warp: np.ndarray, payload:str, min_sim=0.82) -> bool:
    target = reencode_bitmap(payload)
    obs = dpm_enhance(warp)
    _, obw = cv2.threshold(obs,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    obw = cv2.resize(obw, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)

    best = 0.0
    for k in range(4):
        r = np.rot90(obw, k)
        # align black/white
        if np.mean(r) < 128: r = 255 - r
        sim = 1.0 - (np.bitwise_xor(r, target).mean() / 255.0)
        best = max(best, float(sim))
    return best >= min_sim

def consensus_decode(square, min_len=5, accept_regex=None):
    """Require at least 2 matching votes across views, then re-encode validate."""
    votes=[]
    for v in quick_views(square):
        hit = decode_zxing(v)
        if hit: votes.append(hit[0])
    for v in quick_views(square)[:2]:
        hit = decode_libdmtx(v, timeout=200)
        if hit: votes.append(hit[0])

    if not votes:
        return None
    # majority
    cand = max(set(votes), key=votes.count)
    if votes.count(cand) < 2:
        return None
    if len(cand) < min_len:
        return None
    if accept_regex and not re.search(accept_regex, cand):
        return None
    # final structural check
    if not validate_by_reencode(square, cand, min_sim=0.82):
        return None
    return cand

# ---------- simple ROI auto-locator inside search region ----------
def roi_candidates(gray_roi, maxc=6):
    H,W = gray_roi.shape
    if W<32 or H<32: return []
    cla = cv2.createCLAHE(2.0,(8,8)).apply(gray_roi)
    g = cv2.medianBlur(cla, 3)
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    edges = []
    for v in (cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se),
              cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)):
        edges.append(cv2.Canny(v, 40, 120))

    quads=[]
    for ed in edges:
        segs = cv2.HoughLinesP(ed, 1, np.pi/180, threshold=38,
                               minLineLength=max(16, int(0.18*W)),
                               maxLineGap=8)
        if segs is None: continue
        lines=[]
        for x1,y1,x2,y2 in segs[:,0]:
            L = math.hypot(x2-x1, y2-y1)
            if L < 16: continue
            ang = abs((math.degrees(math.atan2(y2-y1,x2-x1)) % 180) - 90)
            if min(ang,90-ang) > 20: continue
            lines.append((x1,y1,x2,y2,L))
        n=len(lines)
        for i in range(n):
            x1,y1,x2,y2,L1 = lines[i]
            for j in range(i+1,n):
                u1,v1,u2,v2,L2 = lines[j]
                a1 = math.atan2(y2-y1, x2-x1); a2 = math.atan2(v2-v1, u2-u1)
                if abs(abs(a1-a2)-np.pi/2) > np.deg2rad(18): continue
                A = np.array([[x1,y1],[x2,y2]],np.float32)
                B = np.array([[u1,v1],[u2,v2]],np.float32)
                d = ((A[:,None,:]-B[None,:,:])**2).sum(-1)
                ai,bj = np.unravel_index(np.argmin(d), d.shape)
                corner = (A[ai]+B[bj])/2.0
                vA = (A[1-ai]-A[ai]); vB = (B[1-bj]-B[bj])
                if max(np.linalg.norm(vA), np.linalg.norm(vB)) < 0.22*W: continue
                p0=corner; p1=corner+vA; p3=corner+vB; p2=p1+vB
                quad = np.array([p0,p1,p2,p3],np.float32)
                w = np.linalg.norm(quad[1]-quad[0]); h = np.linalg.norm(quad[3]-quad[0])
                if max(w,h)/max(1.0,min(w,h)) > 1.6 or w*h < 600: continue
                quads.append(quad)

    # contour backup
    _, bw = cv2.threshold(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), 1)
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 120: continue
        rect = cv2.minAreaRect(c)
        (cx,cy),(w,h),_ = rect
        long, short = max(w,h), max(1.0,min(w,h))
        if long < 0.22*W or long > 0.98*W: continue
        if long/short > 1.45: continue
        quads.append(cv2.boxPoints(rect).astype(np.float32))

    # dedupe by center
    keep, centers = [], []
    for q in sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.int32))):
        c = q.mean(axis=0)
        if all(np.hypot(c[0]-cc[0], c[1]-cc[1])>10 for cc in centers):
            keep.append(q); centers.append(c)
        if len(keep) >= maxc: break
    return keep

# ---------- ROI / UI ----------
class ROI:
    def __init__(self, x,y,w,h):
        self.x,self.y,self.w,self.h = int(x),int(y),int(w),int(h)
        self.result=None; self.state="idle"; self.last_printed=set()

    def rect(self): return (self.x,self.y,self.w,self.h)
    def contains(self, px,py): return (self.x<=px<=self.x+self.w and self.y<=py<=self.y+self.h)
    def draw(self, frame):
        color=(0,255,0) if self.state=="ok" else (0,255,255) if self.state=="scanning" else (0,0,255)
        cv2.rectangle(frame,(self.x,self.y),(self.x+self.w,self.y+self.h),color,2)
        if self.result:
            cv2.putText(frame, sanitize(self.result), (self.x, max(12,self.y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

class ROIEditor:
    EDGE=6
    def __init__(self, img):
        self.img=img; self.rois=[]; self.drag=None
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(WIN, self.on_mouse)

    def near_edge(self, roi, x,y):
        ex=ROIEditor.EDGE
        L = abs(x-roi.x)<=ex and roi.y-ex<=y<=roi.y+roi.h+ex
        R = abs(x-(roi.x+roi.w))<=ex and roi.y-ex<=y<=roi.y+roi.h+ex
        T = abs(y-roi.y)<=ex and roi.x-ex<=x<=roi.x+roi.w+ex
        B = abs(y-(roi.y+roi.h))<=ex and roi.x-ex<=x<=roi.x+roi.w+ex
        return 'resize_left' if L else 'resize_right' if R else 'resize_top' if T else 'resize_bottom' if B else None

    def on_mouse(self, event,x,y,flags,_):
        if event==cv2.EVENT_LBUTTONDOWN:
            for roi in reversed(self.rois):
                mode=self.near_edge(roi,x,y)
                if mode: self.drag=(roi,mode,x,y); return
                if roi.contains(x,y): self.drag=(roi,'move',x-roi.x,y-roi.y); return
            self.drag=(ROI(x,y,1,1),'new',x,y); self.rois.append(self.drag[0])
        elif event==cv2.EVENT_MOUSEMOVE and self.drag:
            roi,mode,a,b=self.drag
            if mode=='move': roi.x=max(0,x-a); roi.y=max(0,y-b)
            elif mode=='resize_left': nx=min(x,roi.x+roi.w-4); roi.w+=roi.x-nx; roi.x=nx
            elif mode=='resize_right': roi.w=max(4,x-roi.x)
            elif mode=='resize_top': ny=min(y,roi.y+roi.h-4); roi.h+=roi.y-ny; roi.y=ny
            elif mode=='resize_bottom': roi.h=max(4,y-roi.y)
            elif mode=='new': roi.w=max(4,x-roi.x); roi.h=max(4,y-roi.y)
        elif event==cv2.EVENT_LBUTTONUP and self.drag:
            self.drag=None
    def clear(self): self.rois.clear()

# ---------- background watcher ----------
class Watcher(threading.Thread):
    def __init__(self, img, editor: ROIEditor, stop_event: threading.Event, hz=3, min_len=5, accept_regex=None):
        super().__init__(daemon=True)
        self.img, self.editor, self.stop_event = img, editor, stop_event
        self.period = 1.0/hz
        self.min_len = min_len
        self.accept_regex = accept_regex

    def run(self):
        while not self.stop_event.is_set():
            t0 = time.time()
            for idx, roi in enumerate(self.editor.rois):
                if roi.w<24 or roi.h<24: continue
                roi.state="scanning"
                crop = self.img[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]

                # candidates inside search region
                quads = roi_candidates(crop, maxc=6)
                ranked=[]
                for q in quads:
                    w = warp_quad(crop, q, out=288)
                    # quick ranking: two dark borders + sharpness
                    def border_black_ratio(img, band=12):
                        _, bw = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        strips = [bw[:band,:], bw[-band:,:], bw[:,:band], bw[:,-band:]]
                        return [(s<128).mean() for s in strips]
                    r = border_black_ratio(w,12)
                    sL = max(sum(sorted(r, reverse=True)[:2]) - sum(sorted(r)[:2]), 0)
                    gx = cv2.Sobel(w, cv2.CV_32F,1,0,3); gy = cv2.Sobel(w, cv2.CV_32F,0,1,3)
                    sF = float(cv2.mean(cv2.magnitude(gx,gy))[0])
                    ranked.append((0.9*sL + 0.1*(sF/100.0), w))
                ranked.sort(key=lambda t: -t[0])

                payload = None
                if ranked:
                    # strong consensus+reencode on best; quick on a couple more
                    first = consensus_decode(ranked[0][1], min_len=self.min_len, accept_regex=self.accept_regex)
                    if first: payload = first
                    else:
                        for s,w in ranked[1:3]:
                            out = consensus_decode(w, min_len=self.min_len, accept_regex=self.accept_regex)
                            if out: payload = out; break

                roi.result = payload
                roi.state  = "ok" if payload else "fail"

                if payload and payload not in roi.last_printed:
                    roi.last_printed.add(payload)
                    raw = sanitize(payload)
                    print(f"[ROI#{idx} x={roi.x} y={roi.y} w={roi.w} h={roi.h}] RAW={raw} | ALNUM={alnum_only(raw)}", flush=True)

                if self.stop_event.is_set(): break

            elapsed = time.time()-t0
            if elapsed < self.period:
                time.sleep(self.period - elapsed)

# ---------- UI ----------
def draw_ui(base, editor: ROIEditor, msg=""):
    vis = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    for i, roi in enumerate(editor.rois):
        roi.draw(vis)
        cv2.putText(vis, f"#{i}", (roi.x+4, roi.y+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0,0), (vis.shape[1], 24), (30,30,30), -1)
    cv2.putText(vis, "LMB: draw/move/resize | C: clear | S: save | H: help | ESC: quit",
                (8,17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
    if msg:
        cv2.putText(vis, msg, (8, vis.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    return vis

def main(path):
    img = imread_gray(path)
    editor = ROIEditor(img)
    stop = threading.Event()
    # Optionally tighten acceptance: e.g. expect a 'K' field -> accept_regex=r"K\d|K[A-Z]"
    watcher = Watcher(img, editor, stop, hz=3, min_len=5, accept_regex=None)
    watcher.start()

    help_text = ("Hotkeys: LMB=draw/move | drag edge=resize | C=clear all | "
                 "S=save results to dm_results.txt | H=toggle help | ESC=quit")
    show_help=False
    while True:
        vis = draw_ui(img, editor, help_text if show_help else "")
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(15) & 0xFF
        if k == 27: break
        elif k in (ord('c'), ord('C')): editor.clear()
        elif k in (ord('h'), ord('H')): show_help = not show_help
        elif k in (ord('s'), ord('S')):
            lines=[]
            for i,r in enumerate(editor.rois):
                if r.result:
                    raw=sanitize(r.result)
                    lines.append(f"ROI#{i} RAW={raw} ALNUM={alnum_only(raw)}")
            with open("dm_results.txt","w",encoding="utf-8") as f:
                f.write("\n".join(lines) if lines else "# (no reads)\n")
            print("Saved -> dm_results.txt")
    stop.set(); watcher.join(timeout=0.5); cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python read_strict.py <image>")
        sys.exit(1)
    main(sys.argv[1])
