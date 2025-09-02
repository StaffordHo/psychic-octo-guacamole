import sys, time, math, threading
import cv2, numpy as np
from collections import deque

# Optional ZXing (script still runs with libdmtx only)
try:
    import zxingcpp
    ZX_OK = True
except Exception:
    ZX_OK = False

from pylibdmtx.pylibdmtx import decode as dmtx_decode

WIN = "DM ROI Watch"

# ---------- helpers ----------
def imread_gray(p):
    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if g is None: raise SystemExit(f"Cannot open {p}")
    return g

def add_qz(img, px=10, val=255):
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=val)

def dpm_enhance(img):
    cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
    den = cv2.medianBlur(cla, 3)
    se9 = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    th = cv2.morphologyEx(den, cv2.MORPH_TOPHAT, se9)
    bh = cv2.morphologyEx(den, cv2.MORPH_BLACKHAT, se9)
    mix = cv2.addWeighted(th, 0.6, bh, 0.6, 0)
    se3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    return cv2.morphologyEx(mix, cv2.MORPH_CLOSE, se3, iterations=1)

# Replace non-printable control chars with Cognex-like tokens
CTRL_TOKENS = {
    0x1D: "<GS>",  # Group Separator
    0x1E: "<RS>",  # Record Separator
    0x1F: "<US>",  # Unit Separator
    0x04: "<EOT>", # End Of Transmission
    0x1C: "<FS>",  # File Separator
}
def sanitize(s: str) -> str:
    return "".join(CTRL_TOKENS.get(ord(ch), ch) if ord(ch) < 32 else ch for ch in s)

# ---------- decoders (Data Matrix only) ----------
def decode_zxing(arr):
    if not ZX_OK: return []
    try:
        res = zxingcpp.read_barcodes(arr, formats=zxingcpp.BarcodeFormat.DataMatrix, try_harder=True)
    except TypeError:
        res = zxingcpp.read_barcodes(arr)
    return [r.text for r in res] if res else []

def decode_libdmtx(arr, timeout_ms=220):
    for inv in (False, True):
        v = cv2.bitwise_not(arr) if inv else arr
        for k in (0,1,2,3):
            r = np.rot90(v, k)
            for shrink in (2,3):
                try:
                    out = dmtx_decode(r, timeout=timeout_ms, max_count=1, shrink=shrink, gap_size=2)
                except TypeError:
                    out = dmtx_decode(r, timeout=timeout_ms, max_count=1, shrink=shrink)
                if out:
                    return [out[0].data.decode("utf-8","ignore")]
    return []

def quick_views(square):
    base = square
    inv  = cv2.bitwise_not(base)
    thrA = cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    _, thrO = cv2.threshold(base,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return [add_qz(base,12,255), add_qz(base,12,0), add_qz(inv,12,255),
            add_qz(thrA,12,255), add_qz(thrO,12,255)]

def try_decode(square):
    for v in quick_views(square):
        hit = decode_zxing(v)
        if hit: return hit
    for v in quick_views(square)[:3]:
        hit = decode_libdmtx(v, timeout_ms=220)
        if hit: return hit
    return []

def heavy_decode(square, max_ms=1200):
    t0 = time.monotonic()
    def left(): return max_ms/1000.0 - (time.monotonic() - t0)

    bases = [square, dpm_enhance(square),
             cv2.resize(square, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_NEAREST)]
    for base in bases:
        if left() <= 0: return []
        for src in (base,
                    cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5),
                    cv2.threshold(base,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]):
            for qz_bg in (255, 0):
                if left() <= 0: return []
                z = add_qz(src, 12, qz_bg)
                hit = decode_zxing(z)
                if hit: return hit
        for src in (base, cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)):
            for qz_bg in (255, 0):
                if left() <= 0: return []
                z = add_qz(src, 12, qz_bg)
                for inv in (False, True):
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

# ---------- ROI model / UI ----------
class ROI:
    def __init__(self, x,y,w,h):
        self.x,self.y,self.w,self.h = int(x),int(y),int(w),int(h)
        self.result = None
        self.state  = "idle"   # idle|scanning|ok|fail
        self.hist   = deque(maxlen=6)
        self.last_printed = None

    def rect(self): return (self.x,self.y,self.w,self.h)
    def contains(self, px,py): return (self.x<=px<=self.x+self.w and self.y<=py<=self.y+self.h)
    def draw(self, frame):
        color = (0,255,0) if self.state=="ok" else (0,255,255) if self.state=="scanning" else (0,0,255)
        cv2.rectangle(frame,(self.x,self.y),(self.x+self.w,self.y+self.h),color,2)
        label = self.result[0] if (self.result and len(self.result)>0) else ("…" if self.state=="scanning" else "")
        if label:
            cv2.putText(frame, sanitize(label), (self.x, max(12,self.y-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

class ROIEditor:
    EDGE=6
    def __init__(self, img):
        self.img = img
        self.rois = []
        self.drag = None
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.setMouseCallback(WIN, self.on_mouse)

    def near_edge(self, roi, x,y):
        ex = ROIEditor.EDGE
        L = abs(x-roi.x)<=ex and roi.y-ex<=y<=roi.y+roi.h+ex
        R = abs(x-(roi.x+roi.w))<=ex and roi.y-ex<=y<=roi.y+roi.h+ex
        T = abs(y-roi.y)<=ex and roi.x-ex<=x<=roi.x+roi.w+ex
        B = abs(y-(roi.y+roi.h))<=ex and roi.x-ex<=x<=roi.x+roi.w+ex
        return 'resize_left' if L else 'resize_right' if R else 'resize_top' if T else 'resize_bottom' if B else None

    def on_mouse(self, event,x,y,flags,_p):
        if event==cv2.EVENT_LBUTTONDOWN:
            for roi in reversed(self.rois):
                mode = self.near_edge(roi,x,y)
                if mode: self.drag=(roi,mode,x,y); return
                if roi.contains(x,y): self.drag=(roi,'move',x-roi.x,y-roi.y); return
            self.drag=(ROI(x,y,1,1),'new',x,y); self.rois.append(self.drag[0])
        elif event==cv2.EVENT_MOUSEMOVE and self.drag:
            roi,mode,a,b = self.drag
            if   mode=='move': roi.x=max(0,x-a); roi.y=max(0,y-b)
            elif mode=='resize_left':  nx=min(x,roi.x+roi.w-4); roi.w+=roi.x-nx; roi.x=nx
            elif mode=='resize_right': roi.w=max(4,x-roi.x)
            elif mode=='resize_top':   ny=min(y,roi.y+roi.h-4); roi.h+=roi.y-ny; roi.y=ny
            elif mode=='resize_bottom':roi.h=max(4,y-roi.y)
            elif mode=='new':          roi.w=max(4,x-roi.x); roi.h=max(4,y-roi.y)
        elif event==cv2.EVENT_LBUTTONUP and self.drag:
            self.drag=None

    def clear(self): self.rois.clear()

# ---------- background decoding ----------
class Watcher(threading.Thread):
    def __init__(self, img, editor: ROIEditor, stop_event: threading.Event, hz=4):
        super().__init__(daemon=True)
        self.img, self.editor, self.stop_event = img, editor, stop_event
        self.period = 1.0/hz

    def run(self):
        while not self.stop_event.is_set():
            t0 = time.time()
            for idx, roi in enumerate(self.editor.rois):
                if roi.w < 12 or roi.h < 12: continue
                roi.state = "scanning"
                crop = self.img[roi.y:roi.y+roi.h, roi.x:roi.x+roi.w]
                up   = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)

                hit = try_decode(up)
                if not hit and idx == 0:
                    hit = heavy_decode(up, max_ms=1000)

                roi.result = hit if hit else None
                roi.state  = "ok" if hit else "fail"
                roi.hist.append(1 if hit else 0)

                # ---- PRINT NEW RESULTS TO CLI ----
                if hit:
                    s = sanitize(hit[0])
                    if s != roi.last_printed:
                        roi.last_printed = s
                        print(f"[ROI#{idx} at x={roi.x},y={roi.y},w={roi.w},h={roi.h}] {s}", flush=True)

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
    watcher = Watcher(img, editor, stop, hz=4)
    watcher.start()

    help_text = ("Hotkeys: LMB=draw/move | drag edge=resize | C=clear all | "
                 "S=save results to dm_results.txt | H=toggle help | ESC=quit")
    show_help = False

    while True:
        vis = draw_ui(img, editor, help_text if show_help else "")
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(15) & 0xFF
        if k == 27: break                    # ESC
        elif k in (ord('c'), ord('C')): editor.clear()
        elif k in (ord('h'), ord('H')): show_help = not show_help
        elif k in (ord('s'), ord('S')):
            lines=[]
            for i,r in enumerate(editor.rois):
                if r.result:
                    for t in r.result:
                        lines.append(f"ROI#{i} -> {sanitize(t)}")
            with open("dm_results.txt","w",encoding="utf-8") as f:
                f.write("\n".join(lines) if lines else "# (no reads)\n")
            print("Saved -> dm_results.txt")

    stop.set(); watcher.join(timeout=0.5); cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python dm_roi_watch_print.py <image>")
        sys.exit(1)
    main(sys.argv[1])
