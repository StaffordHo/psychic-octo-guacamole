from pylibdmtx.pylibdmtx import decode as dmtx_decode
from PIL import Image
import cv2, numpy as np, os, sys, glob

# ---------- utilities ----------
def pil(img_np): return Image.fromarray(img_np)

def uniq(xs):
    out, seen = [], set()
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def rotations_and_polarities(img_bin, timeout=15000):
    outs=[]
    for inv in (False, True):
        v = cv2.bitwise_not(img_bin) if inv else img_bin
        for k in range(4):  # 0/90/180/270
            rot = np.rot90(v, k)
            res = dmtx_decode(pil(rot), timeout=timeout, max_count=8)
            if res:
                outs.extend(r.data.decode("utf-8","ignore") for r in res)
    return outs

def preprocess_variants(gray):
    """Yield a stream of binarized variants (and keep a few raw)."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    base0 = cv2.equalizeHist(gray)
    base1 = clahe.apply(gray)
    bases = [gray, base0, base1, cv2.medianBlur(base0,3)]
    # tophat to pop dot-peen dots
    bases += [cv2.morphologyEx(base0, cv2.MORPH_TOPHAT,
                               cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)))]
    for b in bases:
        yield ("raw", b)

    # adaptive threshold sweeps + light morphology
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    for src in (base0, base1):
        for block in (29,31,35,41):
            for C in (3,5,7):
                thr = cv2.adaptiveThreshold(src,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, block, C)
                thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k3)
                thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k3)
                yield (f"adb{block}_C{C}", thr)
        _, otsu = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        yield ("otsu", otsu)

def forced_top_center_rois(gray):
    """Heuristic for your images: two codes at top center."""
    H,W = gray.shape
    # focus on the bright plate area at top center
    x0 = int(0.20*W); x1 = int(0.80*W)
    y0 = int(0.05*H); y1 = int(0.55*H)
    plate = gray[y0:y1, x0:x1]
    # split into left/right halves
    mid = (x1-x0)//2
    left  = plate[:, :mid]
    right = plate[:, mid:]
    # add small borders (quiet zone)
    pad = 8
    left  = cv2.copyMakeBorder(left,  pad,pad,pad,pad,  cv2.BORDER_REPLICATE)
    right = cv2.copyMakeBorder(right, pad,pad,pad,pad, cv2.BORDER_REPLICATE)
    return [plate, left, right]

def auto_square_rois(gray, max_rois=6):
    """Find square-ish candidates; often isolates a DM block."""
    H,W = gray.shape
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, bw = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        a=w*h
        if a<0.004*W*H or a>0.60*W*H: continue
        ar=w/float(h)
        if 0.6<=ar<=1.4:
            pad=int(0.08*max(w,h))
            x0=max(0,x-pad); y0=max(0,y-pad)
            x1=min(W,x+w+pad); y1=min(H,y+h+pad)
            rois.append(gray[y0:y1, x0:x1])
    rois.sort(key=lambda r:r.shape[0]*r.shape[1], reverse=True)
    return rois[:max_rois]

def grid_rois(gray, nx=2, ny=1):
    """Last-resort tiling (helps when two symbols overlap in search)."""
    H,W = gray.shape
    tiles=[]
    for j in range(ny):
        for i in range(nx):
            x0=i*W//nx; x1=(i+1)*W//nx
            y0=j*H//ny; y1=(j+1)*H//ny
            tiles.append(gray[y0:y1, x0:x1])
    return tiles

def try_decode(gray):
    results=[]
    # upscale for pixels/module
    for scale in (2,3,4):
        up = cv2.resize(gray, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
        for tag, var in preprocess_variants(up):
            outs = rotations_and_polarities(var)
            if outs: results.extend(outs)
    return uniq(results)

def decode_image(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[WARN] cannot open {path}"); return []

    # Candidates: whole, forced top-center (your case), auto square-ish, left/right grid
    cands = [gray]
    cands += forced_top_center_rois(gray)
    cands += auto_square_rois(gray)
    cands += grid_rois(gray, nx=2, ny=1)

    all_out=[]
    for roi in cands:
        out = try_decode(roi)
        if out: all_out += out

    final = uniq(all_out)
    if final:
        print(f"[{os.path.basename(path)}] -> {final}")
    else:
        print(f"[{os.path.basename(path)}] No codes decoded.")
    return final

def collect_images(args):
    files=[]
    for a in args:
        if os.path.isdir(a): files += glob.glob(os.path.join(a,"*.*"))
        else:                files += glob.glob(a)
    exts={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
    return [f for f in files if os.path.splitext(f)[1].lower() in exts]

if __name__=="__main__":
    imgs = collect_images(sys.argv[1:] or [])
    if not imgs:
        print("usage: python decode.py <image(s) or folder or *.png>")
        sys.exit(1)
    for p in imgs:
        decode_image(p)
