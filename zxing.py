import sys, cv2, zxingcpp
from PIL import Image

def read_dm_with_zxing(gray):
    # Prefer Data Matrix only (faster). If your wrapper is older and
    # doesn't accept keyword args, the except block falls back.
    try:
        results = zxingcpp.read_barcodes(
            gray, formats=zxingcpp.BarcodeFormat.DataMatrix
        )
    except TypeError:
        results = zxingcpp.read_barcodes(gray)

    if not results:
        # Some older wheels prefer PIL images
        pil = Image.fromarray(gray)
        try:
            results = zxingcpp.read_barcodes(
                pil, formats=zxingcpp.BarcodeFormat.DataMatrix
            )
        except TypeError:
            results = zxingcpp.read_barcodes(pil)

    return [r.text for r in results] if results else []

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zxing_dm_reader.py IMG")
        sys.exit(1)

    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Cannot open {sys.argv[1]}")

    hits = read_dm_with_zxing(img)
    print(hits if hits else "No codes decoded.")
