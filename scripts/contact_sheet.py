"""
contact_sheet.py — tile all RGB images into a few overview sheets for quick quality review.

Output: <dataset>/contact_00.png, contact_01.png, ... (~100 thumbnails each).

Run:  python3 contact_sheet.py [/path/to/dataset]   (default /data/dataset_500)
"""
import os, sys, glob
from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/dataset_500"
RGB = os.path.join(OUT, "rgb")
COLS, ROWS = 10, 10          # 100 thumbnails per page
TW, TH = 192, 108            # thumbnail size (16:9)
PAD = 2


def main():
    files = sorted(glob.glob(os.path.join(RGB, "*.jpg")))
    if not files:
        print("no images in", RGB); return
    per = COLS * ROWS
    pages = (len(files) + per - 1) // per
    for p in range(pages):
        sheet = Image.new("RGB", (COLS*(TW+PAD)+PAD, ROWS*(TH+PAD)+PAD), (30, 30, 30))
        for i, f in enumerate(files[p*per:(p+1)*per]):
            try:
                im = Image.open(f).convert("RGB").resize((TW, TH))
            except Exception:
                continue
            r, c = divmod(i, COLS)
            sheet.paste(im, (PAD + c*(TW+PAD), PAD + r*(TH+PAD)))
        out = os.path.join(OUT, f"contact_{p:02d}.png")
        sheet.save(out)
        print("saved", out)
    print(f"done — {len(files)} images in {pages} pages")


if __name__ == "__main__":
    main()