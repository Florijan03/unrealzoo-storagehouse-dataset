"""
clean_void.py — review/remove frames that contain a "void" (smooth bright out-of-world area).

Detection is appearance-based (large bright, texture-less region). Deleting removes the image,
its masks, and its entries in panoptic/legend/metadata so the dataset stays consistent
(gaps in numbering are fine for training).

Folder: /data/dataset_500 (or set env DATASET=/path)

Commands:
  list              -> print candidate frames + score (deletes nothing)
  review <thr>      -> montage of candidates with ids into void_candidates.png (deletes nothing)
  drop <id...>      -> delete exactly these frames
  auto <thr>        -> delete all frames above the threshold
"""
import os, sys, json, glob, math
import numpy as np
from PIL import Image, ImageDraw

OUT = os.environ.get("DATASET", "/data/dataset_500")
BRIGHT = 185      # brightness threshold for "bright"
FLAT = 6          # gradient threshold for "texture-less" (lower = smoother)


def void_score(jpg):
    g = np.asarray(Image.open(jpg).convert("L"), float)
    gx = np.abs(np.diff(g, axis=1)); gy = np.abs(np.diff(g, axis=0))
    grad = np.zeros_like(g); grad[:, :-1] += gx; grad[:-1, :] += gy
    return float(((g > BRIGHT) & (grad < FLAT)).mean())


def all_scores():
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "rgb", "*.jpg"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        rows.append((void_score(f), stem))
    rows.sort(reverse=True)
    return rows


def do_list():
    rows = all_scores()
    print(f"{'frame':8} void_score   (higher = more smooth bright void)")
    for s, stem in rows[:40]:
        flag = "  <-- candidate" if s > 0.03 else ""
        print(f"  {stem:8} {100*s:5.1f}%{flag}")
    cand = [stem for s, stem in rows if s > 0.03]
    print(f"\nCandidates (score > 3%): {len(cand)} -> {' '.join(cand)}")


def do_review(thr):
    rows = [(s, stem) for s, stem in all_scores() if s > thr]
    if not rows:
        print(f"no candidates above {100*thr:.0f}%."); return
    cols = 6
    tw, th, pad, lh = 300, 169, 4, 16
    n = len(rows); rn = math.ceil(n / cols)
    W = cols * (tw + pad) + pad
    H = rn * (th + lh + pad) + pad
    sheet = Image.new("RGB", (W, H), (25, 25, 25))
    draw = ImageDraw.Draw(sheet)
    for i, (s, stem) in enumerate(rows):
        try:
            im = Image.open(os.path.join(OUT, "rgb", f"{stem}.jpg")).convert("RGB").resize((tw, th))
        except Exception:
            continue
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + lh + pad)
        draw.text((x + 2, y + 1), f"{stem}  {100*s:.1f}%", fill=(255, 255, 0))
        sheet.paste(im, (x, y + lh))
    out = os.path.join(OUT, "void_candidates.png")
    sheet.save(out)
    ids = " ".join(stem for s, stem in rows)
    print(f"saved {out}  ({n} candidates, threshold {100*thr:.0f}%)")
    print("Look at that image, then delete the ones you confirm:")
    print(f"  python3 /data/clean_void.py drop {ids}")


def do_drop(ids):
    ids = set(ids)
    n = 0
    for sub, ext in [("rgb", "jpg"), ("romb_semantic", "png"), ("instance", "png"), ("panoptic", "png")]:
        for i in ids:
            p = os.path.join(OUT, sub, f"{i}.{ext}")
            if os.path.exists(p):
                os.remove(p); n += 1
    pj_path = os.path.join(OUT, "panoptic.json")
    left = "?"
    if os.path.exists(pj_path):
        pj = json.load(open(pj_path))
        pj["images"] = [im for im in pj["images"] if os.path.splitext(im["file_name"])[0] not in ids]
        pj["annotations"] = [an for an in pj["annotations"] if os.path.splitext(an["file_name"])[0] not in ids]
        json.dump(pj, open(pj_path, "w"), indent=2)
        left = len(pj["images"])
    lg_path = os.path.join(OUT, "romb_legend.json")
    if os.path.exists(lg_path):
        lg = json.load(open(lg_path))
        for i in ids:
            lg.pop(i, None)
        json.dump(lg, open(lg_path, "w"), indent=2)
    md_path = os.path.join(OUT, "metadata.json")
    if os.path.exists(md_path):
        md = [m for m in json.load(open(md_path)) if m.get("id") not in ids]
        json.dump(md, open(md_path, "w"), indent=2)
    print(f"deleted {len(ids)} frames ({n} files). Remaining images: {left}")
    print("Refresh preview:  python3 /data/contact_sheet.py /data/dataset_500")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "list"
    if mode == "list":
        do_list()
    elif mode == "review":
        do_review(float(sys.argv[2]) if len(sys.argv) > 2 else 0.03)
    elif mode == "drop":
        if len(sys.argv) < 3:
            print("provide ids: ... drop 000012 000047"); return
        do_drop(sys.argv[2:])
    elif mode == "auto":
        thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
        ids = [stem for s, stem in all_scores() if s > thr]
        print(f"AUTO threshold {100*thr:.0f}% -> {len(ids)}: {' '.join(ids)}")
        if ids:
            do_drop(ids)
    else:
        print("commands: list | review <thr> | drop <id...> | auto <thr>")


if __name__ == "__main__":
    main()