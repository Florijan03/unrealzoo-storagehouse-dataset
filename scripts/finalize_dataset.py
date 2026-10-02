"""
finalize_dataset.py — convert dataset_500 into the ROMB dataset layout.

Output (mirrors the ROMB structure; our resolution is 1080p):
  /data/dataset_romb/1080p/
    rgb/<stem>.jpg          RGB
    labels/<stem>.png       semantic mask (8-bit indices 0-10, ignore=11)  <- ROMB 'labels'
    instance/<stem>.png     instance mask (16-bit)       [extra, not in the plain ROMB set]
    panoptic/<stem>.png     COCO panoptic (id-encoded)   [extra]
    panoptic.json           COCO panoptic annotations    [extra]
    romb.yaml               names/colors/ignore_id (ROMB)
    train.txt  val.txt      lists of <stem>.png (85/15 split)
    README.md

Run:
  python3 finalize_dataset.py                        # defaults below
  python3 finalize_dataset.py /data/dataset_500 /data/dataset_romb
"""
import os, sys, json, glob, shutil, random

SRC = sys.argv[1] if len(sys.argv) > 1 else "/data/dataset_500"
OUT_ROOT = sys.argv[2] if len(sys.argv) > 2 else "/data/dataset_romb"
RES = "1080p"
VAL_FRAC = 0.15
SEED = 42

NAMES = {0: "drivable", 1: "person", 2: "pallet-face", 3: "pallet-empty", 4: "pallet-full",
         5: "cargo", 6: "vehicle-forklift", 7: "vehicle-other", 8: "ego-forklift",
         9: "other-object", 10: "vertical"}
COLORS = {0: [65, 97, 101], 1: [211, 63, 73], 2: [255, 119, 61], 3: [242, 226, 159],
          4: [255, 200, 87], 5: [173, 191, 151], 6: [0, 121, 145], 7: [34, 46, 80],
          8: [53, 45, 57], 9: [193, 180, 174], 10: [224, 226, 219]}
# Classes with no examples in the synthetic Storagehouse scene:
MISSING = [2, 3, 6, 8]   # pallet-face, pallet-empty, vehicle-forklift, ego-forklift


def main():
    dst = os.path.join(OUT_ROOT, RES)
    for sub in ("rgb", "labels", "instance", "panoptic"):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)

    stems = sorted(os.path.splitext(os.path.basename(f))[0]
                   for f in glob.glob(os.path.join(SRC, "rgb", "*.jpg")))
    if not stems:
        print("No images in", os.path.join(SRC, "rgb")); return
    print(f"Found {len(stems)} images in {SRC}", flush=True)

    # copy
    n_inst = n_pan = 0
    for s in stems:
        shutil.copy(os.path.join(SRC, "rgb", f"{s}.jpg"), os.path.join(dst, "rgb", f"{s}.jpg"))
        shutil.copy(os.path.join(SRC, "romb_semantic", f"{s}.png"), os.path.join(dst, "labels", f"{s}.png"))
        pi = os.path.join(SRC, "instance", f"{s}.png")
        if os.path.exists(pi):
            shutil.copy(pi, os.path.join(dst, "instance", f"{s}.png")); n_inst += 1
        pp = os.path.join(SRC, "panoptic", f"{s}.png")
        if os.path.exists(pp):
            shutil.copy(pp, os.path.join(dst, "panoptic", f"{s}.png")); n_pan += 1
    for extra in ("panoptic.json", "romb_legend.json", "metadata.json"):
        p = os.path.join(SRC, extra)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, extra))

    # train/val split (deterministic)
    rng = random.Random(SEED)
    order = stems[:]; rng.shuffle(order)
    n_val = max(1, int(len(order) * VAL_FRAC))
    val = set(order[:n_val]); train = [s for s in stems if s not in val]
    val_list = [s for s in stems if s in val]
    with open(os.path.join(dst, "train.txt"), "w") as f:
        f.write("\n".join(f"{s}.png" for s in train) + "\n")
    with open(os.path.join(dst, "val.txt"), "w") as f:
        f.write("\n".join(f"{s}.png" for s in val_list) + "\n")

    # romb.yaml (mirror the ROMB convention)
    lines = [f"path: ./{RES} # dataset root directory", "train: train.txt", "val: val.txt", "",
             "# Classes", "names:"]
    for i in range(11):
        lines.append(f"  {i}: {NAMES[i]}")
    lines += ["", "colors:"]
    for i in range(11):
        r, g, b = COLORS[i]
        lines.append(f"  {i}: [ {r}, {g}, {b} ]")
    lines += ["", "ignore_id: 11", ""]
    with open(os.path.join(dst, "romb.yaml"), "w") as f:
        f.write("\n".join(lines))

    # class presence (from romb_legend.json if present)
    stats = {}
    lgp = os.path.join(SRC, "romb_legend.json")
    if os.path.exists(lgp):
        leg = json.load(open(lgp))
        for frame, insts in leg.items():
            for v in set(x["romb"] for x in insts.values()):
                stats[v] = stats.get(v, 0) + 1

    # README
    rd = [f"# Synthetic warehouse dataset (ROMB format) — {len(stems)} images",
          "",
          "Synthetic warehouse dataset generated from the UnrealZoo/UnrealCV **Storagehouse**",
          "scene for autonomous-forklift perception. Labels follow the ROMB scheme (11 classes + ignore).",
          "",
          "## Structure",
          "```",
          f"{RES}/",
          "  rgb/<id>.jpg         RGB (1920x1080)",
          "  labels/<id>.png      semantic mask (8-bit indices 0-10, ignore=11) — ROMB 'labels'",
          "  instance/<id>.png    instance mask (16-bit)         [extra]",
          "  panoptic/<id>.png    COCO panoptic (id-encoded)     [extra]",
          "  panoptic.json        COCO panoptic annotations      [extra]",
          "  romb.yaml            classes/colors/ignore_id (ROMB)",
          "  train.txt val.txt    split (85/15)",
          "```",
          "",
          f"Train: {len(train)} | Val: {len(val_list)} | Resolution: 1920x1080",
          "",
          "## Classes and coverage (number of images containing the class)"]
    for i in range(11):
        nm = NAMES[i]
        c = stats.get(nm, 0)
        note = "  (absent in scene)" if i in MISSING else ""
        rd.append(f"- {i} {nm}: {c}{note}")
    rd += ["",
           "Note: pallet-face, pallet-empty, vehicle-forklift and ego-forklift do not exist in this",
           "scene (no powered forklift, no empty/face pallets), so they are empty.",
           "",
           "## How it was generated",
           "record_waypoints.py (manual position picking) -> capture_dataset.py (lighting/FOV",
           "randomization + cargo hiding + quality filters) -> clean_void.py (cleanup).",
           "UE mesh -> ROMB class mapping lives in romb_export.py (CLASSIFY_RULES)."]
    with open(os.path.join(dst, "README.md"), "w") as f:
        f.write("\n".join(rd) + "\n")

    print(f"DONE -> {dst}", flush=True)
    print(f"  rgb/labels: {len(stems)} | instance: {n_inst} | panoptic: {n_pan}", flush=True)
    print(f"  train: {len(train)} | val: {len(val_list)}", flush=True)
    print(f"  romb.yaml, train.txt, val.txt, README.md written.", flush=True)


if __name__ == "__main__":
    main()