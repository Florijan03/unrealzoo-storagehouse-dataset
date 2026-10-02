"""
viz_panoptic.py — turn COCO panoptic output into human-readable images.

The panoptic PNG is id-encoded (looks red/black to the eye). This writes two visualizations
per image:
  viz_instances/<i>.png  - a random colour per segment (see individual instances)
  viz_classes/<i>.png    - segments coloured by ROMB class (semantic sanity check)

Run:  python3 viz_panoptic.py [/path/to/dataset]   (default /data/dataset_500)
"""
import os, sys, json, random
import numpy as np
from PIL import Image
import romb_export as R

OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/dataset_500"


def rgb2id(a):
    return (a[..., 0].astype(np.uint32)
            + a[..., 1].astype(np.uint32) * 256
            + a[..., 2].astype(np.uint32) * 65536)


def main():
    j = json.load(open(os.path.join(OUT, "panoptic.json")))
    os.makedirs(os.path.join(OUT, "viz_instances"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "viz_classes"), exist_ok=True)

    for ann in j["annotations"]:
        fn = ann["file_name"]
        a = np.array(Image.open(os.path.join(OUT, "panoptic", fn)).convert("RGB"))
        idm = rgb2id(a)
        seg2cat = {s["id"]: s["category_id"] for s in ann["segments_info"]}
        inst = np.zeros_like(a)
        cls = np.zeros_like(a)
        for sid in np.unique(idm):
            if sid == 0:
                continue
            m = idm == sid
            random.seed(int(sid) * 2654435761 % (2 ** 32))
            inst[m] = [random.randint(50, 255) for _ in range(3)]
            cid = seg2cat.get(int(sid))
            if cid is not None:
                cls[m] = R.ROMB_PALETTE[cid]
        base = os.path.splitext(fn)[0]
        Image.fromarray(inst).save(os.path.join(OUT, "viz_instances", base + ".png"))
        Image.fromarray(cls).save(os.path.join(OUT, "viz_classes", base + ".png"))
    print(f"Done -> {OUT}/viz_instances, {OUT}/viz_classes  ({len(j['annotations'])} images)")


if __name__ == "__main__":
    main()