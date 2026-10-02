"""
coco_panoptic.py — export COCO panoptic segmentation (panopticapi format) from our masks.

"""

import os
import json
import numpy as np
from PIL import Image

import romb_export as R

# category_id = ROMB index (0..10); ignore(11) is void, not a category.
STUFF = {"drivable", "vertical"}
CATEGORIES = [
    {"id": R.ROMB_IDX[name], "name": name,
     "isthing": 0 if name in STUFF else 1,
     "color": list(R.ROMB_PALETTE[R.ROMB_IDX[name]])}   # panopticapi expects a per-category 'color'
    for name in R.ROMB_CLASSES
    if name != "ignore"
]


def id2rgb(idmap):
    """uint32 id map -> HxWx3 uint8 (COCO panoptic encoding)."""
    r = (idmap % 256).astype(np.uint8)
    g = ((idmap // 256) % 256).astype(np.uint8)
    b = ((idmap // 256 // 256) % 256).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _bbox(mask):
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def build_panoptic(semantic, instance, legend):
    """
    Return (panoptic_rgb HxWx3 uint8, segments_info list).
    - things: each instance (from `instance`) is its own segment
    - stuff: all pixels of a class (from `semantic`) form one segment
    """
    h, w = semantic.shape
    pan = np.zeros((h, w), dtype=np.uint32)  # segment id per pixel (0 = void)
    segments_info = []
    seg = 0

    # THINGS — per instance
    for inst_id, info in legend.items():
        cls = info["romb"]
        if cls in STUFF:
            continue
        m = instance == inst_id
        if not m.any():
            continue
        seg += 1
        pan[m] = seg
        segments_info.append({
            "id": seg,
            "category_id": int(info["romb_idx"]),
            "area": int(m.sum()),
            "bbox": _bbox(m),
            "iscrowd": 0,
        })

    # STUFF — per class (single segment)
    for cls in STUFF:
        cidx = R.ROMB_IDX[cls]
        m = (semantic == cidx) & (pan == 0)  # only still-unassigned pixels
        if not m.any():
            continue
        seg += 1
        pan[m] = seg
        segments_info.append({
            "id": seg,
            "category_id": int(cidx),
            "area": int(m.sum()),
            "bbox": _bbox(m),
            "iscrowd": 0,
        })

    return id2rgb(pan), segments_info


def save_panoptic_png(panoptic_rgb, path):
    Image.fromarray(panoptic_rgb, mode="RGB").save(path)


class PanopticDataset:
    """Accumulates images/annotations and writes panoptic.json."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.pan_dir = os.path.join(out_dir, "panoptic")
        os.makedirs(self.pan_dir, exist_ok=True)
        self.images = []
        self.annotations = []

    def add(self, frame_name, rgb_file, semantic, instance, legend):
        h, w = semantic.shape
        img_id = len(self.images) + 1
        pan_rgb, seg_info = build_panoptic(semantic, instance, legend)
        pan_file = f"{frame_name}.png"
        save_panoptic_png(pan_rgb, os.path.join(self.pan_dir, pan_file))
        self.images.append({"id": img_id, "file_name": rgb_file, "width": w, "height": h})
        self.annotations.append({"image_id": img_id, "file_name": pan_file, "segments_info": seg_info})
        return len(seg_info)

    def save_json(self, path=None):
        path = path or os.path.join(self.out_dir, "panoptic.json")
        with open(path, "w") as f:
            json.dump({"images": self.images, "annotations": self.annotations,
                       "categories": CATEGORIES}, f, indent=2)
        return path