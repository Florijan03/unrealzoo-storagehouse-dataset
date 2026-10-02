"""
romb_export.py — build ROMB semantic + instance masks from a UnrealCV object_mask render.

Per frame it produces:
  - instance mask  (16-bit PNG, one unique id per actor)
  - ROMB semantic  (8-bit indexed PNG, class indices 0-10, ignore=11)

Key points: use camera 1 (FusionCam); read each actor's unique object_mask colour once
(colours are stable), then decode a single object_mask render into both instance and
semantic masks. Holds the ROMB class scheme, palette and UE-mesh -> ROMB-class rules.
"""

import os
import re
import numpy as np
from PIL import Image

# --------------------- ROMB scheme (11 classes + ignore) ---------------------
# Official scheme from romb.yaml: classes 0-10, ignore_id = 11.
ROMB_CLASSES = [
    "drivable",          # 0
    "person",            # 1
    "pallet-face",       # 2
    "pallet-empty",      # 3
    "pallet-full",       # 4
    "cargo",             # 5
    "vehicle-forklift",  # 6
    "vehicle-other",     # 7
    "ego-forklift",      # 8
    "other-object",      # 9
    "vertical",          # 10
    "ignore",            # 11
]
ROMB_IDX = {n: i for i, n in enumerate(ROMB_CLASSES)}
# Palette from romb.yaml (0-10); ignore (11) = black.
ROMB_PALETTE = [
    (65, 97, 101), (211, 63, 73), (255, 119, 61), (242, 226, 159), (255, 200, 87),
    (173, 191, 151), (0, 121, 145), (34, 46, 80), (53, 45, 57),
    (193, 180, 174), (224, 226, 219), (0, 0, 0),
]

# --------------------- UE name -> ROMB class mapping ---------------------
CLASSIFY_RULES = [
    (r"SM_floor", "drivable"),
    (r"FloorDecal", "drivable"),
    (r"SM_Palette", "__PALLET__"),
    (r"SM_CardBox", "cargo"),
    (r"SM_CratePlastic", "cargo"),
    (r"SM_CartonDrawer", "cargo"),
    (r"SM_Barel", "cargo"),
    (r"SM_BottlePlastic", "cargo"),
    (r"SM_Book", "cargo"),
    (r"SM_PaperNote", "cargo"),
    (r"SM_Paper_Shortcut", "cargo"),
    (r"SM_Rack", "vertical"),
    (r"SM_Beam", "vertical"),
    (r"SM_Wall", "vertical"),
    (r"SM_Pipe", "vertical"),
    (r"SM_FireExtinguisher", "other-object"),
    (r"SM_Sign", "other-object"),
    (r"Character|Manny|Quinn|SK_|Human|Person", "person"),
    (r"Cart|Trolley|HandTruck|PalletJack|Dolly", "vehicle-other"),
    # "out of world" / void: when the camera peeks outside the geometry the object_mask
    # returns the default physics volume colour -> treat as ignore (void), not other-object.
    (r"DefaultPhysicsVolume", "ignore"),
    (r"PhysicsVolume", "ignore"),
    (r"Light", "ignore"),
    (r"ReflectionCapture", "ignore"),
    (r"DirtStain", "ignore"),
]
MISSING_CLASSES = ["vehicle-forklift", "ego-forklift"]  # no powered forklift in the scene


def classify_to_romb(name):
    for pat, cls in CLASSIFY_RULES:
        if re.search(pat, name, re.IGNORECASE):
            return cls
    return "other-object"   # unknown physical object -> other-object (scheme has no generic 'other')


def classify_pallet(name, client=None):
    """v1: all pallets -> 'pallet-full'. TODO: empty/full from bounds, face from rotation."""
    return "pallet-full"


# --------------------- low-level UnrealCV commands ---------------------
def get_objects(client):
    res = client.request("vget /objects")
    return res.split() if res else []


def _parse_color(res):
    if not res:
        return None
    nums = re.findall(r"\d+", res)
    return (int(nums[0]), int(nums[1]), int(nums[2])) if len(nums) >= 3 else None


def get_color_dict(client, objects):
    """Unique object_mask colour per actor. Done once (colours are stable)."""
    d = {}
    for o in objects:
        c = _parse_color(client.request(f"vget /object/{o}/color"))
        if c is not None:
            d[o] = c
    return d


def get_image(client, cam_id, mode, tmp_dir="/tmp"):
    """Proven file-based capture: set viewmode, let UE write a PNG, read it back."""
    client.request(f"vset /viewmode {mode}")
    out = os.path.join(tmp_dir, f"_uecap_{mode}.png")
    res = client.request(f"vget /camera/{cam_id}/{mode} {out}")
    path = res if (isinstance(res, str) and os.path.isfile(res)) else out
    if not os.path.isfile(path):
        raise RuntimeError(f"UnrealCV did not write {mode}; response: {str(res)[:120]}")
    return np.array(Image.open(path).convert("RGB"))


# --------------------- mask building ---------------------
def build_masks(client, cam_id, objects, color_dict, min_pixels=1, om=None):
    """
    Return (semantic uint8, instance uint16, legend).
    Only processes colours actually present in the frame, so the legend and counts are real.
    Instance ids are per-frame (order of appearance); legend[id]['obj'] links to the actor.
    om: optional pre-captured object_mask (HxWx3) to avoid rendering twice.
    """
    if om is None:
        om = get_image(client, cam_id, "object_mask")
    h, w = om.shape[:2]
    key = (om[..., 0].astype(np.uint32) << 16) | (om[..., 1].astype(np.uint32) << 8) | om[..., 2].astype(np.uint32)

    # colour (key) -> actor name
    keymap = {}
    for obj, col in color_dict.items():
        keymap[(int(col[0]) << 16) | (int(col[1]) << 8) | int(col[2])] = obj

    semantic = np.full((h, w), ROMB_IDX["ignore"], dtype=np.uint8)
    instance = np.zeros((h, w), dtype=np.uint16)
    legend = {}
    inst = 0
    for k in np.unique(key):
        obj = keymap.get(int(k))
        if obj is None:
            continue  # unknown colour / background -> stays ignore
        m = key == k
        px = int(m.sum())
        if px < min_pixels:
            continue
        cls = classify_to_romb(obj)
        if cls == "__PALLET__":
            cls = classify_pallet(obj, client)
        if cls == "ignore":
            continue  # stays ignore, no instance id
        inst += 1
        romb_i = ROMB_IDX.get(cls, ROMB_IDX["other-object"])
        instance[m] = inst
        semantic[m] = romb_i
        legend[inst] = {"obj": obj, "romb": cls, "romb_idx": int(romb_i), "pixels": px}
    return semantic, instance, legend


# --------------------- saving (ROMB format) ---------------------
def save_semantic_png(semantic, path):
    im = Image.fromarray(semantic, mode="P")
    pal = []
    for (r, g, b) in ROMB_PALETTE:
        pal += [r, g, b]
    pal += [0, 0, 0] * (256 - len(ROMB_PALETTE))
    im.putpalette(pal)
    im.save(path)


def save_instance_png(instance, path):
    Image.fromarray(instance, mode="I;16").save(path)


def save_lit(client, cam_id, path):
    Image.fromarray(get_image(client, cam_id, "lit")).save(path)