"""
generate_dataset.py (v1, superseded) — auto-generate TARGET images via corridor routes.

Drives the camera along fixed aisle lines at forklift height with jitter, varies exposure,
reshuffles the scene per variant (hides a subset of objects) and filters weak/dark/near-
duplicate frames. Superseded by the waypoint pipeline (record_waypoints.py + capture_dataset.py),
which gives much higher keep rates in dense scenes; kept for reference.

Run (background, survives logout):
  podman exec -d uz-scene bash -c "cd /data && nohup python3 generate_dataset.py > gen.log 2>&1"
  tail -f /mnt/sdc/fstankir/UnrealEnv/gen.log
"""

import os, json, time, socket, random, math
import numpy as np
from unrealcv import Client
import romb_export as R
import coco_panoptic as CP

# --------------------------- CONFIG ---------------------------
HOST, PORT, SOCK = "127.0.0.1", 9104, "/tmp/unrealcv_9104.socket"
CAM = 1
MAP = "Storagehouse"
OUT_DIR = "/data/dataset_coco"
TARGET = 550                       # how many useful images we want
SET_SIZE = (1920, 1080)            # 1080p (ROMB-match); None = leave as is
EXPOSURE_BIAS = [2, 3]             # brighter (was too dark before)

# aisle routes (Storagehouse measured: X -2173..903, Y -6560..287)
CORRIDORS_X = [-1800, -1300, -800, -300, 200, 650]
Y_MIN, Y_MAX = -6560 + 300, 287 - 300
STEP = 110                         # cm between poses along a route
Z_RANGE = (90, 140)                # forklift height
PITCH_RANGE = (-11, -3)
YAW_JITTER = 40                    # +/- degrees around travel direction

# scene randomization
SCENE_VARIANTS = 4                 # how many times to reshuffle the scene (hide a subset)
HIDE_PER_VARIANT = 50              # how many cargo/pallets to hide per variant

# filter + dedup
MIN_FOREGROUND = 0.12
MIN_CLASSES = 2
MAX_ONE_CLASS = 0.90
MIN_MOVE = 130                     # cm; skip if camera is closer than this to the last saved
MAX_ATTEMPTS = TARGET * 12
# --------------------------------------------------------------


def connect():
    try:
        socket.create_connection((HOST, PORT), timeout=5).close()
    except Exception:
        pass
    time.sleep(1)
    c = Client(SOCK, "unix"); c.connect()
    if not c.isconnected():
        raise RuntimeError("Cannot connect to UnrealCV.")
    return c


def make_pose_pool():
    poses = []
    for x in CORRIDORS_X:
        for direction, y0, y1, yaw in (("+", Y_MIN, Y_MAX, 90), ("-", Y_MAX, Y_MIN, 270)):
            n = int(abs(y1 - y0) / STEP)
            for i in range(n):
                y = y0 + (y1 - y0) * i / n
                poses.append((x, y, yaw))
    random.shuffle(poses)
    return poses


def quality_ok(semantic):
    ign = R.ROMB_IDX["ignore"]
    fg = (semantic != ign)
    if fg.mean() < MIN_FOREGROUND:
        return False
    vals, counts = np.unique(semantic[fg], return_counts=True)
    if len(vals) < MIN_CLASSES:
        return False
    if counts.max() / semantic.size > MAX_ONE_CLASS:
        return False
    return True


def main():
    random.seed(2026)
    c = connect()
    cur = c.request("vget /level/name") or ""
    if MAP and MAP.lower() not in cur.lower():
        print(f"Loading {MAP}...")
        c.request(f"vset /action/game/level {MAP}", timeout=180)
        time.sleep(30); c.disconnect(); c = connect()

    if SET_SIZE:
        c.request(f"vset /camera/{CAM}/size {SET_SIZE[0]} {SET_SIZE[1]}")
        time.sleep(1)

    objects = R.get_objects(c)
    print(f"Actors: {len(objects)} — reading colours...", flush=True)
    color_dict = R.get_color_dict(c, objects)
    print(f"Colours for {len(color_dict)} actors.", flush=True)

    # actors we may hide for randomization (cargo + pallets)
    movable = [o for o in objects
               if R.classify_to_romb(o) in ("cargo", "__PALLET__")]
    print(f"Hideable actors: {len(movable)}", flush=True)

    for sub in ("rgb", "romb_semantic", "instance"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    pano = CP.PanopticDataset(OUT_DIR)

    legend_all, meta = {}, []
    class_counts = {n: 0 for n in R.ROMB_CLASSES}
    saved = 0
    skipped = {"empty_or_weak": 0, "dup": 0}
    last_saved_xy = None
    attempts = 0

    for variant in range(SCENE_VARIANTS):
        if saved >= TARGET:
            break
        # scene randomization: hide a subset
        hidden = random.sample(movable, min(HIDE_PER_VARIANT, len(movable))) if movable else []
        for o in hidden:
            c.request(f"vset /object/{o}/hide")
        print(f"[variant {variant+1}/{SCENE_VARIANTS}] hidden {len(hidden)} actors", flush=True)

        for (x, y, yaw_base) in make_pose_pool():
            if saved >= TARGET or attempts >= MAX_ATTEMPTS:
                break
            attempts += 1
            z = random.uniform(*Z_RANGE)
            yaw = yaw_base + random.uniform(-YAW_JITTER, YAW_JITTER)
            pitch = random.uniform(*PITCH_RANGE)
            bias = random.choice(EXPOSURE_BIAS)

            c.request(f"vset /camera/{CAM}/location {x} {y} {z}")
            c.request(f"vset /camera/{CAM}/rotation {pitch} {yaw} 0")
            c.request(f"vrun r.DefaultFeature.AutoExposure.Bias {bias}")
            time.sleep(0.35)

            semantic, instance, legend = R.build_masks(c, CAM, objects, color_dict)
            if not quality_ok(semantic):
                skipped["empty_or_weak"] += 1
                continue
            if last_saved_xy is not None:
                dx, dy = x - last_saved_xy[0], y - last_saved_xy[1]
                if math.hypot(dx, dy) < MIN_MOVE:
                    skipped["dup"] += 1
                    continue

            name = f"{saved:06d}"
            R.save_semantic_png(semantic, os.path.join(OUT_DIR, "romb_semantic", f"{name}.png"))
            R.save_instance_png(instance, os.path.join(OUT_DIR, "instance", f"{name}.png"))
            R.save_lit(c, CAM, os.path.join(OUT_DIR, "rgb", f"{name}.jpg"))
            n_seg = pano.add(name, f"{name}.jpg", semantic, instance, legend)

            legend_all[name] = legend
            present = set(v["romb"] for v in legend.values())
            for cn in present:
                class_counts[cn] = class_counts.get(cn, 0) + 1
            meta.append({"id": name, "variant": variant, "x": round(x, 1), "y": round(y, 1),
                         "z": round(z, 1), "pitch": round(pitch, 1), "yaw": round(yaw, 1),
                         "bias": bias, "segments": n_seg})
            last_saved_xy = (x, y)
            saved += 1
            if saved % 25 == 0:
                print(f"  saved {saved}/{TARGET} (attempts {attempts}, skipped {skipped})", flush=True)

        # restore hidden actors
        for o in hidden:
            c.request(f"vset /object/{o}/show")

    pano.save_json()
    with open(os.path.join(OUT_DIR, "romb_legend.json"), "w") as f:
        json.dump(legend_all, f, indent=2)
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    c.request(f"vset /viewmode lit")
    c.disconnect()
    print(f"\nDONE: {saved} images (attempts {attempts}, skipped {skipped}).", flush=True)
    print("Images containing class:", flush=True)
    for cn, k in class_counts.items():
        if k:
            print(f"  {cn}: {k} ({100*k/max(saved,1):.0f}%)", flush=True)
    print("Output ->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()