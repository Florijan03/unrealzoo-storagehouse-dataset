"""
capture_from_waypoints.py (superseded) — single-pass capture around recorded waypoints.

From waypoints.json it jitters a few poses around each waypoint and saves only good, fully
labelled frames (void / blown-out / too dark / weak / too much ignore / duplicate filters).
Superseded by capture_dataset.py, which adds lighting/FOV randomization and scene variants.

Output: /data/dataset_v2/  rgb/ romb_semantic/ instance/ panoptic/ + json files

Run (background):
  podman exec -d uz-scene bash -c "cd /data && nohup python3 capture_from_waypoints.py > cap.log 2>&1"
  tail -f /mnt/sdc/fstankir/UnrealEnv/cap.log
"""
import os, json, time, socket, random, math
import numpy as np
from PIL import Image
from unrealcv import Client
import romb_export as R
import coco_panoptic as CP

# --------------------------- CONFIG ---------------------------
HOST, PORT, SOCK = "127.0.0.1", 9104, "/tmp/unrealcv_9104.socket"
CAM = 1
MAP = "Storagehouse"
OUT_DIR = "/data/dataset_v2"
WAYPOINTS = "/data/waypoints.json"
TARGET = 100
SET_SIZE = (1920, 1080)
EXPOSURE_BIAS = 2

# jitter around each waypoint
PER_WP = 12           # attempts per waypoint (incl. the exact pose)
JIT_XY = 40           # cm
JIT_Z = 15            # cm
JIT_YAW = 15          # degrees
JIT_PITCH = 4         # degrees

# quality filters
MIN_FOREGROUND = 0.15
MIN_CLASSES = 2
MAX_ONE_CLASS = 0.85
MAX_IGNORE = 0.05
VOID_MAX = 0.001
VOID_NAMES = ("DefaultPhysicsVolume", "PhysicsVolume")
BLOWN_MAX = 0.30
DARK_MIN_MEAN = 18
MIN_MOVE = 60
# --------------------------------------------------------------


def connect():
    try:
        socket.create_connection((HOST, PORT), timeout=5).close()
    except Exception:
        pass
    time.sleep(1)
    c = Client(SOCK, "unix")
    c.connect()
    if not c.isconnected():
        raise RuntimeError("Cannot connect to UnrealCV.")
    return c


def void_keys_from(color_dict):
    keys = set()
    for obj, col in color_dict.items():
        if any(v.lower() in obj.lower() for v in VOID_NAMES):
            keys.add((int(col[0]) << 16) | (int(col[1]) << 8) | int(col[2]))
    return keys


def void_fraction(om, void_keys):
    if not void_keys:
        return 0.0
    key = (om[..., 0].astype(np.uint32) << 16) | (om[..., 1].astype(np.uint32) << 8) | om[..., 2].astype(np.uint32)
    return float(np.isin(key, list(void_keys)).mean())


def rgb_ok(rgb):
    lum = rgb.mean(axis=2)
    if (lum > 250).mean() > BLOWN_MAX:
        return False
    if lum.mean() < DARK_MIN_MEAN:
        return False
    return True


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
    random.seed(7)
    wps = json.load(open(WAYPOINTS))
    print(f"waypoints: {len(wps)}", flush=True)
    if not wps:
        print("waypoints.json is empty — record positions first."); return

    c = connect()
    cur = c.request("vget /level/name") or ""
    if MAP and MAP.lower() not in cur.lower():
        print(f"Loading {MAP}...", flush=True)
        c.request(f"vset /action/game/level {MAP}", timeout=180)
        time.sleep(30); c.disconnect(); c = connect()

    if SET_SIZE:
        c.request(f"vset /camera/{CAM}/size {SET_SIZE[0]} {SET_SIZE[1]}"); time.sleep(1)
    c.request(f"vrun r.DefaultFeature.AutoExposure.Bias {EXPOSURE_BIAS}")

    objects = R.get_objects(c)
    print(f"Actors: {len(objects)} — reading colours...", flush=True)
    color_dict = R.get_color_dict(c, objects)
    print(f"Colours for {len(color_dict)} actors.", flush=True)
    void_keys = void_keys_from(color_dict)
    print(f"Void actors (DefaultPhysicsVolume etc.): {len(void_keys)}", flush=True)

    for sub in ("rgb", "romb_semantic", "instance"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    pano = CP.PanopticDataset(OUT_DIR)

    legend_all, meta = {}, []
    class_counts = {n: 0 for n in R.ROMB_CLASSES}
    saved, attempts = 0, 0
    skipped = {"blown_or_dark": 0, "void": 0, "weak": 0, "too_much_ignore": 0, "dup": 0}
    last_xy = None

    # plan: exact pose + (PER_WP-1) jitter variants per waypoint, then shuffle
    plan = []
    for wp in wps:
        plan.append((wp, True))
        for _ in range(PER_WP - 1):
            plan.append((wp, False))
    random.shuffle(plan)

    for wp, exact in plan:
        if saved >= TARGET:
            break
        attempts += 1
        if exact:
            x, y, z = wp["x"], wp["y"], wp["z"]
            pitch, yaw = wp["pitch"], wp["yaw"]
        else:
            x = wp["x"] + random.uniform(-JIT_XY, JIT_XY)
            y = wp["y"] + random.uniform(-JIT_XY, JIT_XY)
            z = wp["z"] + random.uniform(-JIT_Z, JIT_Z)
            yaw = wp["yaw"] + random.uniform(-JIT_YAW, JIT_YAW)
            pitch = wp["pitch"] + random.uniform(-JIT_PITCH, JIT_PITCH)

        c.request(f"vset /camera/{CAM}/location {x:.1f} {y:.1f} {z:.1f}")
        c.request(f"vset /camera/{CAM}/rotation {pitch:.1f} {yaw:.1f} 0")
        time.sleep(0.3)

        rgb = R.get_image(c, CAM, "lit")
        if not rgb_ok(rgb):
            skipped["blown_or_dark"] += 1
            continue
        om = R.get_image(c, CAM, "object_mask")
        if void_fraction(om, void_keys) > VOID_MAX:
            skipped["void"] += 1
            continue
        semantic, instance, legend = R.build_masks(c, CAM, objects, color_dict, om=om)
        if not quality_ok(semantic):
            skipped["weak"] += 1
            continue
        if (semantic == R.ROMB_IDX["ignore"]).mean() > MAX_IGNORE:
            skipped["too_much_ignore"] += 1
            continue
        if last_xy is not None and math.hypot(x - last_xy[0], y - last_xy[1]) < MIN_MOVE:
            skipped["dup"] += 1
            continue

        name = f"{saved:06d}"
        Image.fromarray(rgb).save(os.path.join(OUT_DIR, "rgb", f"{name}.jpg"), quality=92)
        R.save_semantic_png(semantic, os.path.join(OUT_DIR, "romb_semantic", f"{name}.png"))
        R.save_instance_png(instance, os.path.join(OUT_DIR, "instance", f"{name}.png"))
        n_seg = pano.add(name, f"{name}.jpg", semantic, instance, legend)

        legend_all[name] = legend
        for cn in set(v["romb"] for v in legend.values()):
            class_counts[cn] = class_counts.get(cn, 0) + 1
        meta.append({"id": name, "x": round(x, 1), "y": round(y, 1), "z": round(z, 1),
                     "pitch": round(pitch, 1), "yaw": round(yaw, 1), "exact": exact, "segments": n_seg})
        last_xy = (x, y)
        saved += 1
        if saved % 10 == 0:
            print(f"  saved {saved}/{TARGET} (attempts {attempts}, skipped {skipped})", flush=True)

    pano.save_json()
    json.dump(legend_all, open(os.path.join(OUT_DIR, "romb_legend.json"), "w"), indent=2)
    json.dump(meta, open(os.path.join(OUT_DIR, "metadata.json"), "w"), indent=2)
    c.request("vset /viewmode lit")
    c.disconnect()
    print(f"\nDONE: {saved} images (attempts {attempts}, skipped {skipped}).", flush=True)
    print("Images containing class:", flush=True)
    for cn, k in class_counts.items():
        if k:
            print(f"  {cn}: {k} ({100*k/max(saved,1):.0f}%)", flush=True)
    print("Output ->", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()