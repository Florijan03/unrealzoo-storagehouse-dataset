"""
capture_dataset.py (server, in container) — generate ~500 labelled images with randomization.

From waypoints.json, for each scene variant it varies exposure and FOV and hides a random
subset of cargo/pallets, plus pose jitter around each waypoint. Keeps only good, fully
labelled frames (filters: void / blown-out / too dark / weak / too much ignore / duplicate).

Output (new folder):
  /data/dataset_500/  rgb/ romb_semantic/ instance/ panoptic/
                      panoptic.json romb_legend.json metadata.json

Run (background, survives logout):
  podman exec -d uz-scene bash -c "cd /data && nohup python3 capture_dataset.py > cap500.log 2>&1"
  tail -f /mnt/sdc/fstankir/UnrealEnv/cap500.log
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
OUT_DIR = "/data/dataset_500"
WAYPOINTS = "/data/waypoints.json"
TARGET = 500
SET_SIZE = (1920, 1080)

# scene randomization per variant / frame
SCENE_VARIANTS = 6          # how many times to reshuffle the scene (hide subset + new seed)
HIDE_FRACTION = 0.20        # fraction of cargo/pallets to hide per variant (occupancy variety)
PER_WP = 12                 # frames per waypoint within one variant
EXPOSURE_CHOICES = [1, 2, 3]      # vary lighting (AutoExposure bias)
FOV_CHOICES = [70, 90, 110]       # vary lens angle
JIT_XY, JIT_Z, JIT_YAW, JIT_PITCH = 40, 15, 15, 4

# quality filters
MIN_FOREGROUND = 0.15
MIN_CLASSES = 2
MAX_ONE_CLASS = 0.85
MAX_IGNORE = 0.05
VOID_MAX = 0.001
VOID_NAMES = ("DefaultPhysicsVolume", "PhysicsVolume")
BLOWN_MAX = 0.30
DARK_MIN_MEAN = 18
MIN_MOVE = 60               # dedup WITHIN a variant (same spot in another variant is fine)
MAX_ATTEMPTS = TARGET * 8
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
    if (lum > 250).mean() > BLOWN_MAX:      # too much white (blown out / skybox)
        return False
    if lum.mean() < DARK_MIN_MEAN:          # too dark
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
    random.seed(11)
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

    objects = R.get_objects(c)
    print(f"Actors: {len(objects)} — reading colours...", flush=True)
    color_dict = R.get_color_dict(c, objects)
    void_keys = void_keys_from(color_dict)
    movable = [o for o in objects if R.classify_to_romb(o) in ("cargo", "__PALLET__")]
    print(f"Colours: {len(color_dict)} | void actors: {len(void_keys)} | hideable (cargo/pallets): {len(movable)}", flush=True)

    for sub in ("rgb", "romb_semantic", "instance"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
    pano = CP.PanopticDataset(OUT_DIR)

    legend_all, meta = {}, []
    class_counts = {n: 0 for n in R.ROMB_CLASSES}
    saved, attempts = 0, 0
    skipped = {"blown_or_dark": 0, "void": 0, "weak": 0, "too_much_ignore": 0, "dup": 0}

    for variant in range(SCENE_VARIANTS):
        if saved >= TARGET or attempts >= MAX_ATTEMPTS:
            break
        hidden = random.sample(movable, int(HIDE_FRACTION * len(movable))) if movable else []
        for o in hidden:
            c.request(f"vset /object/{o}/hide")
        print(f"[variant {variant+1}/{SCENE_VARIANTS}] hidden {len(hidden)} actors", flush=True)

        var_positions = []      # dedup within variant
        plan = []
        for wp in wps:
            plan.append((wp, True))
            for _ in range(PER_WP - 1):
                plan.append((wp, False))
        random.shuffle(plan)

        for wp, exact in plan:
            if saved >= TARGET or attempts >= MAX_ATTEMPTS:
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

            if any(math.hypot(x - px, y - py) < MIN_MOVE for px, py in var_positions):
                skipped["dup"] += 1
                continue

            fov = random.choice(FOV_CHOICES)
            bias = random.choice(EXPOSURE_CHOICES)
            c.request(f"vset /camera/{CAM}/location {x:.1f} {y:.1f} {z:.1f}")
            c.request(f"vset /camera/{CAM}/rotation {pitch:.1f} {yaw:.1f} 0")
            c.request(f"vset /camera/{CAM}/fov {fov}")
            c.request(f"vrun r.DefaultFeature.AutoExposure.Bias {bias}")
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

            name = f"{saved:06d}"
            Image.fromarray(rgb).save(os.path.join(OUT_DIR, "rgb", f"{name}.jpg"), quality=92)
            R.save_semantic_png(semantic, os.path.join(OUT_DIR, "romb_semantic", f"{name}.png"))
            R.save_instance_png(instance, os.path.join(OUT_DIR, "instance", f"{name}.png"))
            n_seg = pano.add(name, f"{name}.jpg", semantic, instance, legend)

            legend_all[name] = legend
            for cn in set(v["romb"] for v in legend.values()):
                class_counts[cn] = class_counts.get(cn, 0) + 1
            meta.append({"id": name, "variant": variant, "x": round(x, 1), "y": round(y, 1),
                         "z": round(z, 1), "pitch": round(pitch, 1), "yaw": round(yaw, 1),
                         "fov": fov, "bias": bias, "segments": n_seg})
            var_positions.append((x, y))
            saved += 1
            if saved % 25 == 0:
                print(f"  saved {saved}/{TARGET} (attempts {attempts}, skipped {skipped})", flush=True)

        for o in hidden:
            c.request(f"vset /object/{o}/show")

    pano.save_json()
    json.dump(legend_all, open(os.path.join(OUT_DIR, "romb_legend.json"), "w"), indent=2)
    json.dump(meta, open(os.path.join(OUT_DIR, "metadata.json"), "w"), indent=2)
    c.request(f"vset /camera/{CAM}/fov 90")
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