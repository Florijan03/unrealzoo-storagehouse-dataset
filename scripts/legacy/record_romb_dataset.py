"""
record_romb_dataset.py — minimal recorder (a few fixed poses) for a quick sanity check.

Connects to a running UnrealZoo binary (unix socket, port 9104), loads Storagehouse, and
for each pose saves RGB + ROMB semantic + instance mask. Superseded for real datasets by
capture_dataset.py; kept as a simple smoke test.

Run (inside the container, scene must be up ~5-15 min):
  podman exec uz-scene python3 /data/record_romb_dataset.py
"""

import os
import time
import json
import socket
from unrealcv import Client

import romb_export as R

# --------------------------- CONFIG ---------------------------
HOST, PORT = "127.0.0.1", 9104
SOCK_PATH = "/tmp/unrealcv_9104.socket"
CAM = 1                       # FusionCam (camera 0 is attached to the player, does not move!)
MAP = "Storagehouse"
OUT_DIR = "/data/romb_synth"
# Test poses around the scene centre (Storagehouse centre ~ -635 -3137, forklift height ~120).
POSES = [
    (-635, -3137, 120, -6, 0),
    (-635, -3137, 120, -6, 90),
    (-635, -3137, 120, -6, 180),
    (-635, -3137, 120, -6, 270),
    (-300, -2000, 120, -6, 135),
]
# --------------------------------------------------------------


def tcp_touch():
    try:
        socket.create_connection((HOST, PORT), timeout=5).close()
        print("TCP touch ok")
    except Exception as e:
        print("TCP touch warning:", e)


def main():
    tcp_touch()
    time.sleep(1)
    c = Client(SOCK_PATH, "unix")
    c.connect()
    if not c.isconnected():
        raise RuntimeError("Cannot connect to UnrealCV (binary not up yet or socket missing).")

    print("status:", c.request("vget /unrealcv/status"))
    cur = c.request("vget /level/name") or ""
    print("current scene:", cur)
    if MAP and MAP.lower() not in cur.lower():
        print(f"Loading {MAP} (may take a while)...")
        c.request(f"vset /action/game/level {MAP}", timeout=180)
        time.sleep(30)
        try:
            c.disconnect()
        except Exception:
            pass
        tcp_touch(); time.sleep(1)
        c = Client(SOCK_PATH, "unix"); c.connect()
        print("scene now:", c.request("vget /level/name"))

    objects = R.get_objects(c)
    print(f"Actors in scene: {len(objects)}")
    print("Reading unique actor colours (once, may take a while)...")
    color_dict = R.get_color_dict(c, objects)
    print(f"Colours read for {len(color_dict)} actors.")

    for sub in ("rgb", "semantic", "instance"):
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

    legend_all = {}
    for i, (x, y, z, pitch, yaw) in enumerate(POSES):
        c.request(f"vset /camera/{CAM}/location {x} {y} {z}")
        c.request(f"vset /camera/{CAM}/rotation {pitch} {yaw} 0")
        time.sleep(0.6)

        fn = f"frame_{i:06d}"
        semantic, instance, legend = R.build_masks(c, CAM, objects, color_dict)
        R.save_semantic_png(semantic, os.path.join(OUT_DIR, "semantic", f"{fn}.png"))
        R.save_instance_png(instance, os.path.join(OUT_DIR, "instance", f"{fn}.png"))
        R.save_lit(c, CAM, os.path.join(OUT_DIR, "rgb", f"{fn}.jpg"))
        legend_all[fn] = legend
        print(f"  {fn}: {len(legend)} instances visible")

    c.request(f"vset /viewmode lit")
    with open(os.path.join(OUT_DIR, "legend.json"), "w") as f:
        json.dump(legend_all, f, indent=2)
    c.disconnect()
    print("Done ->", OUT_DIR)


if __name__ == "__main__":
    main()