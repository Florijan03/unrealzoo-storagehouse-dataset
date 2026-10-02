"""
flythrough_recorder.py (Mac) — FPS-style camera-path recorder.

Open a window and fly through the Storagehouse scene with WASD + mouse; the tool drives
camera 1 over UnrealCV and (while recording) logs poses to camera_path.json, which can then
be fed into the dataset generator.

Prereqs (server): binary running in the container, UnrealCV on 9104.
SSH tunnel (Mac, keep open):  ssh -L 9104:127.0.0.1:9104 user@server
Packages (Mac):               pip install unrealcv pygame pillow numpy
Run (Mac, with tunnel up):    python3 flythrough_recorder.py

Controls:
  W/S forward/back     A/D strafe       E/Q up/down       mouse look
  SHIFT faster         +/- change speed
  R start/stop recording the path (logs a pose every frame)
  SPACE add one waypoint manually
  ESC quit and save camera_path.json
"""

import math
import json
import time
import sys
from io import BytesIO

import pygame
from PIL import Image
from unrealcv import Client

# --------------------------- CONFIG ---------------------------
HOST, PORT = "127.0.0.1", 9104     # via SSH tunnel to the Mac
CAM = 1                            # FusionCam
START = [-635.0, -3137.0, 120.0]   # Storagehouse centre, forklift height
START_PITCH_YAW = [-6.0, 0.0]
MOVE_SPEED = 30.0                  # cm per frame (hold key)
MOUSE_SENS = 0.15
OUT = "camera_path.json"
# --------------------------------------------------------------


def forward_vec(pitch, yaw):
    p, y = math.radians(pitch), math.radians(yaw)
    return (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), math.sin(p))


def right_vec(yaw):
    y = math.radians(yaw + 90.0)
    return (math.cos(y), math.sin(y), 0.0)


def get_lit(client, size_hint=None):
    res = client.request(f"vget /camera/{CAM}/lit png")
    if isinstance(res, str):
        return None
    im = Image.open(BytesIO(res)).convert("RGB")
    return im


def main():
    c = Client((HOST, PORT))
    c.connect()
    if not c.isconnected():
        print("Cannot connect to UnrealCV on 127.0.0.1:9104.")
        print("Check: (1) binary running on server, (2) SSH tunnel open:")
        print("  ssh -L 9104:127.0.0.1:9104 user@server")
        sys.exit(1)
    print("status:", c.request("vget /unrealcv/status"))

    pos = list(START)
    pitch, yaw = START_PITCH_YAW
    speed = MOVE_SPEED

    # set initial pose and grab the first frame for window size
    c.request(f"vset /camera/{CAM}/location {pos[0]} {pos[1]} {pos[2]}")
    c.request(f"vset /camera/{CAM}/rotation {pitch} {yaw} 0")
    c.request("vset /viewmode lit")
    im = get_lit(c)
    W, H = (im.size if im else (640, 480))

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("UnrealCV flythrough — R record, SPACE waypoint, ESC save+quit")
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()

    path = []
    recording = False
    running = True

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    recording = not recording
                    print("RECORDING:", "ON" if recording else "OFF", f"({len(path)} points)")
                elif e.key == pygame.K_SPACE:
                    path.append({"x": round(pos[0], 1), "y": round(pos[1], 1), "z": round(pos[2], 1),
                                 "pitch": round(pitch, 1), "yaw": round(yaw, 1)})
                    print("waypoint added:", len(path))
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed *= 1.5; print("speed:", round(speed, 1))
                elif e.key == pygame.K_MINUS:
                    speed /= 1.5; print("speed:", round(speed, 1))

        # mouse = look
        dx, dy = pygame.mouse.get_rel()
        yaw += dx * MOUSE_SENS
        pitch -= dy * MOUSE_SENS
        pitch = max(-89.0, min(89.0, pitch))

        # keys = movement
        keys = pygame.key.get_pressed()
        s = speed * (3.0 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 1.0)
        fx, fy, fz = forward_vec(pitch, yaw)
        rx, ry, _ = right_vec(yaw)
        if keys[pygame.K_w]: pos[0] += fx * s; pos[1] += fy * s; pos[2] += fz * s
        if keys[pygame.K_s]: pos[0] -= fx * s; pos[1] -= fy * s; pos[2] -= fz * s
        if keys[pygame.K_d]: pos[0] += rx * s; pos[1] += ry * s
        if keys[pygame.K_a]: pos[0] -= rx * s; pos[1] -= ry * s
        if keys[pygame.K_e]: pos[2] += s
        if keys[pygame.K_q]: pos[2] -= s

        # push pose + fetch image
        c.request(f"vset /camera/{CAM}/location {pos[0]:.1f} {pos[1]:.1f} {pos[2]:.1f}")
        c.request(f"vset /camera/{CAM}/rotation {pitch:.2f} {yaw:.2f} 0")
        im = get_lit(c)
        if im is not None:
            surf = pygame.image.fromstring(im.tobytes(), im.size, "RGB")
            screen.blit(surf, (0, 0))

        if recording:
            path.append({"x": round(pos[0], 1), "y": round(pos[1], 1), "z": round(pos[2], 1),
                         "pitch": round(pitch, 1), "yaw": round(yaw, 1)})

        # HUD
        hud = f"xyz=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f}) yaw={yaw:.0f} pitch={pitch:.0f} " \
              f"{'REC' if recording else '   '} points={len(path)} v={speed:.0f}"
        screen.blit(font.render(hud, True, (255, 255, 0)), (8, 8))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    with open(OUT, "w") as f:
        json.dump(path, f, indent=2)
    print(f"Saved {len(path)} poses to {OUT}")
    try:
        c.disconnect()
    except Exception:
        pass


if __name__ == "__main__":
    main()