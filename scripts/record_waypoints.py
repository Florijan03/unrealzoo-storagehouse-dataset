"""
record_waypoints.py (Mac) — fly through the scene and record good camera positions.

Control and rendering are decoupled: a background thread continuously pulls a preview and
pushes the current pose, while the main thread handles keys/mouse at 60 fps, so navigation
stays smooth even when the render lags over the tunnel. Preview is low-res (the final
dataset is captured separately at full resolution).

Controls:
  W/A/S/D move (relative to facing), Q/E down/up
  arrows look (yaw/pitch) — or mouse
  SHIFT faster
  SPACE record waypoint     BACKSPACE undo last
  P cycle preview resolution (smaller = faster)
  ENTER save waypoints.json     ESC quit

Tunnel: server -> python3 uds_bridge.py ; Mac -> ssh -N -L 9105:127.0.0.1:9105 user@server
Run:
  python3 record_waypoints.py 127.0.0.1 9105
"""
import os, sys, io, json, math, time, threading
import pygame
from PIL import Image
from unrealcv import Client

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9104
CAM = 1
SIZES = [(480, 270), (320, 180), (640, 360)]   # P cycles; first = default
OUT = "waypoints.json"
EXPOSURE_BIAS = 2

MOVE_SPEED = 300.0     # cm/s
FAST_SPEED = 900.0
LOOK_SPEED = 90.0      # deg/s (arrows)
MOUSE_SENS = 0.15

# ---- shared state between main thread and render thread ----
S = {"x": 0.0, "y": 0.0, "z": 120.0, "pitch": 0.0, "yaw": 0.0,
     "frame": None, "size": SIZES[0], "new_size": None, "fps": 0.0, "run": True}
LOCK = threading.Lock()


def req(c, cmd):
    try:
        return c.request(cmd, timeout=10)
    except Exception:
        return None


def render_thread(c):
    """Continuously push the pose and pull a preview — the only thread touching the socket."""
    req(c, f"vset /camera/{CAM}/size {S['size'][0]} {S['size'][1]}")
    req(c, f"vrun r.DefaultFeature.AutoExposure.Bias {EXPOSURE_BIAS}")
    t0 = time.time(); frames = 0
    while S["run"]:
        if S["new_size"]:
            w, h = S["new_size"]; S["new_size"] = None; S["size"] = (w, h)
            req(c, f"vset /camera/{CAM}/size {w} {h}")
        with LOCK:
            x, y, z, pitch, yaw = S["x"], S["y"], S["z"], S["pitch"], S["yaw"]
        req(c, f"vset /camera/{CAM}/location {x:.1f} {y:.1f} {z:.1f}")
        req(c, f"vset /camera/{CAM}/rotation {pitch:.1f} {yaw:.1f} 0")
        data = req(c, f"vget /camera/{CAM}/lit png")
        if data:
            try:
                if isinstance(data, str):
                    data = data.encode("latin-1")
                im = Image.open(io.BytesIO(data)).convert("RGB")
                S["frame"] = (im.tobytes(), im.size)
            except Exception:
                pass
        frames += 1
        if time.time() - t0 >= 1.0:
            S["fps"] = frames / (time.time() - t0); frames = 0; t0 = time.time()


def main():
    c = Client((HOST, PORT))
    c.connect(timeout=5)
    if not c.isconnected():
        print(f"Cannot connect to {HOST}:{PORT}. Check tunnel/bridge."); return
    if not req(c, "vget /unrealcv/status"):
        print("Connected to tunnel but UnrealCV does not respond."); return
    loc = (req(c, f"vget /camera/{CAM}/location") or "0 0 120").split()
    rot = (req(c, f"vget /camera/{CAM}/rotation") or "0 0 0").split()
    S["x"], S["y"], S["z"] = float(loc[0]), float(loc[1]), float(loc[2])
    S["pitch"], S["yaw"] = float(rot[0]), float(rot[1])

    th = threading.Thread(target=render_thread, args=(c,), daemon=True)
    th.start()

    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    pygame.display.set_caption("record_waypoints — WASD/arrows | SPACE record | P res | ENTER save | ESC")
    font = pygame.font.SysFont("monospace", 16)
    clock = pygame.time.Clock()
    pygame.event.set_grab(True); pygame.mouse.set_visible(False)

    waypoints = []
    if os.path.exists(OUT):
        try:
            waypoints = json.load(open(OUT)); print(f"continuing from {len(waypoints)} waypoints")
        except Exception:
            waypoints = []
    size_idx = 0
    flash = 0
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    with LOCK:
                        wp = {"x": round(S["x"], 1), "y": round(S["y"], 1), "z": round(S["z"], 1),
                              "pitch": round(S["pitch"], 1), "yaw": round(S["yaw"], 1)}
                    waypoints.append(wp); flash = 10; print(f"waypoint #{len(waypoints)}: {wp}")
                elif e.key == pygame.K_BACKSPACE and waypoints:
                    waypoints.pop(); print("undo ->", len(waypoints))
                elif e.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    json.dump(waypoints, open(OUT, "w"), indent=2); print(f"saved {len(waypoints)} to {OUT}")
                elif e.key == pygame.K_p:
                    size_idx = (size_idx + 1) % len(SIZES); S["new_size"] = SIZES[size_idx]
            elif e.type == pygame.MOUSEMOTION:
                mx, my = e.rel
                with LOCK:
                    S["yaw"] += mx * MOUSE_SENS
                    S["pitch"] = max(-85, min(85, S["pitch"] - my * MOUSE_SENS))

        keys = pygame.key.get_pressed()
        speed = (FAST_SPEED if keys[pygame.K_LSHIFT] else MOVE_SPEED) * dt
        look = LOOK_SPEED * dt
        with LOCK:
            yaw = S["yaw"]
            if keys[pygame.K_LEFT]:  S["yaw"] -= look
            if keys[pygame.K_RIGHT]: S["yaw"] += look
            if keys[pygame.K_UP]:    S["pitch"] = min(85, S["pitch"] + look)
            if keys[pygame.K_DOWN]:  S["pitch"] = max(-85, S["pitch"] - look)
            rad = math.radians(yaw)
            fx, fy = math.cos(rad), math.sin(rad)
            rx, ry = math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)
            if keys[pygame.K_w]: S["x"] += fx * speed; S["y"] += fy * speed
            if keys[pygame.K_s]: S["x"] -= fx * speed; S["y"] -= fy * speed
            if keys[pygame.K_a]: S["x"] -= rx * speed; S["y"] -= ry * speed
            if keys[pygame.K_d]: S["x"] += rx * speed; S["y"] += ry * speed
            if keys[pygame.K_e]: S["z"] += speed
            if keys[pygame.K_q]: S["z"] -= speed
            px, py, pz, ppitch, pyaw = S["x"], S["y"], S["z"], S["pitch"], S["yaw"]

        fr = S["frame"]
        if fr is not None:
            surf = pygame.image.frombytes(fr[0], fr[1], "RGB")
            surf = pygame.transform.scale(surf, (960, 540))
            screen.blit(surf, (0, 0))
        else:
            screen.fill((20, 20, 20))
        hud = [f"pos ({px:.0f},{py:.0f},{pz:.0f})  pitch {ppitch:.0f}  yaw {pyaw:.0f}   preview {S['size'][0]}x{S['size'][1]}  {S['fps']:.1f} fps",
               f"waypoints: {len(waypoints)}   SPACE record | BKSP undo | P res | ENTER save | ESC"]
        for i, t in enumerate(hud):
            screen.blit(font.render(t, True, (255, 255, 0)), (8, 8 + i * 18))
        if flash > 0:
            pygame.draw.rect(screen, (0, 255, 0), screen.get_rect(), 6); flash -= 1
        pygame.display.flip()

    S["run"] = False
    time.sleep(0.3)
    json.dump(waypoints, open(OUT, "w"), indent=2)
    print(f"done — saved {len(waypoints)} to {OUT}")
    try:
        c.request(f"vset /camera/{CAM}/size 1920 1080")
    except Exception:
        pass
    pygame.quit()


if __name__ == "__main__":
    main()