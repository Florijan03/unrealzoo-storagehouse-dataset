"""
diag_void.py — find which actor covers the "void" (white out-of-world area).

Reads romb_legend.json (no rendering) and prints the largest segments and the actors that
often cover a big part of the image. The one at the top is the culprit (physics volume / void).

Run:
  podman exec uz-scene python3 /data/diag_void.py [/path/to/dataset]
"""
import sys, json
from collections import Counter

OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/dataset_v2"
W, H = 1920, 1080
TOTAL = W * H

leg = json.load(open(f"{OUT}/romb_legend.json"))

rows = []
for frame, insts in leg.items():
    for iid, info in insts.items():
        rows.append((info["pixels"], info["obj"], info["romb"], frame))
rows.sort(reverse=True)

print("=== 25 largest single segments (px, %image, actor, class, frame) ===")
for px, obj, romb, frame in rows[:25]:
    print(f"  {px:>9}  {100*px/TOTAL:5.1f}%  {obj:42} {romb:14} {frame}")

print("\n=== Actors that OFTEN cover >12% of the image (count, actor, class) ===")
big = Counter()
for px, obj, romb, frame in rows:
    if px > 0.12 * TOTAL:
        big[(obj, romb)] += 1
for (obj, romb), n in big.most_common(20):
    print(f"  {n:>4}x  {obj:42} {romb}")

print("\n=== Total pixels per class (which class dominates) ===")
by_cls = Counter()
for px, obj, romb, frame in rows:
    by_cls[romb] += px
for cls, px in by_cls.most_common():
    print(f"  {cls:14} {100*px/(TOTAL*len(leg)):5.1f}% of total area")