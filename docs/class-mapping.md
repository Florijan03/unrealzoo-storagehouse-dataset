# UE mesh → ROMB class mapping

Each actor in the scene has a unique colour in the UnrealCV `object_mask` render. We read
those colours once, then map every actor's name to a ROMB class with the rules below (first
match wins; unknown physical objects fall back to `other-object`). The rules live in
[`scripts/romb_export.py`](../scripts/romb_export.py) → `CLASSIFY_RULES`.

## ROMB classes

| id | class | notes |
|----|-------|-------|
| 0 | drivable | floor / drivable ground |
| 1 | person | people / characters |
| 2 | pallet-face | (not present in Storagehouse) |
| 3 | pallet-empty | (not present in Storagehouse) |
| 4 | pallet-full | all pallets map here (v1) |
| 5 | cargo | boxes, crates, barrels, bottles, books, paper |
| 6 | vehicle-forklift | (not present — no powered forklift) |
| 7 | vehicle-other | hand carts / trolleys / pallet jacks |
| 8 | ego-forklift | (not present) |
| 9 | other-object | known props + unknown physical objects |
| 10 | vertical | racks, beams, walls, pipes |
| 11 | ignore | lights, reflection captures, dirt decals, out-of-world (physics volume) |

## Name patterns (regex, case-insensitive)

| pattern | class |
|---------|-------|
| `SM_floor`, `FloorDecal` | drivable |
| `SM_Palette` | pallet-full (via pallet sub-classifier) |
| `SM_CardBox`, `SM_CratePlastic`, `SM_CartonDrawer`, `SM_Barel`, `SM_BottlePlastic`, `SM_Book`, `SM_PaperNote`, `SM_Paper_Shortcut` | cargo |
| `SM_Rack`, `SM_Beam`, `SM_Wall`, `SM_Pipe` | vertical |
| `SM_FireExtinguisher`, `SM_Sign` | other-object |
| `Character`, `Manny`, `Quinn`, `SK_`, `Human`, `Person` | person |
| `Cart`, `Trolley`, `HandTruck`, `PalletJack`, `Dolly` | vehicle-other |
| `DefaultPhysicsVolume`, `PhysicsVolume` | ignore (void / out-of-world) |
| `Light`, `ReflectionCapture`, `DirtStain` | ignore |
| *(anything else)* | other-object |

## Adapting to a new scene

1. List actors with `vget /objects` and inspect their names.
2. Add or adjust patterns in `CLASSIFY_RULES` so each mesh maps to the right ROMB class.
3. Re-read colours (done automatically at the start of each capture run).

The pallet sub-classifier (`classify_pallet`) currently maps every pallet to `pallet-full`;
distinguishing `pallet-empty` / `pallet-face` would require geometry/rotation checks and a
scene that actually contains them.
