# UnrealZoo Storagehouse — Synthetic Warehouse Dataset

A synthetic warehouse dataset for **autonomous-forklift perception**, generated from the
UnrealZoo / UnrealCV **Storagehouse** scene. Images are labelled at three levels, all using
the **ROMB** class scheme (11 classes + ignore):

- **Semantic** segmentation — ROMB-format indexed masks (`labels/`, indices 0–10, ignore=11)
- **Instance** segmentation — one id per object (`instance/`)
- **Panoptic** segmentation — COCO panoptic (`panoptic/` + `panoptic.json`)

The dataset layout mirrors the real ROMB semantic dataset, so it is a drop-in synthetic
companion for the same models.

## Examples

See [`sample/`](sample/) for a few example frames (RGB, semantic overlay, instance view).

## Dataset

- **530 images**, 1920×1080, from a single scene (Storagehouse) with domain randomization
  (lighting, lens angle, cargo occupancy, viewpoint).
- **Formats:** ROMB semantic (`labels/`) + instance + COCO panoptic.
- **Split:** 451 train / 79 val (85/15).

**Structure** (ROMB layout):
```
1080p/
  rgb/<id>.jpg         RGB
  labels/<id>.png      semantic mask (indices 0-10, ignore=11)   <- ROMB 'labels'
  instance/<id>.png    instance mask (16-bit)
  panoptic/<id>.png    COCO panoptic (id-encoded)
  panoptic.json        COCO panoptic annotations
  romb.yaml            classes / colors / ignore_id
  train.txt  val.txt   split
```

**Download:** the image data is not stored in this repository (too large for git).
See the latest [**Release**](../../releases) for `dataset_romb.zip`, or the link in the
Release notes.

### Classes and coverage

ROMB scheme (0–10 + ignore=11). "Coverage" = share of images containing the class.

| id | class | coverage |
|----|-------|----------|
| 0 | drivable | ~99% |
| 1 | person | ~16% |
| 2 | pallet-face | — (absent in scene) |
| 3 | pallet-empty | — (absent in scene) |
| 4 | pallet-full | ~97% |
| 5 | cargo | ~99% |
| 6 | vehicle-forklift | — (absent in scene) |
| 7 | vehicle-other | ~60% |
| 8 | ego-forklift | — (absent in scene) |
| 9 | other-object | ~99% |
| 10 | vertical | ~100% |
| 11 | ignore | — |

The Storagehouse scene has no powered forklift and no empty/face pallets, so those four
classes have no examples. The UE-mesh → ROMB-class mapping is in
[`scripts/romb_export.py`](scripts/romb_export.py) (see also [docs/class-mapping.md](docs/class-mapping.md)).

## Quickstart (reproduce)

Full instructions: [docs/SETUP.md](docs/SETUP.md) (English) · [docs/SETUP_hr.md](docs/SETUP_hr.md) (hrvatski).

1. Run the UnrealZoo Storagehouse binary headless in the Podman container (see SETUP).
2. **Record waypoints** from your laptop:
   `python3 scripts/record_waypoints.py 127.0.0.1 9105` → `waypoints.json`.
3. **Generate** on the server:
   `python3 scripts/capture_dataset.py` → `dataset_500/`.
4. **Clean & package:**
   `python3 scripts/clean_void.py review 0.03` (then `drop`), `python3 scripts/finalize_dataset.py` → `dataset_romb/1080p/`.

`waypoints.json` in this repo contains the exact 60 positions used, for reproducibility.

## Repository layout

```
scripts/           pipeline scripts
  container/        Containerfile for the render environment
  legacy/           older / auxiliary scripts
docs/              SETUP (EN + HR, md + pdf), class mapping
sample/            a few example frames
waypoints.json     the camera positions used
```

## Requirements

Python 3.10+, see [`requirements.txt`](requirements.txt). Rendering needs a Linux host with an
NVIDIA GPU + the UnrealZoo UE5.6 binary (see SETUP); the scripts themselves only need
`unrealcv`, `pillow`, `numpy`, `pygame`.

## Acknowledgements

Built on [UnrealZoo](https://github.com/UnrealZoo/unrealzoo-gym) and
[UnrealCV](https://unrealcv.org/). Panoptic format per
[cocodataset/panopticapi](https://github.com/cocodataset/panopticapi). Class scheme follows
the ROMB semantic dataset. Developed as part of an autonomous-forklift research project at FER.

## License

Code: MIT (see [LICENSE](LICENSE)). Dataset: CC BY 4.0 (stated in the Release).
