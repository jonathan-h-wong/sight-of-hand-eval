# hoi-audit

Minimal, dataset-agnostic **hand-object penetration audit** for MANO-annotated grasp datasets. This is the extracted
core of `xray-hands/analysis` (`hoi_metrics.py`, `mano_np.py` are unchanged ports). It covers **GRAB**, **DexYCB** and
**GraspXL**, and ARCTIC slots in as one more source.

The raw data is never modified. The audit writes a per-frame mask that indexes straight into the raw files.

```
hoiaudit/
  metrics.py          SDF penetration (fast winding number, 1x-subdivided MANO) + smoothness   [= analysis/hoi_metrics.py]
  mano_np.py          NumPy MANO, no chumpy / torch                                          [= analysis/mano_np.py]
  s3zip.py            read zip central directories / single entries from S3 with range requests
  sources/
    common.py         MANO cache, pose helpers
    grab.py           GRAB      (subject v_template, flat_hand_mean=True, 120 fps, both hands)
    dexycb.py         DexYCB    (PCA-45, flat_hand_mean=False, subject betas, all scene objects)
    graspxl.py        GraspXL   (betas 0; flat_hand_mean False for mano_dataset_N, True for mano_tabletop)
scripts/
  prepare_data.sh     pull only what the audit needs from s3://xray-hands/raw (no images)
  fetch_graspxl_sample.py   stratified GraspXL sample via ranged reads (no 80 GB download)
  audit.py            the one central script: --dataset {grab,dexycb,graspxl}
```

## Run

```bash
pip install -r requirements.txt
bash scripts/prepare_data.sh data
python scripts/audit.py --dataset grab    --root data/grab/unz --out out/grab    --mano-dir data/mano/mano_v1_2/models
python scripts/audit.py --dataset dexycb  --root data/dexycb   --out out/dexycb  --mano-dir data/mano/mano_v1_2/models
python scripts/audit.py --dataset graspxl --root data/graspxl  --out out/graspxl --mano-dir data/mano/mano_v1_2/models
aws s3 sync out/grab s3://xray-hands/eval/grab/penetration_v1/   # etc.
```

## Outputs (`s3://xray-hands/eval/<dataset>/penetration_v1/`)

| file | content |
|---|---|
| `mask.parquet` | `dataset, seq, hand, frame, pen_max_mm, min_sd_mm, contact, grounded` (grounded = `pen_max_mm < 2.0`) |
| `frames.parquet` | every metric from `metrics.evaluate` (penetration depth/area/volume, smoothness, dataset extras) |
| `summary.json` | parameters, grounded/contact fractions, penetration percentiles, errors |

`seq` and `frame` address the raw data directly:

| dataset | `seq` | `frame` |
|---|---|---|
| grab | `sN/<obj>_<intent>` → `raw/grab/grab__sN.zip:sN/<seq>.npz` | row in that file's arrays (120 fps) |
| dexycb | `<date>-subject-NN/<seq>` → `<seq>/pose.npz` in `raw/dexycb/<date>-subject-NN.tar.gz` | row of `pose_m` / image index |
| graspxl | `<archive>/<size>/<obj_id>/<file>` → `raw/graspxl/<archive>.zip` | row in the `.npy` dict arrays |

Frames without a valid GT hand (DexYCB `pose_m == 0`) are absent from the mask. To use a different threshold, read
`pen_max_mm` and threshold it yourself; no rerun is needed.

## Conventions and how they were checked

* **GRAB**: hands use `fullpose` (45-d), which equals `hand_pose(24 PCA) @ components` to 1e-7. Objects come from
  `objects_info.npz`, whose vertex count matches GRAB's own per-vertex `contact` arrays. Object pose is
  `v @ R + t` (GRAB `ObjectModel`). Check: in frames that GRAB's contact labels mark as touched, the median hand-object
  signed distance is about −2.6 mm, and the idle hand sits at ≥30 mm.
* **DexYCB**: follows `dex-ycb-toolkit` (manopth `use_pca=True, ncomps=45, flat_hand_mean=False`, betas from
  `calibration/mano_*`, object quaternions in `xyzw`). The hand is tested against the union of all scene objects.
  Check: penetration is 0 throughout the approach and appears only on the grasped object.
* **GraspXL**: follows `GraspXL_visualization`. The documented `flat_hand_mean` per archive was checked empirically.
  With it, the final grasp frames are in contact (100% for mano_dataset, 73% for tabletop). The swapped setting leaves
  about half the hands open.

## GT rectification (`hoiaudit/rectify`, `scripts/rectify.py`)

Turns penetrating GT into a non-penetrating **pseudo-GT** for full-sequence evaluation (contact metrics on every
frame, smoothness/drift on unbroken sequences). It covers ARCTIC and DexYCB, the datasets with real calibrated cameras.

```bash
python scripts/rectify.py --dataset arctic --root data/arctic --out out/rectify/arctic --mano-dir data/mano/mano_v1_2/models --device cuda
python scripts/rectify.py --dataset dexycb --root data/dexycb --out out/rectify/dexycb --mano-dir data/mano/mano_v1_2/models --device cuda
python scripts/rectify.py --dataset dexycb --out out/rectify/dexycb --root data/dexycb --summarize
```
For several processes, use `--shard i/n`. A rerun skips sequences that are already done.

**Frame status** (per hand-frame, columns `frozen` / `unresolved` in the report):

| status | rule | output |
|---|---|---|
| grounded (frozen) | GT max penetration ≤ 2 mm (exact metric) | GT parameters, **bit-identical** |
| rectified | GT > 2 mm, fixed to ≤ 2 mm (exact metric, checked) | corrected parameters |
| unresolved | GT > 2 mm that no local correction fixes, e.g. a finger passing through a thin part, where the surface pulls it both ways | GT parameters, bit-identical, flagged |

**Method.** This is a ContactOpt-style pose optimization over the whole sequence at once, batched over frames on the GPU.
For each frame it corrects MANO global rotation, the 45-d articulation and translation. Shape and the object never change.
* **Penetration:** squared depth beyond 1 mm, over the dense hand surface (the same 1x-subdivided MANO the audit
  measures), against cached SDF grids of every object part (`sdfgrid.py`, trilinear, about 0.1–0.5 mm from exact).
* **Contact:** each of the 16 hand regions that touched the object keeps its closest point on the surface. A finger
  may lift out of the object, but may not leave it.
* **Camera consistency:** in every camera that sees a joint, its pixel change, converted to mm at the joint's depth.
  ARCTIC uses 8 allocentric cameras plus the per-frame egocentric camera; DexYCB uses 8 cameras.
* **Temporal:** first and second differences of the joint correction. Frozen and unresolved frames have zero
  correction, so fixes fade smoothly into the untouched frames around them.
* **Rounds:** frames still above 2 mm get a stronger penetration weight. Whatever is left is set to unresolved and
  frozen, and the rest is re-optimized.

**Outputs:**
* `seqs/<seq>.npz`: rectified MANO parameters for all frames, with the status masks and penetration before and after.
* `frames/<seq>.parquet`, merged into `frames.parquet`: per-frame penetration before and after, joint shift
  (mean and max), worst pixel change over cameras, contact, and jitter before and after.
* `summary.json`.

Every reported number comes from the exact metric, not the grid.
