# hoi-audit

Minimal, dataset-agnostic **hand-object penetration audit** for MANO-annotated grasp datasets. This is the extracted
core of `xray-hands/analysis` (`hoi_metrics.py`, `mano_np.py` are unchanged ports). It covers **ARCTIC**, **GRAB**,
**DexYCB**, **GraspXL**, **HO3D v3**, **HOI4D** (rigid objects) and **OakInk-Image** (once its annotations are uploaded),
and turns each audit into a **physically grounded subset** that a training or evaluation loader can fetch.

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
    ho3d.py           HO3D v3   (absolute 48-d pose = flat_hand_mean=True; evaluation split from the released GT verts)
    hoi4d.py          HOI4D     (manopth flat_hand_mean=True, both hands; rigid categories; Euler-XYZ object poses)
    oakink.py         OakInk    (released 778-vertex hands + T_c_o; one camera per frame; needs anno_v2_1.zip)
  subset.py           grounded-subset criteria, selection, raw/ pointers
scripts/
  prepare_data.sh     pull only what the audit needs from s3://xray-hands/raw (no images)
  fetch_graspxl_sample.py   stratified GraspXL sample via ranged reads (no 80 GB download)
  audit.py            the one central script: --dataset {arctic,grab,dexycb,graspxl,ho3d,hoi4d,oakink}
  ho3d_eval_verts.py  split HO3D's evaluation_verts.json per sequence (used by prepare_data.sh)
  build_subset.py     audit -> grounded subset index (+ --publish to S3)
  fetch_grounded.py   subset index -> local labels (hand meshes, object poses/meshes) and images
  runpod/audit_bootstrap.sh   CPU pod job: prepare, audit, publish, build subsets
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

## Physically grounded subsets

The audit measures; the subset decides. `scripts/build_subset.py` applies a `Criteria` (hoiaudit/subset.py) to an audit
table and writes an index of the grounded hand-frames, each with pointers back into `raw/`:

| criterion | default | rule |
|---|---|---|
| contact | 5 mm | `min_sd_mm <= contact_mm` (the audit's contact definition) |
| penetration | 1 mm | `pen_max_mm <= pen_mm` |
| gap | 2 mm | `min_sd_mm <= gap_mm`, so the hand touches instead of hovering |
| jitter | off | `rel_hf_mm <= jitter_mm` (frames without a jitter estimate pass) |
| run | 1 | keep runs of at least `min_run` consecutive grounded frames |
| labels | original | `rectified` uses the ARCTIC/DexYCB pseudo-GT (`scripts/rectify.py`), unresolved frames excluded |

```bash
python scripts/build_subset.py --all --publish        # every audited dataset: original 1 mm, original 2 mm, rectified 2 mm
python scripts/build_subset.py --dataset hoi4d --pen-mm 1 --min-run 15 --publish
```

Published at `s3://xray-hands/eval/<ds>/grounded_v1/<tag>/` (`tag` like `original_pen1_gap2`):
`index.parquet` (one row per grounded seq/hand/frame: audit metrics, `run_id`, `run_len`, `label_key`/`label_member`,
`image_key`/`image_member`/`image_frame`; `rectified_key` for rectified labels) and `summary.json` (counts, runs,
per-object frames).

Fetch the frames themselves with `scripts/fetch_grounded.py`. It reruns the dataset's source loader on the prepared
labels for just the grounded frames and writes hand vertices, object poses and object meshes, plus the images
straight from the raw archives (ARCTIC and DexYCB tars streamed once, HO3D/OakInk zips by ranged reads, HOI4D videos
decoded at the grounded frames):

```bash
bash scripts/prepare_data.sh data arctic
python scripts/fetch_grounded.py --dataset arctic --tag original_pen1_gap2 --label-root data/arctic \
    --out grounded/arctic --images cams --cams 0          # egocentric view only
```

In Python, `subset.read_index(ds, tag)` returns the published index, and `subset.build_index(ds, Criteria(...))`
builds one with any other thresholds without rerunning the audit.

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
