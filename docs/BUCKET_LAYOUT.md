# s3://xray-hands layout

```
raw/<dataset>/...                       untouched upstream downloads (zips/tars exactly as released)
  arctic/   dexycb/   grab/   graspxl/   mano/   ho3d_v3/   hoi4d/   oakink_v1/   mow/
  hoisdf/   baselines/   polyhaven/   ambientcg/
eval/<dataset>/penetration_v1/          audit output over raw/ (this repo: scripts/audit.py)
  mask.parquet    seq, hand, frame, pen_max_mm, min_sd_mm, contact, grounded(<2 mm)
  frames.parquet  all per-frame metrics
  summary.json
eval/graspxl/penetration_v1_sample/      same, over a stratified 8,436-sequence sample (+ sample_manifest.json); biased,
                                        drawn while mano_dataset_N.part00 was missing
eval/graspxl/penetration_v2_sample/      unbiased stratified sample over the complete archives (audit_bootstrap.sh)
eval/<dataset>/grounded_v1/<tag>/       physically grounded subset: index.parquet + summary.json (build_subset.py)
eval/<dataset>/benchmarks/<run_id>/     model benchmark runs on an eval subset (predictions, per-frame scores)
train/                                  (empty)
```

`seq` + `frame` in every mask point back into `raw/<dataset>`. See the README table for the per-dataset key format.
A downstream loader only needs `raw/` plus `mask.parquet`, filtered with `grounded` or with its own threshold on
`pen_max_mm`.

## State on 2026-11-17

| prefix | status |
|---|---|
| `raw/dexycb` | complete (10 subjects + calibration + models + bop) |
| `raw/grab` | `grab__s1..s10` and objects/subject meshes complete. Audit only needs these. `data__train__data__s1` (GrabNet) was absent at audit time, and uploads were still landing. |
| `raw/graspxl` | **`mano_dataset_{1,2,3}.zip.part01` are truncated.** Each object is only the last 15 GiB chunk of a 64.5 GB split archive, so 48.3 GB (45 GiB) is missing from the front. That leaves 21–24% of sequences readable (central directory intact). `mano_tabletop.zip` and `object_dataset.zip` are complete. The likely cause is that every `.partNN` was uploaded to the same key. Fix: re-upload all parts under their own names, or upload the re-joined `.zip`. |
| `raw/mano` | `bodyHands_SCANS_{A01,A03,A10,A11,B07,B15,B28}.zip` are absent (their `.mp4` previews exist). Uploads were still landing. |
| `raw/arctic` | **absent.** ARCTIC raw data has only ever lived on the local workstation (`data/arctic/{downloads,unpack}` in xray-hands). |
| `eval/{dexycb,grab}/penetration_v1`, `eval/graspxl/penetration_v1_sample` | written 2026-11-17 by `scripts/audit.py`, 0 errors |
| `eval/arctic/20261113_ego_s5` | benchmark run (HaMeR / WiLoR / HOPformer on `arctic_p2_val_v0_s5`). It belongs under `eval/arctic/benchmarks/`. |

## Moving ARCTIC into raw/

Run these from the xray-hands checkout on the workstation that holds the data:

```bash
aws s3 sync data/arctic/downloads s3://xray-hands/raw/arctic/    # raw_seqs.zip, meta.zip, splits_json.zip (+ images if fetched)
aws s3 cp results/arctic_frames.parquet s3://xray-hands/eval/arctic/penetration_v1/frames.parquet
aws s3 sync data/arctic/unpack/arctic_data/data/grounded_mask s3://xray-hands/eval/arctic/penetration_v1/grounded_mask/
```

Then re-file the benchmark run. This copies first and deletes only after the copy is checked:

```bash
aws s3 sync s3://xray-hands/eval/arctic/20261113_ego_s5/ s3://xray-hands/eval/arctic/benchmarks/20261113_ego_s5/
aws s3 rm --recursive s3://xray-hands/eval/arctic/20261113_ego_s5/
```

`baselines/pod_env.sh` and `baselines/README.md` in xray-hands still point at `s3://xray-hands/eval/arctic`.

## State on 2026-11-23

* `raw/graspxl/mano_dataset_{1,2,3}.zip.part00` are now present: all 5,964,698 sequences of mano_dataset_1-3 are readable (shift 0); mano_tabletop adds 868,046.
* `raw/arctic` holds the full release (`data/*.zip`, `unpacked/arctic_data/data/{images,cropped_images}/sXX/<seq>.tar`).
* Added: `raw/ho3d_v3` (HO3D_v3.zip + HO3D_V3_GT), `raw/hoi4d` (hand pose, annotations, RGB release, CAD, cameras;
  no depth), `raw/oakink_v1` (Hugging Face release; the image annotations `anno_v2_1.zip` are Google-Form gated and
  missing), `raw/mow`, `raw/hoisdf`, `raw/baselines`, `raw/polyhaven`, `raw/ambientcg`.
* `eval/arctic/penetration_v1/frames.parquet` exists (301 sequences).
