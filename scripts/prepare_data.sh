#!/usr/bin/env bash
# Pull the minimal pieces of each raw dataset from s3://xray-hands/raw/ that the audit needs (no images).
#   bash scripts/prepare_data.sh data            # all datasets
#   bash scripts/prepare_data.sh data grab       # one dataset
# Needs: aws cli with read access to the bucket, ~8 GB free (GRAB 4 GB unpacked, DexYCB labels <1 GB).
set -euo pipefail
DATA=${1:?data dir}; ONLY=${2:-all}
B=s3://xray-hands/raw
mkdir -p "$DATA"

if [[ $ONLY == all || $ONLY == mano ]]; then
  mkdir -p "$DATA/mano" && aws s3 cp --quiet $B/mano/mano_v1_2.zip "$DATA/mano/" && unzip -q -o "$DATA/mano/mano_v1_2.zip" -d "$DATA/mano"
fi

if [[ $ONLY == all || $ONLY == arctic ]]; then
  A="$DATA/arctic"; mkdir -p "$A"
  for k in raw_seqs.zip meta.zip; do
    aws s3 cp --quiet $B/arctic/data/$k "$A/" && unzip -q -o "$A/$k" -d "$A" && rm "$A/$k"
  done
fi

if [[ $ONLY == all || $ONLY == grab ]]; then
  G="$DATA/grab"; mkdir -p "$G/zips" "$G/unz"
  for k in grab__objects_info.zip tools__subject_meshes__male.zip tools__subject_meshes__female.zip \
           grab__s1.zip grab__s2.zip grab__s3.zip grab__s4.zip grab__s5.zip grab__s6.zip grab__s7.zip grab__s8.zip \
           grab__s9.zip grab__s10.zip; do
    aws s3 cp --quiet $B/grab/$k "$G/zips/" && unzip -q -o "$G/zips/$k" -d "$G/unz" && rm "$G/zips/$k"
  done
fi

if [[ $ONLY == all || $ONLY == dexycb ]]; then
  X="$DATA/dexycb"; mkdir -p "$X"
  aws s3 cp --quiet $B/dexycb/calibration.tar.gz - | tar -xzf - -C "$X"
  aws s3 cp --quiet $B/dexycb/models.tar.gz - | tar -xzf - -C "$X" --wildcards '*textured_simple.obj'
  # each subject tarball is ~12 GB of images; stream it and keep only per-sequence labels (5 in parallel)
  aws s3 ls $B/dexycb/ | grep -oE '20[0-9]{6}-subject-[0-9]+\.tar\.gz' | \
    xargs -P 5 -I{} sh -c "aws s3 cp --quiet $B/dexycb/{} - | tar -xzf - -C '$X' --wildcards --no-anchored pose.npz meta.yml && echo done {}"
fi

if [[ $ONLY == all || $ONLY == graspxl ]]; then
  # all four archives are complete since 2026-11-23 (mano_dataset_N.zip.part00 re-uploaded), so the stratified sample
  # is no longer biased toward the back half of mano_dataset_1-3. GRASPXL_OBJECTS objects per archive.
  python scripts/fetch_graspxl_sample.py --out "$DATA/graspxl" --objects-per-archive ${GRASPXL_OBJECTS:-1500} --seed 0
fi

# move the single directory under $1 that contains $2 to $3 (archives differ in their top-level folder)
_adopt() { local d; d=$(find "$1" -maxdepth 4 -type d -name "$2" -print -quit); [[ -n $d ]] || { echo "no $2 in $1"; exit 1; }
           mkdir -p "$(dirname "$3")"; rm -rf "$3"; mv "$(dirname "$d")" "$3"; }

if [[ $ONLY == all || $ONLY == ho3d ]]; then
  # labels only: meta/*.pkl + split lists (no rgb/depth/seg), YCB meshes from DexYCB, released evaluation GT verts
  H="$DATA/ho3d"; T="$DATA/_ho3d_tmp"; mkdir -p "$T"
  aws s3 cp --quiet $B/ho3d_v3/HO3D_v3/HO3D_v3.zip "$T/"
  unzip -q -o "$T/HO3D_v3.zip" '*/meta/*' '*.txt' -d "$T/unz" && rm "$T/HO3D_v3.zip"
  _adopt "$T/unz" train "$H"
  aws s3 cp --quiet $B/dexycb/models.tar.gz - | tar -xzf - -C "$H" --wildcards '*textured_simple.obj'
  aws s3 cp --quiet $B/ho3d_v3/HO3D_V3_GT/evaluation_verts.json "$T/"
  python scripts/ho3d_eval_verts.py --root "$H" --gt "$T/evaluation_verts.json" && rm -rf "$T"
fi

if [[ $ONLY == all || $ONLY == hoi4d ]]; then
  # labels only: hand poses, per-frame object poses (objpose/*.json), rigid CAD models. No video, depth or segs.
  X="$DATA/hoi4d"; T="$DATA/_hoi4d_tmp"; mkdir -p "$X" "$T"
  aws s3 cp --quiet $B/hoi4d/HOI4D_Hand_pose.zip "$T/" && unzip -q -o "$T/HOI4D_Hand_pose.zip" -d "$X" && rm "$T/HOI4D_Hand_pose.zip"
  aws s3 cp --quiet $B/hoi4d/HOI4D_CAD_Model_for_release.zip "$T/"
  unzip -q -o "$T/HOI4D_CAD_Model_for_release.zip" '*/rigid/*' -d "$T/cad" && rm "$T/HOI4D_CAD_Model_for_release.zip"
  mkdir -p "$X/CAD" && rm -rf "$X/CAD/rigid" && mv "$(find "$T/cad" -maxdepth 2 -type d -name rigid -print -quit)" "$X/CAD/rigid"
  aws s3 cp --quiet $B/hoi4d/HOI4D_annotations.zip "$T/"
  unzip -q -o "$T/HOI4D_annotations.zip" '*/objpose/*' -d "$T/ann" && rm "$T/HOI4D_annotations.zip"
  d=$(find "$T/ann" -maxdepth 3 -type d -name 'ZY2021080000*' -print -quit); rm -rf "$X/annotations"; mv "$(dirname "$d")" "$X/annotations"
  rm -rf "$T"
fi

if [[ $ONLY == all || $ONLY == oakink ]]; then
  # needs raw/oakink_v1/image/anno_v2_1.zip (Google-Form gated, see hoiaudit/sources/oakink.py); skipped without it
  O="$DATA/oakink"; mkdir -p "$O/image"
  if aws s3 ls $B/oakink_v1/image/anno_v2_1.zip > /dev/null; then
    aws s3 cp --quiet $B/oakink_v1/image/anno_v2_1.zip "$O/" && unzip -q -o "$O/anno_v2_1.zip" -d "$O/_a" && rm "$O/anno_v2_1.zip"
    _adopt "$O/_a" hand_v "$O/image/anno" && rm -rf "$O/_a"
    aws s3 cp --quiet $B/oakink_v1/image/obj.zip "$O/" && unzip -q -o "$O/obj.zip" -d "$O/image" && rm "$O/obj.zip"
  else
    echo "oakink: raw/oakink_v1/image/anno_v2_1.zip not uploaded yet; skipped"
  fi
fi
