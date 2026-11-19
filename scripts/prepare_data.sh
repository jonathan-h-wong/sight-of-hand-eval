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
  python scripts/fetch_graspxl_sample.py --out "$DATA/graspxl" --objects-per-archive 500 --seed 0
fi
