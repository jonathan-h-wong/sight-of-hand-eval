#!/usr/bin/env bash
# RunPod (CPU) bootstrap for the penetration audits of the datasets added on 2026-11-23, plus the grounded subsets.
# Control and results go through S3 only: logs/status -> $W/status. Results are published to
#   s3://xray-hands/eval/<ds>/penetration_v1/            ho3d, hoi4d, oakink (if its annotations are uploaded)
#   s3://xray-hands/eval/graspxl/penetration_v2_sample/  unbiased GraspXL sample (all archives complete now)
#   s3://xray-hands/eval/<ds>/grounded_v1/<tag>/         subset indexes for every audited dataset (build_subset --all)
# Needs AWS_* in the pod env. On success the pod removes itself; on failure it stops (rerun resumes: audit.py skips
# finished sequences, kept in $R/out/<ds>/parts).
#   start command:  bash -c "pip install -q awscli && aws s3 cp s3://xray-hands/work/audit_v2/bootstrap.sh /b.sh && bash /b.sh"
set -uo pipefail
W=${WORK:-s3://xray-hands/work/audit_v2}
R=/workspace/audit; mkdir -p $R/status && cd $R
DATASETS=${DATASETS:-"ho3d hoi4d oakink graspxl"}
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/status/pod.log; }
push_status() { aws s3 sync --quiet $R/status $W/status; }
finish() {  # $1 = DONE|FAILED
  log "$1"; echo "$1" > $R/status/STATE; push_status
  if [ "$1" = DONE ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi
  sleep infinity
}
publish() {  # ds out_dir s3_prefix
  aws s3 cp --quiet $2/frames.parquet $3/frames.parquet && aws s3 cp --quiet $2/mask.parquet $3/mask.parquet && \
  aws s3 cp --quiet $2/summary.json $3/summary.json
}

log "start; vCPU: $(nproc); RAM: $(free -g | awk '/Mem/{print $2}') GB"
echo RUNNING > $R/status/STATE; push_status
pip install -q libigl==2.6.3 trimesh pandas pyarrow scipy pyyaml boto3 >> $R/status/pip.log 2>&1 || finish FAILED
apt-get -qq update >/dev/null 2>&1; apt-get -qq install -y unzip >/dev/null 2>&1
aws s3 cp --quiet $W/code.tar.gz . && rm -rf code && mkdir -p code && tar -xzf code.tar.gz -C code || finish FAILED
cd code
bash scripts/prepare_data.sh $R/data mano > $R/status/prep_mano.log 2>&1 || finish FAILED
M=$R/data/mano/mano_v1_2/models
syncer() { while true; do sleep 300; push_status; done; }
syncer &

for ds in $DATASETS; do
  out=$R/out/$ds
  if [ -f $R/status/$ds.done ]; then continue; fi
  if [ ! -f $R/status/prep_$ds.done ]; then
    log "prepare $ds"; push_status
    bash scripts/prepare_data.sh $R/data $ds > $R/status/prep_$ds.log 2>&1 || finish FAILED
    touch $R/status/prep_$ds.done
  fi
  if [ "$ds" = oakink ] && [ ! -d $R/data/oakink/image/anno ]; then log "oakink: no annotations, skipped"; continue; fi
  log "audit $ds"; push_status
  python -W ignore scripts/audit.py --dataset $ds --root $R/data/$ds --out $out --mano-dir $M --workers $(nproc) \
    > $R/status/audit_$ds.log 2>&1 || finish FAILED
  if [ "$ds" = graspxl ]; then
    dst=s3://xray-hands/eval/graspxl/penetration_v2_sample
    aws s3 cp --quiet $R/data/graspxl/sample_manifest.json $dst/sample_manifest.json
  else
    dst=s3://xray-hands/eval/$ds/penetration_v1
  fi
  publish $ds $out $dst || finish FAILED
  cp $out/summary.json $R/status/summary_$ds.json
  touch $R/status/$ds.done; log "audit $ds done -> $dst"; push_status
done

log "subsets"
python -W ignore scripts/build_subset.py --all --publish --out $R/out > $R/status/subsets.log 2>&1 || finish FAILED
finish DONE
