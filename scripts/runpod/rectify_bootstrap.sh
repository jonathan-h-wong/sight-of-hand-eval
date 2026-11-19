#!/usr/bin/env bash
# Runpod bootstrap for the full ARCTIC + DexYCB rectification.
# Pod start command (image runpod/pytorch, env AWS_* from Runpod secrets):
#   bash -c 'git clone -q --depth 1 -b claude/rectify-v1 https://github.com/jonathan-h-wong/sight-of-hand-eval /workspace/rect/code \
#            && bash /workspace/rect/code/scripts/runpod/rectify_bootstrap.sh'
# Inputs (extracted ARCTIC/DexYCB labels, SDF cache, ARCTIC mocap) come from $W/inputs.tar.gz, MANO from raw/mano.
# Control and results go through S3 only: logs/status -> $W/status, outputs -> $W/out.
#   phase 1 (bench): 5 ARCTIC + 5 DexYCB sequences, one process each, timed
#   phase 2 (full):  every sequence, sharded over NPROC processes, resumable (finished sequences are skipped)
# On success the pod removes itself; on failure it stops (restart = rerun with whatever code is in S3).
set -uo pipefail
W=${WORK:-s3://xray-hands/work/rectify_v1}
R=/workspace/rect; mkdir -p $R/status && cd $R
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $R/status/pod.log; }
push_status() { aws s3 sync --quiet $R/status $W/status; }
finish() {  # $1 = DONE|FAILED
  log "$1"; echo "$1" > $R/status/STATE; push_status
  aws s3 sync --quiet $R/out $W/out --exclude "*/sdf_cache/*" --exclude "*.tmp.npz"
  if [ "$1" = DONE ]; then runpodctl remove pod "$RUNPOD_POD_ID"; else runpodctl stop pod "$RUNPOD_POD_ID"; fi
  sleep infinity
}

log "start; GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader); vCPU: $(nproc)"
echo RUNNING > $R/status/STATE
pip install -q awscli >/dev/null 2>&1 || finish FAILED
push_status
pip install -q libigl==2.6.3 trimesh pandas pyarrow scipy pyyaml >> $R/status/pip.log 2>&1 || finish FAILED
[ -d code ] || git clone -q --depth 1 -b ${BRANCH:-claude/rectify-v1} https://github.com/jonathan-h-wong/sight-of-hand-eval code || finish FAILED
git -C code log --oneline -1 >> $R/status/pod.log
if [ ! -d data/arctic ]; then
  mkdir -p data && aws s3 cp --quiet $W/inputs.tar.gz - | tar -xzf - -C data || finish FAILED
  aws s3 cp --quiet s3://xray-hands/raw/mano/mano_v1_2.zip data/ && (cd data && unzip -q -o mano_v1_2.zip) || finish FAILED
fi
python -c "import torch, igl; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda ok')" >> $R/status/pod.log 2>&1 || finish FAILED
log "setup done"; push_status

M=$R/data/mano_v1_2/models; C=$R/data/sdf_cache
run() {  # dataset out extra...
  local ds=$1 out=$2; shift 2
  python -W ignore code/scripts/rectify.py --dataset $ds --root $R/data/$ds --out $out --mano-dir $M --sdf-cache $C --device cuda "$@"
}
gpu_sampler() { while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed "s/^/$(date -u +%T) /" >> $R/status/gpu.log; \
                echo "$(date -u +%T) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/status/gpu.log; sleep 30; done; }
syncer() { while true; do sleep 600; push_status; aws s3 sync --quiet $R/out $W/out --exclude "*/sdf_cache/*" --exclude "*.tmp.npz"; done; }
gpu_sampler & syncer &

# ---- phase 1: bench (outputs land in the final dirs, so they count toward the full run)
if [ ! -f $R/status/bench.done ]; then
  log "bench start"; push_status
  run arctic $R/out/arctic --limit 5 > $R/status/bench_arctic.log 2>&1 & A=$!
  run dexycb $R/out/dexycb --limit 5 > $R/status/bench_dexycb.log 2>&1 & B=$!
  wait $A $B
  grep -q ERROR $R/status/bench_*.log && finish FAILED
  touch $R/status/bench.done; log "bench done"; push_status
fi

# ---- phase 2: full run. CPU (exact penetration check, SDF lookups setup) is the other bottleneck, so the
# number of processes follows the vCPU count; ARCTIC gets a share proportional to its bench cost.
NPROC=${NPROC:-$(( $(nproc) / 2 > 12 ? 12 : $(nproc) / 2 ))}; [ $NPROC -lt 2 ] && NPROC=2
NA=${NA:-$(( NPROC / 2 ))}; ND=$(( NPROC - NA ))
export OMP_NUM_THREADS=$(( $(nproc) / NPROC > 0 ? $(nproc) / NPROC : 1 ))
log "full start: $NA arctic + $ND dexycb processes, OMP_NUM_THREADS=$OMP_NUM_THREADS"; push_status
for i in $(seq 0 $((NA - 1))); do run arctic $R/out/arctic --shard $i/$NA > $R/status/full_arctic_$i.log 2>&1 & done
for i in $(seq 0 $((ND - 1))); do run dexycb $R/out/dexycb --shard $i/$ND > $R/status/full_dexycb_$i.log 2>&1 & done
while pgrep -f "rectify.py" > /dev/null; do
  sleep 300
  echo "$(date -u +%FT%TZ) arctic $(find $R/out/arctic/seqs -name '*.npz' ! -name '*.tmp.npz' | wc -l)/301 dexycb $(find $R/out/dexycb/seqs -name '*.npz' ! -name '*.tmp.npz' | wc -l)/1000" >> $R/status/progress.log
done
log "full done"
for ds in arctic dexycb; do python -W ignore code/scripts/rectify.py --dataset $ds --root $R/data/$ds --out $R/out/$ds --summarize > $R/status/summary_$ds.log 2>&1; done
python -W ignore code/scripts/check_marker_fit.py --rect $R/out/arctic --root $R/data/arctic --mocap $R/data/mocap_npy \
  --mano-dir $M --out $R/out/arctic/marker_fit.parquet > $R/status/marker_arctic.log 2>&1
finish DONE
