#!/bin/bash
# RUN-0034 driver (B200 E3 prep), worktree pinned at the task's start commit (/dev/shm/visint/tmp/wt0034); launched detached (setsid nohup).
# 2026-10-07 04:3x: --out renamed to $OUT/0034_e3_consistency, $OUT/0034_e3_throughput (train scripts require a run-numbered basename;
# the first pass with $OUT/consistency exited rc=2); landing/dedupe guarded by done_land/done_dedupe markers.
# Phases: mem before -> land all (land_train.py) -> mem after -> manifest compare with A800 -> link E1 data -> dedupe -> exclusions compare
# (stop if not identical) -> push -> qwen3.5-2b: download+verify, consistency e3b/e3a (50 steps), throughput 6 variants (20 steps) ->
# delete -> push -> qwen3.5-9b: download+verify, throughput 6 variants -> delete -> push -> rm checkpoints -> push.
# Heartbeat "running 0034 <phase>" at every step and every 30 min. Watchdog every 30 s: MemAvailable < 60 GB -> stop the driver's process
# group and delete the training data from the RAM disk (task book §6). Disk red line (/ < 3 GB): stop.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
source $PROJ/venv/bin/activate
export TMPDIR=/dev/shm/visint/tmp
export PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
L=$PROJ/logs/0034; DC=$PROJ/scripts/diskcheck34.sh; ST=$PROJ/logs/standby.status
WT=/dev/shm/visint/tmp/wt0034; OUT=$REPO/experiments/E3/0034_b200_prep; PUSH="bash $OUT/probe_scripts/push_phase.sh"
TRAIN=/dev/shm/visint/train_data; CK=/dev/shm/visint/tmp/e3_ckpt_0034
A800=experiments/P0/0026_a800_delta/results_0026
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l) pid $$ pgid $(ps -o pgid= $$ | tr -d ' ')" | tee -a $L/driver.log
echo starting > $L/current.txt
( while kill -0 $$ 2>/dev/null; do echo "$(date -u +%FT%TZ) running 0034 $(cat $L/current.txt)" > $ST; sleep 1800; done ) &
HB=$!
PG=$(ps -o pgid= $$ | tr -d ' ')
( trap '' TERM; while kill -0 $$ 2>/dev/null; do m=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
    if [ "$m" -lt 60000000000 ]; then echo "=== MEMORY RED LINE $(date -Is): MemAvailable ${m} B < 60 GB; stopping driver, deleting $TRAIN" | tee -a $L/driver.log
      touch $L/mem_red; echo "$(date -u +%FT%TZ) blocked mem-red-line" > $ST; kill -TERM -- -$PG; sleep 3; rm -rf $TRAIN $CK; $DC "after mem red line cleanup" >> $L/driver.log; exit 0; fi
    sleep 30; done ) &
WD=$!; trap "kill $HB $WD 2>/dev/null" EXIT
beat() { echo "$1" > $L/current.txt; echo "$(date -u +%FT%TZ) running 0034 $1" > $ST; }
disk() { $DC "$1" | tee -a $L/driver.log; r=${PIPESTATUS[0]}; if [ $r -eq 3 ]; then echo "DISK RED LINE" | tee -a $L/driver.log; echo "$(date -u +%FT%TZ) blocked disk-red-line" > $ST; exit 3; fi
         if [ $r -eq 4 ]; then echo "MEMORY RED LINE (ledger)" | tee -a $L/driver.log; rm -rf $TRAIN $CK; echo "$(date -u +%FT%TZ) blocked mem-red-line" > $ST; exit 4; fi; }
# step <name> <gpu 1|0> cmd...
step() { name=$1; gpu=$2; shift 2; t0=$(date +%s); beat "$name"; echo "=== $name start $(date -Is)" | tee -a $L/driver.log
         ( time "$@" ) > $L/$name.log 2>&1; RC=$?; t1=$(date +%s)
         echo "=== $name rc=$RC $(date -Is) ($((t1 - t0))s; gpu=$gpu)" | tee -a $L/driver.log; disk "after $name"; }
fetch() { beat "download $1"; disk "before download $1"; HF_HUB_OFFLINE=0 python $VER "$1" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
          echo "=== $1 weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log; disk "after download $1"; }
drop() { python - "$1" <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import scan_cache_dir
e = registry.get_model(sys.argv[1]); ci = scan_cache_dir('/dev/shm/visint/hf-hub')
s = ci.delete_revisions(e.revision); print('delete', e.repo, e.revision, 'expected_freed', s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
  disk "after delete $1"; }
push() { beat "push $1"; ( cd $REPO && $PUSH "$1" "$2" ) > $L/push_$1.txt 2>&1; echo "=== push $1 rc=$? $(tail -1 $L/push_$1.txt)" | tee -a $L/driver.log; }
trn() { local name=$1; shift; step "$name" 1 env HF_HUB_OFFLINE=1 python "$@" --data-root $TRAIN --resume; }

# --- §2-§4: memory, landing, manifests, dedupe ---
if [ ! -f $L/done_land ]; then
  { date -u; free -g; df -h /dev/shm; } > $L/mem_before_land.txt; cat $L/mem_before_land.txt >> $L/driver.log
  m=$(awk '/MemAvailable/{print $2*1024}' /proc/meminfo)
  if [ "$m" -lt 120000000000 ]; then echo "=== MemAvailable ${m} B < 120 GB; not landing (task book §2)" | tee -a $L/driver.log; echo "$(date -u +%FT%TZ) blocked mem-below-120GB" > $ST; exit 0; fi
  mkdir -p $TRAIN
  step land 0 python scripts/data/land_train.py all --root $TRAIN --tmpdir $TMPDIR --workers 8
  echo $RC > $L/land_rc.txt
  { date -u; free -g; df -h /dev/shm; } > $L/mem_after_land.txt; cat $L/mem_after_land.txt >> $L/driver.log
  touch $L/done_land
fi
if [ ! -f $L/done_dedupe ]; then
step manifest_compare 0 python $OUT/probe_scripts/p33_manifest_compare.py $TRAIN $WT $OUT/land_compare
[ -e $TRAIN/refcoco ] || ln -s $PROJ/data/refcoco $TRAIN/refcoco
[ -e $TRAIN/nyuv2 ] || ln -s $PROJ/data/nyuv2 $TRAIN/nyuv2
ls -la $TRAIN | tee -a $L/driver.log
step dedupe 0 python scripts/data/dedupe_report.py --train-root $TRAIN --eval-manifest experiments/P0/0023_eval_hashes/eval_hashes_b200.json --out $TRAIN/dedupe
mkdir -p $OUT/dedupe; cp $TRAIN/dedupe/* $OUT/dedupe/ 2>/dev/null
step exclusions_compare 0 python $OUT/probe_scripts/p34_exclusions_compare.py $TRAIN/dedupe/train_exclusions.json $A800/dedupe/train_exclusions.json $OUT/dedupe_compare
EXC=$RC
for s in refcoco_train nyu_depth_train nyu_normals ade20k; do [ -f $TRAIN/$s/MANIFEST.json ] && { mkdir -p $OUT/land_manifests; gzip -9 -n -c $TRAIN/$s/MANIFEST.json > $OUT/land_manifests/MANIFEST_$s.json.gz; }; done
push land_dedupe "land rc=$(cat $L/land_rc.txt), manifest compare, dedupe, exclusions compare rc=$EXC"
if [ $EXC -ne 0 ]; then echo "=== exclusions differ from A800; stopping before §5 (task book §4)" | tee -a $L/driver.log; echo "$(date -u +%FT%TZ) blocked dedupe-mismatch" > $ST; exit 0; fi
touch $L/done_dedupe
fi

# --- §5: consistency and throughput ---
M=qwen3.5-2b; fetch $M
trn consistency_e3b_$M scripts/train_e3b.py --model $M --supervision grid --seed 0 --out $OUT/0034_e3_consistency --ckpt-root $CK/consistency --stop-after-steps 50
trn consistency_e3a_$M scripts/train_e3a.py --model $M --variant full --seed 0 --out $OUT/0034_e3_consistency --ckpt-root $CK/consistency --stop-after-steps 50
for M in qwen3.5-2b qwen3.5-9b; do
  [ $M = qwen3.5-9b ] && fetch $M
  T=$OUT/0034_e3_throughput; TC=$CK/throughput
  trn tput_e3a_full_$M scripts/train_e3a.py --model $M --variant full --seed 0 --out $T --ckpt-root $TC --stop-after-steps 20
  trn tput_e3a_c1_$M   scripts/train_e3a.py --model $M --variant c1   --seed 0 --out $T --ckpt-root $TC --stop-after-steps 20
  trn tput_e3a_c2_$M   scripts/train_e3a.py --model $M --variant c2   --seed 0 --out $T --ckpt-root $TC --stop-after-steps 20
  trn tput_e3b_$M      scripts/train_e3b.py --model $M --supervision grid --seed 0 --out $T --ckpt-root $TC --stop-after-steps 20
  trn tput_c1prime_$M  scripts/train_e3b.py --model $M --supervision grid --variant c1prime --seed 0 --out $T --ckpt-root $TC --stop-after-steps 20
  trn tput_c3_$M       scripts/train_e3b.py --model $M --supervision text --seed 0 --out $T --ckpt-root $TC --stop-after-steps 20
  du -sb $CK 2>/dev/null | tee -a $L/driver.log
  drop $M
  RCS=$(grep -oE "=== (consistency|tput)_[a-z0-9_]+_$M rc=[0-9]+" $L/driver.log | tail -8 | sed -E "s/=== //; s/_$M//" | paste -sd' ')
  push $M "$M steps: $RCS"
done
echo "checkpoints before delete: $(du -sb $CK 2>/dev/null | cut -f1) B, $(find $CK -type f 2>/dev/null | wc -l) files" | tee -a $L/driver.log
rm -rf $CK; disk "after delete checkpoints"
{ date -u; free -g; df -h /dev/shm; } > $L/mem_end.txt
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
push final "driver done; checkpoints deleted; training data kept on /dev/shm"
echo "$(date -u +%FT%TZ) running 0034 report" > $ST
