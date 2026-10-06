#!/bin/bash
# RUN-0032 driver (E5 test split) (worktree pinned at the task's start commit, /dev/shm/visint/tmp/wt0032; --out = main repo's run dir, absolute).
# 0) trivial baselines (CPU, not counted). Then items in priority order; before each item: stop if cumulative GPU time > 15 h
#    (task book §3/§4: finish the current model, then stop; writes cap_reached). GPU time = run_e5 zeroshot/cot walls + LISA inference walls.
#    model: ledger -> download + verify -> zeroshot+cot a c d e -> zeroshot+cot b (200/bin) -> delete (except qwen3.5-0.8b) -> done_<model>.
#    lisa:  download + verify (p18_download.py) -> export a b c d (eval) and b (200/bin) -> rehearsal --limit 5 + overlays -> full (both
#           exports) -> score external (a c d from the first export, b from the 200/bin one) -> delete LISA + CLIP -> delete exports -> done_lisa.
# Disk red line: stop.   usage: bash run_e5_32.sh <item> [...]   (item = model key or "lisa")
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0032; DC=$PROJ/scripts/diskcheck32.sh; CAP=54000
WT=/dev/shm/visint/tmp/wt0032; OUT=$REPO/experiments/E5/0032_e5_test
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
DL=$REPO/experiments/P0/0015_lisa_smoke/probe_scripts/p18_download.py
EXP=$PROJ/tmp/e5_inputs_test_0032; EXPB=$PROJ/tmp/e5_inputs_test_b200_0032
LISA_W=/dev/shm/visint/hf-hub/models--xinlai--LISA-7B-v1/snapshots/43c754eef75871fcc11c84d3930a402f7b0a754f
CLIP_W=/dev/shm/visint/hf-hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l)" | tee -a $L/driver.log
[ -f $L/gpu_s.txt ] || echo 0 > $L/gpu_s.txt
cap() { g=$(cat $L/gpu_s.txt); if [ $g -gt $CAP ]; then echo "=== cumulative GPU ${g}s > 15h; stopping before $1" | tee -a $L/driver.log; touch $L/cap_reached; echo "=== driver stopped $(date -Is)" | tee -a $L/driver.log; exit 0; fi; }
disk() { $DC "$1" | tee -a $L/driver.log; if [ ${PIPESTATUS[0]} -eq 3 ]; then echo "DISK RED LINE" | tee -a $L/driver.log; touch $L/disk_red; exit 3; fi; }
# step <name> <gpu:1|0> cmd...   (rc in $RC)
step() { name=$1; gpu=$2; shift 2; t0=$(date +%s); echo "=== $name start $(date -Is) (GPU cumulative $(cat $L/gpu_s.txt)s)" | tee -a $L/driver.log
         ( time "$@" ) > $L/$name.log 2>&1; RC=$?; t1=$(date +%s); [ $gpu = 1 ] && echo $(( $(cat $L/gpu_s.txt) + t1 - t0 )) > $L/gpu_s.txt
         echo "=== $name rc=$RC $(date -Is) ($((t1 - t0))s; gpu=$gpu; GPU cumulative $(cat $L/gpu_s.txt)s)" | tee -a $L/driver.log; disk "after $name"; }
delete_rev() { python - "$@" <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from huggingface_hub import scan_cache_dir
revs = sys.argv[1:]
if revs == ['lisa']:
    revs = ['43c754eef75871fcc11c84d3930a402f7b0a754f', '32bd64288804d66eefd0ccbe215aa642df71cc41']
else:
    from src.adapters import registry
    revs = [registry.get_model(revs[0]).revision]
ci = scan_cache_dir('/dev/shm/visint/hf-hub'); s = ci.delete_revisions(*revs)
print('delete', revs, 'expected_freed', s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
}
if [ ! -f $L/done_baseline ]; then
  step baseline__acde 0 python scripts/run_e5.py --system baseline --task a c d e --out $OUT
  step baseline__b 0 python scripts/run_e5.py --system baseline --task b --n-per-bin 200 --out $OUT
  touch $L/done_baseline
fi
lisa() {
  disk "before download lisa"
  ( HF_HOME=$ROOT/.hf python $DL $OUT/weights_download.json ) > $L/lisa__download.log 2>&1; rc=$?
  echo "=== lisa weights download+verify rc=$rc" | tee -a $L/driver.log; disk "after download lisa"
  [ $rc -ne 0 ] && { echo "=== lisa: weights verify failed; skipping LISA" | tee -a $L/driver.log; return; }
  step lisa__export 0 python scripts/run_e5.py --export-inputs $EXP --task a b c d --subset eval
  step lisa__export_b200 0 python scripts/run_e5.py --export-inputs $EXPB --task b --subset eval --n-per-bin 200
  LENV="env TMPDIR=$PROJ/tmp PIP_NO_CACHE_DIR=1 HF_HOME=$ROOT/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
  LRUN="$PROJ/ext/lisa/venv/bin/python scripts/external/lisa_e5_infer.py --lisa-src $PROJ/ext/lisa/LISA --lisa-weights $LISA_W --clip-weights $CLIP_W"
  step lisa__limit5 1 $LENV $LRUN --inputs $EXP --out $OUT/lisa_predictions_limit5 --limit 5
  if [ $RC -ne 0 ]; then echo "=== lisa: rehearsal rc=$RC; skipping LISA full/score" | tee -a $L/driver.log
  else
    step lisa__overlays 0 python $OUT/probe_scripts/p23_overlays.py $OUT/lisa_predictions_limit5 $EXP $OUT/overlays 2
    step lisa__full 1 $LENV $LRUN --inputs $EXP --out $OUT/lisa_predictions
    step lisa__full_b200 1 $LENV $LRUN --inputs $EXPB --out $OUT/lisa_predictions_b200
    step lisa__score_acd 0 python scripts/run_e5.py --system external --model lisa-7b-v1 --external-dir $OUT/lisa_predictions --task a c d --out $OUT
    step lisa__score_b 0 python scripts/run_e5.py --system external --model lisa-7b-v1 --external-dir $OUT/lisa_predictions_b200 --task b --n-per-bin 200 --out $OUT
  fi
  delete_rev lisa; disk "after delete lisa"
  for d in $EXP $EXPB; do echo "$(basename $d) before delete: $(du -sb $d | cut -f1) B, $(find $d -type f | wc -l) files" | tee -a $L/driver.log; rm -rf $d; done
  disk "after delete exports"
}
for M in "$@"; do
  cap "$M"
  echo "=== $M item start $(date -Is)" | tee -a $L/driver.log
  if [ "$M" = lisa ]; then lisa; touch $L/done_lisa; echo "=== lisa item done $(date -Is)" | tee -a $L/driver.log; continue; fi
  disk "before download $M"
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  disk "after download $M"
  step "${M}__acde" 1 python scripts/run_e5.py --system zeroshot cot --model "$M" --task a c d e --out $OUT
  step "${M}__b" 1 python scripts/run_e5.py --system zeroshot cot --model "$M" --task b --n-per-bin 200 --out $OUT
  if [ "$M" != "qwen3.5-0.8b" ]; then delete_rev "$M"; disk "after delete $M"; fi
  touch $L/done_$M
  echo "=== $M item done $(date -Is)" | tee -a $L/driver.log
done
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
