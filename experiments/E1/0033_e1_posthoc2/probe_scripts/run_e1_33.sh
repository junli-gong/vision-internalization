#!/bin/bash
# RUN-0033 driver (E1 post-hoc follow-ups; worktree pinned at the task's start commit, /dev/shm/visint/tmp/wt0033; --out = main repo's run dir).
# Launched detached (setsid nohup; STANDBY "长时间运行的驱动"). Per model: ledger -> download + verify -> letter_swap (rel_depth_nyu)
# -> [Qwen: forced_poly (refexp_poly)] -> delete (except qwen3.5-0.8b) -> done_<model> -> push (probe_scripts/push_model.sh, main checkout)
# -> heartbeat "running 0033 k/8". Heartbeat also every 30 min with the current step. Before each model: stop if cumulative run_e1
# wall > 2 h (task book §4; cap_reached). After all models: the two summaries (CPU, in the worktree), pushed. Disk red line: stop.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
export VISINT_DATA_ROOT=$PROJ/data
export NLTK_DATA=$PROJ/nltk_data
export TMPDIR=/dev/shm/visint/tmp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$PROJ/logs/0033; DC=$PROJ/scripts/diskcheck33.sh; CAP=7200; ST=$PROJ/logs/standby.status
WT=/dev/shm/visint/tmp/wt0033; OUT=$REPO/experiments/E1/0033_e1_posthoc2; PUSH="bash $OUT/probe_scripts/push_model.sh"
VER=$REPO/experiments/P0/0009_ifeval_rescore/probe_scripts/p12_verify_weights.py
cd $WT
echo "=== worktree $(git rev-parse HEAD) dirty_files=$(git status --short | wc -l) pid $$" | tee -a $L/driver.log
[ -f $L/gpu_s.txt ] || echo 0 > $L/gpu_s.txt
[ -f $L/ndone.txt ] || echo 0 > $L/ndone.txt
echo "starting" > $L/current.txt
( while kill -0 $$ 2>/dev/null; do echo "$(date -u +%FT%TZ) running 0033 $(cat $L/ndone.txt)/8 ($(cat $L/current.txt))" > $ST; sleep 1800; done ) &
HB=$!; trap "kill $HB 2>/dev/null" EXIT
beat() { echo "$1" > $L/current.txt; echo "$(date -u +%FT%TZ) running 0033 $(cat $L/ndone.txt)/8 ($1)" > $ST; }
cap() { g=$(cat $L/gpu_s.txt); if [ $g -gt $CAP ]; then echo "=== cumulative run_e1 wall ${g}s > 2h; stopping before $1" | tee -a $L/driver.log; touch $L/cap_reached; echo "=== driver stopped $(date -Is)" | tee -a $L/driver.log; echo "$(date -u +%FT%TZ) blocked e1-posthoc2-over-budget" > $ST; exit 0; fi; }
disk() { $DC "$1" | tee -a $L/driver.log; if [ ${PIPESTATUS[0]} -eq 3 ]; then echo "DISK RED LINE" | tee -a $L/driver.log; touch $L/disk_red; echo "$(date -u +%FT%TZ) blocked disk-red-line" > $ST; exit 3; fi; }
step() { name=$1; shift; t0=$(date +%s); beat "$name"; echo "=== $name start $(date -Is) (run_e1 cumulative $(cat $L/gpu_s.txt)s)" | tee -a $L/driver.log
         ( time "$@" ) > $L/$name.log 2>&1; RC=$?; t1=$(date +%s); echo $(( $(cat $L/gpu_s.txt) + t1 - t0 )) > $L/gpu_s.txt
         echo "=== $name rc=$RC $(date -Is) ($((t1 - t0))s; cumulative $(cat $L/gpu_s.txt)s)" | tee -a $L/driver.log; disk "after $name"; }
cstep() { name=$1; shift; t0=$(date +%s); beat "$name"; echo "=== $name start $(date -Is) (CPU, not counted)" | tee -a $L/driver.log
          ( time "$@" ) > $L/$name.log 2>&1; RC=$?; echo "=== $name rc=$RC $(date -Is) ($(( $(date +%s) - t0 ))s; CPU)" | tee -a $L/driver.log; }
for M in "$@"; do
  [ -f $L/done_$M ] && { echo "=== $M already done; skip" | tee -a $L/driver.log; continue; }
  cap "$M"
  echo "=== $M model start $(date -Is)" | tee -a $L/driver.log; beat "$M download"
  disk "before download $M"
  python $VER "$M" --download 2>&1 | grep -vE "Warn|it/s\]" | tee -a $L/driver.log
  echo "=== $M weights verify rc=${PIPESTATUS[0]}" | tee -a $L/driver.log
  disk "after download $M"
  step "${M}__letter_swap" python scripts/run_e1.py --model "$M" --task rel_depth_nyu --subset letter_swap --data-root $PROJ/data --out $OUT
  R1=$RC; R2=-
  case "$M" in qwen3.5-*)
    step "${M}__forced_poly" python scripts/run_e1.py --model "$M" --task refexp_poly --subset forced_poly --data-root $PROJ/data --out $OUT; R2=$RC ;;
  esac
  if [ "$M" != "qwen3.5-0.8b" ]; then
    python - "$M" <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import scan_cache_dir
e = registry.get_model(sys.argv[1]); ci = scan_cache_dir('/dev/shm/visint/hf-hub')
s = ci.delete_revisions(e.revision); print('delete', e.repo, e.revision, 'expected_freed', s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
    disk "after delete $M"
  fi
  touch $L/done_$M; echo $(( $(cat $L/ndone.txt) + 1 )) > $L/ndone.txt
  echo "=== $M model done $(date -Is)" | tee -a $L/driver.log
  beat "push $M"
  ( cd $REPO && $PUSH "$M" "E1 post-hoc 2: letter_swap NYU rc=$R1; forced_poly rc=$R2" ) > $L/push_$M.log 2>&1
  echo "=== $M push rc=$? $(tail -1 $L/push_$M.log)" | tee -a $L/driver.log
done
beat "summaries"
cstep summary_validity_d72 python scripts/summarize_e1_posthoc.py validity --debiased experiments/E1/0030_e1_posthoc --letter-swap $OUT \
  --free experiments/E1/0027_e1_main --main experiments/E1/0027_e1_main --out-json $OUT/validity_d72.json --out-md $OUT/validity_d72.md
cstep summary_ownbox python scripts/summarize_e1_posthoc.py ownbox --main experiments/E1/0027_e1_main --think experiments/E1/0028_e1_think \
  --forced experiments/E1/0030_e1_posthoc --forced-poly $OUT --data-root $PROJ/data --out-json $OUT/ownbox.json --out-md $OUT/ownbox.md
echo "=== driver done $(date -Is)" | tee -a $L/driver.log
echo "$(date -u +%FT%TZ) running 0033 8/8 (report)" > $ST
