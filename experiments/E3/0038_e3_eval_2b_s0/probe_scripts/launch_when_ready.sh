#!/bin/bash
# RUN-0038 §1/§2 (server CC): wait (detached) until RUN-0036's done.txt has all 7 seed-0 runs, then check each final checkpoint against
# ckpt_ledger.tsv and its <final>.sha256.txt, then launch eval.sh detached (setsid nohup). Any check failing -> do not launch, write status.
# Log: $PROJ/logs/0038/launch.log; status line in $PROJ/logs/0038/launch.status.
PROJ=/home/ubuntu/workspace/wzg/gjl/vision-internalization; REPO=$PROJ/repo; L=$PROJ/logs/0038
Q=$REPO/experiments/E3/0036_e3_train_2b; CK=$PROJ/ckpt/0036; M=qwen3.5-2b
NEED="warmup_s0 e3b_s0 e3a_full_s0 c3_s0 c1prime_s0 e3a_c1_s0 e3a_c2_s0"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L/launch.log; }
st() { echo "$(date -u +%FT%TZ) $*" > $L/launch.status; }
log "launcher pid $$ waiting for: $NEED"
while :; do
  miss=""; for n in $NEED; do grep -qxF "$n" $Q/driver/done.txt || miss="$miss $n"; done
  [ -z "$miss" ] && break
  st "waiting for RUN-0036 seed 0:$miss"
  if grep -qE "^[^ ]+ (blocked|done)" $PROJ/logs/train_0036.status 2>/dev/null && [ -n "$miss" ]; then
    log "train_0036.status is '$(cat $PROJ/logs/train_0036.status)' while still missing:$miss"; st "blocked: queue not running, missing:$miss"; exit 1; fi
  sleep 60
done
log "all 7 seed-0 runs in done.txt"
declare -A EXP=([warmup_s0]="$CK/warmup/$M/seed0/ckpt/final" [e3b_s0]="$CK/e3b/$M/seed0/ckpt/final" [e3a_full_s0]="$CK/e3a/full/$M/seed0/ckpt/final"
  [c3_s0]="$CK/c3/$M/seed0/ckpt/final" [c1prime_s0]="$CK/c1prime/$M/seed0/ckpt/final" [e3a_c1_s0]="$CK/e3a/c1/$M/seed0/ckpt/final" [e3a_c2_s0]="$CK/e3a/c2/$M/seed0/ckpt/final")
bad=0
for n in $NEED; do
  led=$(awk -F'\t' -v n=$n '$1==n{print $2}' $Q/ckpt_ledger.tsv | tail -1); fin=${EXP[$n]}
  case "$led" in "$fin"|"$fin/model") okp=yes;; *) okp=no;; esac
  if [ -f "$fin.sha256.txt" ]; then (cd $fin && sha256sum --quiet -c $fin.sha256.txt) >> $L/launch.log 2>&1 && oks=yes || oks=no; nf=$(wc -l < $fin.sha256.txt); else oks=missing; nf=0; fi
  log "$n ledger=$led expected=$fin path_ok=$okp sha256_ok=$oks files=$nf kb=$(du -sk $fin 2>/dev/null | cut -f1)"
  [ $okp = yes ] && [ $oks = yes ] || bad=1
done
if [ $bad -ne 0 ]; then log "checkpoint check failed; not launching"; st "blocked: checkpoint check failed"; exit 1; fi
log "checks passed; launching eval.sh (setsid nohup)"
setsid nohup bash /dev/shm/visint/tmp/wt0038/prompts/server/0038-b200-e3-eval-2b-s0/eval.sh > $PROJ/logs/eval_0038.driver.log 2>&1 < /dev/null &
sleep 5
log "eval.sh launched: $(ps -eo pid,sid,etime,cmd | grep '[0]038-b200-e3-eval-2b-s0/eval.sh' | head -1 | cut -c1-120)"
st "launched eval.sh"
