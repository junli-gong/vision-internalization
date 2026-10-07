#!/bin/bash
# RUN-0032: server-CC hold. Waits for done_<model>, then stops the driver before it starts the next item (pending the lead's
# confirmation of the 15 GPU h cap overshoot), and removes any partial download of the next model. usage: bash hold_before_next.sh <model> <next model>
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
L=$PROJ/logs/0032; M=$1; NEXT=$2; D=$(cat $L/driver.pid)
until [ -f $L/done_$M ]; do kill -0 $D 2>/dev/null || exit 1; sleep 0.2; done
pkill -TERM -P $D; kill -TERM $D; sleep 2; pkill -KILL -f "p12_verify_weights.py $NEXT" 2>/dev/null
echo "=== HOLD (server CC) $(date -Is): driver stopped after $M, before $NEXT; GPU cumulative $(cat $L/gpu_s.txt)s; projected $NEXT ~7 h would exceed the 15 h cap (D59-4) -> waiting for confirmation" | tee -a $L/driver.log
cd /dev/shm/visint/tmp/wt0032 && python - "$NEXT" <<'PY' 2>&1 | grep -v Warn | tee -a $L/driver.log
import sys; sys.path.insert(0, '.')
from src.adapters import registry
from huggingface_hub import scan_cache_dir
e = registry.get_model(sys.argv[1]); ci = scan_cache_dir('/dev/shm/visint/hf-hub')
revs = [rv.commit_hash for r in ci.repos for rv in r.revisions if r.repo_id == e.repo]
if revs:
    s = ci.delete_revisions(*revs); print('partial download removed', e.repo, revs, s.expected_freed_size); s.execute()
print('remaining', [(r.repo_id, r.size_on_disk) for r in scan_cache_dir('/dev/shm/visint/hf-hub').repos])
PY
rm -rf /dev/shm/visint/hf-hub/models--$(cd /dev/shm/visint/tmp/wt0032 && python -c "import sys; sys.path.insert(0,'.'); from src.adapters import registry; print(registry.get_model('$NEXT').repo.replace('/','--'))") 2>/dev/null
$PROJ/scripts/diskcheck32.sh "hold before $NEXT" | tee -a $L/driver.log
