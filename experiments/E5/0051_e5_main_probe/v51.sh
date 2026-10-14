#!/bin/bash
VI=/home/mail/chenyuma/visint; O=$VI/logs/0051_prep_verify.txt
{ echo "## squeue $(date -u +%FT%TZ) [0051 verify]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
echo "wt0051 HEAD $(git -C $VI/wt/wt0051 rev-parse HEAD) status_lines=$(git -C $VI/wt/wt0051 status --short | wc -l)"
git -C $VI/repo merge-base --is-ancestor 43826228 origin/main && echo "43826228 in origin/main: yes"
cd $VI/wt/wt0051 && sha256sum prompts/server_hpc/0051-hpc-e5-main/queue.sh scripts/server/train_queue.sh configs/e5_train.json configs/e5_prompts.json
cd $VI/b200; echo "manifest lines for 12 starts: $(wc -l < $VI/tmp/sha_0051_starts.txt)"
for p in 0040/e3b_c3 0040/c3_warmup 0036/warmup 0036/e3b; do for s in 0 1 2; do echo "  $p seed$s: $(grep -c "  ckpt/$p/qwen3.5-2b/seed$s/" $VI/tmp/sha_0051_starts.txt) files"; done; done
t0=$(date +%s); nice -n 10 sha256sum -c $VI/tmp/sha_0051_starts.txt > $VI/logs/0051_starts_sha256c.txt 2>&1; rc=$?
echo "sha256sum -c rc=$rc $(( $(date +%s)-t0 ))s OK=$(grep -c ': OK$' $VI/logs/0051_starts_sha256c.txt) FAILED=$(grep -c FAILED $VI/logs/0051_starts_sha256c.txt) missing=$(grep -c 'No such file' $VI/logs/0051_starts_sha256c.txt)"
ls -d $VI/logs/e5_claims_0051 2>&1 | tail -1
echo "## squeue $(date -u +%FT%TZ) [0051 verify after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; echo "== done"; } > $O 2>&1
