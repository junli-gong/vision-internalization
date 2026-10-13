#!/bin/bash
# RUN-0053 addendum-1 steps 1-5
VI=/home/mail/chenyuma/visint; O=$VI/logs/0053a_v3.txt; L=$VI/logs/git.lock; V3=$VI/wt/sync3/scripts/server/sync_push.sh
{ echo "## squeue $(date -u +%FT%TZ) [0053a before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
git -C $VI/repo status --short > $VI/logs/repo_status_before_v3.txt; echo "repo status lines before: $(wc -l < $VI/logs/repo_status_before_v3.txt)"
timeout 600 flock -w 300 $L -c "git -C $VI/repo fetch -q origin main"; echo "fetch rc=$? origin/main $(git -C $VI/repo rev-parse --short origin/main)"
git -C $VI/repo merge-base --is-ancestor 51e05fbf origin/main && echo "51e05fbf in origin/main: yes" || { echo "51e05fbf missing - STOP"; exit 1; }
[ -e $VI/wt/sync3 ] || git -C $VI/repo worktree add -q --no-checkout --detach $VI/wt/sync3 origin/main
git -C $VI/wt/sync3 sparse-checkout set --no-cone '/*' '!/experiments/*/' '/experiments/P0/'; echo "sparse rc=$?"
echo "sync3 HEAD $(git -C $VI/wt/sync3 rev-parse HEAD) status_lines=$(git -C $VI/wt/sync3 status --short | wc -l) experiments/: $(ls $VI/wt/sync3/experiments | tr '\n' ' ') du $(du -s --apparent-size -B1 $VI/wt/sync3 | cut -f1)"
a=$(sha256sum $V3 | cut -d' ' -f1); echo "v3 sha256 $a"; [ "$a" = 21e5892ef539b5095b21974978a36d44f28f197c7bd000ab6ddf1240a16092d4 ] || { echo "v3 sha mismatch - STOP"; exit 1; }
stopv2() { local tag=$1 P; P=$(cut -d= -f2 $VI/logs/sync_$tag.pid)
  for i in $(seq 1 120); do C=$(ps -o pid=,args= --ppid $P 2>/dev/null | awk '{print $1, $2}')
    if [ -z "$(ps -o pid= -p $P)" ]; then echo "v2 $tag pid $P already gone"; return 0; fi
    if [[ "$C" == *sleep* ]] && flock -n $L true; then kill $P ${C%% *}; sleep 2; ps -p $P -o pid= >/dev/null && echo "v2 $tag pid $P STILL ALIVE" || echo "$(date -u +%FT%TZ) v2 $tag stopped (pid $P, child ${C%% *})"; return 0; fi
    sleep 20; done; echo "v2 $tag: could not catch it sleeping - STOP"; return 1; }
startv3() { local tag=$1; shift; cd $VI && setsid nohup bash $V3 $VI/repo $L 1800 $tag "$@" > $VI/logs/sync_$tag.log 2>&1 < /dev/null & echo "sync_pid=$!" > $VI/logs/sync_$tag.pid; echo "$(date -u +%FT%TZ) v3 $tag started pid $!"; }
stopv2 0045 && startv3 0045 experiments/E3/0045_hpc_train_hpcgpu06 experiments/E3/0045_hpc_train_hpcgpu07
stopv2 0050 && startv3 0050 experiments/E3/0050_hpc_eval_08b_hpcgpu07 experiments/E3/0050_hpc_e3b_hpcgpu07
for tag in 0045 0050; do for i in $(seq 1 60); do s=$(cat $VI/logs/sync_$tag.status); t=$(date -u -d "${s%% *}" +%s 2>/dev/null || echo 0); [ $t -gt $(( $(date +%s) - 900 )) ] && break; sleep 20; done; echo "sync_$tag status: $s"; tail -3 $VI/logs/sync_$tag.log; done
echo "=========== verify first round on origin"
timeout 600 flock -w 300 $L -c "git -C $VI/repo fetch -q origin main"; echo "origin/main $(git -C $VI/repo log --oneline -1 origin/main)"
for d in experiments/E3/0045_hpc_train_hpcgpu06 experiments/E3/0045_hpc_train_hpcgpu07 experiments/E3/0050_hpc_eval_08b_hpcgpu07 experiments/E3/0050_hpc_e3b_hpcgpu07; do
  big=$(git -C $VI/repo ls-tree -r -l origin/main -- $d | awk '$4>5242880' | wc -l); man=$(git -C $VI/repo cat-file -e origin/main:$d/LARGE_FILES.manifest.tsv 2>/dev/null && echo yes || echo no)
  n=$( [ $man = yes ] && git -C $VI/repo show origin/main:$d/LARGE_FILES.manifest.tsv | grep -vc '^#' || echo 0); echo "$d: files>5MB in tree=$big manifest=$man manifest_rows=$n"; done
d=experiments/E3/0050_hpc_eval_08b_hpcgpu07; line=$(git -C $VI/repo show origin/main:$d/LARGE_FILES.manifest.tsv 2>/dev/null | grep -v '^#' | head -2 | tail -1); echo "sample manifest row: $line"
f=$(echo "$line" | cut -f1); w=$(echo "$line" | cut -f3); [ -n "$f" ] && echo "lustre sha256 $(sha256sum "$VI/repo/$d/$f" | cut -d' ' -f1) manifest $w"
git -C $VI/repo status --short > $VI/logs/repo_status_after_v3.txt; echo "repo status lines before/after: $(wc -l < $VI/logs/repo_status_before_v3.txt)/$(wc -l < $VI/logs/repo_status_after_v3.txt)"
echo "## squeue $(date -u +%FT%TZ) [0053a after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; echo "== done"; } > $O 2>&1
