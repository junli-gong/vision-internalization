#!/bin/bash
# RUN-0053 steps 1-2 (+ tmp listing for step 3; no deletion here)
VI=/home/mail/chenyuma/visint; O=$VI/logs/0053_step12.txt; P0=experiments/P0/0001_env_probe/raw/p6/A/coco_val2017/coco_val2017_coco_urls.txt
{ echo "## squeue $(date -u +%FT%TZ) [0053 before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
q=$(timeout 60 lfs quota -u chenyuma /home | sed -n 3p); echo "quota line: $q"; echo "quota delta KB before: $(( $(echo $q | awk '{print $2}') - 7059403936 ))"
for w in wt0049 wt0051 wt0052 sync; do W=$VI/wt/$w; pat="$VI/wt/${w:0:1}[${w:1:1}]${w:2}"
  echo "=========== $w"; echo "processes using it:"; pgrep -u chenyuma -af "$pat([/ ]|$)" | cut -c1-150
  for n in hpcgpu06 hpcgpu07; do echo "  $n: $(ssh -n $n "pgrep -u chenyuma -af '$pat([/ ]|\$)' | cut -c1-120")"; done
  st=$(git -C $W status --short | wc -l); echo "status_lines before=$st; du before: $(du -s --apparent-size -B1 $W | cut -f1)"
  if [ "$st" != 0 ]; then echo "SKIP $w: status not empty"; continue; fi
  git -C $W sparse-checkout set --no-cone '/*' '!/experiments/*/' '/experiments/P0/'; echo "sparse rc=$?"
  echo "experiments/: $(ls $W/experiments/ | tr '\n' ' ')"; sha256sum $W/$P0 | cut -c1-16
  echo "status_lines after=$(git -C $W status --short | wc -l); du after: $(du -s --apparent-size -B1 $W | cut -f1)"
done
echo "=========== sha256 of runtime scripts"
sha256sum $VI/wt/wt0051/prompts/server_hpc/0051-hpc-e5-main/queue.sh $VI/wt/wt0052/prompts/server_hpc/0052-hpc-06-4b/run.sh $VI/wt/wt0052/prompts/server_hpc/0052-hpc-06-4b/eval.sh $VI/wt/wt0052/prompts/server_hpc/0052-hpc-06-4b/queue.sh $VI/wt/sync/scripts/server/sync_push.sh
echo "=========== tmp listing"; ls -la --time-style=full-iso $VI/tmp
echo "--- sizes"; for e in $VI/tmp/* $VI/tmp/.[!.]*; do [ -e "$e" ] && echo "$(du -s --apparent-size -B1 "$e" 2>/dev/null | cut -f1) $(find "$e" -newermt '-2 hours' 2>/dev/null | head -1 | wc -l) $e"; done
echo "## squeue $(date -u +%FT%TZ) [0053 step12 after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; echo "== done"; } > $O 2>&1
