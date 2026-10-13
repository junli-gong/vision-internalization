#!/bin/bash
# RUN-0053 step 3 (pt_* only), D95 checkout of wt0051/wt0052, step 4 gc, step 5 checks
VI=/home/mail/chenyuma/visint; O=$VI/logs/0053_step345.txt; P0=experiments/P0/0001_env_probe/raw/p6/A/coco_val2017/coco_val2017_coco_urls.txt
{ echo "## squeue $(date -u +%FT%TZ) [0053 step3 before]"; squeue -h -j 157296,157297 -o "%i %T %M %N"
echo "=========== step 3: delete pytest basetemps"
for d in pt_0049_hpcgpu07 pt_failtest pt_parity_r3 pt_plain_r3 pt_rerun3_1 pt_rerun3_2 pt_rerun3_3 pt_rerun3_4 pt_skips; do p=$VI/tmp/$d
  [ -d "$p" ] || { echo "absent $d"; continue; }
  r=$(find "$p" -newermt '-2 hours' | head -1 | wc -l); b=$(du -s --apparent-size -B1 "$p" | cut -f1)
  if [ "$r" = 0 ]; then rm -rf "$p" && echo "deleted $d ($b B)"; else echo "KEPT $d (recent files)"; fi; done
ls -la $VI/tmp
echo "=========== D95: checkout wt0051 / wt0052 to origin/main"
t0=$(date +%s); timeout 600 flock -w 300 $VI/logs/git.lock -c "git -C $VI/repo fetch -q origin main"; echo "fetch rc=$? $(( $(date +%s)-t0 ))s; origin/main $(git -C $VI/repo rev-parse --short origin/main)"
git -C $VI/repo merge-base --is-ancestor 2c67380d origin/main && echo "2c67380d in origin/main: yes"
for w in wt0051 wt0052; do W=$VI/wt/$w; git -C $W checkout -q --detach origin/main; echo "$w checkout rc=$? HEAD $(git -C $W rev-parse HEAD) status_lines=$(git -C $W status --short | wc -l) sparse=$(git -C $W config core.sparseCheckout) experiments/: $(ls $W/experiments | tr '\n' ' ') P0 $(sha256sum $W/$P0 | cut -c1-8) du $(du -s --apparent-size -B1 $W | cut -f1)"; done
chk() { a=$(sha256sum "$1" | cut -d' ' -f1); [ "$a" = "$2" ] && echo "OK   $1" || echo "DIFF $1 got $a want $2"; }
W=$VI/wt/wt0051
chk $W/prompts/server_hpc/0051-hpc-e5-main/queue.sh 8f33ad28c0f7b95757c0a2555b96f3c01f4c4d840824f0a2b45e39cb9a5ab769
chk $W/scripts/server/train_queue.sh fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa
chk $W/configs/e5_train.json e7f28997935f7281a6d0f4fe1d29132cc2d14e4b500000f00882dd7b2b2b94c3
chk $W/configs/e5_prompts.json c0e59d3589cfb51a29eb271d1e1527604899a062408f5d429c9acc4f80467ec4
W=$VI/wt/wt0052
chk $W/prompts/server_hpc/0052-hpc-06-4b/run.sh a4ae09a0117790047babd15ba5fd21e6a6368b58c4f678d4452c12a661919ffc
chk $W/prompts/server_hpc/0052-hpc-06-4b/eval.sh 51b0db5a39854d731831dafb4c5e40b6768c52fea0f365082e2202c055ff37bc
chk $W/prompts/server_hpc/0052-hpc-06-4b/queue.sh 5daa6013b7b97022e8fcb144a2d68a02a6deb1fe2936f7aaf91a0f62a7e3e8b4
chk $W/scripts/server/eval_lib.sh 2380464beeabe300f2cfdd17ab1d79921b8d978b454b2d9f16ee42e4ca074a59
chk $W/scripts/server/train_queue.sh fcb10f47c2fb6cf068b29f849230c82611bbcc018eb550116867b17337e0dbaa
chk $W/configs/e3b_train.json b265cd87b2f141a37d8aea4d62b667efe084709d737220903938feaf7811c375
chk $W/configs/e1_prompts.json f3353de0d53de22b8f2226e07afef94b643d02ec106868eaf6bf2766a45bf452
chk $W/configs/e1_posthoc.json 9a5607826bae8b2387cd9b5bbd88cd71d5887ebf86237b786546ab13ac613ce1
echo "=========== step 4: gc"
echo "--- count-objects before"; git -C $VI/repo count-objects -vH
t0=$(date +%s); flock $VI/logs/git.lock -c "git -C $VI/repo gc --prune=now" 2>&1 | tail -3; echo "gc rc=${PIPESTATUS[0]} $(( $(date +%s)-t0 ))s"
echo "--- count-objects after"; git -C $VI/repo count-objects -vH
echo "=========== step 5"
q=$(timeout 60 lfs quota -u chenyuma /home | sed -n 3p); echo "quota delta KB after: $(( $(echo $q | awk '{print $2}') - 7059403936 ))"
for f in train_0045-hpcgpu06 train_0050-hpcgpu07 sync_0045 sync_0050; do echo "$f: $(cat $VI/logs/$f.status)"; done
echo "## squeue $(date -u +%FT%TZ) [0053 after]"; squeue -h -j 157296,157297 -o "%i %T %M %N"; echo "== done"; } > $O 2>&1
