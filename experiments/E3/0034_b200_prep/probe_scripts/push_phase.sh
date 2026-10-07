#!/bin/bash
# RUN-0034: push the run dir after a phase, from the main checkout. usage: bash push_phase.sh <phase> "<commit summary>"
# prep_push.py over the run dir (gzips files > 20 MB; originals stay on the server) -> copy logs -> pull -> HOLD check -> token scan
# -> add paths one by one -> commit -> push.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
PH=$1; MSG=$2; O=experiments/E3/0034_b200_prep; L=$PROJ/logs/0034; LIST=$PROJ/tmp/add34_$PH.txt
cd $REPO
mkdir -p $O/logs; cp $L/*.log $L/*.txt $L/disk_ledger.tsv $O/logs/ 2>/dev/null; cp $L/disk_ledger.tsv $O/ 2>/dev/null
python $O/probe_scripts/prep_push.py $O $LIST || exit 1
git pull --ff-only 2>&1 | tail -1
if git ls-tree origin/main --name-only | grep -qx HOLD; then echo "HOLD present"; fi
TOK=$(cat $ROOT/.hf/token)
hits=$(while read f; do case "$f" in *.gz) zcat "$f" | grep -qF "$TOK" && echo "$f"; zcat "$f" | grep -qE "hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}" && echo "$f";; *) grep -lF "$TOK" "$f"; grep -lE "hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}" "$f";; esac; done < $LIST)
if [ -n "$hits" ]; then echo "TOKEN SCAN HIT: $hits"; exit 2; fi
echo "scan done ($(wc -l < $LIST) files)"
big=$(while read f; do [ $(stat -c %s "$f") -gt 20000000 ] && echo "$f"; done < $LIST); if [ -n "$big" ]; then echo "STILL > 20 MB: $big"; exit 3; fi
while read f; do git add "$f"; done < $LIST
git diff --cached --stat | tail -1
git commit -q -F - <<MSGEOF
[results] 0034: $PH — $MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSGEOF
git push 2>&1 | tail -1; git log --oneline -1
