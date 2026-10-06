#!/bin/bash
# RUN-0032: push one finished item from the main checkout. usage: bash push_item.sh <model key | lisa | baseline> "<commit summary>"
# prep_push.py per output dir (gzips files > 20 MB; originals stay on the server, not added) -> copy that item's logs -> pull -> HOLD check
# -> token scan -> add paths one by one -> commit -> push.
source /home/ubuntu/workspace/wzg/gjl/vision-internalization/scripts/env.sh
M=$1; MSG=$2; O=experiments/E5/0032_e5_test; L=$PROJ/logs/0032; LIST=$PROJ/tmp/add32_$M.txt
cd $REPO
case "$M" in
  lisa) DIRS="$O/external $O/lisa_predictions $O/lisa_predictions_b200 $O/lisa_predictions_limit5 $O/overlays"; EXTRA="$O/weights_download.json" ;;
  baseline) DIRS="$O/baseline"; EXTRA="" ;;
  *) DIRS="$O/zeroshot/$M $O/cot/$M"; EXTRA="" ;;
esac
[ -d $O/errors ] && DIRS="$DIRS $O/errors"
: > $LIST.dirs
for d in $DIRS; do [ -d $d ] || continue; python $O/probe_scripts/prep_push.py $d $LIST.part || exit 1; cat $LIST.part >> $LIST.dirs; done
cp $L/${M}__*.log $L/driver.log $O/logs/
{ cat $LIST.dirs; for f in $EXTRA; do [ -f $f ] && echo $f; done; ls $O/logs/${M}__*.log $O/logs/driver.log; } | grep -v '^$' | sort -u > $LIST
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
[results] 0032: $M — $MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
MSGEOF
git push 2>&1 | tail -1; git log --oneline -1
