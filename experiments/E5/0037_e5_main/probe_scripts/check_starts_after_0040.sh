#!/bin/bash
# RUN-0037 §1 (server CC): after RUN-0040 finishes (train_0040.done), check the 9 E5 start checkpoints (.sha256.txt, model/, leaf config.yaml)
# and push reports/0037-b200-e5-main-progress-1.md (facts only). Detached; git under $PROJ/logs/git.lock.
PROJ=/home/ubuntu/workspace/wzg/gjl/vision-internalization; REPO=$PROJ/repo; L=$PROJ/logs/0037; M=qwen3.5-2b
until [ -e $PROJ/logs/train_0040.done ]; do sleep 120; done
out=$L/start_ckpt_check.txt; : > $out; bad=0
for init in e3b_c3 c3_warmup base_warmup; do for s in 0 1 2; do
  case $init in base_warmup) leaf=$PROJ/ckpt/0036/warmup/$M/seed$s ;; *) leaf=$PROJ/ckpt/0040/$init/$M/seed$s ;; esac
  fin=$leaf/ckpt/final
  if [ -f $fin.sha256.txt ]; then (cd $fin && sha256sum --quiet -c $fin.sha256.txt) >/dev/null 2>&1 && sha=ok || sha=FAIL; nf=$(wc -l < $fin.sha256.txt); else sha=missing; nf=0; fi
  echo "$init s$s model=$([ -d $fin/model ] && echo y || echo n) sha256=$sha files=$nf config.yaml=$([ -f $leaf/config.yaml ] && echo y || echo n) kb=$(du -sk $fin 2>/dev/null | cut -f1)" >> $out
  { [ -d $fin/model ] && [ $sha = ok ] && [ -f $leaf/config.yaml ]; } || bad=1
done; done
R=$REPO/reports/0037-b200-e5-main-progress-1.md
{ echo "# RUN-0037 进度报告 1 — E5 起点检查点的核对（RUN-0040 跑完之后）"; echo
  echo "服务器 CC 的核对脚本（与 CC 脱钩），$(date -u +%FT%TZ)。只记事实。任务书 §1：启动时 RUN-0040 还没跑完，起点检查点在它跑完之后核对。"; echo
  echo "- 9 个起点：\`.sha256.txt\` 逐文件核对、\`model/\` 在、叶目录有 \`config.yaml\`。全部通过：$([ $bad = 0 ] && echo 是 || echo 否)。"; echo
  echo '```'; cat $out; echo '```'; echo
  echo "- 当时的心跳："; echo; echo '```'; for n in 0037 0040 0041; do cat $PROJ/logs/train_$n.status; done; echo '```'; } > $R
( flock 9; cd $REPO && git pull -q --rebase --autostash origin main && git add reports/0037-b200-e5-main-progress-1.md && \
  git commit -q -m "[results] 0037: progress 1 — E5 start checkpoints checked after RUN-0040 (all ok: $([ $bad = 0 ] && echo yes || echo NO))" -- reports/0037-b200-e5-main-progress-1.md && \
  git push -q origin HEAD:main ) 9>$PROJ/logs/git.lock
echo "$(date -u +%FT%TZ) pushed progress-1 bad=$bad" >> $L/check_starts.log
