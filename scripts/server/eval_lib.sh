#!/usr/bin/env bash
# B200 评测脚本的公共函数（D72-4：评测独占 GPU，训练只在评测的间隙里跑；RUN-0038 起通用）。由评测脚本 source，不单独运行。
#
#   claim_gpu <任务名>   写暂停文件 $LOGS/train.pause，等后台训练队列停在段的边界（train_*.status 不再是 running）、GPU 上没有任何
#                        进程；退出时（含出错）自动删暂停文件，队列一分钟内自己接着跑。等超过 CLAIM_MAX_S（默认 3 小时）就返回 1。
#   step <名字> <命令…>  跑一条评测命令，rc 与墙钟记进 $OUT/steps.log；不因为 rc ≠ 0 而停（由调用方决定）。
#                        每条命令之前先查停止线：拿到 GPU 之后的墙钟超过 EV_CAP_S（任务书的上限），根盘剩余低于 3 GB（D6），
#                        本项目根盘占用超过 15 GB（D28-8；给了 PROJ 才查）。触线就推送已有结果、写 blocked 心跳，退出码 3。
#   shrink_large         $OUT 下超过 20 MB 的文件压成 .gz（gzip -n），同目录 LARGE_FILES.sha256 记原文件与 .gz 的 sha256 与字节数，
#                        核对之后删原文件（RUN-0027 的约定；省根盘份额，D28-8）。
#   push_out <说明>      只提交 $OUT 下的改动，在 git 锁里 rebase 后推送（同 train_queue.sh）。
#   progress <说明>      心跳写 $LOGS/eval_<RUN_ID>.status。
# 调用方先设好：RUN_ID、REPO、OUT（在 REPO 里）、LOGS；TINY=1 时不查 GPU、不写暂停文件。
set -uo pipefail
: "${RUN_ID:?}" "${REPO:?}" "${OUT:?}" "${LOGS:?}"
case "$OUT" in "$REPO"/*) ;; *) echo "OUT 必须在 REPO 里：$OUT" >&2; exit 2;; esac
TINY=${TINY:-0}
CLAIM_MAX_S=${CLAIM_MAX_S:-10800}
# CPU 线程（RUN-0036 的教训）：不设时 torch 与 numpy 按全部核开线程，B200 只有 30 核、多人共用，E3a 的 build_inputs 每步从 0.22 s
# 慢到 0.8–1.6 s。与 RUN-0034、A800 的 common.sh 相同，取 16；调用方已设的照用。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
EV_CAP_S=${EV_CAP_S:-}
PROJ=${PROJ:-}
EV_T0=$(date +%s)
EV_STATUS=$LOGS/eval_${RUN_ID}.status
OUT_REL=${OUT#"$REPO"/}
mkdir -p "$OUT" "$LOGS"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
progress() { printf '%s %s\n' "$(now)" "$*" > "$EV_STATUS.tmp" && mv -f "$EV_STATUS.tmp" "$EV_STATUS"; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
locked() { if command -v flock >/dev/null 2>&1; then ( flock 9 || exit 1; "$@" ) 9>"$LOGS/git.lock"; else ( "$@" ); fi; }

release_gpu() { [ "$TINY" = 1 ] || rm -f "$LOGS/train.pause"; }
claim_gpu() {
  [ "$TINY" = 1 ] && return 0
  echo "$1" > "$LOGS/train.pause"
  trap release_gpu EXIT
  local t0 st running procs
  t0=$(date +%s)
  while :; do
    running=""
    for st in "$LOGS"/train_*.status; do
      [ -e "$st" ] || continue
      case "$(cut -d' ' -f2- "$st")" in running*) running="$running $(basename "$st")";; esac
    done
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | paste -sd, -)
    if [ -z "$running" ] && [ -z "$procs" ]; then EV_T0=$(date +%s); progress "gpu claimed"; return 0; fi
    progress "waiting for gpu (running:${running:- none}; pids: ${procs:-none})"
    [ $(( $(date +%s) - t0 )) -ge "$CLAIM_MAX_S" ] && return 1
    sleep 30
  done
}

stop_line() {  # 触线就推送、写 blocked、退出 3
  local why="" d g
  if [ -n "$EV_CAP_S" ] && [ $(( $(date +%s) - EV_T0 )) -ge "$EV_CAP_S" ]; then why="cap ${EV_CAP_S}s reached"; fi
  d=$(df -Pk / | awk 'NR==2{print int($4/1048576)}'); [ "$d" -lt 3 ] && why="disk free ${d}GB<3GB (D6)"
  if [ -n "$PROJ" ]; then g=$(du -sk "$PROJ" 2>/dev/null | awk '{print int($1/1048576)}'); [ "$g" -gt 15 ] && why="project share ${g}GB>15GB (D28-8)"; fi
  [ -z "$why" ] && return 0
  echo "$(now) STOP before $1: $why" | tee -a "$OUT/steps.log"
  progress "blocked: $why"
  push_out "stopped before $1: $why"
  exit 3
}

step() {
  local name=$1; shift
  local t0 rc
  stop_line "$name"
  t0=$(date +%s); progress "running $name"
  mkdir -p "$OUT/logs"
  "$@" >> "$OUT/logs/$name.log" 2>&1
  rc=$?
  echo "$(now) $name rc=$rc wall_s=$(( $(date +%s) - t0 ))" | tee -a "$OUT/steps.log"
  return $rc
}

shrink_large() {
  local f d b o g
  find "$OUT" -type f -size +20M ! -name '*.gz' | LC_ALL=C sort | while IFS= read -r f; do
    d=$(dirname "$f"); b=$(basename "$f")
    o=$(sha "$f" | awk '{print $1}')
    gzip -n -c "$f" > "$f.gz.part" && mv "$f.gz.part" "$f.gz" || { echo "gzip 失败：$f" >&2; continue; }
    if [ "$(gzip -dc "$f.gz" | sha | awk '{print $1}')" != "$o" ]; then echo "解压核对不通过：$f" >&2; rm -f "$f.gz"; continue; fi
    g=$(sha "$f.gz" | awk '{print $1}')
    printf '%s  %s  %s  %s  %s\n' "$o" "$(wc -c < "$f" | tr -d ' ')" "$b" "$g" "$(wc -c < "$f.gz" | tr -d ' ')" >> "$d/LARGE_FILES.sha256"
    rm -f "$f"
  done
}

_push() {
  local msg=$1 i
  cd "$REPO" || return 1
  git add -A -- "$OUT_REL" || return 1
  if git diff --cached --quiet -- "$OUT_REL"; then return 0; fi
  git commit -q -m "$msg" -- "$OUT_REL" || return 1
  for i in 1 2 3; do
    git pull -q --rebase --autostash origin main && git push -q origin HEAD:main && return 0
    sleep 20
  done
  return 1
}
push_out() { shrink_large; locked _push "[results] $RUN_ID: $1" || echo "$(now) 推送失败（$1）" | tee -a "$OUT/steps.log"; }
