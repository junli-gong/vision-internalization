#!/usr/bin/env bash
# B200 后台训练队列（D72-4、D76、D77-4/5；RUN-0036 起通用）。由每个训练 RUN 的队列脚本 source，不单独运行。
#
# 做什么：
#   - 逐条跑训练命令，每 CHUNK 步一停（--resume --stop-after-steps CHUNK；停下时检查点已写，下一段从它接着跑）；
#   - 两段之间过闸：暂停文件（评测要用卡，STANDBY"后台训练"一节）、origin 上的 HOLD、别人的 GPU 进程、
#     内存（可用 < 60 GB 停，D77-5）、根盘（剩 < 3 GB 停，D6）、本项目根盘份额（> 15 GB 停，D28-8）；
#   - 跑一段时每 30 秒看一次内存，每 5 分钟写一次心跳；
#   - 一条跑完：final/ 从内存盘拷到根盘（按训练叶目录的结构，见 finalize），逐文件 sha256 核对，删掉内存盘上的那份，
#     记账（ckpt_ledger.tsv 的 checkpoint 列 = 训练脚本打印的那一层，可以直接给评测或 E5 的 --start-checkpoint），推结果；
#   - 一条失败就从检查点重试一次；还失败就记进 failed.txt、跳到下一条；连续两条失败就停。
#   - 用量上限（可选，D81-2）：给了 QUEUE_CAP_H 时，累计各段的墙钟（只算跑命令的时间，不算暂停与等待）记在 $DRV/gpu_seconds；
#     每条运行开始前，已用 + 这一条的预估（run/run_cmd 前设 EST_H，没设按 0）超过上限就停，写 blocked 并推送，不等跑完再停。
#
# 队列脚本先设好下面的变量，再 source 本文件，然后逐条调用  run <名字> <脚本> <参数…>，最后调用 finish。
#   RUN_ID    运行编号，例如 0036
#   REPO      主 checkout（只用来提交与推送结果，git 操作都拿 $LOGS/git.lock）
#   OUT       结果目录，必须在 $REPO 里（绝对路径）
#   CK_TMP    检查点的工作目录（内存盘；含续跑状态）
#   CK_FINAL  最终检查点的存放目录（根盘）
#   CHUNK     每段的步数
#   PY        python 解释器
#   LOGS      心跳、暂停文件、git 锁所在的目录（真跑时 $PROJ/logs）
#   可选：PROJ（算份额用）、EXTRA（每条命令都加的参数，数组）、TQ_TRAIN_DATA（内存低于停止线时删掉的训练数据目录）、
#         TINY=1（开发机彩排：不查 GPU）、POLL_S、WATCH_S、BEAT_S、MEM_STOP_GB、DISK_STOP_GB、SHARE_STOP_GB
# 运行的命令与工作目录：调用 run 时的当前目录（队列脚本先 cd 到固定 commit 的工作树）。
set -uo pipefail
: "${RUN_ID:?}" "${REPO:?}" "${OUT:?}" "${CK_TMP:?}" "${CK_FINAL:?}" "${CHUNK:?}" "${PY:?}" "${LOGS:?}"
case "$OUT" in "$REPO"/*) ;; *) echo "OUT 必须在 REPO 里：$OUT" >&2; exit 2;; esac
TINY=${TINY:-0}
PROJ=${PROJ:-}
# TQ_NO_GIT=1（ITSC HPC，D89）：驱动所在的计算节点不碰 git（hpcgpu07 连不上 github），推送由登录节点上的同步循环做；
# HOLD 改看本地开关文件 $LOGS/HOLD。TQ_QUOTA_CMD：给了就用它算份额（输出整数 GB；Lustre 上用 lfs quota，不能对大目录 du）。
TQ_NO_GIT=${TQ_NO_GIT:-0}
TQ_QUOTA_CMD=${TQ_QUOTA_CMD:-}
# CPU 线程（RUN-0036 的教训）：不设时 torch 与 numpy 按全部核开线程，B200 只有 30 核、多人共用，E3a 的 build_inputs 每步从 0.22 s
# 慢到 0.8–1.6 s。与 RUN-0034、A800 的 common.sh 相同，取 16；调用方已设的照用。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${MKL_NUM_THREADS:-16}
TQ_TRAIN_DATA=${TQ_TRAIN_DATA:-}
POLL_S=${POLL_S:-60}; WATCH_S=${WATCH_S:-30}; BEAT_S=${BEAT_S:-300}
MEM_STOP_GB=${MEM_STOP_GB:-60}; DISK_STOP_GB=${DISK_STOP_GB:-3}; SHARE_STOP_GB=${SHARE_STOP_GB:-15}
MAX_CHUNKS=${MAX_CHUNKS:-200}
QUEUE_CAP_H=${QUEUE_CAP_H:-}
EST_H=${EST_H:-0}
TQ_STATUS=$LOGS/train_${RUN_ID}.status
TQ_PAUSE=$LOGS/train.pause
GITLOCK=$LOGS/git.lock
DRV=$OUT/driver
OUT_REL=${OUT#"$REPO"/}
mkdir -p "$DRV/logs" "$CK_TMP" "$CK_FINAL" "$LOGS"
touch "$DRV/done.txt"
[ -f "$OUT/ckpt_ledger.tsv" ] || printf 'name\tcheckpoint\tkbytes\tmanifest\tmanifest_sha256\n' > "$OUT/ckpt_ledger.tsv"
FAILS_IN_ROW=0

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
beat() { printf '%s %s\n' "$(now)" "$*" > "$TQ_STATUS.tmp" && mv -f "$TQ_STATUS.tmp" "$TQ_STATUS"; }
note() { echo "$(now) $*" | tee -a "$DRV/driver.log"; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
locked() { if command -v flock >/dev/null 2>&1; then ( flock 9 || exit 1; "$@" ) 9>"$GITLOCK"; else ( "$@" ); fi; }  # 子 shell 里跑，函数可用
mem_avail_gb() { if [ -r /proc/meminfo ]; then awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo; else echo 9999; fi; }
disk_free_gb() { df -Pk / | awk 'NR==2{print int($4/1048576)}'; }
share_gb() { if [ -n "$TQ_QUOTA_CMD" ]; then bash -c "$TQ_QUOTA_CMD" 2>/dev/null || echo 0; elif [ -n "$PROJ" ]; then du -sk "$PROJ" 2>/dev/null | awk '{print int($1/1048576)}'; else echo 0; fi; }
gpu_foreign() {  # 过闸时我们自己的训练没在跑，GPU 上的任何计算进程都算别人的
  [ "$TINY" = 1 ] && return 0
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$' | paste -sd, -
}
hold_on_origin() {
  if [ "$TQ_NO_GIT" = 1 ]; then [ -e "$LOGS/HOLD" ]; return; fi
  locked git -C "$REPO" fetch -q origin main 2>/dev/null; git -C "$REPO" cat-file -e origin/main:HOLD 2>/dev/null
}
dir_sums() { (cd "$1" && find . -type f | LC_ALL=C sort | while IFS= read -r f; do sha "$f"; done); }

_push() {  # 在锁里：只提交 OUT 下的改动，rebase 后推送，最多试 3 次
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
push_results() {
  if [ "$TQ_NO_GIT" = 1 ]; then note "（不推送，TQ_NO_GIT=1：由登录节点的同步循环推）$1"; return 0; fi
  locked _push "[results] $RUN_ID: $1" || note "推送失败（$1），下次推送时一并带上"
}

die() { beat "blocked $*"; note "停止：$*"; push_results "driver stopped: $*"; exit 1; }
mem_stop() {
  note "内存可用 $1 GB < $MEM_STOP_GB GB（D77-5）"
  if [ -n "$TQ_TRAIN_DATA" ] && [ -d "$TQ_TRAIN_DATA" ]; then rm -rf "$TQ_TRAIN_DATA" && note "已删除内存盘上的训练数据 $TQ_TRAIN_DATA"; fi
  die "memory ${1}GB<${MEM_STOP_GB}GB (D77-5), training data removed"
}

used_s() { [ -f "$DRV/gpu_seconds" ] && cat "$DRV/gpu_seconds" || echo 0; }
add_used() { echo $(( $(used_s) + $1 )) > "$DRV/gpu_seconds"; }
cap_check() {  # $1 = 名字；已用 + 预估（EST_H，小时，可带小数）超过 QUEUE_CAP_H 就停
  [ -n "$QUEUE_CAP_H" ] || return 0
  local u; u=$(used_s)
  if awk -v u="$u" -v e="${EST_H:-0}" -v c="$QUEUE_CAP_H" 'BEGIN{exit !((u/3600 + e) > c)}'; then
    die "cap: used $(awk -v u="$u" 'BEGIN{printf "%.2f", u/3600}') h + estimate ${EST_H:-0} h for $1 > cap $QUEUE_CAP_H h (D81-2)"
  fi
}

gate() {  # 不满足就等；触到停止线就停
  local why last="" m d s g
  while :; do
    m=$(mem_avail_gb); [ "$m" -lt "$MEM_STOP_GB" ] && mem_stop "$m"
    d=$(disk_free_gb); [ "$d" -lt "$DISK_STOP_GB" ] && die "disk free ${d}GB<${DISK_STOP_GB}GB (D6)"
    s=$(share_gb); [ "$s" -gt "$SHARE_STOP_GB" ] && die "project share ${s}GB>${SHARE_STOP_GB}GB (D28-8)"
    why=""
    if [ -e "$TQ_PAUSE" ]; then why="paused (pause file: $(head -c 80 "$TQ_PAUSE" | tr '\n' ' '))"
    elif hold_on_origin; then why="paused (HOLD$( [ "$TQ_NO_GIT" = 1 ] && echo " file $LOGS/HOLD" || echo " on origin"))"
    else g=$(gpu_foreign); [ -n "$g" ] && why="waiting (GPU busy, pids $g)"
    fi
    [ -z "$why" ] && return 0
    [ "$why" != "$last" ] && note "$why"
    last=$why; beat "$why"
    sleep "$POLL_S"
  done
}

chunk() {  # $1 = 名字，其余 = 命令；输出写进 $DRV/logs/<名字>.cur，再追加进 <名字>.log
  local name=$1; shift
  local cur="$DRV/logs/$name.cur" t0 tb pid rc m
  t0=$(date +%s); tb=$t0
  "$@" > "$cur" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$WATCH_S"
    m=$(mem_avail_gb)
    if [ "$m" -lt "$MEM_STOP_GB" ]; then kill "$pid" 2>/dev/null; wait "$pid"; cat "$cur" >> "$DRV/logs/$name.log"; mem_stop "$m"; fi
    if [ $(( $(date +%s) - tb )) -ge "$BEAT_S" ]; then
      tb=$(date +%s); beat "running $name ($(( (tb - t0) / 60 )) min into chunk; mem avail ${m}GB)"
    fi
  done
  wait "$pid"; rc=$?
  add_used $(( $(date +%s) - t0 ))
  { echo "=== $(now) chunk rc=$rc wall_s=$(( $(date +%s) - t0 ))"; cat "$cur"; } >> "$DRV/logs/$name.log"
  return $rc
}

finalize() {  # $1 = 名字，$2 = 训练脚本打印的检查点（<ckpt>/final 或 <ckpt>/final/model）
  # 根盘上按训练叶目录的结构存：$CK_FINAL/<叶目录相对 OUT 的路径>/ckpt/final，旁边放叶目录的 config.yaml 与 train_summary.json。
  # 这样 <新位置>/ckpt/final/model 与默认布局（不给 --ckpt-root）一样能找到训练时的运行身份：E5 的配种子与变体核对读
  # <叶>/config.yaml（src/e5/stack.py 的 _leaf_config）。叶目录由 logs/run_*.json 记下的 ckpt_dir 找到。
  local name=$1 printed=$2 fin sub ckdir f leaf rel dst man kb
  case "$printed" in
    */final) fin=$printed; sub="" ;;
    */final/model) fin=${printed%/model}; sub=/model ;;
    *) die "$name：认不出检查点路径（$printed）" ;;
  esac
  case "$fin" in "$CK_TMP"/*) ;; *) die "$name：final 目录不在 CK_TMP 下（$fin）" ;; esac
  ckdir=$(dirname "$fin")
  f=$(grep -rlF "\"ckpt_dir\": \"$ckdir\"" "$OUT" --include='run_*.json' 2>/dev/null | head -n 1)
  [ -n "$f" ] || die "$name：在 $OUT 里找不到 ckpt_dir = $ckdir 的训练叶目录"
  leaf=$(dirname "$(dirname "$f")"); rel=${leaf#"$OUT"/}
  dst="$CK_FINAL/$rel/ckpt/final"
  if [ -d "$fin" ]; then
    mkdir -p "$(dirname "$dst")"; rm -rf "$dst.part"
    cp -a "$fin" "$dst.part" || die "$name：拷贝 final 失败"
    if ! cmp -s <(dir_sums "$fin") <(dir_sums "$dst.part"); then die "$name：拷贝后 sha256 不一致"; fi
    rm -rf "$dst"; mv "$dst.part" "$dst"
    dir_sums "$dst" > "$dst.sha256.txt"
    rm -rf "$fin"
    f=$ckdir; while [ "$f" != "$CK_TMP" ] && rmdir "$f" 2>/dev/null; do f=$(dirname "$f"); done
  elif [ ! -d "$dst" ]; then
    die "$name：final 目录既不在内存盘（$fin）也不在根盘（$dst）"
  fi
  cp "$leaf/config.yaml" "$CK_FINAL/$rel/config.yaml" || die "$name：叶目录没有 config.yaml（$leaf）"
  [ -f "$leaf/train_summary.json" ] && cp "$leaf/train_summary.json" "$CK_FINAL/$rel/train_summary.json"
  man=$(cd "$dst" && find . -name manifest.json | LC_ALL=C sort | head -n 1)
  kb=$(du -sk "$dst" | awk '{print $1}')
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$dst$sub" "$kb" "$man" "$( [ -n "$man" ] && sha "$dst/$man" | awk '{print $1}')" >> "$OUT/ckpt_ledger.tsv"
  echo "$name" >> "$DRV/done.txt"
  note "$name 完成：final → $dst$sub（${kb} KB）"
}

run() {  # $1 = 名字（队列内唯一），$2 = 脚本，其余 = 参数（不含 --out/--ckpt-root/--resume/--stop-after-steps）
  local name=$1; shift
  local tries=0 n=0 rc cur ck
  if grep -qxF "$name" "$DRV/done.txt"; then note "$name 已完成（driver 记录），跳过"; return 0; fi
  cap_check "$name"
  cur="$DRV/logs/$name.cur"
  while :; do
    gate
    n=$((n + 1)); [ "$n" -gt "$MAX_CHUNKS" ] && die "$name：超过 $MAX_CHUNKS 段仍未完成"
    beat "running $name (chunk $n)"
    chunk "$name" "$PY" "$@" ${EXTRA[@]+"${EXTRA[@]}"} --out "$OUT" --ckpt-root "$CK_TMP" --resume --stop-after-steps "$CHUNK"
    rc=$?
    if [ $rc -eq 0 ] && grep -q '按 --stop-after-steps 停下' "$cur"; then continue; fi
    if [ $rc -eq 0 ] && ck=$(grep -E ' steps=[0-9]+ .*ckpt=' "$cur" | tail -n 1 | grep -oE 'ckpt=[^ ]+' | cut -d= -f2-) && [ -n "$ck" ]; then
      finalize "$name" "$ck"
      rm -f "$cur"; FAILS_IN_ROW=0
      push_results "$name complete"
      return 0
    fi
    tries=$((tries + 1))
    note "$name rc=$rc，没有完成也没有按段停下（第 $tries 次），日志 $DRV/logs/$name.log"
    if [ $tries -ge 2 ]; then
      echo "$(now) $name rc=$rc" >> "$DRV/failed.txt"; rm -f "$cur"
      FAILS_IN_ROW=$((FAILS_IN_ROW + 1))
      push_results "$name FAILED rc=$rc"
      [ $FAILS_IN_ROW -ge 2 ] && die "连续两条失败（最后一条 $name）"
      return 1
    fi
  done
}

run_cmd() {  # 一条不分段的命令（例如评测）：$1 = 名字（队列内唯一），其余 = 完整命令。过闸后跑一次，rc = 0 记完成并推送；
  # 否则重试一次，还失败就记进 failed.txt、跳到下一条（连续失败的计数与 run 共用）。输出追加进 $DRV/logs/<名字>.log。
  local name=$1; shift
  local tries=0 rc
  if grep -qxF "$name" "$DRV/done.txt"; then note "$name 已完成（driver 记录），跳过"; return 0; fi
  cap_check "$name"
  while :; do
    gate
    beat "running $name"
    chunk "$name" "$@"
    rc=$?
    rm -f "$DRV/logs/$name.cur"
    if [ $rc -eq 0 ]; then echo "$name" >> "$DRV/done.txt"; note "$name 完成"; FAILS_IN_ROW=0; push_results "$name complete"; return 0; fi
    tries=$((tries + 1)); note "$name rc=$rc（第 $tries 次）"
    if [ $tries -ge 2 ]; then
      echo "$(now) $name rc=$rc" >> "$DRV/failed.txt"
      FAILS_IN_ROW=$((FAILS_IN_ROW + 1)); push_results "$name FAILED rc=$rc"
      [ $FAILS_IN_ROW -ge 2 ] && die "连续两条失败（最后一条 $name）"
      return 1
    fi
  done
}

ckpt_of() {  # 已完成的运行的检查点（ckpt_ledger.tsv 的 checkpoint 列，最后一条记录）
  awk -F'\t' -v n="$1" '$1 == n {c = $2} END {print c}' "$OUT/ckpt_ledger.tsv"
}

wait_for_queue() {  # 等另一个队列跑完（$LOGS/train_<编号>.done）
  while [ ! -e "$LOGS/train_$1.done" ]; do beat "waiting for queue $1"; sleep "$POLL_S"; done
}

finish() {
  note "队列跑完：完成 $(wc -l < "$DRV/done.txt" | tr -d ' ') 条，失败 $( [ -f "$DRV/failed.txt" ] && wc -l < "$DRV/failed.txt" | tr -d ' ' || echo 0) 条"
  touch "$LOGS/train_${RUN_ID}.done"
  beat "done $RUN_ID"
  push_results "queue complete"
}
