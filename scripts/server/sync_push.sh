#!/usr/bin/env bash
# 登录节点上的结果同步循环（ITSC HPC，D89；v3，D94）：计算节点上的驱动用 TQ_NO_GIT=1，结果写进<源仓库>的工作区，
# 这里每 INTERVAL 秒把给定的结果目录提交并推送一次。
# 历史：
#   v1 在源仓库里 commit 后 pull --rebase --autostash，autostash 换掉了驱动开着写的文件（RUN-0045 的事故，0045-addendum-1）。
#   v2 把目录 rsync 到一个专用的推送工作树再提交，但推送工作树是完整签出（约 3.4 GB）外加一份结果的拷贝（D94：共享盘份额）。
# v3：
#   - 不用工作树，不碰源仓库的工作区与它自己的索引。每轮在一个单独的索引文件里，以 origin/main 为底，把给定目录换成源目录里的现状，
#     write-tree、commit-tree，推到 main；推送被拒（别人先推了）就整轮重来。
#   - **大的结果文件不进 git**（D94）：每个目录下大于 LARGE_MB（默认 5）的文件留在 Lustre 上原处，不加进提交；
#     目录里另写一份 LARGE_FILES.manifest.tsv（相对路径、字节数、sha256、修改时间），清单进 git。sha256 按（大小、修改时间）缓存，没变就不重算。
#   - 不同步：*.cur（驱动的单段输出）、*.tmp、*.part，以及任何 ckpt/ 目录下的文件。
#   - 以前的提交里已有的大文件，在 v3 的第一轮里从树上删掉（历史里还在），之后只出现在清单里。
# 用法：sync_push.sh <源仓库> <锁文件> <间隔秒> <说明前缀> <相对路径>…   （可选环境变量 LARGE_MB）
#   例：sync_push.sh $VI/repo $VI/logs/git.lock 1800 0051 experiments/E5/0051_e5_main_hpcgpu07
#   git 操作都拿 $LOCK（与 CC 共用）。一轮之后若存在 <锁文件>.stop 就退出；单轮：ONCE=1。
#   心跳写 <锁文件所在目录>/sync_<说明前缀>.status；索引与 sha256 缓存也放在那个目录里。
set -uo pipefail
SRC=${1:?源仓库}; LOCK=${2:?锁文件}; INTERVAL=${3:?间隔秒}; TAG=${4:?说明前缀}; shift 4
[ $# -ge 1 ] || { echo "至少给一个相对路径" >&2; exit 2; }
LARGE_MB=${LARGE_MB:-5}
LARGE_B=$((LARGE_MB * 1048576))
D=$(dirname "$LOCK")
ST="$D/sync_${TAG}.status"
IDX="$D/sync_${TAG}.index"
CACHE="$D/sync_${TAG}.shacache"
beat() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$ST.tmp" && mv -f "$ST.tmp" "$ST"; }
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
fstat() { if stat -c '%s %Y' "$1" >/dev/null 2>&1; then stat -c '%s %Y' "$1"; else stat -f '%z %m' "$1"; fi; }  # 字节数 修改时间（GNU / BSD）
touch "$CACHE"

skip() {  # 不同步的文件（相对路径）
  case "$1" in *.cur|*.tmp|*.part|*/ckpt/*) return 0 ;; esac
  return 1
}

manifest() {  # $1 = 相对目录：在 $SRC/$1/LARGE_FILES.manifest.tsv 写大文件清单（内容不变就不重写）；stdout 无输出
  local p=$1 f size mt h line tmp out
  out="$SRC/$p/LARGE_FILES.manifest.tsv"; tmp="$out.new"
  printf 'path\tbytes\tsha256\tmtime_epoch\n' > "$tmp"
  while IFS= read -r -d '' f; do
    skip "$f" && continue
    read -r size mt < <(fstat "$SRC/$f")
    [ "$size" -ge "$LARGE_B" ] || continue
    line=$(awk -F'\t' -v f="$f" -v s="$size" -v m="$mt" '$1==f && $2==s && $3==m {print $4; exit}' "$CACHE")
    if [ -n "$line" ]; then h=$line; else
      h=$(sha "$SRC/$f"); printf '%s\t%s\t%s\t%s\n' "$f" "$size" "$mt" "$h" >> "$CACHE"
    fi
    printf '%s\t%s\t%s\t%s\n' "${f#"$p"/}" "$size" "$h" "$mt" >> "$tmp"
  done < <(cd "$SRC" && find "$p" -type f ! -name LARGE_FILES.manifest.tsv -print0 | LC_ALL=C sort -z)
  if [ "$(wc -l < "$tmp")" -le 1 ]; then rm -f "$tmp" "$out"; return 0; fi   # 没有大文件就不要清单
  if [ -f "$out" ] && cmp -s "$tmp" "$out"; then rm -f "$tmp"; else mv -f "$tmp" "$out"; fi
}

small_files() {  # $1 = 相对目录：列出要进 git 的文件（NUL 分隔）
  local p=$1 f size mt
  while IFS= read -r -d '' f; do
    skip "$f" && continue
    read -r size mt < <(fstat "$SRC/$f")
    if [ "$size" -lt "$LARGE_B" ]; then printf '%s\0' "$f"; fi
  done < <(cd "$SRC" && find "$p" -type f -print0 | LC_ALL=C sort -z)
  return 0  # 最后一个是大文件时，不要让函数的退出码变成 1（pipefail 下整条管道会被判失败）
}

one_round() {
  (
    if command -v flock >/dev/null 2>&1; then flock 9 || exit 1; fi  # 开发机（macOS）没有 flock，彩排时不加锁
    export GIT_DIR="$SRC/.git" GIT_INDEX_FILE="$IDX"
    local base p tree msg commit i
    for i in 1 2 3; do
      git fetch -q origin main || exit 1
      base=$(git rev-parse origin/main) || exit 1
      rm -f "$IDX"
      git read-tree "$base" || exit 1
      for p in "$@"; do
        git rm -r -q --cached --ignore-unmatch -- "$p" >/dev/null || exit 1
        [ -d "$SRC/$p" ] || continue
        manifest "$p"
        small_files "$p" | git --work-tree="$SRC" -C "$SRC" update-index --add -z --stdin || exit 1
      done
      tree=$(git write-tree) || exit 1
      [ "$tree" = "$(git rev-parse "$base^{tree}")" ] && { rm -f "$IDX"; exit 0; }
      msg="[results] $TAG: sync $(date -u +%Y-%m-%dT%H:%M:%SZ)"
      commit=$(git commit-tree "$tree" -p "$base" -m "$msg") || exit 1
      git push -q origin "$commit:refs/heads/main" 2>/dev/null && { rm -f "$IDX"; exit 0; }
      sleep 20   # 别人先推了：重新取 origin/main，整轮重来
    done
    rm -f "$IDX"; exit 1
  ) 9>"$LOCK"
}

while :; do
  if one_round "$@"; then beat "ok"; else beat "push failed (will retry)"; fi
  [ "${ONCE:-0}" = 1 ] && break
  [ -e "$LOCK.stop" ] && { beat "stopped ($LOCK.stop)"; break; }
  sleep "$INTERVAL"
done
