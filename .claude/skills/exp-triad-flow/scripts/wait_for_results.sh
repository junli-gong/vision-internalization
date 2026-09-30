#!/usr/bin/env bash
# 管理员值守（模式 B）：等 origin 上出现本地还没有的 [results] 提交；找到就打印这些提交并退出 0。
# 用法: wait_for_results.sh <repo_dir> [poll_s=180] [max_s=21600] [ssh_host] [server_status_file] [stale_s=1800] [stale_running_s=43200]
#   "本地还没有" = HEAD..origin/<当前分支>；只 fetch，不动工作区。
#   给了 ssh_host 与 server_status_file 时，每轮只读查看服务器值守的状态文件
#   （内容为 "<UTC时间> <状态> <说明>"，状态取 waiting / hold / found / running / blocked）：
#     - waiting/hold 且超过 stale_s 秒没更新      → 退出 4（值守脚本可能已停）
#     - running/found 且超过 stale_running_s 秒没更新 → 退出 4（服务器 CC 可能已失联）
#     - blocked 且是本次监听开始之后写的          → 退出 5（服务器停下等管理员）
# 退出码: 0 有新结果；3 等满 max_s；4 心跳过期；5 服务器 blocked；2 参数错误。
set -u
REPO="${1:?usage: wait_for_results.sh <repo> [poll_s] [max_s] [ssh_host] [server_status_file] [stale_s] [stale_running_s]}"
POLL="${2:-180}"; MAX="${3:-21600}"; HOST="${4:-}"; SFILE="${5:-}"; STALE="${6:-1800}"; STALE_RUN="${7:-43200}"
cd "$REPO" 2>/dev/null || { echo "no such dir: $REPO" >&2; exit 2; }
BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null) || { echo "not a git repo: $REPO" >&2; exit 2; }
start=$(date +%s)
while :; do
  if git fetch -q origin "$BR" 2>/dev/null; then
    new=$(git log --format='%h %s' "HEAD..origin/$BR" 2>/dev/null | grep -E '^[0-9a-f]+ \[results\]' || true)
    if [ -n "$new" ]; then printf '%s\n' "$new"; exit 0; fi
  fi
  if [ -n "$HOST" ] && [ -n "$SFILE" ]; then
    # 远端输出一行: "<距上次修改的秒数> <UTC时间> <状态> <说明...>"；文件不存在或 ssh 失败则为空
    hb=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$HOST" \
          "f='$SFILE'; [ -f \"\$f\" ] && echo \$(( \$(date +%s) - \$(stat -c %Y \"\$f\") )) \$(head -n 1 \"\$f\")" 2>/dev/null || true)
    age=$(printf '%s\n' "$hb" | awk 'NR==1{print $1}')
    state=$(printf '%s\n' "$hb" | awk 'NR==1{print $3}')
    elapsed=$(( $(date +%s) - start ))
    case "$age" in ''|*[!0-9]*) state="" ;; esac
    case "$state" in
      waiting|hold)
        [ "$age" -gt "$STALE" ] && { echo "server standby heartbeat stale (${age}s): $hb"; exit 4; } ;;
      running|found)
        [ "$age" -gt "$STALE_RUN" ] && { echo "server running but silent for ${age}s: $hb"; exit 4; } ;;
      blocked)
        [ "$age" -le "$elapsed" ] && { echo "server blocked: $hb"; exit 5; } ;;
    esac
  fi
  if [ $(( $(date +%s) - start )) -ge "$MAX" ]; then echo "max wait ${MAX}s reached"; exit 3; fi
  sleep "$POLL"
done
