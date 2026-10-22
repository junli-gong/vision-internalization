#!/usr/bin/env bash
# 门报告的精简审阅包（D113-5）：仓库某个 commit 的全部已跟踪文件，去掉 .git 与 experiments/，再附上 experiments/ 下所有已跟踪的
# metrics.json（保持原路径）。打的是 commit 的内容（git archive），与工作区里没提交的改动无关，可复现。
#
# 用法：scripts/make_review_bundle.sh [名字] [commit]
#   名字默认 review；commit 默认 HEAD。输出 dist/review_bundle_<名字>_<UTC 日期>_<短 commit>.tar.gz（dist/ 已在 .gitignore），
#   包内顶层目录同名；另有 BUNDLE_MANIFEST.txt：commit、生成时间、文件数、各 metrics.json 的路径与 sha256。
set -euo pipefail
NAME=${1:-review}
REV=${2:-HEAD}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
SHA=$(git rev-parse --verify "$REV^{commit}")
SHORT=${SHA:0:8}
STAMP=$(date -u +%Y%m%d)
BASE="review_bundle_${NAME}_${STAMP}_${SHORT}"
mkdir -p dist
TMP=$(mktemp -d "${TMPDIR:-/tmp}/bundle.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/$BASE"
# 1. 除 experiments/ 以外的全部已跟踪文件
git archive --format=tar "$SHA" -- . ':(exclude)experiments' | tar -x -C "$TMP/$BASE"
# 2. experiments/ 下所有已跟踪的 metrics.json（分批交给 git archive，避免参数过长）
git ls-tree -r --name-only "$SHA" -- experiments | grep '/metrics\.json$' > "$TMP/metrics.list" || true
batch=()
flush() { [ ${#batch[@]} -gt 0 ] && git archive --format=tar "$SHA" -- "${batch[@]}" | tar -x -C "$TMP/$BASE"; batch=(); }
while IFS= read -r f; do
  batch+=("$f")
  [ ${#batch[@]} -ge 200 ] && flush
done < "$TMP/metrics.list"
flush
# 3. 清单
sha() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }
{
  echo "commit: $SHA"
  echo "generated_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "excluded: .git, experiments/ (except metrics.json)"
  echo "n_files: $(find "$TMP/$BASE" -type f | wc -l | tr -d ' ')"
  echo "n_metrics_json: $(wc -l < "$TMP/metrics.list" | tr -d ' ')"
  echo "metrics_json_sha256:"
  ( cd "$TMP/$BASE" && while IFS= read -r f; do [ -f "$f" ] && sha "$f"; done < "$TMP/metrics.list" )
} > "$TMP/$BASE/BUNDLE_MANIFEST.txt"
tar -czf "dist/$BASE.tar.gz" -C "$TMP" "$BASE"
echo "dist/$BASE.tar.gz  $(du -h "dist/$BASE.tar.gz" | cut -f1)  files $(find "$TMP/$BASE" -type f | wc -l | tr -d ' ')  metrics.json $(wc -l < "$TMP/metrics.list" | tr -d ' ')"
