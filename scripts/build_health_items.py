#!/usr/bin/env python
"""重建健康检查的固定题目文件 configs/health_check_items.json（简报 §8.11；DEV-0008）。

原始文件按 src/health/protocol.py 里固定的 revision 用 huggingface_hub 下载（不用 datasets），逐个核对 sha256，再按
src/health/items.py 的抽样规则挑题。parquet 用 src/health/parquet_lite.py 读（不依赖 pyarrow）。

    # 下载到指定目录（不入库）并写出题目文件与离线复现用的总体索引
    python scripts/build_health_items.py --raw-dir <目录>
    # 复现核对：重新构造，与入库的两个文件逐字节比较，不写文件；不一致时退出码 1
    python scripts/build_health_items.py --raw-dir <目录> --check

--raw-dir 下的布局：<raw-dir>/<仓库名里的 / 换成 __>/<文件>；已存在且 sha256 相符就不再下载。不给 --raw-dir 时用 HF 缓存。
重建后题目文件的 sha256 若变了，要同步改 protocol.ITEMS_SHA256（脚本会打印），并开 D 条说明原因。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.health import items as I  # noqa: E402
from src.health import protocol as P  # noqa: E402
from src.health.parquet_lite import read_rows  # noqa: E402

DEFAULT_INDEX = "tests/fixtures/health/population_index.json"


def fetch(name: str, raw_dir: Path | None) -> Path:
    """取一个来源的原始文件（必要时下载），核对 sha256。"""
    src = P.SOURCES[name]
    if raw_dir is not None:
        local = raw_dir / src.repo.replace("/", "__")
        path = local / src.file
    if raw_dir is None or not path.exists():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(src.repo, src.file, repo_type="dataset", revision=src.revision,
                                    local_dir=None if raw_dir is None else str(local)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != src.file_sha256:
        raise SystemExit(f"{src.repo}@{src.revision}:{src.file} 的 sha256={digest}，与固定的 {src.file_sha256} 不一致")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", help="原始文件目录（不入库）；不给则用 HF 缓存")
    ap.add_argument("--out", default=P.ITEMS_PATH, help=f"题目文件（默认 {P.ITEMS_PATH}，相对仓库根）")
    ap.add_argument("--index-out", default=DEFAULT_INDEX, help=f"总体索引 fixture（默认 {DEFAULT_INDEX}）")
    ap.add_argument("--check", action="store_true", help="只核对：与已有的两个文件逐字节比较，不写")
    args = ap.parse_args(argv)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    paths = {name: fetch(name, raw_dir) for name in P.BENCHMARKS}
    mmlu = read_rows(paths["mmlu_pro"])
    gsm = read_rows(paths["gsm8k"])
    ife = read_jsonl(paths["ifeval"])
    text = I.dumps_items(I.build_items(mmlu, gsm, ife))
    index_text = json.dumps(I.population_index(mmlu, gsm, ife), ensure_ascii=False) + "\n"

    outputs = ((REPO_ROOT / args.out, text), (REPO_ROOT / args.index_out, index_text))
    digest = I.sha256_text(text)
    if args.check:
        bad = [str(p) for p, t in outputs if not p.exists() or p.read_text(encoding="utf-8") != t]
        print(f"items sha256={digest}  protocol.ITEMS_SHA256={P.ITEMS_SHA256}")
        if bad or digest != P.ITEMS_SHA256:
            print(f"不一致：{bad or '（文件相同，但 protocol.ITEMS_SHA256 未同步）'}")
            return 1
        print("一致：重新构造的题目文件与总体索引与入库版本逐字节相同")
        return 0
    for p, t in outputs:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t, encoding="utf-8")
        print(f"写出 {p}（{len(t.encode('utf-8'))} 字节）")
    print(f"items sha256={digest}")
    if digest != P.ITEMS_SHA256:
        print("注意：与 protocol.ITEMS_SHA256 不同，需同步更新")
    return 0


if __name__ == "__main__":
    sys.exit(main())
