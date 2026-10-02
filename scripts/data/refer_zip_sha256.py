#!/usr/bin/env python
"""refer 三个 zip 的完整性核对（DEV-0026）。合作者拿到 refcoco.zip、refcoco+.zip、refcocog.zip 后先跑它：

    python scripts/data/refer_zip_sha256.py <目录>

对目录里的每个 zip（官方文件名）：
- 打印字节数与整个文件的 sha256；
- 与 src/data/land/refcoco.SPEC 的钉住值比对：字节数与中央目录（尾部）的大小、条目数、sha256（refer_zip_pins），
  整个文件的 sha256（refer_zip_sha256；还没钉住时只打印，由管理员补钉）；
- 逐个成员解压，核对 CRC32（zipfile.testzip）。

只读，不联网，不改任何文件。最后一行打印 JSON {zip 名: {bytes, sha256}}，供管理员补钉 SPEC。

退出码：0 全部相符（整个文件的 sha256 未钉住时也是 0，会提示）；1 有文件缺失或任何一项不符。
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.data.land import net, refcoco  # noqa: E402
from src.data.land.common import LocalZip, read_zip_index  # noqa: E402


def check_zip(path: Path, name: str, spec: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """核对一个 zip。返回 (是否全部相符, 记录)。"""
    rec: dict[str, Any] = {"file": str(path)}
    if not path.is_file():
        rec["problems"] = ["文件不存在"]
        return False, rec
    problems: list[str] = []
    size = path.stat().st_size
    full = net.sha256_file(path)
    rec.update(bytes=size, sha256=full)
    pins = spec["refer_zip_pins"][name]
    if size != pins["size"]:
        problems.append(f"字节数 {size} ≠ 钉住的 {pins['size']}")
    try:
        got = read_zip_index(LocalZip(path)).pins()
        bad = {k: (got[k], v) for k, v in pins.items() if k in got and got[k] != v}
        if bad:
            problems.append(f"中央目录与钉住的不符 {bad}")
    except zipfile.BadZipFile as e:
        problems.append(f"读不出中央目录：{e}")
    try:
        with zipfile.ZipFile(path) as zf:
            first_bad = zf.testzip()
        if first_bad is not None:
            problems.append(f"成员 {first_bad} 的 CRC32 不符")
    except (zipfile.BadZipFile, zlib.error, EOFError, OSError) as e:
        problems.append(f"解压成员失败：{type(e).__name__}: {e}")
    pinned = spec["refer_zip_sha256"][name]
    rec["sha256_pinned"] = pinned
    if pinned is not None and full != pinned:
        problems.append(f"整个文件的 sha256 {full} ≠ 钉住的 {pinned}")
    rec["problems"] = problems
    return not problems, rec


def main(argv: list[str] | None = None, spec: dict[str, Any] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path, help="放着 refcoco.zip、refcoco+.zip、refcocog.zip 的目录")
    a = ap.parse_args(argv)
    spec = spec or refcoco.SPEC
    all_ok = True
    out: dict[str, dict[str, Any]] = {}
    for name in spec["refer_zip_pins"]:
        ok, rec = check_zip(refcoco.refer_zip_path(a.dir, name), name, spec)
        all_ok &= ok
        fn = f"{name}.zip"
        if "sha256" not in rec:
            print(f"[缺失] {rec['file']}")
            continue
        out[fn] = {"bytes": rec["bytes"], "sha256": rec["sha256"]}
        full_state = "未钉住，请交管理员补钉" if rec["sha256_pinned"] is None else ("= 钉住的值" if rec["sha256"] == rec["sha256_pinned"] else "≠ 钉住的值")
        print(f"[{'相符' if ok else '不符'}] {fn}  {rec['bytes']} B  sha256 {rec['sha256']}（{full_state}）")
        for p in rec["problems"]:
            print(f"    - {p}")
    if all_ok:
        unpinned = "" if all(spec["refer_zip_sha256"].values()) else "（整个文件的 sha256 还没钉住）"
        print(f"全部相符：字节数、中央目录 sha256、成员 CRC32{unpinned}")
    else:
        print("有不符或缺失的文件：不要用它们落地")
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
