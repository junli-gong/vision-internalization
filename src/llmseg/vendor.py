"""引入的 LLMSeg 官方代码（third_party/llmseg）：出处、逐文件 sha256 的核对与加载（DEV-0066）。

- 来源 https://github.com/ajhamdi/llmseg-code，固定在 commit f2191e6605e40a5be5d932187fc055708ae098cf，MIT 许可（LICENSE 原样保留）。
- 只放运行管线必需的文件：方法本身（llm.py、pipeline.py、prompts.py、geometry.py）与它自己的离线测试（tests/test_offline.py）及其依赖的
  两个评测模块（evals/common.py、evals/gtmask.py）。全部原样，一个字节都没改；本地提供方从外部注入（src/llmseg/provider.install）。
- 逐文件 sha256 记在 third_party/llmseg/PROVENANCE.md 的表里；check_vendor() 现算并与表逐项比较，不一致就报 VendorError（运行前必查）。
- 官方模块之间用顶层名字互相 import（`import llm`、`import geometry as G`、`import prompts as P`），所以 load() 把 third_party/llmseg 放到
  sys.path 最前面再 import，并核对每个模块的 __file__ 确实在 third_party/llmseg 里（同名的别的模块先被 import 了就报错，不混用）。
"""

from __future__ import annotations

import hashlib
import importlib
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = REPO_ROOT / "third_party" / "llmseg"
PROVENANCE_PATH = VENDOR_DIR / "PROVENANCE.md"

UPSTREAM_URL = "https://github.com/ajhamdi/llmseg-code"
UPSTREAM_COMMIT = "f2191e6605e40a5be5d932187fc055708ae098cf"
UPSTREAM_COMMIT_DATE = "2026-10-22T10:09:44+03:00"
UPSTREAM_LICENSE = "MIT"
PAPER = "arXiv 2610.29409（LLMSeg: Training-Free Text-Prompted Image Segmentation with B-Splines）"

METHOD_FILES: tuple[str, ...] = ("llm.py", "pipeline.py", "prompts.py", "geometry.py")
TEST_FILES: tuple[str, ...] = ("tests/test_offline.py", "evals/common.py", "evals/gtmask.py")
VENDORED_FILES: tuple[str, ...] = ("LICENSE", *METHOD_FILES, *TEST_FILES)
MODULES: tuple[str, ...] = ("llm", "prompts", "geometry", "pipeline")  # import 顺序：pipeline 依赖前三个

_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")


class VendorError(RuntimeError):
    """引入的官方文件与 PROVENANCE.md 不符，或官方模块加载到了别处的同名模块。"""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_digests(root: str | Path = VENDOR_DIR) -> dict[str, str]:
    """引入的每个文件的 sha256（VENDORED_FILES 的顺序）。"""
    root = Path(root)
    return {f: sha256_file(root / f) for f in VENDORED_FILES}


def provenance_table(path: str | Path = PROVENANCE_PATH) -> dict[str, str]:
    """PROVENANCE.md 里"文件与 sha256"表（形如 | `llm.py` | `<64 位十六进制>` |）→ {文件: sha256}。"""
    out: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line.strip())
        if m:
            if m.group(1) in out:
                raise VendorError(f"{path}：{m.group(1)} 在表里出现两次")
            out[m.group(1)] = m.group(2)
    return out


def check_vendor() -> dict[str, Any]:
    """现算的逐文件 sha256 必须与 PROVENANCE.md 的表逐项相同（文件集合也相同）。返回运行身份里的 llmseg.code 一节。"""
    have = file_digests()
    want = provenance_table()
    if set(want) != set(have):
        raise VendorError(f"PROVENANCE.md 列的文件 {sorted(want)} 与引入的文件 {sorted(have)} 不同")
    bad = sorted(f for f in have if have[f] != want[f])
    if bad:
        raise VendorError(f"{bad} 的 sha256 与 PROVENANCE.md 记的不同（官方文件被改过？）")
    return {"upstream": UPSTREAM_URL, "commit": UPSTREAM_COMMIT, "license": UPSTREAM_LICENSE, "dir": "third_party/llmseg",
            "provenance_sha256": sha256_file(PROVENANCE_PATH), "files_sha256": have, "modified_files": []}


def load() -> SimpleNamespace:
    """import 官方的 llm、prompts、geometry、pipeline（先 check_vendor）。重复调用返回同一批模块。"""
    check_vendor()
    vdir = str(VENDOR_DIR)
    if vdir not in sys.path:
        sys.path.insert(0, vdir)
    mods = {}
    for name in MODULES:
        mod = sys.modules.get(name) or importlib.import_module(name)
        f = Path(getattr(mod, "__file__", "") or "").resolve()
        if f != (VENDOR_DIR / f"{name}.py").resolve():
            raise VendorError(f"模块 {name!r} 加载自 {f}，不是 {VENDOR_DIR / (name + '.py')}（同名模块冲突）")
        mods[name] = mod
    return SimpleNamespace(**mods)


__all__ = [
    "METHOD_FILES",
    "MODULES",
    "PAPER",
    "PROVENANCE_PATH",
    "TEST_FILES",
    "UPSTREAM_COMMIT",
    "UPSTREAM_LICENSE",
    "UPSTREAM_URL",
    "VENDORED_FILES",
    "VENDOR_DIR",
    "VendorError",
    "check_vendor",
    "file_digests",
    "load",
    "provenance_table",
    "sha256_file",
]
