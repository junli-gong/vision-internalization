"""实验目录与元数据（简报 §11.1）。

目录：experiments/<phase>/<NNNN_slug>/，含 config.yaml、meta.json、logs/、figures/。
meta.json 记 seed、git commit（含 dirty 标记与 diff 统计）、数据版本（划分文件 sha256）、主机名、
GPU 名、关键包版本、UTC 时间戳。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import json
import platform
import re
import socket
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

META_SCHEMA_VERSION = 1

# 记版本的关键包（发行包名）；没装的记 null
KEY_PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "pillow",
    "shapely",
    "pycocotools",
    "pyyaml",
    "transformers",
    "accelerate",
    "peft",
    "vllm",
)

_RUN_ID = re.compile(r"[0-9]{4}")  # 不用 \d：它匹配全角等 Unicode 数字
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def create_run_dir(phase: str, run_id: str, slug: str, *, root: str | Path = "experiments") -> Path:
    """创建 <root>/<phase>/<run_id>_<slug>/，并建 logs/、figures/。

    run_id 必须是恰好 4 位数字的字符串（如 "0001"），否则 ValueError。
    目标目录已存在且非空时 FileExistsError（不覆盖）；已存在但为空时沿用。
    """
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"运行编号必须是 4 位数字字符串，收到 {run_id!r}")
    for label, v in (("phase", phase), ("slug", slug)):
        if not isinstance(v, str) or not _NAME.fullmatch(v) or ".." in v:
            raise ValueError(f"{label} 只能含字母、数字、_ . -，收到 {v!r}")
    run_dir = Path(root) / phase / f"{run_id}_{slug}"
    if run_dir.exists():
        if not run_dir.is_dir():
            raise FileExistsError(f"{run_dir} 已存在且不是目录")
        if any(run_dir.iterdir()):
            raise FileExistsError(f"{run_dir} 已存在且非空，拒绝覆盖")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir()
    (run_dir / "figures").mkdir()
    return run_dir


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(args: Sequence[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout


def git_info(repo_root: str | Path) -> dict[str, Any]:
    """git commit、dirty 标记、相对 HEAD 的 diff 统计、未跟踪文件数。不是 git 仓库时 commit 为 null。"""
    cwd = Path(repo_root)
    head = _git(["rev-parse", "HEAD"], cwd)
    if head is None:
        return {"commit": None, "dirty": None, "diff_shortstat": None, "n_untracked": None}
    status = _git(["status", "--porcelain"], cwd) or ""
    lines = [ln for ln in status.splitlines() if ln]
    shortstat = (_git(["diff", "--shortstat", "HEAD"], cwd) or "").strip()
    return {
        "commit": head.strip(),
        "dirty": bool(lines),
        "diff_shortstat": shortstat,
        "n_untracked": sum(1 for ln in lines if ln.startswith("??")),
    }


def gpu_name() -> str | None:
    """第 0 张 CUDA 卡的名字；没有 CUDA（或没装 torch）返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def package_versions(names: Sequence[str] = KEY_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = None
    return out


def write_run_metadata(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any],
    seed: int,
    data_files: Sequence[str | Path],
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """写 config.yaml（配置快照）与 meta.json，返回 meta 字典。

    data_files：本次运行用到的划分文件；数据版本 = 每个文件的 sha256（文件不存在则 FileNotFoundError）。
    config.yaml 或 meta.json 已存在时 FileExistsError（不覆盖）。
    """
    run_dir = Path(run_dir)
    cfg_path, meta_path = run_dir / "config.yaml", run_dir / "meta.json"
    for p in (cfg_path, meta_path):
        if p.exists():
            raise FileExistsError(f"{p} 已存在，拒绝覆盖")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed 必须是 int，收到 {seed!r}")
    if "seed" in config and config["seed"] != seed:
        raise ValueError(f"config 里的 seed={config['seed']!r} 与参数 seed={seed!r} 不一致")
    data = {str(p): sha256_file(p) for p in data_files}
    meta = {
        "schema_version": META_SCHEMA_VERSION,
        "seed": seed,
        "git": git_info(repo_root),
        "data_sha256": data,
        "hostname": socket.gethostname(),
        "gpu": gpu_name(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    cfg_path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True))
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta
