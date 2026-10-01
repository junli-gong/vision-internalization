"""落地的清单、磁盘记账与预算（DEV-0012）。

每个源一份 <root>/<源>/MANIFEST.json，内容：
- upstream：来源 URL、revision 或 commit、上游文件的 sha256 与字节数，以及与官方的核对方式和结果；
- license：许可原文（或"未声明"）与出处；
- files：落进 --root 的每个文件（相对源目录的路径、sha256、字节数，图像另记宽高与感知哈希）；
- landed_at_utc、script（路径、argv、git commit 与 dirty 标记）；
- estimate（开跑前的体积估算与预算）、df_log（每一步的剩余空间）、failures（不可达的 URL，逐条）；
- selection / stats / dedupe_facts：抽样规则与结果、计数、去重相关的事实。

续传与幂等：每落一个文件就往 MANIFEST.journal.jsonl 追加一行；重跑时先读上一次的 MANIFEST.json 与 journal，
文件已在且 sha256 相符就跳过。收尾时写出完整的 MANIFEST.json（原子替换），删掉 journal。
注意：upstream、selection、stats 等只描述最近一次运行；正式落地每个源跑一次完整的（不带 --limit 与数据集筛选）。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from src.runlog import git_info

MANIFEST_NAME = "MANIFEST.json"
JOURNAL_NAME = "MANIFEST.journal.jsonl"
MANIFEST_SCHEMA = 1
GB = 10**9
REPO_ROOT = Path(__file__).resolve().parents[3]


class BudgetExceeded(RuntimeError):
    """估算体积超过 --budget-gb，或落地后剩余空间会低于红线。开跑前抛出，不下载任何东西。"""


class DiskRedLine(RuntimeError):
    """运行中剩余空间跌破红线，立即停止。"""


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def disk_free(path: str | Path) -> int:
    """path 所在文件系统的剩余字节（path 不存在时向上找存在的祖先）。"""
    p = Path(path).resolve()
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def dir_bytes(path: str | Path) -> int:
    total = 0
    for dp, _dn, fns in os.walk(path):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total


def check_budget(estimate_bytes: int, *, budget_gb: float, root: str | Path, min_free_gb: float) -> dict[str, Any]:
    """开跑前的预算检查：估算 > 预算，或 剩余 − 估算 < 红线，都拒绝（BudgetExceeded）。返回写进清单的记录。"""
    budget = int(budget_gb * GB)
    free = disk_free(root)
    rec = {
        "estimate_bytes": int(estimate_bytes),
        "budget_bytes": budget,
        "root_free_bytes_before": free,
        "min_free_bytes": int(min_free_gb * GB),
    }
    if estimate_bytes > budget:
        raise BudgetExceeded(f"估算 {estimate_bytes / GB:.3f} GB 超过预算 --budget-gb {budget_gb}，拒绝开跑")
    if free - estimate_bytes < min_free_gb * GB:
        raise BudgetExceeded(
            f"剩余 {free / GB:.3f} GB − 估算 {estimate_bytes / GB:.3f} GB 低于红线 {min_free_gb} GB，拒绝开跑"
        )
    return rec


class Manifest:
    """一个源的清单。线程安全地登记文件；每登记一个就写 journal，便于中断后续传。"""

    def __init__(self, source_dir: str | Path, source: str, *, root: str | Path, tmpdir: str | Path, min_free_gb: float, log: Callable[[str], None] | None = None) -> None:
        self.dir = Path(source_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.root = Path(root)
        self.tmpdir = Path(tmpdir)
        self.min_free = int(min_free_gb * GB)
        self.log = log or (lambda s: print(s, file=sys.stderr))
        self._lock = threading.Lock()
        prev = self.load_previous()
        self.prev_files: dict[str, dict[str, Any]] = prev
        self.data: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "source": source,
            "landed_at_utc": None,
            "script": {
                "path": "scripts/data/land_e1.py",
                "argv": sys.argv[1:],
                "git": git_info(REPO_ROOT),
                "python": sys.version.split()[0],
            },
            "upstream": [],
            "license": {},
            "estimate": None,
            "df_log": [],
            "failures": [],
            "selection": {},
            "stats": {},
            "dedupe_facts": {},
            "verification": [],
            "notes": [],
            "files": {},
        }

    # --- 上一次的记录（续传/幂等） ---
    def load_previous(self) -> dict[str, dict[str, Any]]:
        files: dict[str, dict[str, Any]] = {}
        mf = self.dir / MANIFEST_NAME
        if mf.exists():
            try:
                files.update(json.loads(mf.read_text())["files"])
            except (json.JSONDecodeError, KeyError):
                pass
        jr = self.dir / JOURNAL_NAME
        if jr.exists():
            for line in jr.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # 中断时写了半行
                files[rec["path"]] = rec
        return files

    def known_sha256(self, rel: str) -> str | None:
        rec = self.prev_files.get(rel)
        return rec.get("sha256") if rec else None

    def known(self, rel: str) -> dict[str, Any] | None:
        return self.prev_files.get(rel)

    # --- 登记 ---
    def add_file(self, rel: str, *, sha256: str, nbytes: int, **extra: Any) -> None:
        rec = {"path": rel, "sha256": sha256, "bytes": int(nbytes), **extra}
        with self._lock:
            self.data["files"][rel] = rec
            with open(self.dir / JOURNAL_NAME, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    def df(self, step: str) -> None:
        rec = {"t": utc_now(), "step": step, "root_free": disk_free(self.root), "tmp_free": disk_free(self.tmpdir)}
        with self._lock:
            self.data["df_log"].append(rec)
        self.log(f"[df] {step}: root 剩 {rec['root_free'] / GB:.2f} GB，tmp 剩 {rec['tmp_free'] / GB:.2f} GB")
        if rec["root_free"] < self.min_free:
            raise DiskRedLine(f"{step}：根盘剩余 {rec['root_free'] / GB:.2f} GB 低于红线 {self.min_free / GB:.1f} GB，停止")

    def fail(self, **rec: Any) -> None:
        with self._lock:
            self.data["failures"].append(rec)

    def note(self, text: str) -> None:
        self.data["notes"].append(text)

    def verify(self, what: str, ok: bool, **detail: Any) -> None:
        self.data["verification"].append({"what": what, "ok": bool(ok), **detail})
        if not ok:
            self.log(f"[核对失败] {what}: {detail}")

    def write(self) -> Path:
        """写出完整清单（原子替换），删掉 journal。文件条目按路径排序。

        上一次清单里登记过、这次没碰到、但仍在盘上且字节数相符的文件原样带过来（标 carried_over），
        这样只重跑部分数据集（--refcoco-datasets、--odinw-configs、--limit）不会把其余已落地的文件从清单里丢掉。
        """
        carried = 0
        for rel, rec in self.prev_files.items():
            if rel in self.data["files"]:
                continue
            p = self.dir / rel
            if p.is_file() and p.stat().st_size == rec.get("bytes"):
                self.data["files"][rel] = {**rec, "carried_over": True}
                carried += 1
        self.data["n_files_carried_over"] = carried
        self.data["landed_at_utc"] = utc_now()
        self.data["files"] = dict(sorted(self.data["files"].items()))
        self.data["n_files"] = len(self.data["files"])
        self.data["total_bytes"] = sum(r["bytes"] for r in self.data["files"].values())
        out = self.dir / MANIFEST_NAME
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1, sort_keys=False) + "\n")
        os.replace(tmp, out)
        (self.dir / JOURNAL_NAME).unlink(missing_ok=True)
        return out


def read_manifest(source_dir: str | Path) -> dict[str, Any]:
    mf = Path(source_dir) / MANIFEST_NAME
    if not mf.exists():
        raise FileNotFoundError(f"{mf} 不存在：这个源还没有落地（scripts/data/land_e1.py）")
    return json.loads(mf.read_text())
