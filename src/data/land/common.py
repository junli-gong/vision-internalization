"""各数据源落地共用的上下文与小工具（DEV-0012）。"""

from __future__ import annotations

import io
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from src.data.dedupe import phash_image
from src.data.land import net
from src.data.land.manifest import GB, Manifest, check_budget


@dataclass
class LandContext:
    """一次落地运行的参数。limit 不为 None 时是开发机小样本（每个源只落 limit 条）。"""

    root: Path
    tmpdir: Path
    budget_gb: float = 10.0
    min_free_gb: float = 3.0
    workers: int = net.MAX_CONCURRENCY
    limit: int | None = None
    retries: int = net.DEFAULT_RETRIES
    backoff: float = net.DEFAULT_BACKOFF
    dry_run: bool = False
    log: Callable[[str], None] = field(default=lambda s: print(s, file=sys.stderr))
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.tmpdir = Path(self.tmpdir)
        net.check_workers(self.workers)
        if self.limit is not None and self.limit < 1:
            raise ValueError("--limit 必须 ≥ 1")

    def source_dir(self, name: str) -> Path:
        return self.root / name

    def scratch(self, name: str) -> Path:
        p = self.tmpdir / "visint_land" / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def manifest(self, name: str) -> Manifest:
        return Manifest(self.source_dir(name), name, root=self.root, tmpdir=self.tmpdir, min_free_gb=self.min_free_gb, log=self.log)

    def take(self, items: Sequence[Any]) -> list[Any]:
        return list(items if self.limit is None else items[: self.limit])


class DryRun(Exception):
    """--dry-run：估算与预算检查做完就停，不下载。"""


def budget_gate(ctx: LandContext, m: Manifest, estimate_bytes: int, detail: dict[str, Any] | None = None) -> None:
    """开跑前的预算检查；通过后记进清单。--dry-run 时在这里停下。"""
    rec = check_budget(estimate_bytes, budget_gb=ctx.budget_gb, root=ctx.root, min_free_gb=ctx.min_free_gb)
    rec["detail"] = detail or {}
    m.data["estimate"] = rec
    ctx.log(f"[估算] {m.data['source']}: {estimate_bytes / GB:.3f} GB（预算 {ctx.budget_gb} GB）{detail or ''}")
    m.df("开跑前")
    if ctx.dry_run:
        raise DryRun(m.data["source"])


# ---------------------------------------------------------------------------
# HF Hub（只用 REST API 与 resolve 直链，revision 一律钉 commit sha）
# ---------------------------------------------------------------------------

HF_BASE = "https://huggingface.co"


def hf_kind_prefix(kind: str) -> str:
    if kind not in ("dataset", "model"):
        raise ValueError(kind)
    return "datasets/" if kind == "dataset" else ""


def hf_resolve_url(repo: str, rev: str, path: str, kind: str = "dataset", base: str = HF_BASE) -> str:
    from urllib.parse import quote

    return f"{base}/{hf_kind_prefix(kind)}{repo}/resolve/{rev}/{quote(path)}"


def hf_tree(repo: str, rev: str, path: str = "", kind: str = "dataset", base: str = HF_BASE) -> dict[str, dict[str, Any]]:
    """列出 repo@rev 下 path 目录的文件：{路径: {size, sha256（LFS 文件）, git_sha1（非 LFS 文件）}}。"""
    from urllib.parse import quote

    plural = "datasets" if kind == "dataset" else "models"
    url = f"{base}/api/{plural}/{repo}/tree/{rev}/{quote(path)}"
    entries = json.loads(net.get_bytes(url))
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        if e.get("type") != "file":
            continue
        lfs = e.get("lfs")
        out[e["path"]] = {
            "size": int(e["size"]),
            "sha256": lfs["oid"] if lfs else None,
            "git_sha1": None if lfs else e.get("oid"),
        }
    return out


def verify_hf_file(local: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """按 HF tree 的元数据核对本地文件：LFS 文件比 sha256，非 LFS 文件比 git blob sha1，并比字节数。"""
    size = local.stat().st_size
    rec: dict[str, Any] = {"bytes": size, "expected_bytes": meta["size"]}
    ok = size == meta["size"]
    if meta.get("sha256"):
        got = net.sha256_file(local)
        rec.update(sha256=got, expected_sha256=meta["sha256"], method="LFS sha256")
        ok = ok and got == meta["sha256"]
    else:
        got = net.git_blob_sha1(local)
        rec.update(sha256=net.sha256_file(local), git_sha1=got, expected_git_sha1=meta["git_sha1"], method="git blob sha1")
        ok = ok and got == meta["git_sha1"]
    rec["ok"] = ok
    return rec


def fetch_hf_file(ctx: LandContext, m: Manifest, repo: str, rev: str, path: str, dest: Path, meta: dict[str, Any], kind: str = "dataset", base: str = HF_BASE) -> dict[str, Any]:
    """下载 HF 上的一个文件到 dest（通常在 TMPDIR），按 LFS sha256 跳过/核对。返回核对记录。"""
    url = hf_resolve_url(repo, rev, path, kind, base)
    res = net.download(url, dest, expected_sha256=meta.get("sha256"), expected_size=meta["size"], retries=ctx.retries, backoff=ctx.backoff, log=ctx.log)
    rec = verify_hf_file(dest, meta)
    rec.update(url=url, repo=repo, revision=rev, path=path, skipped=res.skipped)
    m.verify(f"HF {repo}@{rev[:10]} {path}", rec["ok"], **{k: v for k, v in rec.items() if k != "ok"})
    if not rec["ok"]:
        raise net.FetchError(url, f"与 HF 元数据不符：{rec}")
    return rec


# ---------------------------------------------------------------------------
# 远端 zip：只读中央目录与需要的成员
# ---------------------------------------------------------------------------


def open_remote_zip(url: str, ctx: LandContext, *, block_size: int = 1 << 20) -> tuple[zipfile.ZipFile, net.HTTPRangeFile]:
    f = net.HTTPRangeFile(url, block_size=block_size, retries=ctx.retries, backoff=ctx.backoff)
    return zipfile.ZipFile(f), f


def extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> int:
    """把 zip 成员解到 dest（先写 .part 再改名）。zipfile 在读完时核对 CRC32，不符抛 BadZipFile。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with zf.open(member) as src, open(part, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    part.replace(dest)
    return dest.stat().st_size


# ---------------------------------------------------------------------------
# 文件登记
# ---------------------------------------------------------------------------


def image_meta(path: Path) -> dict[str, Any]:
    """图像的宽高、格式与感知哈希；打不开就抛错（坏图不进清单）。"""
    with Image.open(path) as im:
        im.load()
        return {"kind": "image", "width": im.width, "height": im.height, "format": im.format, "phash": phash_image(im)}


def image_ext(data: bytes) -> str:
    """按文件头判断扩展名（不转码，原样保存）。"""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:2] == b"BM":
        return ".bmp"
    raise ValueError(f"认不出的图像格式，文件头 {data[:12]!r}")


def register(m: Manifest, src_dir: Path, path: Path, **extra: Any) -> dict[str, Any]:
    rel = path.relative_to(src_dir).as_posix()
    digest = net.sha256_file(path)
    m.add_file(rel, sha256=digest, nbytes=path.stat().st_size, **extra)
    return m.data["files"][rel]


def write_bytes_idempotent(m: Manifest, src_dir: Path, path: Path, data: bytes, **extra: Any) -> dict[str, Any]:
    """写一个由内容决定的文件；已存在且字节相同就不重写。登记进清单。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.read_bytes() == data):
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    return register(m, src_dir, path, **extra)


def write_json(m: Manifest, src_dir: Path, path: Path, obj: Any, **extra: Any) -> dict[str, Any]:
    data = (json.dumps(obj, ensure_ascii=False, indent=None, sort_keys=False, separators=(",", ":")) + "\n").encode("utf-8")
    return write_bytes_idempotent(m, src_dir, path, data, kind="metadata", **extra)


def write_jsonl(m: Manifest, src_dir: Path, path: Path, rows: Iterable[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=False, separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    return write_bytes_idempotent(m, src_dir, path, data, kind="metadata", **extra)


def save_image_bytes(m: Manifest, src_dir: Path, rel_stem: str, data: bytes, **extra: Any) -> dict[str, Any]:
    """把打包在 parquet/json 里的图像字节原样落成文件（扩展名按文件头），登记宽高与感知哈希。"""
    path = src_dir / (rel_stem + image_ext(data))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (path.exists() and path.stat().st_size == len(data) and path.read_bytes() == data):
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        meta = {"kind": "image", "width": im.width, "height": im.height, "format": im.format, "phash": phash_image(im)}
    return register(m, src_dir, path, **meta, **extra)


def download_images(ctx: LandContext, m: Manifest, src_dir: Path, jobs: Sequence[tuple[str, str, dict[str, Any]]], step: str) -> list[dict[str, Any]]:
    """并发下载图像：jobs = [(url, 相对源目录的路径, 额外清单字段)]。已落地且与上次清单哈希相符的跳过。

    每张图下载后核对能否解码，记宽高与感知哈希。失败的 URL 记进清单 failures 并返回，不静默跳过。
    """
    dl_jobs = []
    for url, rel, extra in jobs:
        dl_jobs.append(net.DownloadJob(url=url, dest=src_dir / rel, expected_sha256=m.known_sha256(rel), meta={"rel": rel, **extra}))

    done = [0]

    def on_done(job: net.DownloadJob, res: net.DownloadResult) -> None:
        rel = job.meta["rel"]
        prev = m.known(rel)
        if res.skipped and prev and "phash" in prev:
            meta = {k: prev[k] for k in ("kind", "width", "height", "format", "phash") if k in prev}
        else:
            meta = image_meta(res.path)
        extra = {k: v for k, v in job.meta.items() if k != "rel"}
        m.add_file(rel, sha256=res.sha256, nbytes=res.bytes, url=job.url, **meta, **extra)
        done[0] += 1
        if done[0] % 500 == 0:  # 下载途中也盯着红线；DiskRedLine 会从线程池传回主线程，整个落地停止
            m.df(f"{step}（已完成 {done[0]}）")

    ok, failures = net.download_many(dl_jobs, workers=ctx.workers, retries=ctx.retries, backoff=ctx.backoff, log=ctx.log, on_done=on_done)
    for f in failures:
        m.fail(step=step, **f)
        ctx.log(f"[失败] {f['url']}: {f['error']}")
    ctx.log(f"[{step}] 成功 {len(ok)}（其中跳过 {sum(r.skipped for _, r in ok)}），失败 {len(failures)}")
    m.df(step)
    return failures
