"""各数据源落地共用的上下文与小工具（DEV-0012；DEV-0022 加了文末"按成员偏移批量取 zip 成员"一节）。"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import sys
import threading
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
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


# ---------------------------------------------------------------------------
# 按成员偏移批量取 zip 成员（DEV-0022）
#
# 一次读出 zip 的"尾部"（中央目录起点到文件末尾，含 zip64 记录与 EOCD），算 sha256 与 SPEC 钉住的值比对：尾部相同
# ⇒ 每个成员的名字、偏移、大小与 CRC32 都与开发机核对过的那份相同。之后把要取的成员按偏移排序，相邻的合并成少量
# Range 请求（span），并发取回，逐个解析本地头、解压、核对 CRC32 与大小，先写 .part 再改名。
# 来源可以是远端 URL（HTTP Range）或本地已有的 zip（国内网络连不上时的备选，同样核对尾部 sha256 与成员 CRC32）。
# ---------------------------------------------------------------------------

_EOCD_SIG = b"PK\x05\x06"
_ZIP64_LOC_SIG = b"PK\x06\x07"
_ZIP64_EOCD_SIG = b"PK\x06\x06"
_CD_SIG = b"PK\x01\x02"
_LOCAL_SIG = b"PK\x03\x04"
_EOCD_SCAN = 22 + 65_535 + 20  # EOCD + 最长注释 + zip64 locator


class ZipSource:
    """zip 的字节来源。read(start, end) 返回 [start, end) 的字节；size 是文件总长；where 写进清单。"""

    kind = "?"
    where = "?"
    size = 0
    n_requests = 0
    bytes_fetched = 0

    def read(self, start: int, end: int) -> bytes:  # pragma: no cover - 接口
        raise NotImplementedError

    def record(self) -> dict[str, Any]:
        return {"access": self.kind, "where": self.where, "n_requests": self.n_requests, "bytes_fetched": self.bytes_fetched}


class RemoteZip(ZipSource):
    """远端 zip：先跟随重定向解析出最终直链（HF 的签名 CDN 地址），之后的 Range 请求都打它。

    签名直链过期（401/403/410）时重新解析一次再试。每次 Range 都核对 Content-Range 的总长。"""

    kind = "remote"

    def __init__(self, url: str, *, retries: int = net.DEFAULT_RETRIES, backoff: float = net.DEFAULT_BACKOFF, timeout: float = net.DEFAULT_TIMEOUT) -> None:
        self.url = self.where = url
        self.retries, self.backoff, self.timeout = retries, backoff, timeout
        self._lock = threading.Lock()
        self._final, self.size = net.resolve_final_url(url, retries=retries, backoff=backoff, timeout=timeout)
        self.n_requests = 0
        self.bytes_fetched = 0
        self.n_refresh = 0

    def _refresh(self, stale: str) -> str:
        with self._lock:
            if self._final == stale:
                self._final, size = net.resolve_final_url(self.url, retries=self.retries, backoff=self.backoff, timeout=self.timeout)
                self.n_refresh += 1
                if size != self.size:
                    raise net.FetchError(self.url, f"重新解析后文件总长 {size} ≠ {self.size}")
            return self._final

    def read(self, start: int, end: int) -> bytes:
        if end <= start:
            return b""
        final = self._final
        kw = dict(expect_total=self.size, retries=self.retries, backoff=self.backoff, timeout=self.timeout)
        try:
            data = net.range_get(final, start, end - 1, **kw)
        except net.HTTPStatusError as e:
            if e.status not in (401, 403, 410) or final == self.url:
                raise
            data = net.range_get(self._refresh(final), start, end - 1, **kw)
        with self._lock:
            self.n_requests += 1
            self.bytes_fetched += len(data)
        return data

    def record(self) -> dict[str, Any]:
        return {**super().record(), "url": self.url, "n_url_refresh": self.n_refresh}


class LocalZip(ZipSource):
    """本地已有的 zip（只读）。"""

    kind = "local"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.where = str(self.path)
        self.size = self.path.stat().st_size
        self.n_requests = 0
        self.bytes_fetched = 0

    def read(self, start: int, end: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
        if len(data) != end - start:
            raise zipfile.BadZipFile(f"{self.path}：读 [{start}, {end}) 只得到 {len(data)} 字节")
        self.n_requests += 1
        self.bytes_fetched += len(data)
        return data


def zip_source(ctx: LandContext, url: str, local: str | Path | None = None) -> ZipSource:
    """local 给了就用本地 zip，否则远端 URL（Range）。"""
    if local:
        return LocalZip(local)
    return RemoteZip(url, retries=ctx.retries, backoff=ctx.backoff)


class _TailFile(io.RawIOBase):
    """只装着 zip 尾部（中央目录起点到文件末尾）的只读文件对象，偏移与原文件相同，供 zipfile 解析中央目录。"""

    def __init__(self, tail: bytes, base: int, size: int) -> None:
        super().__init__()
        self.tail, self.base, self.size, self.pos = tail, base, size, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:
        mv = memoryview(b).cast("B")
        n = min(len(mv), max(self.size - self.pos, 0))
        if n <= 0:
            return 0
        if self.pos < self.base:
            raise zipfile.BadZipFile(f"解析中央目录时读到尾部之外（偏移 {self.pos} < 中央目录起点 {self.base}）")
        o = self.pos - self.base
        mv[:n] = self.tail[o : o + n]
        self.pos += n
        return n


@dataclass(frozen=True)
class ZipMember:
    name: str
    header_offset: int
    compress_size: int
    file_size: int
    crc: int
    compress_type: int
    end_bound: int  # 下一个条目的本地头偏移（或中央目录起点）：本成员的本地头、数据与数据描述符都在 [header_offset, end_bound) 里


@dataclass
class ZipIndex:
    size: int
    start_dir: int
    tail_bytes: int
    tail_sha256: str
    n_entries: int
    members: dict[str, ZipMember]  # 只含文件（不含目录条目）

    def pins(self) -> dict[str, Any]:
        return {"size": self.size, "n_entries": self.n_entries, "start_dir": self.start_dir, "tail_bytes": self.tail_bytes, "tail_sha256": self.tail_sha256}


def read_zip_index(src: ZipSource) -> ZipIndex:
    """读 EOCD（必要时读 zip64 记录）定位中央目录，一次读出尾部，算 sha256，用 zipfile 解析成员表。"""
    n = min(src.size, _EOCD_SCAN)
    buf = src.read(src.size - n, src.size)
    i = buf.rfind(_EOCD_SIG)
    if i < 0 or len(buf) - i < 22:
        raise zipfile.BadZipFile(f"{src.where}：末尾 {n} 字节里找不到 EOCD")
    _sig, _d, _dcd, _n_this, n_total, cd_size, cd_offset, _clen = struct.unpack("<4s4H2LH", buf[i : i + 22])
    if cd_offset == 0xFFFFFFFF or n_total == 0xFFFF or cd_size == 0xFFFFFFFF:
        loc = buf[i - 20 : i] if i >= 20 else src.read(src.size - n + i - 20, src.size - n + i)
        if loc[:4] != _ZIP64_LOC_SIG:
            raise zipfile.BadZipFile(f"{src.where}：EOCD 指向 zip64，但找不到 zip64 locator")
        rec_off = struct.unpack("<4sLQL", loc)[2]
        rec = src.read(rec_off, rec_off + 56)
        if rec[:4] != _ZIP64_EOCD_SIG:
            raise zipfile.BadZipFile(f"{src.where}：zip64 EOCD 记录的签名不对")
        n_total, cd_size, cd_offset = struct.unpack("<4sQ2H2L4Q", rec)[7:10]
    tail = src.read(cd_offset, src.size)
    if n_total and tail[:4] != _CD_SIG:
        raise zipfile.BadZipFile(f"{src.where}：偏移 {cd_offset} 处不是中央目录（zip 前面有拼接的数据？）")
    infos = zipfile.ZipFile(_TailFile(tail, cd_offset, src.size)).infolist()
    if len(infos) != n_total:
        raise zipfile.BadZipFile(f"{src.where}：中央目录条目数 {len(infos)} ≠ EOCD 记的 {n_total}")
    order = sorted(infos, key=lambda z: z.header_offset)
    bound = {id(z): (order[k + 1].header_offset if k + 1 < len(order) else cd_offset) for k, z in enumerate(order)}
    members = {z.filename: ZipMember(z.filename, z.header_offset, z.compress_size, z.file_size, z.CRC, z.compress_type, bound[id(z)]) for z in infos if not z.is_dir()}
    return ZipIndex(src.size, cd_offset, len(tail), hashlib.sha256(tail).hexdigest(), len(infos), members)


def check_zip_pins(m: Manifest, what: str, index: ZipIndex, pins: dict[str, Any]) -> None:
    """尾部（中央目录）与 SPEC 钉住的一致才继续；不一致说明上游换了文件，记核对失败并停下。"""
    got = index.pins()
    want = {k: v for k, v in pins.items() if k in got}
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    m.verify(f"{what}：zip 大小、条目数与中央目录 sha256 = SPEC 钉住的值", not bad, got=got, expected=want)
    if bad:
        raise zipfile.BadZipFile(f"{what}：zip 与钉住的不符 {bad}")


def crc32_file(path: str | Path) -> int:
    crc = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def member_bytes(buf: bytes, base: int, mm: ZipMember) -> bytes:
    """从覆盖 [base, base+len(buf)) 的字节里取出成员 mm：核对本地头签名与文件名，解压，核对大小与 CRC32。"""
    o = mm.header_offset - base
    if o < 0 or buf[o : o + 4] != _LOCAL_SIG:
        raise zipfile.BadZipFile(f"{mm.name}：偏移 {mm.header_offset} 处不是本地文件头")
    flags = struct.unpack("<H", buf[o + 6 : o + 8])[0]
    n_name, n_extra = struct.unpack("<HH", buf[o + 26 : o + 30])
    name = buf[o + 30 : o + 30 + n_name].decode("utf-8" if flags & 0x800 else "cp437")
    if name != mm.name:
        raise zipfile.BadZipFile(f"本地头里的文件名 {name!r} ≠ 中央目录的 {mm.name!r}")
    d0 = o + 30 + n_name + n_extra
    comp = buf[d0 : d0 + mm.compress_size]
    if len(comp) != mm.compress_size:
        raise zipfile.BadZipFile(f"{mm.name}：数据越过了取回的范围")
    if mm.compress_type == zipfile.ZIP_STORED:
        data = bytes(comp)
    elif mm.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(comp, -15)
    else:
        raise zipfile.BadZipFile(f"{mm.name}：不支持的压缩方式 {mm.compress_type}")
    if len(data) != mm.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != mm.crc:
        raise zipfile.BadZipFile(f"{mm.name}：大小或 CRC32 与中央目录不符")
    return data


@dataclass
class MemberJob:
    member: str
    dest: Path
    meta: dict[str, Any] = field(default_factory=dict)


def plan_spans(members: Sequence[ZipMember], *, gap: int, max_span: int) -> list[list[ZipMember]]:
    """按偏移排序，相邻成员间隔 ≤ gap 且合并后不超过 max_span 就并进同一个 Range 请求。"""
    spans: list[list[ZipMember]] = []
    cur: list[ZipMember] = []
    start = end = 0
    for mm in sorted(members, key=lambda z: z.header_offset):
        if cur and mm.header_offset - end <= gap and mm.end_bound - start <= max_span:
            cur.append(mm)
            end = mm.end_bound
            continue
        if cur:
            spans.append(cur)
        cur, start, end = [mm], mm.header_offset, mm.end_bound
    if cur:
        spans.append(cur)
    return spans


def fetch_zip_members(
    src: ZipSource,
    index: ZipIndex,
    jobs: Sequence[MemberJob],
    *,
    workers: int = net.MAX_CONCURRENCY,
    gap: int = 0,
    max_span: int = 32 << 20,
    process: Callable[[MemberJob, Path, dict[str, Any]], dict[str, Any]] | None = None,
    on_done: Callable[[MemberJob, dict[str, Any]], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[tuple[MemberJob, dict[str, Any]]], list[dict[str, Any]]]:
    """取回 jobs 里的成员到各自的 dest。返回 (成功 [(job, rec)]，失败 [{member, dest, error, **job.meta}])。

    - 幂等：dest 已存在且大小与 CRC32 都与中央目录相同就跳过（不联网），rec["skipped"]=True。
    - rec：sha256、bytes、crc32、zip_member、skipped。process(job, dest, rec) 在工作线程里跑（解码图像、算感知哈希、
      格式核对），返回的字段并进 rec；它抛 ValueError 记为该成员失败（文件删掉）。on_done 在锁里调用（登记清单）；
      它抛的异常（例如 DiskRedLine）传回主线程，整个落地停止。
    - 一个 span 的网络读取失败，span 里的成员都记失败；单个成员 CRC 不符只记它自己。不静默跳过。
    - on_done 或写盘抛出的异常（DiskRedLine、磁盘满）会让还没开始的 span 全部不再执行，异常传回调用方。
    """
    net.check_workers(workers)
    lock = threading.Lock()
    ok: list[tuple[MemberJob, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    stop = threading.Event()
    errors: list[BaseException] = []

    def guarded(fn: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def run(arg: Any) -> Any:
            if stop.is_set():
                return None
            try:
                return fn(arg)
            except BaseException as e:  # noqa: BLE001 — 记下来、叫停其余任务，主线程再抛出
                stop.set()
                with lock:
                    errors.append(e)
                return None

        return run

    def fail(job: MemberJob, error: str) -> None:
        with lock:
            failures.append({"member": job.member, "dest": str(job.dest), "error": error, **job.meta})

    def finish(job: MemberJob, rec: dict[str, Any]) -> None:
        if process is not None:
            try:
                rec.update(process(job, job.dest, rec))
            except (ValueError, OSError) as e:  # 格式核对不通过、图像解不开（PIL 的 UnidentifiedImageError 是 OSError）
                job.dest.unlink(missing_ok=True)
                fail(job, f"核对不通过：{e}")
                return
        with lock:
            ok.append((job, rec))
            if on_done is not None:
                on_done(job, rec)

    def already_there(job: MemberJob) -> bool:
        mm = index.members[job.member]
        if not (job.dest.exists() and job.dest.stat().st_size == mm.file_size and crc32_file(job.dest) == mm.crc):
            return False
        finish(job, {"sha256": net.sha256_file(job.dest), "bytes": mm.file_size, "crc32": f"{mm.crc:08x}", "zip_member": mm.name, "skipped": True})
        return True

    known = []
    for job in jobs:
        if job.member in index.members:
            known.append(job)
        else:
            fail(job, "zip 里没有这个成员")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        skipped = list(ex.map(guarded(already_there), known))
    if errors:
        raise errors[0]
    todo: dict[str, list[MemberJob]] = {}
    for job, sk in zip(known, skipped):
        if not sk:
            todo.setdefault(job.member, []).append(job)

    def one_span(span: list[ZipMember]) -> None:
        start, end = span[0].header_offset, span[-1].end_bound
        try:
            buf = src.read(start, end)
        except (net.FetchError, OSError, zipfile.BadZipFile) as e:
            for mm in span:
                for job in todo[mm.name]:
                    fail(job, f"读取 [{start}, {end}) 失败：{e}")
            return
        for mm in span:
            try:
                data = member_bytes(buf, start, mm)
            except (zipfile.BadZipFile, zlib.error) as e:
                for job in todo[mm.name]:
                    fail(job, str(e))
                continue
            digest = hashlib.sha256(data).hexdigest()
            for job in todo[mm.name]:
                job.dest.parent.mkdir(parents=True, exist_ok=True)
                part = job.dest.with_name(job.dest.name + ".part")
                part.write_bytes(data)
                os.replace(part, job.dest)
                finish(job, {"sha256": digest, "bytes": len(data), "crc32": f"{mm.crc:08x}", "zip_member": mm.name, "skipped": False})

    spans = plan_spans([index.members[k] for k in todo], gap=gap, max_span=max_span)
    if log:
        log(f"[zip] {src.where}：{len(known) - len(todo)} 个成员已在且 CRC 相符（跳过），要取 {len(todo)} 个，合并成 {len(spans)} 个 Range 请求")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(guarded(one_span), spans))
    if errors:
        raise errors[0]
    return ok, failures
