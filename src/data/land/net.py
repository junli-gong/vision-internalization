"""落地用的网络层：断点续传下载、带退避的重试、并发上限、HTTP Range 随机读（DEV-0012）。

只用标准库 urllib，不引入 requests。四件事：
- download：下到 dest.part，按 Range 续传；完整且长度、哈希相符后才改名为 dest。dest 已存在且哈希与期望相符就跳过。
- download_many：线程池，并发不超过 MAX_CONCURRENCY；失败的逐条记下来返回，不静默跳过。
- HTTPRangeFile：只读、可 seek 的远端文件对象（按块缓存的 Range 读）。zipfile、h5py、pyarrow 都能直接打开它，
  用来定点取 zip 成员、读 HDF5 元数据、读 parquet 的单列，不用整包下载（lessons："zip 用 Range 抽取 central
  directory 定点取文件"）。
- 失败分两类：可重试（网络错误、5xx、408、429、读到一半断开）按指数退避重试；其余 4xx 立即失败。
"""

from __future__ import annotations

import hashlib
import http.client
import io
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

# 并发上限（任务书：不用浏览器式的并发轰炸，并发 ≤ 8）。
MAX_CONCURRENCY = 8
USER_AGENT = "visint-e1-landing/1 (non-commercial research)"
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF = 2.0  # 第 k 次重试前等 backoff * 2**(k-1) 秒
_CHUNK = 1 << 20
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    """下载或 Range 读最终失败（重试用尽，或遇到不可重试的状态码）。"""

    def __init__(self, url: str, msg: str) -> None:
        super().__init__(f"{url}: {msg}")
        self.url = url
        self.msg = msg


class _Retryable(Exception):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: str | Path) -> str:
    """git 的 blob 哈希（HF 上非 LFS 文件的 oid 就是它）。"""
    size = os.path.getsize(path)
    h = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None) -> urllib.request.Request:
    h = {"User-Agent": USER_AGENT}
    if headers:
        h.update(headers)
    return urllib.request.Request(url, method=method, headers=h)


def _open(url: str, *, method: str = "GET", headers: dict[str, str] | None = None, timeout: float = DEFAULT_TIMEOUT):
    """打开 URL；把可重试的失败统一成 _Retryable，不可重试的 HTTP 错误原样抛出。"""
    try:
        return urllib.request.urlopen(_request(url, method=method, headers=headers), timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in _RETRY_STATUS:
            raise _Retryable(f"HTTP {e.code}") from e
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        raise _Retryable(f"{type(e).__name__}: {e}") from e


def _with_retries(url: str, fn: Callable[[], Any], *, retries: int, backoff: float, log: Callable[[str], None] | None):
    last = ""
    for attempt in range(retries + 1):
        if attempt:
            delay = backoff * 2 ** (attempt - 1)
            if log:
                log(f"重试 {attempt}/{retries}（{last}），{delay:.1f}s 后：{url}")
            time.sleep(delay)
        try:
            return fn()
        except _Retryable as e:
            last = str(e)
        except urllib.error.HTTPError as e:
            raise FetchError(url, f"HTTP {e.code}（不可重试）") from e
    raise FetchError(url, f"重试 {retries} 次后仍失败：{last}")


def head(url: str, *, retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """HEAD（跟随重定向）。返回 size（Content-Length，没有则 None）、accept_ranges、final_url。"""

    def go():
        with _open(url, method="HEAD", timeout=timeout) as r:
            cl = r.headers.get("Content-Length")
            return {
                "size": int(cl) if cl is not None else None,
                "accept_ranges": (r.headers.get("Accept-Ranges") or "").lower() == "bytes",
                "final_url": r.geturl(),
                "content_type": r.headers.get("Content-Type"),
            }

    return _with_retries(url, go, retries=retries, backoff=backoff, log=None)


def get_bytes(url: str, *, retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    """整段读进内存（只用于 KB 级的小文件与 API 响应）。"""

    def go():
        try:
            with _open(url, timeout=timeout) as r:
                return r.read()
        except (ConnectionError, TimeoutError, OSError, http.client.HTTPException) as e:  # 读到一半断开
            if isinstance(e, urllib.error.HTTPError):
                raise
            raise _Retryable(f"{type(e).__name__}: {e}") from e

    return _with_retries(url, go, retries=retries, backoff=backoff, log=None)


@dataclass
class DownloadResult:
    url: str
    path: Path
    bytes: int
    sha256: str
    skipped: bool  # dest 已存在且哈希相符，没有联网
    resumed_from: int = 0  # 续传时 .part 已有的字节数


def download(
    url: str,
    dest: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    timeout: float = DEFAULT_TIMEOUT,
    log: Callable[[str], None] | None = None,
) -> DownloadResult:
    """把 url 下到 dest，可断点续传、幂等。

    - dest 已存在：给了 expected_sha256 且相符 → 跳过；不相符 → 删掉重下。没给 expected_sha256 时不跳过
      （调用方从上一次的清单里取哈希传进来，就能跳过）。
    - 续传：dest.part 存在时发 Range: bytes=<已有字节>-；服务器回 206 就接着写，回 200 就从头写。
    - 完成后核对 expected_size 与 expected_sha256，不符就删掉 .part 并抛 FetchError（不重试：内容本身不对）。
    """
    dest = Path(dest)
    if dest.exists():
        if expected_sha256 is not None and sha256_file(dest) == expected_sha256:
            return DownloadResult(url, dest, dest.stat().st_size, expected_sha256, skipped=True)
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    first_offset = part.stat().st_size if part.exists() else 0

    def go():
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else None
        try:
            with _open(url, headers=headers, timeout=timeout) as r:
                status = getattr(r, "status", 200)
                if offset and status != 206:
                    offset = 0  # 服务器不支持 Range，从头来
                total = None
                cl = r.headers.get("Content-Length")
                if cl is not None:
                    total = offset + int(cl)
                with open(part, "ab" if offset else "wb") as f:
                    for chunk in iter(lambda: r.read(_CHUNK), b""):
                        f.write(chunk)
        except urllib.error.HTTPError as e:
            if e.code == 416 and offset and expected_size is not None and offset == expected_size:
                return  # .part 已完整
            if e.code == 416:
                part.unlink(missing_ok=True)
                raise _Retryable("HTTP 416，丢弃 .part 从头下") from e
            raise
        except _Retryable:
            raise
        except (ConnectionError, TimeoutError, OSError, http.client.HTTPException) as e:
            raise _Retryable(f"读到一半断开：{type(e).__name__}: {e}") from e
        got = part.stat().st_size
        if total is not None and got != total:
            raise _Retryable(f"长度不足：{got}/{total}")

    _with_retries(url, go, retries=retries, backoff=backoff, log=log)
    size = part.stat().st_size
    digest = sha256_file(part)
    if expected_size is not None and size != expected_size:
        part.unlink()
        raise FetchError(url, f"长度 {size} 与期望 {expected_size} 不符")
    if expected_sha256 is not None and digest != expected_sha256:
        part.unlink()
        raise FetchError(url, f"sha256 {digest} 与期望 {expected_sha256} 不符")
    os.replace(part, dest)
    return DownloadResult(url, dest, size, digest, skipped=False, resumed_from=first_offset)


@dataclass
class DownloadJob:
    url: str
    dest: Path
    expected_sha256: str | None = None
    expected_size: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def download_many(
    jobs: Sequence[DownloadJob],
    *,
    workers: int = MAX_CONCURRENCY,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    timeout: float = DEFAULT_TIMEOUT,
    log: Callable[[str], None] | None = None,
    on_done: Callable[[DownloadJob, DownloadResult], None] | None = None,
) -> tuple[list[tuple[DownloadJob, DownloadResult]], list[dict[str, Any]]]:
    """并发下载。返回 (成功列表，按 jobs 顺序；失败列表 [{url, dest, error}])。并发超过 MAX_CONCURRENCY 报错。"""
    check_workers(workers)
    results: list[tuple[DownloadJob, DownloadResult] | None] = [None] * len(jobs)
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()

    def one(i: int) -> None:
        job = jobs[i]
        try:
            res = download(
                job.url,
                job.dest,
                expected_sha256=job.expected_sha256,
                expected_size=job.expected_size,
                retries=retries,
                backoff=backoff,
                timeout=timeout,
                log=log,
            )
        except FetchError as e:
            with lock:
                failures.append({"url": job.url, "dest": str(job.dest), "error": e.msg, **job.meta})
            return
        results[i] = (job, res)
        if on_done is not None:
            with lock:
                on_done(job, res)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, range(len(jobs))))
    return [r for r in results if r is not None], failures


def check_workers(workers: int) -> None:
    if isinstance(workers, bool) or int(workers) != workers or not 1 <= workers <= MAX_CONCURRENCY:
        raise ValueError(f"并发数必须在 1..{MAX_CONCURRENCY}，收到 {workers}")


class HTTPRangeFile(io.RawIOBase):
    """只读、可 seek 的远端文件：按 block_size 对齐的块做 Range 读，LRU 缓存 max_blocks 块。

    大于 4 个块的单次读直接发一个 Range 请求，不进缓存（读 zip 成员时常见）。bytes_fetched 记网络读取量。
    服务器不回 206 或 Content-Range 对不上就报错（不支持 Range 的服务器不能这样用）。
    """

    def __init__(
        self,
        url: str,
        *,
        size: int | None = None,
        block_size: int = 1 << 20,
        max_blocks: int = 64,
        retries: int = DEFAULT_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.retries, self.backoff, self.timeout = retries, backoff, timeout
        self.size = size if size is not None else self._probe_size()
        self.pos = 0
        self.bytes_fetched = 0
        self.n_requests = 0
        self._cache: dict[int, bytes] = {}
        self._order: list[int] = []

    def _probe_size(self) -> int:
        def go():
            with _open(self.url, headers={"Range": "bytes=0-0"}, timeout=self.timeout) as r:
                cr = r.headers.get("Content-Range")
                if getattr(r, "status", 200) != 206 or not cr or "/" not in cr:
                    raise FetchError(self.url, "服务器不支持 Range（探测 bytes=0-0 没有得到 206）")
                return int(cr.rsplit("/", 1)[1])

        return _with_retries(self.url, go, retries=self.retries, backoff=self.backoff, log=None)

    def _get_range(self, start: int, end_incl: int) -> bytes:
        def go():
            try:
                with _open(self.url, headers={"Range": f"bytes={start}-{end_incl}"}, timeout=self.timeout) as r:
                    if getattr(r, "status", 200) != 206:
                        raise FetchError(self.url, f"Range 请求得到 {getattr(r, 'status', '?')}，不是 206")
                    data = r.read()
            except (ConnectionError, TimeoutError, http.client.HTTPException) as e:
                raise _Retryable(f"{type(e).__name__}: {e}") from e
            if len(data) != end_incl - start + 1:
                raise _Retryable(f"Range 读到 {len(data)} 字节，期望 {end_incl - start + 1}")
            return data

        data = _with_retries(self.url, go, retries=self.retries, backoff=self.backoff, log=None)
        self.bytes_fetched += len(data)
        self.n_requests += 1
        return data

    def _block(self, k: int) -> bytes:
        if k in self._cache:
            self._order.remove(k)
            self._order.append(k)
            return self._cache[k]
        start = k * self.block_size
        end = min(start + self.block_size, self.size) - 1
        data = self._get_range(start, end)
        self._cache[k] = data
        self._order.append(k)
        while len(self._order) > self.max_blocks:
            del self._cache[self._order.pop(0)]
        return data

    # --- io.RawIOBase 接口 ---
    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new = offset
        elif whence == 1:
            new = self.pos + offset
        elif whence == 2:
            new = self.size + offset
        else:
            raise ValueError(f"whence={whence}")
        if new < 0:
            raise ValueError("seek 到负位置")
        self.pos = new
        return new

    def readinto(self, b) -> int:
        mv = memoryview(b).cast("B")
        n = min(len(mv), max(self.size - self.pos, 0))
        if n <= 0:
            return 0
        if n > 4 * self.block_size:
            data = self._get_range(self.pos, self.pos + n - 1)
            mv[:n] = data
            self.pos += n
            return n
        out = 0
        while out < n:
            k, off = divmod(self.pos, self.block_size)
            blk = self._block(k)
            take = min(len(blk) - off, n - out)
            mv[out : out + take] = blk[off : off + take]
            out += take
            self.pos += take
        return out


# ---------------------------------------------------------------------------
# DEV-0022：给"按成员偏移批量取 zip 成员"用的两个通用函数（上面的函数行为不变）
# ---------------------------------------------------------------------------


class HTTPStatusError(FetchError):
    """不可重试的 HTTP 状态（带状态码），调用方可据此决定是否刷新签名直链后重试。"""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(url, f"HTTP {status}（不可重试）")
        self.status = status


def resolve_final_url(url: str, *, retries: int = DEFAULT_RETRIES, backoff: float = DEFAULT_BACKOFF, timeout: float = DEFAULT_TIMEOUT) -> tuple[str, int]:
    """跟随重定向，返回 (最终 URL, 文件总长)。用 Range: bytes=0-0 探测，只取 1 个字节。

    HF 的 resolve 直链 302 到带签名的 CDN 地址：先解析一次，之后的 Range 请求直接打 CDN，免得每个请求都过一遍
    resolve（resolve 有频率限制）。服务器不回 206 就报错（不支持 Range 的服务器不能这样取成员）。"""

    def go():
        try:
            with _open(url, headers={"Range": "bytes=0-0"}, timeout=timeout) as r:
                cr = r.headers.get("Content-Range")
                if getattr(r, "status", 200) != 206 or not cr or "/" not in cr:
                    raise FetchError(url, "服务器不支持 Range（探测 bytes=0-0 没有得到 206）")
                r.read()
                return r.geturl(), int(cr.rsplit("/", 1)[1])
        except (ConnectionError, TimeoutError, http.client.HTTPException) as e:
            raise _Retryable(f"{type(e).__name__}: {e}") from e

    return _with_retries(url, go, retries=retries, backoff=backoff, log=None)


def range_get(
    url: str,
    start: int,
    end_incl: int,
    *,
    expect_total: int | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    timeout: float = DEFAULT_TIMEOUT,
    log: Callable[[str], None] | None = None,
) -> bytes:
    """一次 Range 读 [start, end_incl]，带退避重试。expect_total 给了就核对 Content-Range 的总长（上游换了文件就报错）。

    不可重试的 4xx 抛 HTTPStatusError（带状态码）；内容不对（不是 206、总长不符）抛 FetchError，不重试。"""

    def go():
        try:
            with _open(url, headers={"Range": f"bytes={start}-{end_incl}"}, timeout=timeout) as r:
                status = getattr(r, "status", 200)
                if status != 206:
                    raise FetchError(url, f"Range 请求得到 {status}，不是 206")
                cr = r.headers.get("Content-Range") or ""
                if expect_total is not None and cr.rsplit("/", 1)[-1] != str(expect_total):
                    raise FetchError(url, f"Content-Range {cr!r} 的总长与期望 {expect_total} 不符（上游文件变了？）")
                data = r.read()
        except urllib.error.HTTPError as e:
            raise HTTPStatusError(url, e.code) from e
        except (ConnectionError, TimeoutError, http.client.HTTPException) as e:
            raise _Retryable(f"读到一半断开：{type(e).__name__}: {e}") from e
        if len(data) != end_incl - start + 1:
            raise _Retryable(f"Range 读到 {len(data)} 字节，期望 {end_incl - start + 1}")
        return data

    return _with_retries(url, go, retries=retries, backoff=backoff, log=log)
