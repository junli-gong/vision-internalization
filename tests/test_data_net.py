"""落地网络层（src/data/land/net.py，DEV-0012）：断点续传、幂等、重试与退避、并发上限、Range 随机读。

用本地的假 HTTP 服务器（FakeServer，支持 Range、HEAD、重定向与故障注入）模拟，不联网。其他 test_data_* 模块也复用它。
"""

from __future__ import annotations

import hashlib
import io
import threading
import time
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.data.land import net


class FakeServer:
    """本地假服务器。files：{路径: 字节}；faults：{路径: [动作…]}，每次请求弹出一个动作：
    "503"（回 503）、"drop"（发完整的头、只发一半正文就断开）、"norange"（忽略 Range，回 200 全文）、"404"。
    redirects：{路径: 目标路径}（302）。log 记 (方法, 路径, Range 头)。"""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.faults: dict[str, list[str]] = {}
        self.redirects: dict[str, str] = {}
        self.log: list[tuple[str, str, str | None]] = []
        self.active = 0
        self.max_active = 0
        self.delay = 0.0
        self._lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):  # 安静
                pass

            def _serve(self, head_only: bool) -> None:
                path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
                rng = self.headers.get("Range")
                with outer._lock:
                    outer.log.append((self.command, path, rng))
                    outer.active += 1
                    outer.max_active = max(outer.max_active, outer.active)
                    q = outer.faults.get(path)
                    action = q.pop(0) if q else None
                try:
                    if outer.delay:
                        time.sleep(outer.delay)
                    if path in outer.redirects:
                        self.send_response(302)
                        self.send_header("Location", outer.redirects[path])
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if action == "503":
                        self.send_response(503)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    if action == "404" or path not in outer.files:
                        self.send_response(404)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    data = outer.files[path]
                    start, end = 0, len(data) - 1
                    partial = False
                    if rng and action != "norange":
                        spec = rng.split("=", 1)[1]
                        a, b = spec.split("-", 1)
                        start = int(a)
                        end = int(b) if b else len(data) - 1
                        end = min(end, len(data) - 1)
                        if start >= len(data):
                            self.send_response(416)
                            self.send_header("Content-Range", f"bytes */{len(data)}")
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return
                        partial = True
                    body = data[start : end + 1]
                    self.send_response(206 if partial else 200)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(len(body)))
                    if partial:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                    self.end_headers()
                    if head_only:
                        return
                    if action == "drop":
                        self.wfile.write(body[: len(body) // 2])
                        self.wfile.flush()
                        self.close_connection = True
                        return
                    self.wfile.write(body)
                finally:
                    with outer._lock:
                        outer.active -= 1

            def do_GET(self):
                self._serve(False)

            def do_HEAD(self):
                self._serve(True)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def url(self, path: str) -> str:
        return self.base + urllib.parse.quote(path)

    def gets(self, path: str) -> list[tuple[str, str, str | None]]:
        return [e for e in self.log if e[0] == "GET" and e[1] == path]

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def _blob(n: int, seed: int = 0) -> bytes:
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"{seed}:{i}".encode()).digest()
        i += 1
    return bytes(out[:n])


FAST = dict(retries=3, backoff=0.0, timeout=5.0)


def test_download_fresh_and_hash(server, tmp_path):
    data = _blob(300_000)
    server.files["/a.bin"] = data
    res = net.download(server.url("/a.bin"), tmp_path / "a.bin", **FAST)
    assert (tmp_path / "a.bin").read_bytes() == data
    assert res.sha256 == hashlib.sha256(data).hexdigest() and res.bytes == len(data) and not res.skipped
    assert not (tmp_path / "a.bin.part").exists()


def test_download_resumes_from_part_with_range(server, tmp_path):
    data = _blob(500_000, 1)
    server.files["/b.bin"] = data
    (tmp_path / "b.bin.part").write_bytes(data[:123_456])
    res = net.download(server.url("/b.bin"), tmp_path / "b.bin", expected_sha256=hashlib.sha256(data).hexdigest(), **FAST)
    assert (tmp_path / "b.bin").read_bytes() == data
    assert res.resumed_from == 123_456
    assert server.gets("/b.bin") == [("GET", "/b.bin", "bytes=123456-")]


def test_download_restarts_when_server_ignores_range(server, tmp_path):
    data = _blob(200_000, 2)
    server.files["/c.bin"] = data
    server.faults["/c.bin"] = ["norange"]
    (tmp_path / "c.bin.part").write_bytes(b"garbage-prefix")
    net.download(server.url("/c.bin"), tmp_path / "c.bin", **FAST)
    assert (tmp_path / "c.bin").read_bytes() == data


def test_download_retries_transient_errors_and_mid_body_drop(server, tmp_path):
    data = _blob(400_000, 3)
    server.files["/d.bin"] = data
    server.faults["/d.bin"] = ["503", "drop", "503"]
    res = net.download(server.url("/d.bin"), tmp_path / "d.bin", expected_sha256=hashlib.sha256(data).hexdigest(), **FAST)
    assert (tmp_path / "d.bin").read_bytes() == data and not res.skipped
    gets = server.gets("/d.bin")
    assert len(gets) == 4
    # 断开之后的重试从 .part 已有的字节续传
    assert gets[2][2] is not None and gets[2][2].startswith("bytes=")


def test_download_404_fails_fast_without_retry(server, tmp_path):
    with pytest.raises(net.FetchError, match="404"):
        net.download(server.url("/missing"), tmp_path / "m.bin", **FAST)
    assert len(server.gets("/missing")) == 1
    assert not (tmp_path / "m.bin").exists()


def test_download_gives_up_after_retries(server, tmp_path):
    server.files["/e.bin"] = b"x" * 10
    server.faults["/e.bin"] = ["503"] * 10
    with pytest.raises(net.FetchError, match="重试 3 次"):
        net.download(server.url("/e.bin"), tmp_path / "e.bin", **FAST)
    assert len(server.gets("/e.bin")) == 4


def test_download_idempotent_skip_when_hash_matches(server, tmp_path):
    data = _blob(50_000, 4)
    server.files["/f.bin"] = data
    sha = hashlib.sha256(data).hexdigest()
    net.download(server.url("/f.bin"), tmp_path / "f.bin", expected_sha256=sha, **FAST)
    n = len(server.log)
    res = net.download(server.url("/f.bin"), tmp_path / "f.bin", expected_sha256=sha, **FAST)
    assert res.skipped and len(server.log) == n  # 没有发请求


def test_download_replaces_corrupt_existing_file(server, tmp_path):
    data = _blob(50_000, 5)
    server.files["/g.bin"] = data
    (tmp_path / "g.bin").write_bytes(b"corrupt")
    res = net.download(server.url("/g.bin"), tmp_path / "g.bin", expected_sha256=hashlib.sha256(data).hexdigest(), **FAST)
    assert not res.skipped and (tmp_path / "g.bin").read_bytes() == data


def test_download_wrong_hash_is_rejected(server, tmp_path):
    server.files["/h.bin"] = b"hello"
    with pytest.raises(net.FetchError, match="sha256"):
        net.download(server.url("/h.bin"), tmp_path / "h.bin", expected_sha256="0" * 64, **FAST)
    assert not (tmp_path / "h.bin").exists() and not (tmp_path / "h.bin.part").exists()


def test_download_follows_redirect(server, tmp_path):
    server.files["/real.bin"] = b"payload"
    server.redirects["/alias.bin"] = "/real.bin"
    net.download(server.url("/alias.bin"), tmp_path / "r.bin", **FAST)
    assert (tmp_path / "r.bin").read_bytes() == b"payload"


def test_download_many_records_failures_and_caps_concurrency(server, tmp_path):
    server.delay = 0.05
    jobs = []
    for i in range(12):
        server.files[f"/img{i}.jpg"] = _blob(1000, i)
        jobs.append(net.DownloadJob(server.url(f"/img{i}.jpg"), tmp_path / f"img{i}.jpg"))
    jobs.append(net.DownloadJob(server.url("/nope.jpg"), tmp_path / "nope.jpg", meta={"rel": "nope.jpg"}))
    ok, failures = net.download_many(jobs, workers=3, **FAST)
    assert len(ok) == 12 and [j.dest.name for j, _ in ok] == [f"img{i}.jpg" for i in range(12)]
    assert len(failures) == 1 and failures[0]["url"].endswith("/nope.jpg") and "404" in failures[0]["error"]
    assert failures[0]["rel"] == "nope.jpg"
    assert 1 <= server.max_active <= 3


@pytest.mark.parametrize("w", [0, 9, 2.5, True])
def test_concurrency_limit_enforced(w):
    with pytest.raises(ValueError):
        net.check_workers(w)
    assert net.MAX_CONCURRENCY == 8


def test_range_file_random_access_and_zip_members(server):
    buf = io.BytesIO()
    payload = {f"dir/file{i}.txt": _blob(70_000 + i, i) for i in range(5)}
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k, v in payload.items():
            zf.writestr(k, v)
    raw = buf.getvalue()
    server.files["/x.zip"] = raw
    f = net.HTTPRangeFile(server.url("/x.zip"), block_size=4096, max_blocks=4, **FAST)
    assert f.size == len(raw)
    f.seek(100)
    assert f.read(50) == raw[100:150]
    f.seek(-10, 2)
    assert f.read() == raw[-10:]
    zf = zipfile.ZipFile(f)
    assert sorted(zf.namelist()) == sorted(payload)
    assert zf.read("dir/file3.txt") == payload["dir/file3.txt"]
    assert f.bytes_fetched < len(raw)  # 只取了中央目录与一个成员


def test_range_file_zip_crc_mismatch_detected(server):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("a.txt", b"A" * 5000)
    raw = bytearray(buf.getvalue())
    raw[100] ^= 0xFF  # 改坏成员内容，CRC 对不上
    server.files["/bad.zip"] = bytes(raw)
    zf = zipfile.ZipFile(net.HTTPRangeFile(server.url("/bad.zip"), **FAST))
    with pytest.raises(zipfile.BadZipFile):
        zf.read("a.txt")


def test_range_file_requires_range_support(server):
    server.files["/n.bin"] = b"abc" * 100
    server.faults["/n.bin"] = ["norange"]
    with pytest.raises(net.FetchError, match="Range"):
        net.HTTPRangeFile(server.url("/n.bin"), **FAST)


def test_head_and_git_blob_sha1(server, tmp_path):
    server.files["/t.txt"] = b"hello\n"
    h = net.head(server.url("/t.txt"), **FAST)
    assert h["size"] == 6 and h["accept_ranges"]
    p = tmp_path / "t.txt"
    p.write_bytes(b"hello\n")
    # `git hash-object` 对 "hello\n" 的结果
    assert net.git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"
