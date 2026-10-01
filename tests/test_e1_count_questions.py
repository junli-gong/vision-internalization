"""DEV-0015 §4：CountBench 人工问句的落地（src/data/land/countbenchqa.py，scripts/data/land_e1.py countbenchqa_questions）与
E1 读取（src/e1/tasks.load_count_questions）。上游文件由本地假服务器提供（tests/test_data_net.FakeServer），不联网。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from src.data.land import countbenchqa as CQ
from src.data.land import net
from src.data.land.common import LandContext
from src.data.land.manifest import read_manifest
from src.e1.prompts import COUNTBENCHQA_FIELD
from src.e1.tasks import FIXTURE_DIR, DataPaths, load_count_questions, select
from tests.test_data_net import FakeServer

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def upstream(server: FakeServer, n: int = 6) -> tuple[dict, bytes]:
    data = json.dumps([{"question": f"How many cats{k} are there in the image?"} for k in range(n)]).encode()
    server.files["/bv/q.json"] = data
    spec = {"url": server.url("/bv/q.json"), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "n_questions": n}
    return spec, data


def ctx_for(tmp_path: Path, **kw) -> LandContext:
    return LandContext(root=tmp_path / "root", tmpdir=tmp_path / "tmp", retries=2, backoff=0.0, log=lambda s: None, **kw)


def test_land_writes_file_and_manifest_and_is_idempotent(server, tmp_path):
    spec, data = upstream(server)
    ctx = ctx_for(tmp_path)
    CQ.land(ctx, spec)
    src = ctx.root / CQ.SOURCE
    assert (src / CQ.FILE_NAME).read_bytes() == data  # 原样字节
    man = read_manifest(src)
    assert all(v["ok"] for v in man["verification"]) and not man["failures"]
    assert man["files"][CQ.FILE_NAME]["sha256"] == spec["sha256"] and man["upstream"][0]["commit"] == CQ.SPEC["commit"]
    assert man["license"]["text"].startswith("Apache-2.0") and man["stats"]["n_questions"] == 6
    assert any("countbench 还没有落地" in n for n in man["notes"])
    n_get = sum(1 for m, p, _ in server.log if m == "GET" and p == "/bv/q.json")
    CQ.land(ctx, spec)  # 已落地且哈希相符：不再联网
    assert sum(1 for m, p, _ in server.log if m == "GET" and p == "/bv/q.json") == n_get
    assert read_manifest(src)["upstream"][0]["skipped"] is True


def test_land_cross_checks_countbench_rows(server, tmp_path):
    spec, _ = upstream(server, n=4)
    ctx = ctx_for(tmp_path)
    shutil.copytree(FIXTURE_DIR / "root" / "countbench", ctx.root / "countbench")  # 4 行
    CQ.land(ctx, spec)
    v = {x["what"]: x for x in read_manifest(ctx.root / CQ.SOURCE)["verification"]}
    assert v["条数 = 已落地的 countbench.jsonl 的行数（第 k 条对应第 k 行）"]["ok"] is True
    spec5, _ = upstream(server, n=5)
    CQ.land(ctx, spec5)
    v = {x["what"]: x for x in read_manifest(ctx.root / CQ.SOURCE)["verification"]}
    assert v["条数 = 已落地的 countbench.jsonl 的行数（第 k 条对应第 k 行）"]["ok"] is False


def test_land_rejects_wrong_hash_and_flags_bad_structure(server, tmp_path):
    spec, _ = upstream(server)
    with pytest.raises(net.FetchError):
        CQ.land(ctx_for(tmp_path), {**spec, "sha256": "0" * 64})
    assert not (tmp_path / "root" / CQ.SOURCE / CQ.FILE_NAME).exists()
    bad = json.dumps([{"question": "x"}, {"q": "y"}]).encode()
    server.files["/bv/bad.json"] = bad
    ctx = ctx_for(tmp_path / "b")
    CQ.land(ctx, {"url": server.url("/bv/bad.json"), "sha256": hashlib.sha256(bad).hexdigest(), "bytes": len(bad),
                  "n_questions": 2})
    v = [x for x in read_manifest(ctx.root / CQ.SOURCE)["verification"] if x["what"].startswith("结构")]
    assert v[0]["ok"] is False and "第 1 条" in v[0]["problems_head"][0]


def test_check_questions_rules():
    assert CQ.check_questions([{"question": "a?"}], 1) == []
    assert CQ.check_questions({"question": "a?"}, 1)[0].startswith("顶层应为列表")
    assert any("应有 2 条" in p for p in CQ.check_questions([{"question": "a?"}], 2))
    assert CQ.check_questions([{"question": "  "}], 1) and CQ.check_questions([{"question": "a", "x": 1}], 1)


def test_spec_is_pinned():
    """开发机 2026-10-01 下载核对过的值（commit 是该文件在 GitHub 上唯一的一次提交）。"""
    assert CQ.SPEC["commit"] == "ec104fd760a1272796e12e5fcde42f6bc3f78f9b"
    assert CQ.SPEC["sha256"] == "d3d94521899ff05bcd3facc8712e3cb4c1f9fb5cfa17ce35b14310a882090046"
    assert CQ.SPEC["bytes"] == 31_595 and CQ.SPEC["n_questions"] == 540
    assert CQ.SPEC["url"] == ("https://raw.githubusercontent.com/google-research/big_vision/" + CQ.SPEC["commit"] + "/"
                              + CQ.SPEC["path"])


def test_land_cli_lists_optional_source_outside_all():
    spec = importlib.util.spec_from_file_location("_land_e1_cq", REPO / "scripts/data/land_e1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_args(["countbenchqa_questions", "--root", "/tmp/x"]).source == "countbenchqa_questions"
    assert mod.OPTIONAL_LANDERS == {"countbenchqa_questions": CQ.land} and "countbenchqa_questions" not in mod.LANDERS


def test_load_count_questions_checks_manifest_and_rows(tmp_path):
    root = tmp_path / "root"
    shutil.copytree(FIXTURE_DIR / "root" / "countbench", root / "countbench")
    shutil.copytree(FIXTURE_DIR / "root" / CQ.SOURCE, root / CQ.SOURCE)
    qs, st = load_count_questions(root)
    assert len(qs) == 4 and st["sha256"] == read_manifest(root / CQ.SOURCE)["files"][CQ.FILE_NAME]["sha256"]
    f = root / CQ.SOURCE / CQ.FILE_NAME
    f.write_bytes(f.read_bytes().replace(b"apples", b"pears", 1))  # 改了内容，清单没改
    with pytest.raises(ValueError, match="sha256"):
        load_count_questions(root)
    data = json.dumps([{"question": "How many x are there in the image?"}] * 3).encode()  # 3 条 ≠ 4 行
    f.write_bytes(data)
    man = json.loads((root / CQ.SOURCE / "MANIFEST.json").read_text())
    man["files"][CQ.FILE_NAME]["sha256"] = hashlib.sha256(data).hexdigest()
    (root / CQ.SOURCE / "MANIFEST.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="一一对应"):
        load_count_questions(root)


def test_select_without_landed_questions_fails_loudly(tmp_path):
    root = tmp_path / "root"
    shutil.copytree(FIXTURE_DIR / "root" / "countbench", root / "countbench")
    paths = DataPaths(root=root, dev_subsets=FIXTURE_DIR / "dev_subsets.json", eval_subsets=FIXTURE_DIR / "eval_subsets.json",
                      nyu_pairs=FIXTURE_DIR / "nyu_pairs.json")
    assert all(COUNTBENCHQA_FIELD not in s.fields for s in select("count", "dev", paths).samples)
    with pytest.raises(FileNotFoundError):  # 没落地就报错，不静默退回启发式
        select("count", "dev", paths, count_questions="countbenchqa")


def test_fixture_manifest_matches_fixture_file():
    src = FIXTURE_DIR / "root" / CQ.SOURCE
    man = read_manifest(src)
    data = (src / CQ.FILE_NAME).read_bytes()
    assert man["files"][CQ.FILE_NAME]["sha256"] == hashlib.sha256(data).hexdigest()
    assert man["files"][CQ.FILE_NAME]["bytes"] == len(data) and CQ.check_questions(json.loads(data), 4) == []
