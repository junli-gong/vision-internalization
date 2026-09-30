import hashlib
import json
import subprocess
import sys
import types

import pytest
import yaml

from src import runlog
from src.runlog import create_run_dir, git_info, write_run_metadata


def test_create_run_dir_layout(tmp_path):
    d = create_run_dir("phase0", "0007", "smoke_test", root=tmp_path)
    assert d == tmp_path / "phase0" / "0007_smoke_test"
    assert (d / "logs").is_dir() and (d / "figures").is_dir()


@pytest.mark.parametrize("bad", ["7", "00007", "abcd", "12a4", 7, "０００１"])
def test_create_run_dir_rejects_bad_run_id(tmp_path, bad):
    with pytest.raises(ValueError):
        create_run_dir("phase0", bad, "x", root=tmp_path)


def test_create_run_dir_rejects_bad_names(tmp_path):
    for phase, slug in (("../up", "x"), ("phase0", "a/b"), ("", "x"), ("phase0", "..")):
        with pytest.raises(ValueError):
            create_run_dir(phase, "0001", slug, root=tmp_path)


def test_create_run_dir_no_overwrite(tmp_path):
    empty = tmp_path / "e1" / "0002_empty"
    empty.mkdir(parents=True)
    assert create_run_dir("e1", "0002", "empty", root=tmp_path) == empty  # 空目录可沿用
    with pytest.raises(FileExistsError):
        create_run_dir("e1", "0002", "empty", root=tmp_path)  # 现在非空


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args], cwd=cwd, check=True, capture_output=True)


def test_git_info_clean_and_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    assert git_info(repo)["commit"] is None  # 不是 git 仓库
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("one\n")
    _git(repo, "add", "a.txt")
    _git(repo, "commit", "-q", "-m", "init")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    info = git_info(repo)
    assert info == {"commit": head, "dirty": False, "diff_shortstat": "", "n_untracked": 0}
    (repo / "a.txt").write_text("one\ntwo\n")
    (repo / "new.txt").write_text("u\n")
    info = git_info(repo)
    assert info["dirty"] is True and info["n_untracked"] == 1
    assert "1 file changed" in info["diff_shortstat"] and "1 insertion" in info["diff_shortstat"]


def test_write_run_metadata(tmp_path):
    d = create_run_dir("e1", "0003", "meta", root=tmp_path)
    split = tmp_path / "split.txt"
    split.write_bytes(b"img_001\nimg_017\n")
    cfg = {"model": "tiny", "grid": 16, "seed": 11}
    meta = write_run_metadata(d, config=cfg, seed=11, data_files=[split], repo_root=tmp_path)
    assert yaml.safe_load((d / "config.yaml").read_text()) == cfg
    on_disk = json.loads((d / "meta.json").read_text())
    assert on_disk == meta
    assert meta["seed"] == 11
    assert meta["data_sha256"] == {str(split): hashlib.sha256(b"img_001\nimg_017\n").hexdigest()}
    assert meta["git"]["commit"] is None
    assert meta["packages"]["numpy"] is not None
    assert "transformers" in meta["packages"]  # 没装时记 null，键仍在
    assert meta["timestamp_utc"].endswith("+00:00")
    for k in ("hostname", "gpu", "python"):
        assert k in meta
    with pytest.raises(FileExistsError):
        write_run_metadata(d, config=cfg, seed=11, data_files=[split], repo_root=tmp_path)


def test_write_run_metadata_checks(tmp_path):
    d = create_run_dir("e1", "0004", "meta", root=tmp_path)
    with pytest.raises(ValueError):
        write_run_metadata(d, config={"seed": 1}, seed=2, data_files=[], repo_root=tmp_path)
    with pytest.raises(TypeError):
        write_run_metadata(d, config={}, seed="1", data_files=[], repo_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        write_run_metadata(d, config={}, seed=1, data_files=[tmp_path / "missing"], repo_root=tmp_path)


def test_gpu_name_both_branches(monkeypatch):
    # 开发机没有 CUDA，两条分支都用假 torch 模块走一遍（真卡上的名字由服务器运行核对）
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, get_device_name=lambda i: "X"))
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert runlog.gpu_name() is None
    fake.cuda.is_available = lambda: True
    fake.cuda.get_device_name = lambda i: f"FakeGPU-{i}"
    assert runlog.gpu_name() == "FakeGPU-0"
