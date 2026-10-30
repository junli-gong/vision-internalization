"""DEV-0075：引入的 LISA 官方代码（third_party/lisa @ 3cb2d430）、协议的出处、权重清单、单独环境的约定。

- 出处：PROVENANCE.md 的表与现算的逐文件 sha256 相同、文件集合相同（我们自己写的只有 PROVENANCE.md 与 requirements_lisa.txt）；
  LICENSE 是 Apache 2.0；commit 与权重的 revision 同 scripts/external/lisa_e5_infer.py（RUN-0015 核对过的那一份）。
- 协议：src/lisa/common.CITATIONS 的每一条在引入的文件的那一行逐字存在，且都写进了 PROVENANCE.md 的"协议"一节；PROTOCOL 的常量与
  官方代码的取值一致。
- 权重清单：configs/lisa_weights.json 的结构、总大小；verify_weights 拒绝多文件、缺文件、字节数或 sha256 不符。
- 环境：requirements_lisa.txt 全部钉版本、没有 tokenizers；src/lisa/common.py 只 import 标准库与 numpy；scripts/run_lisa.py 的顶层只
  import 标准库、numpy 与 src.lisa.common；hide_tokenizers 屏蔽后 find_spec 为 None，且必须在 import transformers 之前。
- 未压缩 RLE（LISA 环境写）与 src/executors/raster.rle_to_mask（计分时读）往返一致。
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.executors.raster import rle_to_mask
from src.lisa import common as C

REPO = Path(__file__).resolve().parents[1]
E5_SCRIPT = REPO / "scripts" / "external" / "lisa_e5_infer.py"


def test_provenance_table_matches_files():
    code = C.check_vendor()
    assert code["commit"] == "3cb2d4301f1af4691bd4f3938335ef06e76f155a" == C.UPSTREAM_COMMIT
    assert code["license"] == "Apache-2.0" and code["modified_files"] == [] and code["n_files"] == 49
    table = C.provenance_table()
    assert sorted(table) == C.files_on_disk()
    for f, h in table.items():
        assert C.sha256_file(C.VENDOR_DIR / f) == h, f
    assert (C.VENDOR_DIR / "LICENSE").read_text().lstrip().startswith("Apache License")
    assert C.UPSTREAM_COMMIT in C.PROVENANCE_PATH.read_text(encoding="utf-8")
    on_disk = sorted(p.relative_to(C.VENDOR_DIR).as_posix() for p in C.VENDOR_DIR.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    assert on_disk == sorted([*table, *C.OURS_IN_VENDOR])
    # 推理路径、官方评测的预处理与模板、协议出处都在
    for f in ("model/LISA.py", "utils/dataset.py", "utils/utils.py", "model/llava/conversation.py", "model/segment_anything/modeling/sam.py",
              "train_ds.py", "chat.py"):
        assert f in table


def test_check_vendor_detects_changes(tmp_path):
    import shutil

    root = tmp_path / "lisa"
    shutil.copytree(C.VENDOR_DIR, root, ignore=shutil.ignore_patterns("__pycache__"))
    C.check_vendor(root)
    (root / "utils" / "dataset.py").write_bytes((root / "utils" / "dataset.py").read_bytes() + b"\n")
    with pytest.raises(C.VendorError, match="sha256"):
        C.check_vendor(root)
    shutil.copyfile(C.VENDOR_DIR / "utils" / "dataset.py", root / "utils" / "dataset.py")
    (root / "extra.py").write_text("x = 1\n")
    with pytest.raises(C.VendorError, match="多"):
        C.check_vendor(root)


def test_citations_verbatim_and_in_provenance():
    prov = C.PROVENANCE_PATH.read_text(encoding="utf-8")
    assert len(C.CITATIONS) >= 40
    for f, n, snippet, _ in C.CITATIONS:
        lines = (C.VENDOR_DIR / f).read_text(encoding="utf-8").splitlines()
        assert snippet in lines[n - 1], f"{f}:{n} 不是 {snippet!r}：{lines[n - 1]!r}"
        assert f"`{f}:{n}`" in prov, f"{f}:{n} 没写进 PROVENANCE.md 的协议一节"


def test_protocol_constants_match_official_code():
    ds = (C.VENDOR_DIR / "utils" / "dataset.py").read_text(encoding="utf-8")
    train = (C.VENDOR_DIR / "train_ds.py").read_text(encoding="utf-8")
    assert f'conv.append_message(conv.roles[1], "{C.PROTOCOL["answer"]}")' in ds
    q = C.PROTOCOL["question"].replace("<image>", "").replace("\\n", "\\n").replace("{sent}", "{}")
    assert f'"{q}"' in ds
    assert re.search(r'"--conv_type",\s*default="%s"' % C.PROTOCOL["conv_type"], train)
    assert re.search(r'"--precision",\s*default="%s"' % C.PROTOCOL["precision"], train)
    assert f'"--image_size", default={C.PROTOCOL["image_size"]}' in train
    assert f'"--model_max_length", default={C.PROTOCOL["model_max_length"]}' in train
    assert '"--use_mm_start_end", action="store_true", default=True' in train and C.PROTOCOL["use_mm_start_end"] is True
    # 视觉 token：CLIP ViT-L/14 在 224 上 16 × 16 = 256；model/LISA.py 的偏移 255 = 256 − 占位的 1 个
    lisa = (C.VENDOR_DIR / "model" / "LISA.py").read_text(encoding="utf-8")
    assert "torch.zeros((seg_token_mask.shape[0], 255))" in lisa and C.CLIP_PATCH_TOKENS == (224 // 14) ** 2 == 256
    pre = json.loads(json.dumps(C.PROTOCOL))
    assert pre["name"] == C.PROTOCOL_NAME == "refer_seg_validate"


def _e5_constants() -> dict:
    tree = ast.parse(E5_SCRIPT.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else (node.targets if isinstance(node, ast.Assign) else [])
        for t in targets:
            if isinstance(t, ast.Name) and t.id in ("LISA_COMMIT", "WEIGHTS"):
                out[t.id] = ast.literal_eval(node.value)
    return out


def test_commit_and_revisions_same_as_run_0015():
    e5 = _e5_constants()
    doc = C.load_weights_manifest()
    assert e5["LISA_COMMIT"] == C.UPSTREAM_COMMIT
    assert e5["WEIGHTS"]["lisa"] == (doc["models"]["lisa"]["repo"], doc["models"]["lisa"]["revision"])
    assert e5["WEIGHTS"]["clip"] == (doc["models"]["clip"]["repo"], doc["models"]["clip"]["revision"])


def test_weights_manifest_structure_and_totals():
    doc = C.load_weights_manifest()
    lisa = {f["path"]: f for f in doc["models"]["lisa"]["files"]}
    clip = {f["path"]: f for f in doc["models"]["clip"]["files"]}
    assert set(lisa) == {"added_tokens.json", "config.json", "generation_config.json", "pytorch_model-00001-of-00002.bin",
                         "pytorch_model-00002-of-00002.bin", "pytorch_model.bin.index.json", "special_tokens_map.json",
                         "tokenizer.model", "tokenizer_config.json"}
    assert set(clip) == {"config.json", "preprocessor_config.json", "model.safetensors"}
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) for f in (*lisa.values(), *clip.values()))
    assert C.weights_total_bytes(doc) == {"lisa": 16_121_904_615, "clip": 1_710_545_415}


def test_verify_weights_rejects_mismatch(tmp_path):
    d = tmp_path / "w"
    d.mkdir()
    (d / "a.bin").write_bytes(b"abc")
    (d / "b.json").write_text("{}")
    doc = {"models": {"lisa": {"repo": "x/y", "revision": "0" * 40, "files": [
        {"path": "a.bin", "bytes": 3, "sha256": C.sha256_bytes(b"abc")}, {"path": "b.json", "bytes": 2, "sha256": C.sha256_bytes(b"{}")}]}}}
    ident = C.verify_weights(d, "lisa", doc)
    assert ident["n_files"] == 2 and ident["sha256_verified"] and ident["revision"] == "0" * 40
    (d / ".cache").mkdir()
    (d / ".cache" / "x").write_text("hidden")  # 隐藏的元数据目录不算
    C.verify_weights(d, "lisa", doc)
    (d / "c.bin").write_bytes(b"")
    with pytest.raises(C.WeightsError, match="文件"):
        C.verify_weights(d, "lisa", doc)
    (d / "c.bin").unlink()
    (d / "a.bin").write_bytes(b"abd")
    with pytest.raises(C.WeightsError, match="sha256"):
        C.verify_weights(d, "lisa", doc)
    (d / "a.bin").write_bytes(b"abcd")
    with pytest.raises(C.WeightsError, match="字节"):
        C.verify_weights(d, "lisa", doc)


def test_requirements_all_pinned_without_tokenizers():
    lines = [x.split("#")[0].strip() for x in C.REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()]
    reqs = [x for x in lines if x]
    assert reqs and all(re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9A-Za-z_.+\-]+", x) for x in reqs), reqs
    names = {x.split("==")[0].lower() for x in reqs}
    assert len(names) == len(reqs)
    assert "transformers==4.31.0" in reqs and "numpy==1.26.4" in reqs and "huggingface-hub==0.16.4" in reqs
    assert "tokenizers" not in names and "torch" not in names and "torchvision" not in names  # torch 继承 $VI/env
    assert "matplotlib" in names  # utils/refer.py 在模块顶层 import matplotlib.pyplot


_STDLIB = set(sys.stdlib_module_names)


def _top_imports(path: Path, *, top_level_only: bool) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    out = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.add(node.module or "")
            out |= {f"{node.module}.{a.name}" for a in node.names}
    return out


def test_lisa_side_imports_are_minimal():
    common = _top_imports(REPO / "src" / "lisa" / "common.py", top_level_only=False)
    assert {m.split(".")[0] for m in common} <= _STDLIB | {"__future__", "numpy"}, common
    run = _top_imports(REPO / "scripts" / "run_lisa.py", top_level_only=True)
    bad = {m for m in run if m.split(".")[0] not in _STDLIB | {"__future__", "numpy"} and not m.startswith("src.lisa")}
    assert bad <= set(), bad
    assert {m for m in run if m.startswith("src")} <= {"src.lisa", "src.lisa.common"}
    init = ast.parse((REPO / "src" / "lisa" / "__init__.py").read_text(encoding="utf-8"))
    assert all(isinstance(n, ast.Expr) for n in init.body)  # 只有说明，import 时不拉别的模块


def _py(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, timeout=300)


def test_hide_tokenizers_before_transformers():
    p = _py("import importlib.util, sys\nfrom src.lisa import common as C\ns = C.hide_tokenizers()\n"
            "assert importlib.util.find_spec('tokenizers') is None\n"
            "try:\n    import tokenizers\n    raise SystemExit('import 成功了')\nexcept ImportError:\n    pass\n"
            "assert C.hide_tokenizers() == s\nprint(s['hidden'], s['found_in_environment'])")
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.split()[0] == "True"
    p = _py("import transformers\nfrom src.lisa import common as C\ntry:\n    C.hide_tokenizers()\nexcept C.LisaError as e:\n"
            "    print('refused', e)\n")
    assert p.returncode == 0 and "refused" in p.stdout, p.stderr[-2000:]


@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (36, 48)])
@pytest.mark.parametrize("kind", ["empty", "full", "random", "first_on"])
def test_uncompressed_rle_round_trip(shape, kind):
    rng = np.random.default_rng(0)
    m = {"empty": np.zeros(shape, bool), "full": np.ones(shape, bool), "random": rng.random(shape) > 0.5,
         "first_on": np.eye(*shape, dtype=bool)}[kind]
    rle = C.mask_to_uncompressed_rle(m)
    assert rle["size"] == list(shape) and sum(rle["counts"]) == m.size
    np.testing.assert_array_equal(rle_to_mask(rle), m)


def test_weights_command_downloads_pinned_revisions_and_verifies(tmp_path, monkeypatch, capsys):
    """scripts/lisa_w2.py weights：按清单的 repo、revision 与文件列表调用 snapshot_download，再逐文件核对（这里用本地的假快照）。"""
    import importlib.util

    import huggingface_hub

    doc = {"version": 1, "models": {}}
    snaps = {}
    for key, files in (("lisa", {"config.json": b"{}", "w.bin": b"weights"}), ("clip", {"model.safetensors": b"clip"})):
        d = tmp_path / key
        d.mkdir()
        for name, data in files.items():
            (d / name).write_bytes(data)
        rev = ("a" if key == "lisa" else "b") * 40
        doc["models"][key] = {"repo": f"org/{key}", "revision": rev,
                              "files": [{"path": n, "bytes": len(b), "sha256": C.sha256_bytes(b)} for n, b in files.items()]}
        snaps[(f"org/{key}", rev)] = d
    man = tmp_path / "weights.json"
    man.write_text(json.dumps(doc), encoding="utf-8")
    calls = []

    def fake_snapshot_download(repo, revision, allow_patterns, local_files_only):
        calls.append((repo, revision, sorted(allow_patterns), local_files_only))
        return str(snaps[(repo, revision)])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    spec = importlib.util.spec_from_file_location("lisa_w2_cli", REPO / "scripts" / "lisa_w2.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["weights", "--download", "--manifest", str(man)]) == 0
    assert calls == [("org/lisa", "a" * 40, ["config.json", "w.bin"], False), ("org/clip", "b" * 40, ["model.safetensors"], False)]
    res = json.loads(capsys.readouterr().out)
    assert res["ok"] and res["total_bytes"] == {"lisa": 9, "clip": 4}
    assert res["models"]["lisa"]["dir"] == str(snaps[("org/lisa", "a" * 40)])
    (snaps[("org/clip", "b" * 40)] / "model.safetensors").write_bytes(b"CLIP")
    assert cli.main(["weights", "--verify", "--manifest", str(man)]) == 1
    assert calls[-1][3] is True  # --verify 只读本地
    assert json.loads(capsys.readouterr().out)["models"]["clip"]["ok"] is False


def test_run_dir_and_approval_rules():
    C.check_run_dir("experiments/W2/0099_lisa_hpcgpu06")
    for bad in ("experiments/W2/lisa_hpcgpu06", "experiments/W2/99_lisa", "experiments/W2/0099_llmseg_hpcgpu06", "0099lisa"):
        with pytest.raises(C.LisaError):
            C.check_run_dir(bad)
    C.check_approval("D154")
    for bad in (None, "D147", "d154", "D154 "):
        with pytest.raises(C.ApprovalError):
            C.check_approval(bad)
    assert C.TEST_APPROVAL == "D154"
