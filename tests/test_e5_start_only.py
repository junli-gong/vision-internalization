"""DEV-0040（D81-1；E5 预注册 §8 第 2 项）：E5 的 N = 0 读数——scripts/eval_e5_trained.py --start-only（CPU fp32 小模型，--tiny）。

- 关键的一条：同一个起点，"只评起点"的 predictions 与"E5 LoRA 为恒等"的训练后评测逐字节相同（三个家族）。恒等的 E5 检查点是刚初始化就保存的：
  src/e5/train.E5Trainer.setup_fresh（训练开头的同一个函数：合并起点 LoRA、恢复网格 token 行、注入新 LoRA，peft 默认 B = 0）之后立刻
  save_trained，e5_stack.json 的字段同 run_e5_training。另在权重上核对：两条路径加载出来的模型（训练后评测的在 E5 LoRA 合并之后）全部参数
  逐位相同、token 表相同。起点用合成的检查点（网格 token 行与 LoRA 都随机）：受限解码的输出长短不一、token 各异；--tiny 训几步的检查点
  一开口就是停止 token，逐字节相同说明不了什么；
- 用法：--init e3a / base 被拒；--checkpoint 与 --start-only 同时给出（或都不给）被拒；只属于一种方式的选项给错了方式被拒；
- 配种子：不配被拒（核对不了也拒）；--allow-seed-mismatch 放开后记进运行身份；
- 起点检查点的合并核对不过：拒绝评测（退出码 1，不写任务目录）；已通过的复用（与 train_e5 同一个位置）；
- 不用 --start-only 时，训练后评测的确定输出与改动前（a8b0e12 的脚本与模块，原样存在 tests/fixtures/e5/）逐字节相同；
- 训练叶目录的两种布局（默认；按叶目录结构搬到别处、旁边放 config.yaml，同 scripts/server/train_queue.sh 的 finalize）结果逐字节相同；
- 续跑逐字节相同。
DEV-0043 起 eval_e5_trained 默认用结构强制解码（configs/e5_train.json 的 decode 一节）；本文件测的是 N = 0 的入口与改动前的一致，读数一律加
--decode restricted（RESTRICTED；受限解码的输出与改动前逐字节相同）。结构强制下的同一组性质见 tests/test_e5_v3.py。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src.adapters import registry
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e3.tokens import add_vocab
from src.e5 import stack as STK
from src.e5 import trained_eval as TE
from src.e5.protocol import E5_TASKS
from src.e5.train import E5Trainer
from src.e5.train_config import STARTS, load_train_config, resolve_run
from src.e5.train_data import TrainSet

REPO = Path(__file__).resolve().parents[1]
CFG = load_train_config()
KEY = "qwen3.5-0.8b"
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
DET = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")  # 确定的输出（compute.json、meta.json、logs/ 含墙钟与主机）
SNAP = REPO / "tests" / "fixtures" / "e5"
RESTRICTED = ("--decode", "restricted")  # DEV-0043：受限解码（诊断口径；见模块说明）
SNAP_SHA256 = {  # a8b0e12（DEV-0040 之前）的文件原样
    "eval_e5_trained_a8b0e12.py": "bd4866758e30780cf7bca84c77c79499df16aad9a6dc86066793506f614a650e",
    "trained_eval_a8b0e12.py": "db56c2680c5d919a4d7dee6590397f0f98939c515b5f087da3323f45501c0c91",
    "stack_a8b0e12.py": "77b901d4ef173f95e1bfecd533df68ae837893adc7e5170f7921ed7b955c4697",
}
_SCRIPTS: dict[str, object] = {}


def _script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_e5so", REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.path[:] = saved
        _SCRIPTS[name] = mod
    return _SCRIPTS[name]


def _load_path(path: Path, name: str):
    """按文件加载模块（登记进 sys.modules：快照里有 dataclass），sys.path 原样恢复。"""
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def _tiny(key: str):
    t = CFG.e3b.doc["tiny"]
    fam = registry.get_model(key).family
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _mdir(out: Path, init: str = "e3b", key: str = KEY, tag: str = TE.START_ONLY_TAG, seed: int = 0) -> Path:
    return out / "trained" / init / key / tag / f"seed{seed}"


def _so_args(ck: Path, out: Path, *, init: str = "e3b", key: str = KEY, seed: int = 0, extra: tuple[str, ...] = ()) -> list[str]:
    return ["--tiny", "--start-only", "--init", init, "--start-checkpoint", str(ck), "--model", key, "--seed", str(seed),
            "--out", str(out), *RESTRICTED, *extra]


def _sharp_start(root: Path, key: str) -> Path:
    """合成的起点检查点（e3b 的结构：LoRA + E3b 词表的 67 个网格 token 行）：网格 token 行随机（标准差同小模型的 init_std，不是均值初始化）、
    LoRA 随机（B 不为 0，合并确实改权重）。按训练叶目录的布局存，旁边放 config.yaml（variant e3b、seed 0），变体与配种子的核对照常。"""
    ad = _tiny(key)
    tokens = add_vocab(ad, CFG.e3b.vocab)
    torch.manual_seed(0)
    ad.inject_lora(r=8, lora_alpha=16, init_lora_weights=False)
    g = torch.Generator().manual_seed(1)
    idx = torch.tensor(sorted(tokens.grid_ids))
    std = float(CFG.e3b.doc["tiny"]["init_std"])
    with torch.no_grad():
        for p in ad.trainable_token_rows(list(CFG.e3b.vocab.names)):
            p[idx] = torch.randn(len(idx), p.shape[1], generator=g) * std
    leaf = root / "e3b" / key / "seed0"
    ck = ad.save_trained(leaf / "ckpt" / "final" / "model")
    (leaf / "config.yaml").write_text(yaml.safe_dump({"variant": "e3b", "seed": 0}), encoding="utf-8")
    return ck


def _identity_e5_checkpoint(start_ck: Path, key: str, final: Path, kind: str = "e3b", perturb: float = 0.0) -> None:
    """刚初始化就保存的 E5 检查点（E5 LoRA 为恒等）：E5Trainer.setup_fresh 之后立刻 save_trained；e5_stack.json 的字段同 run_e5_training。
    perturb > 0（只用于核对区分力）：B 换成标准差 perturb 的随机数，E5 LoRA 不再是恒等。"""
    start = STK.resolve_start(CFG, kind, start_ck, model_key=key, tiny=True)
    ad = STK.make_base(start.manifest["build_spec"], device="cpu", dtype=torch.float32)
    ts = TrainSet.build(CFG, 20, None, 0)
    tr = E5Trainer(ad, CFG, resolve_run(CFG, n_train=20, n_examples=len(ts), tiny=True), seed=0, start=start)
    tr.setup_fresh()
    bs = [p for n, p in ad.model.named_parameters() if "lora_B" in n]
    assert bs and all(torch.count_nonzero(p) == 0 for p in bs)  # peft 默认：B = 0，E5 LoRA 是恒等
    assert any(torch.count_nonzero(p) > 0 for n, p in ad.model.named_parameters() if "lora_A" in n)
    if perturb:
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            for p in bs:
                p.copy_(torch.randn(p.shape, generator=g) * perturb)
    ad.save_trained(final / "model", extra=STK.marker_extra(start))
    pairing = STK.check_seed_pairing(start, 0, allow_mismatch=False)
    STK.write_stack_file(final, {"model_dir": "model", "start": start.record(),
                                 "fingerprint": {"value": tr.stack_info["fingerprint"], "device_key": STK.device_key(ad)},
                                 "model": {"key": key, "tiny": True}, "seed": 0, "tag": ts.tag, "data": ts.identity(),
                                 "identity_sha256": "identity-e5-lora（DEV-0040 测试：训练 0 步）", "config": CFG.identity(),
                                 "train_leaf": str(final.parent), "tokens_e5": tr.e5.layout(), "steps": 0, "selection": None,
                                 "seed_pairing": pairing})


@pytest.fixture(scope="module")
def real_starts(tmp_path_factory):
    """scripts/train_e3b.py --tiny 的网格监督与网格格式预热（种子 0），默认布局；另按 train_queue.sh 的 finalize 搬一份：
    <搬到的根>/<叶目录相对 out 的路径>/ckpt/final + config.yaml + train_summary.json。"""
    root = tmp_path_factory.mktemp("e5so_real")
    e3 = _script("train_e3b")
    for sup in ("grid", "warmup"):
        assert e3.main(["--tiny", "--model", KEY, "--supervision", sup, "--seed", "0", "--out", str(root / "e3")]) == 0
    res = {}
    for init, var in (("e3b", "e3b"), ("base_warmup", "warmup")):
        rel = Path(var) / KEY / "seed0"
        leaf = root / "e3" / rel
        dst = root / "moved" / rel
        shutil.copytree(leaf / "ckpt" / "final", dst / "ckpt" / "final")
        shutil.copy(leaf / "config.yaml", dst / "config.yaml")
        shutil.copy(leaf / "train_summary.json", dst / "train_summary.json")
        res[init] = leaf / "ckpt" / "final" / "model"
        res[f"{init}_moved"] = dst / "ckpt" / "final" / "model"
    return res


@pytest.fixture(scope="module")
def sharp(tmp_path_factory):
    root = tmp_path_factory.mktemp("e5so_sharp")
    return {key: _sharp_start(root / key, key) for key in FAMILIES}


# ---------------------------------------------------------------------------
# 规则与用法
# ---------------------------------------------------------------------------


def test_rules_and_usage(tmp_path, monkeypatch):
    # 只评起点的起点 = 配置里带检查点、网格 token 行来自检查点（训练过）的起点
    assert TE.START_ONLY_KINDS == ("base_warmup", "e3b_c3", "c3_warmup", "e3b") == tuple(  # DEV-0043：v3 的两个新起点
        k for k in STARTS if CFG.start(k)["checkpoint"] and CFG.start(k)["grid_tokens"] == "from_checkpoint")
    assert TE.START_ONLY_TAG == "n0" and TrainSet.build(CFG, 10, None, 0).tag == f"n{10}"  # 与数据量档同一种标记、同一层
    assert TE.model_dir(tmp_path, "e3b", KEY, TE.START_ONLY_TAG, 2) == tmp_path / "trained" / "e3b" / KEY / "n0" / "seed2"
    kw = dict(model_key=KEY, seed=0, task="occlusion", subset="eval", n_per_bin=1, n_resamples=0, batch_size=1, tiny=True,
              dtype="fp32")
    TE.EvalSpec(start="e3b", tag="n0", start_only=True, **kw)
    for bad in (dict(start="e3a", tag="n0", start_only=True), dict(start="e3b", tag="n10", start_only=True),
                dict(start="e3b", tag="n0", start_only=True, data_seed=0), dict(start="e3b", tag="n0")):
        with pytest.raises(ValueError):
            TE.EvalSpec(**bad, **kw)
    for kind in ("e3a", "base"):
        st = STK.StartRef(kind=kind, checkpoint=None if kind == "base" else tmp_path, grid_tokens="add")
        with pytest.raises(STK.StartError, match="只评起点"):
            STK.load_start_only(st, CFG, device="cpu", dtype=torch.float32)
    ev = _script("eval_e5_trained")
    monkeypatch.delenv("VISINT_DATA_ROOT", raising=False)
    ck = tmp_path  # parse_args 只核对是目录
    out = str(tmp_path / "o")
    good = ["--tiny", "--start-only", "--init", "e3b", "--start-checkpoint", str(ck), "--model", KEY, "--seed", "0", "--out", out]
    a = ev.parse_args(good)
    assert a.start_only and a.init == "e3b" and a.seed == 0 and a.checkpoint is None and a.data_root is None
    assert ev.parse_args(good[:3] + ["base_warmup"] + good[4:]).init == "base_warmup"
    assert ev.parse_args(["--tiny", "--checkpoint", str(ck), "--model", KEY, "--out", out]).start_only is False
    bad = [good[:3] + ["e3a"] + good[4:],  # e3a 的网格 token 行没训练过
           good[:3] + ["base"] + good[4:],  # base 没有起点检查点
           good[:3] + ["nope"] + good[4:],
           good + ["--checkpoint", str(ck)],  # 二选一
           ["--tiny", "--model", KEY, "--out", out],  # 都不给
           [x for x in good if x not in ("--init", "e3b")],
           [x for x in good if x not in ("--seed", "0")],
           good[:4] + good[6:],  # 没有 --start-checkpoint
           good[:5] + [str(tmp_path / "missing")] + good[6:],
           good + ["--merge-check-n-per-task", "2"],  # 只评起点没有 E5 LoRA 合并前后的核对
           [x for x in good if x != "--tiny"] + ["--out", str(tmp_path / "0099_x")],  # 真模型要 --data-root
           ["--tiny", "--checkpoint", str(ck), "--model", KEY, "--out", out, "--seed", "0"],  # 只属于 --start-only 的选项
           ["--tiny", "--checkpoint", str(ck), "--model", KEY, "--out", out, "--init", "e3b"],
           ["--tiny", "--checkpoint", str(ck), "--model", KEY, "--out", out, "--allow-seed-mismatch"],
           ["--tiny", "--checkpoint", str(ck), "--model", KEY, "--out", out, "--data-root", str(tmp_path)]]
    for argv in bad:
        with pytest.raises(SystemExit) as e:
            ev.parse_args(argv)
        assert e.value.code == 2, argv
    for init in ("e3a", "base"):  # main 也是用法错误（退出码 2），不建任何目录
        with pytest.raises(SystemExit) as e:
            ev.main(good[:3] + [init] + good[4:])
        assert e.value.code == 2
    assert not (tmp_path / "o").exists()
    monkeypatch.setenv("VISINT_DATA_ROOT", str(tmp_path))
    assert ev.parse_args([x for x in good if x != "--tiny"][:-1] + [str(tmp_path / "0099_x")]).data_root == str(tmp_path)


# ---------------------------------------------------------------------------
# 关键的一条：只评起点 = E5 LoRA 为恒等的训练后评测
# ---------------------------------------------------------------------------


def _params(ad) -> dict[str, torch.Tensor]:
    return {n: p.detach() for n, p in ad.model.named_parameters()}


@pytest.mark.parametrize("key", FAMILIES)
def test_start_only_equals_identity_e5_lora(sharp, key, tmp_path):
    ck = sharp[key]
    final = tmp_path / "e5" / "ckpt" / "final"
    _identity_e5_checkpoint(ck, key, final)
    ev = _script("eval_e5_trained")
    a, b = tmp_path / "a", tmp_path / "b"
    common = ["--task", "a", "b", "c", "e", "--limit", "2"]
    assert ev.main(["--tiny", "--checkpoint", str(final), "--model", key, "--out", str(a), *RESTRICTED, *common]) == 0
    assert ev.main(_so_args(ck, b, key=key, extra=tuple(common))) == 0
    ma, mb = _mdir(a, key=key, tag="n20"), _mdir(b, key=key)
    assert json.loads((ma / "merge_check.json").read_text())["pass"] is True  # 训练后评测：E5 LoRA 合并前后的核对照做
    assert not (mb / "merge_check.json").exists()  # 只评起点：没有 E5 LoRA，这一项跳过
    lens, uniq = [], []
    for task in ("rotation3d", "viewpoint", "occlusion", "imagery"):
        pa, pb = (ma / task / "predictions.jsonl").read_bytes(), (mb / task / "predictions.jsonl").read_bytes()
        assert pa == pb, (key, task)
        recs = [json.loads(x) for x in pb.splitlines()]
        lens += [r["n_generated_tokens"] for r in recs]
        uniq += [len(set(r.get("token_ids", []))) for r in recs]
        xa, xb = (json.loads((d / task / "metrics.json").read_text()) for d in (ma, mb))
        assert (xb["tag"], xb["start_only"], xb["n_train"], xa["tag"]) == ("n0", True, 0, "n20") and "start_only" not in xa
        assert {k: v for k, v in xa.items() if k != "tag"} == {k: v for k, v in xb.items() if k not in ("tag", "start_only", "n_train")}
    assert max(lens) > 1 and len(set(lens)) > 1 and max(uniq) > 1  # 输出不是一开口就停：逐字节相同有区分力
    # 权重：两条路径加载出来的模型（训练后评测的在 E5 LoRA 合并之后）全部参数逐位相同、token 表相同
    start = STK.resolve_start(CFG, "e3b", ck, model_key=key, tiny=True)
    so, _, info = STK.load_start_only(start, CFG, device="cpu", dtype=torch.float32)
    st = STK.load_stacked(final, device="cpu", dtype=torch.float32)
    st.adapter.merge_lora()
    p1, p2 = _params(so), _params(st.adapter)
    assert p1.keys() == p2.keys() and all(torch.equal(p1[k], p2[k]) for k in p1)
    assert so.token_table() == st.adapter.token_table() and so.lora_state == "none"
    assert info["fingerprint"] == st.fingerprint["value"]
    base = _params(STK.make_base(start.manifest["build_spec"], device="cpu", dtype=torch.float32))
    assert any(p1[k].shape == base[k].shape and not torch.equal(p1[k], base[k]) for k in base)  # 起点 LoRA 确实合并了
    if key == KEY:  # 区分力：E5 LoRA 不是恒等（B 很小的随机数）时 predictions 就不同
        pert = tmp_path / "e5p" / "ckpt" / "final"
        _identity_e5_checkpoint(ck, key, pert, perturb=0.02)
        c = tmp_path / "c"
        assert ev.main(["--tiny", "--checkpoint", str(pert), "--model", key, "--out", str(c), *RESTRICTED, "--task", "a", "c",
                        "--limit", "2"]) == 0
        assert any((_mdir(c, key=key, tag="n20") / task / "predictions.jsonl").read_bytes() !=
                   (mb / task / "predictions.jsonl").read_bytes() for task in ("rotation3d", "occlusion"))


# ---------------------------------------------------------------------------
# 真起点（train_e3b --tiny）：输出、运行身份、核对、两种布局
# ---------------------------------------------------------------------------


def test_real_starts_outputs_and_layouts(real_starts, tmp_path):
    ev = _script("eval_e5_trained")
    for init in ("base_warmup", "e3b"):  # v3 的 e3b_c3、c3_warmup 见 tests/test_e5_v3.py
        ck = real_starts[init]
        out, moved = tmp_path / f"o_{init}", tmp_path / f"m_{init}"
        assert ev.main(_so_args(ck, out, init=init, extra=("--limit", "2"))) == 0
        assert ev.main(_so_args(real_starts[f"{init}_moved"], moved, init=init, extra=("--task", "a", "c", "--limit", "2"))) == 0
        mdir = _mdir(out, init)
        assert sorted(p.name for p in mdir.iterdir()) == sorted(E5_TASKS)  # 没有 merge_check.json
        sha = _sha(ck / "manifest.json")
        checks = STK.start_checks_dir(out, init, KEY)  # 与 train_e5 开训前的核对同一个位置
        mc = json.loads((checks / f"merge_check_{sha[:16]}.json").read_text())
        ref = json.loads((checks / f"reference_{sha[:16]}.json").read_text())
        assert mc["pass"] is True and sorted(mc["checks"]) == ["merge", "restricted"]
        for task in E5_TASKS:
            leaf = mdir / task
            assert sorted(p.name for p in leaf.iterdir()) == ["README.md", "compute.json", "config.yaml", "logs", "meta.json",
                                                              "metrics.json", "predictions.jsonl"]
            for f in DET if task in ("rotation3d", "occlusion") else ():  # 搬到别处（旁边放 config.yaml）：逐字节相同（运行身份里没有路径）
                assert (leaf / f).read_bytes() == (_mdir(moved, init) / task / f).read_bytes(), (init, task, f)
            ident = yaml.safe_load((leaf / "config.yaml").read_text())
            assert ident["start_only"] is True and ident["n_train"] == 0 and ident["system"] == TE.SYSTEM
            c = ident["checkpoint"]
            assert c["start"]["kind"] == init and c["start"]["manifest_sha256"] == sha and c["n_train"] == 0
            assert c["start"]["leaf_variant"] == CFG.start(init)["variant"] and c["start"]["leaf_seed"] == 0
            assert c["fingerprint"]["value"] == ref["fingerprint"] and c["fingerprint"]["device_key"]["device_type"] == "cpu"
            assert c["seed_pairing"]["status"] == "match" and c["seed"] == 0
            assert c["checks"]["skipped"] == TE.START_ONLY_SKIPPED and c["checks"]["done"]["start_merge_check"]["checks"] == \
                ["merge", "restricted"]
            assert ident["decoding"]["max_new_tokens"] == 1057 and ident["decoding"]["path"] == "generate_restricted"
            m = json.loads((leaf / "metrics.json").read_text())
            assert m["complete"] and m["start_only"] is True and m["n_train"] == 0 and m["tag"] == "n0" and "gate" in m
            readme = (leaf / "README.md").read_text()
            assert "只评起点" in readme and "跳过：e5_lora_merge" in readme
            run = json.loads((leaf / "logs" / "run_001.json").read_text())
            assert run["mode"] == "start_only" and run["start_merge_check"]["pass"] is True
            assert run["reference_fingerprint"]["fingerprint"] == ref["fingerprint"] == run["setup"]["fingerprint"]
        # 已通过的核对复用（同一个 --out 再评一个子集）
        assert ev.main(_so_args(ck, out, init=init, extra=("--task", "c", "--subset", "dev", "--limit", "1"))) == 0
        run = json.loads((mdir / "occlusion__dev" / "logs" / "run_001.json").read_text())
        assert run["start_merge_check"]["reused"] is True
        # 两个真起点也与 E5 LoRA 为恒等的训练后评测一致：权重逐位相同，predictions 逐字节相同
        final = tmp_path / f"e5_{init}" / "ckpt" / "final"
        _identity_e5_checkpoint(ck, KEY, final, kind=init)
        assert ev.main(["--tiny", "--checkpoint", str(final), "--model", KEY, "--out", str(out), *RESTRICTED, "--task", "a", "c",
                        "--limit", "2"]) == 0
        for task in ("rotation3d", "occlusion"):
            assert (_mdir(out, init, tag="n20") / task / "predictions.jsonl").read_bytes() == \
                (mdir / task / "predictions.jsonl").read_bytes(), (init, task)
        so, _, _ = STK.load_start_only(STK.resolve_start(CFG, init, ck, model_key=KEY, tiny=True), CFG, device="cpu",
                                       dtype=torch.float32)
        st = STK.load_stacked(final, device="cpu", dtype=torch.float32)
        st.adapter.merge_lora()
        p1, p2 = _params(so), _params(st.adapter)
        assert p1.keys() == p2.keys() and all(torch.equal(p1[k], p2[k]) for k in p1)
        assert so.token_table() == st.adapter.token_table()


def test_seed_pairing(real_starts, tmp_path):
    ev = _script("eval_e5_trained")
    ck = real_starts["e3b"]
    out = tmp_path / "o"
    args = ("--task", "c", "--limit", "1")
    assert ev.main(_so_args(ck, out, seed=1, extra=args)) == 2  # 起点来自种子 0，--seed 1：拒绝
    (err,) = (out / "trained" / "errors").glob("error_*.json")
    assert "配种子" in json.loads(err.read_text())["error"]
    assert not (out / "trained" / "e3b").exists()
    assert ev.main(_so_args(ck, out, seed=1, extra=(*args, "--allow-seed-mismatch"))) == 0  # 放开，记进运行身份
    ident = yaml.safe_load((_mdir(out, seed=1) / "occlusion" / "config.yaml").read_text())
    p = ident["checkpoint"]["seed_pairing"]
    assert (p["status"], p["start_seed"], p["seed"], p["allow_mismatch"]) == ("mismatch", 0, 1, True)
    assert "配种子 mismatch" in (_mdir(out, seed=1) / "occlusion" / "README.md").read_text()
    assert ev.main(_so_args(ck, out, seed=7, extra=(*args, "--allow-seed-mismatch"))) == 2  # 不在配置的 training.seeds 里
    # 核对不了（检查点不在训练叶目录下）：也拒绝；放开后记 unverifiable
    loose = tmp_path / "loose" / "model"
    shutil.copytree(ck, loose)
    assert ev.main(_so_args(loose, tmp_path / "p", extra=args)) == 2
    assert ev.main(_so_args(loose, tmp_path / "p", extra=(*args, "--allow-seed-mismatch"))) == 0
    p = yaml.safe_load((_mdir(tmp_path / "p") / "occlusion" / "config.yaml").read_text())["checkpoint"]["seed_pairing"]
    assert (p["status"], p["start_seed"], p["allow_mismatch"]) == ("unverifiable", None, True)
    # 预热的检查点冒充 e3b：变体核对拒绝
    assert ev.main(_so_args(real_starts["base_warmup"], tmp_path / "q", extra=args)) == 2


def test_resume_bytewise(real_starts, tmp_path):
    ev = _script("eval_e5_trained")
    out = tmp_path / "o"
    args = _so_args(real_starts["e3b"], out, extra=("--task", "c", "--limit", "3", "--batch-size", "2"))
    assert ev.main(args) == 0
    leaf = _mdir(out) / "occlusion"
    before = {f: (leaf / f).read_bytes() for f in DET}
    assert ev.main(args) == 2  # 非空目录不加 --resume：拒绝
    lines = (leaf / "predictions.jsonl").read_bytes().splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_bytes(lines[0] + lines[1][:7])  # 中断在一批中间、末行不完整
    (leaf / "metrics.json").unlink()
    assert ev.main(args + ["--resume"]) == 0
    assert {f: (leaf / f).read_bytes() for f in DET} == before
    run = json.loads((leaf / "logs" / "run_002.json").read_text())
    assert run["n_done_before"] == 0 and run["start_merge_check"]["reused"] is True
    assert ev.main(args[:-1] + ["3", "--resume"]) == 2  # 换批大小：运行身份不同，拒绝混写


def _pathological_lora(ad) -> None:
    """同 tests/test_e5_train._pathological_lora：合并与不合并两条路径在 fp32 上也分家。"""
    mods = dict(ad.model.named_modules())
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name in ad.lora_target_modules():
            m = mods[name]
            a_w, b_w = m.lora_A["default"].weight, m.lora_B["default"].weight
            u = torch.randn(a_w.shape[1], generator=g)
            a_w.zero_()
            b_w.zero_()
            a_w[0] = u
            a_w[1] = u + 1e-6 * torch.randn(a_w.shape[1], generator=g)
            v = torch.randn(b_w.shape[0], generator=g)
            b_w[:, 0] = v * 1e7
            b_w[:, 1] = -v * 1e7


def test_bad_start_refused(tmp_path):
    """起点检查点的合并核对不过：拒绝评测（退出码 1），不写任何任务目录。"""
    ad = _tiny(KEY)
    add_vocab(ad, CFG.e3b.vocab)
    ad.inject_lora(r=2, lora_alpha=2, init_lora_weights=False)
    _pathological_lora(ad)
    leaf = tmp_path / "e3" / "e3b" / KEY / "seed0"
    bad = ad.save_trained(leaf / "ckpt" / "final" / "model")
    (leaf / "config.yaml").write_text(yaml.safe_dump({"variant": "e3b", "seed": 0}), encoding="utf-8")
    out = tmp_path / "o"
    assert _script("eval_e5_trained").main(_so_args(bad, out, extra=("--task", "c", "--limit", "1"))) == 1
    assert not (out / "trained" / "e3b").exists()
    (mc,) = STK.start_checks_dir(out, "e3b", KEY).glob("merge_check_*.json")
    doc = json.loads(mc.read_text())
    assert doc["pass"] is False and (doc["failures"]["merge"] or doc["failures"]["restricted"])


# ---------------------------------------------------------------------------
# 不用 --start-only：与改动前（a8b0e12）逐字节相同
# ---------------------------------------------------------------------------


def test_checkpoint_mode_identical_to_a8b0e12(sharp, tmp_path, capsys):
    """E5 检查点的评测：新脚本加模块与改动前（a8b0e12）的脚本加模块（trained_eval、stack），在同一进程、同一组参数下，确定的输出
    （README、config.yaml、metrics.json、predictions.jsonl）、评测前的合并核对（去掉墙钟）、运行记录的字段与打印的结果行逐字节相同。
    E5 检查点从合成的起点训练（10 条的档：带 val 选择，运行身份里有 selection 一节）。"""
    for name, sha in SNAP_SHA256.items():
        assert _sha(SNAP / name) == sha, name
    old_stack = _load_path(SNAP / "stack_a8b0e12.py", "_e5_stack_a8b0e12")
    old_te = _load_path(SNAP / "trained_eval_a8b0e12.py", "_e5_trained_eval_a8b0e12")
    old_te.REPO_ROOT = TE.REPO_ROOT  # 快照按自己的位置推出的仓库根不对（只影响 meta.json 里配置文件的路径）
    old_script = _load_path(SNAP / "eval_e5_trained_a8b0e12.py", "_eval_e5_trained_a8b0e12")
    old_script.STK, old_script.TE = old_stack, old_te
    out = tmp_path / "t"
    assert _script("train_e5").main(["--tiny", "--init", "e3b", "--start-checkpoint", str(sharp[KEY]), "--model", KEY,
                                     "--n-train", "10", "--seed", "0", "--out", str(out), *RESTRICTED]) == 0
    final = out / "train" / "e3b" / KEY / "n10" / "seed0" / "ckpt" / "final"
    assert json.loads((final / "e5_stack.json").read_text())["selection"]["selected_step"] in (2, 4, 6)
    capsys.readouterr()
    outs, printed = [], []
    for k, (mod, extra) in enumerate(((old_script, ()), (_script("eval_e5_trained"), RESTRICTED))):
        o = tmp_path / f"o{k}"
        assert mod.main(["--tiny", "--checkpoint", str(final), "--model", KEY, "--out", str(o), "--limit", "2", *extra]) == 0
        printed.append(capsys.readouterr().out)
        outs.append(_mdir(o, tag="n10"))
    assert printed[0] == printed[1] and printed[0].count("\n") == len(E5_TASKS)
    lens = []
    for task in E5_TASKS:
        for f in DET:
            assert (outs[0] / task / f).read_bytes() == (outs[1] / task / f).read_bytes(), (task, f)
        r0, r1 = (json.loads((d / task / "logs" / "run_001.json").read_text()) for d in outs)
        assert list(r0) == list(r1) and r0["merge_check"]["file_sha256"] != "" and r0["fingerprint"] == r1["fingerprint"]
        lens += [json.loads(x)["n_generated_tokens"] for x in (outs[1] / task / "predictions.jsonl").read_text().splitlines()]
    assert max(lens) > 1  # 输出不是一开口就停
    m0, m1 = (json.loads((d / "merge_check.json").read_text()) for d in outs)
    for m in (m0, m1):
        m.pop("timing_s")
        m.pop("written_utc")
    assert m0 == m1 and m1["pass"] is True
