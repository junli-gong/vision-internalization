"""DEV-0042 §2：C3+预热（scripts/train_e3b.py --supervision warmup --start-checkpoint <C3 的 ckpt/final/model>，变体 c3_warmup）与
按秩拼接两份 LoRA（src/adapters/checkpoint.write_rank_concat / check_rank_concat）。

- resolve_run：与基座上的预热只多 parent 一节；c3_warmup 只配预热监督；
- 按秩拼接的代数：数值可精确表示（位数很少的二进制小数）时，拼接后一次合并与"先合并父、再合并子"逐位相同（peft 的同一串合并调用）；
  换成一般的浮点值只差舍入（实测的最大差记在测试里），所以不主张一般情况下逐位相同；拼错块的顺序就不同（测试有判别力）；
- 拼接的检查：前 32 个秩与父的 LoRA 逐位相同，后 32 个与子的逐位相同；父检查点的 manifest sha256 不符时拒绝；父带新增 token 行、缩放
  不同、base 不同时拒绝；
- 端到端（tiny）：C3 → C3+预热：检查点是秩 64、alpha 128 的一个 LoRA 加预热之后的网格 token 行，factory.load_trained(merge=True) 能加载、
  受限解码可用，合并后的权重与依次合并只差舍入；预热的数据与基座上的预热逐步相同，预热 LoRA 的初始值也逐位相同（合并在 manual_seed
  之前）；运行身份记下父检查点的 sha256；E5 的 merge_start_lora 能把它当普通起点合并，与 load_trained(merge=True) 的权重指纹相同；
  续跑与一次跑完逐字节相同，续跑时起点换了（sha256 不同）就拒绝；起点不是 C3、种子不符、构造方式不同、核对不了叶目录时拒绝；
- 命令行：三家族 --tiny 跑通（C3 → C3+预热），检查点能被 eval_e3b 与 run_e1 --checkpoint 加载；用法错误退出码 2。
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src.adapters import factory, registry
from src.adapters.checkpoint import (
    CheckpointError,
    check_rank_concat,
    file_sha256,
    policy_spec,
    rank_concat_lora_state,
    read_manifest,
    write_rank_concat,
)
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e3 import data as D
from src.e3 import warmup as W
from src.e3.config import C3_WARMUP, ConfigError, load_config, resolve_run, resolve_variant
from src.e3.targets import parse_target, seq_len
from src.e3.tokens import add_vocab, attach_vocab
from src.e3.train import (
    PARENT_MARKER,
    ParentError,
    RunSpec,
    Trainer,
    leaf_dir,
    read_parent_marker,
    resolve_parent,
    run_training,
)

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()
N = CFG.grid_n
KEY = "qwen3.5-0.8b"


def _tiny(key=KEY):
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _spec(key, ad, supervision, parent=None, seed=0):
    e = registry.get_model(key)
    return RunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                   image_policy=policy_spec(ad.image_policy), model_dtype="fp32", supervision=supervision, seed=seed,
                   parent_checkpoint=None if parent is None else str(parent))


def _ckpt(leaf: Path) -> Path:
    return leaf / "ckpt" / "final" / "model"


@pytest.fixture(scope="module")
def c3_leaf(tmp_path_factory) -> Path:
    """tiny 的 C3（Qwen，种子 0）：C3+预热的起点。"""
    out = tmp_path_factory.mktemp("c3")
    rc = resolve_run(CFG, supervision="text", tiny=True)
    ad = _tiny()
    run_training(ad, CFG, _spec(KEY, ad, "text"), rc, D.E3Paths.fixtures(), out, log=lambda m: None)
    return leaf_dir(out, "c3", KEY, 0)


def _cw_rc():
    rc = resolve_run(CFG, supervision="warmup", tiny=True, variant=C3_WARMUP)
    rc["data_workers"] = 0
    return rc


@pytest.fixture(scope="module")
def cw_leaf(tmp_path_factory, c3_leaf) -> Path:
    """tiny 的 C3+预热（6 步，留出集的真读数）。"""
    out = tmp_path_factory.mktemp("cw")
    ad = _tiny()
    s = run_training(ad, CFG, _spec(KEY, ad, "warmup", parent=_ckpt(c3_leaf)), _cw_rc(), D.E3Paths.fixtures(), out,
                     log=lambda m: None)
    assert s["complete"] and s["variant"] == C3_WARMUP
    return leaf_dir(out, C3_WARMUP, KEY, 0)


# ---------------------------------------------------------------------------
# 配置与变体
# ---------------------------------------------------------------------------


def test_resolve_run_c3_warmup():
    w = resolve_run(CFG, supervision="warmup", tiny=False)
    rc = resolve_run(CFG, supervision="warmup", tiny=False, variant=C3_WARMUP)
    assert rc["variant"] == C3_WARMUP and "parent" not in w
    assert set(rc) - set(w) == {"parent"} and {k: rc[k] for k in w if k != "variant"} == {k: w[k] for k in w if k != "variant"}
    assert rc["parent"]["variant"] == "c3"
    assert resolve_variant("warmup", C3_WARMUP) == C3_WARMUP
    for sup in ("grid", "text", "grid+text"):
        with pytest.raises(ConfigError, match=C3_WARMUP):
            resolve_variant(sup, C3_WARMUP)
    ad = _tiny()
    with pytest.raises(ValueError, match="预热"):
        _ = _spec(KEY, ad, "grid", parent="x").variant
    assert _spec(KEY, ad, "warmup", parent="x").variant == C3_WARMUP and _spec(KEY, ad, "warmup").variant == "warmup"


# ---------------------------------------------------------------------------
# 按秩拼接：代数与核对
# ---------------------------------------------------------------------------


def _set_lora(ad, gen, *, dyadic: bool) -> None:
    with torch.no_grad():
        for n, p in ad.model.named_parameters():
            if "lora_" in n:
                if dyadic:  # k · 2^-6，|k| ≤ 3：乘积与和都在 fp32 里精确
                    p.copy_(torch.randint(-3, 4, tuple(p.shape), generator=gen).to(p.dtype) * 2.0 ** -6)
                else:
                    p.copy_(torch.randn(tuple(p.shape), generator=gen).to(p.dtype) * 0.02)


def _make_pair(tmp: Path, *, dyadic: bool, child_alpha: float = 64.0) -> tuple[Path, Path]:
    """父（只有 LoRA，像 C3）、子（网格 token + LoRA，像预热）两个检查点，同一个 base。"""
    g = torch.Generator().manual_seed(7)
    ad = _tiny()
    ad.inject_lora(r=32, lora_alpha=64.0, lora_dropout=0.05, init_lora_weights=False)
    _set_lora(ad, g, dyadic=dyadic)
    ad.save_trained(tmp / "parent")
    ad = _tiny()
    add_vocab(ad, CFG.vocab)
    ad.inject_lora(r=32, lora_alpha=child_alpha, lora_dropout=0.05, init_lora_weights=False)
    _set_lora(ad, g, dyadic=dyadic)
    ad.save_trained(tmp / "child")
    return tmp / "parent", tmp / "child"


def _dyadic_base(ad) -> None:
    """LoRA 作用的线性层的 base 权重取整到 2^-10 的倍数（|w| < 1：几位就够，合并时加法精确）。"""
    mods = dict(ad.model.named_modules())
    with torch.no_grad():
        for name in ad.lora_target_modules():
            w = mods[name].weight
            w.copy_(torch.round(w * 2.0 ** 10) / 2.0 ** 10)


def _peft_merge(ad, lora_dir: Path) -> None:
    """与 load_trained(merge=True)、src/e5/stack.merge_start_lora 同一串 peft 调用（只合并 LoRA，不碰 token 行与适配器的 LoRA 状态）。"""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    pm = get_peft_model(ad.model, LoraConfig.from_pretrained(str(lora_dir / "lora")))
    set_peft_model_state_dict(pm, load_file(str(lora_dir / "lora" / "adapter_model.safetensors")))
    assert pm.merge_and_unload() is ad.model


def _target_weights(ad) -> dict[str, torch.Tensor]:
    """LoRA 作用的线性层的 base 权重（注入了 LoRA 时取 base_layer 的）。"""
    mods = dict(ad.model.named_modules())
    return {n: getattr(mods[n], "base_layer", mods[n]).weight.detach().clone() for n in sorted(ad.lora_target_modules())}


def _merged(dirs: list[Path], *, dyadic: bool) -> dict[str, torch.Tensor]:
    ad = _tiny()
    if dyadic:
        _dyadic_base(ad)
    for d in dirs:
        _peft_merge(ad, d)
    return _target_weights(ad)


def test_rank_concat_merge_bitwise_when_exact(tmp_path):
    p, c = _make_pair(tmp_path, dyadic=True)
    out = write_rank_concat(p, c, tmp_path / "cat", record={"why": "test"})
    seq = _merged([p, c], dyadic=True)  # 先合并父、再合并子
    cat = _merged([out], dyadic=True)   # 拼接后一次合并
    assert set(seq) == set(cat) and all(torch.equal(seq[k], cat[k]) for k in seq)
    base = _merged([], dyadic=True)
    assert any(not torch.equal(seq[k], base[k]) for k in seq)  # 合并确实改了权重
    # 判别力：子的 B 换成父的（拼错块），一次合并的结果就不同
    from safetensors.torch import load_file, save_file

    wrong = tmp_path / "wrong"
    shutil.copytree(out, wrong)
    st = load_file(str(out / "lora" / "adapter_model.safetensors"))
    for k, v in st.items():
        if k.endswith("lora_B.weight"):
            st[k] = torch.cat([v[:, :32], v[:, :32]], dim=1).contiguous()
    save_file(st, str(wrong / "lora" / "adapter_model.safetensors"), metadata={"format": "pt"})
    bad = _merged([wrong], dyadic=True)
    assert any(not torch.equal(seq[k], bad[k]) for k in seq)
    man = read_manifest(out)
    lc = man["lora_concat"]
    assert (man["lora"]["r"], man["lora"]["lora_alpha"], lc["scaling"]) == (64, 128.0, 2.0)
    assert lc["parent"]["manifest_sha256"] == file_sha256(p / "manifest.json") and lc["parent"]["ranks"] == [0, 32]
    assert lc["child"]["ranks"] == [32, 64] and lc["record"] == {"why": "test"}
    assert man["lora"]["n_parameters"] == read_manifest(p)["lora"]["n_parameters"] + read_manifest(c)["lora"]["n_parameters"]
    assert json.loads((out / "tokens.json").read_text()) == json.loads((c / "tokens.json").read_text())  # token 表取子的
    assert (out / "token_rows.safetensors").read_bytes() == (c / "token_rows.safetensors").read_bytes()
    assert check_rank_concat(out, p, c)["checked"] == ["parent", "child"]


MAX_ABS_DIFF_REAL = 1e-6  # 一般浮点值：拼接后一次合并与依次合并只差舍入（tiny fp32 实测见完成总结；增量的量级约 1e-2）


def test_rank_concat_merge_close_for_real_values(tmp_path):
    """一般的浮点值：两种合并的求和顺序不同（秩维上的累加、两次舍入对一次），只差末位舍入，不主张逐位相同。"""
    p, c = _make_pair(tmp_path, dyadic=False)
    out = write_rank_concat(p, c, tmp_path / "cat")
    seq, cat, base = _merged([p, c], dyadic=False), _merged([out], dyadic=False), _merged([], dyadic=False)
    diff = max(float((seq[k] - cat[k]).abs().max()) for k in seq)
    delta = max(float((seq[k] - base[k]).abs().max()) for k in seq)
    assert diff <= MAX_ABS_DIFF_REAL and delta > 1e3 * MAX_ABS_DIFF_REAL, (diff, delta)


def test_rank_concat_state_and_rejections(tmp_path):
    a = {"x.lora_A.weight": torch.ones(2, 5), "x.lora_B.weight": torch.ones(3, 2)}
    b = {"x.lora_A.weight": torch.zeros(4, 5), "x.lora_B.weight": torch.zeros(3, 4)}
    cat = rank_concat_lora_state(a, b, r_parent=2, r_child=4)
    assert cat["x.lora_A.weight"].shape == (6, 5) and cat["x.lora_B.weight"].shape == (3, 6)
    assert torch.equal(cat["x.lora_A.weight"][:2], a["x.lora_A.weight"]) and torch.equal(cat["x.lora_B.weight"][:, 2:], b["x.lora_B.weight"])
    with pytest.raises(CheckpointError, match="形状"):
        rank_concat_lora_state(a, b, r_parent=4, r_child=2)
    with pytest.raises(CheckpointError, match="键不同"):
        rank_concat_lora_state(a, {"y.lora_A.weight": b["x.lora_A.weight"]}, r_parent=2, r_child=4)
    with pytest.raises(CheckpointError, match="不认识的键"):
        rank_concat_lora_state({"x.magnitude": torch.ones(2, 2)}, {"x.magnitude": torch.ones(2, 2)}, r_parent=2, r_child=2)
    p, c = _make_pair(tmp_path / "s", dyadic=False, child_alpha=32.0)
    with pytest.raises(CheckpointError, match="缩放"):
        write_rank_concat(p, c, tmp_path / "o1")
    p, c = _make_pair(tmp_path / "t", dyadic=False)
    with pytest.raises(CheckpointError, match="新增 token 行"):  # 父带网格 token：拼接会丢掉父的行
        write_rank_concat(c, c, tmp_path / "o2")
    other = make_adapter("gemma-4-e2b-it", device="cpu", dtype=torch.float32,
                         image_policy=parse_policy(CFG.doc["tiny"]["image_policies"]["gemma4"]), tiny=True, init_std=0.05)
    other.inject_lora(r=32, lora_alpha=64.0, lora_dropout=0.05)
    other.save_trained(tmp_path / "gemma")
    with pytest.raises(CheckpointError, match="base 不同"):
        write_rank_concat(tmp_path / "gemma", c, tmp_path / "o3")
    out = write_rank_concat(p, c, tmp_path / "o4")
    with pytest.raises(CheckpointError, match="不是空目录"):
        write_rank_concat(p, c, tmp_path / "o4")
    with pytest.raises(CheckpointError, match="不是按秩拼接"):
        check_rank_concat(p, p)
    # 父检查点换了（另一份合法的检查点，manifest sha256 不同）：拒绝
    p2, _ = _make_pair(tmp_path / "u", dyadic=True)
    with pytest.raises(CheckpointError, match="sha256"):
        check_rank_concat(out, p2)


# ---------------------------------------------------------------------------
# 端到端（tiny）：C3 → C3+预热
# ---------------------------------------------------------------------------


def test_c3_warmup_checkpoint(cw_leaf, c3_leaf, tmp_path):
    c3 = _ckpt(c3_leaf)
    ck = _ckpt(cw_leaf)
    sha = file_sha256(c3 / "manifest.json")
    s = json.loads((cw_leaf / "train_summary.json").read_text())
    assert s["lora"]["r"] == 64 and s["lora"]["lora_alpha"] == 128.0 and s["parent"]["manifest_sha256"] == sha
    assert s["parent"]["variant"] == "c3" and s["parent"]["seed"] == 0 and "check_rank_concat 通过" in s["lora_concat"]["check"]
    assert "warmup" in s and s["warmup"]["max_steps"] == 6
    files = sorted(str(x.relative_to(ck)) for x in ck.rglob("*") if x.is_file())
    assert files == ["lora/adapter_config.json", "lora/adapter_model.safetensors", "manifest.json", "token_rows.safetensors",
                     "tokens.json"]
    assert sorted(x.name for x in (cw_leaf / "ckpt").iterdir()) == ["final"]  # 中间目录与续跑状态都删了
    man = read_manifest(ck)
    assert man["lora_concat"]["parent"]["manifest_sha256"] == sha and man["lora_concat"]["record"]["variant"] == C3_WARMUP
    assert man["extra_keys"] == [] and check_rank_concat(ck, c3)["r"] == 64
    ident = yaml.safe_load((cw_leaf / "config.yaml").read_text())
    assert ident["variant"] == C3_WARMUP and ident["seed"] == 0 and ident["supervision"] == "warmup"
    assert ident["parent"]["manifest_sha256"] == sha and ident["parent"]["variant"] == "c3" and "checkpoint" not in ident["parent"]
    assert ident["training"]["parent"]["variant"] == "c3"
    readme = (cw_leaf / "README.md").read_text()
    assert "C3+预热的起点" in readme and sha in readme
    # 能当普通检查点加载；受限解码可用
    ad, _ = factory.load_trained(ck, device="cpu", merge=True)
    assert ad.lora_state == "merged" and len(ad.grid_token_ids) == 67
    tok = attach_vocab(ad, CFG.vocab)
    item = W.heldout_items(W.WarmupSpec.from_doc(CFG.doc, n=N, tiny=True))[0]
    res = ad.generate_restricted([ad.build_inputs([], item.instruction, "instruction_first")], tok.allowed_ids(item.task, "task"),
                                 max_new_tokens=seq_len(N), stop_ids=[tok.stop_id], enable_thinking=False)
    assert 0 < len(res[0].token_ids) <= seq_len(N) and set(res[0].token_ids) <= set(tok.allowed_ids(item.task, "task"))
    parse_target(item.task, res[0].token_ids, tok, n=N)  # 解析得了（格式对不对是训练的事）
    # 合并后的权重 = 先合并 C3、再合并预热的那一半（只差舍入）
    seq = _tiny()
    _peft_merge(seq, c3)
    _peft_merge(seq, _split_child(ck, tmp_path / "child_part"))
    a, b = _target_weights(ad), _target_weights(seq)
    assert max(float((a[k] - b[k]).abs().max()) for k in a) <= MAX_ABS_DIFF_REAL


def _split_child(ck: Path, out: Path) -> Path:
    """从拼接的检查点里取出后 32 个秩，存成一个秩 32 的 LoRA 目录（只给 _peft_merge 用）。"""
    from peft import LoraConfig
    from safetensors.torch import load_file, save_file

    man = read_manifest(ck)
    lo, hi = man["lora_concat"]["child"]["ranks"]
    cfg = LoraConfig.from_pretrained(str(ck / "lora"))
    cfg.r, cfg.lora_alpha = hi - lo, man["lora_concat"]["scaling"] * (hi - lo)
    (out / "lora").mkdir(parents=True)
    cfg.save_pretrained(str(out / "lora"))
    st = load_file(str(ck / "lora" / "adapter_model.safetensors"))
    save_file({k: (v[lo:hi] if k.endswith("lora_A.weight") else v[:, lo:hi]).contiguous() for k, v in st.items()},
              str(out / "lora" / "adapter_model.safetensors"), metadata={"format": "pt"})
    return out


def test_c3_warmup_same_data_and_init_as_warmup(cw_leaf, c3_leaf, tmp_path):
    """预热的数据（逐步的样本数、种类、token 数）与基座上的预热相同；合并起点在 manual_seed 之前，所以预热 LoRA 的初始值与网格 token 行
    的初始值也与基座上的预热逐位相同。"""
    rc = resolve_run(CFG, supervision="warmup", tiny=True)
    rc["data_workers"] = 0
    ad = _tiny()
    run_training(ad, CFG, _spec(KEY, ad, "warmup"), rc, D.E3Paths.fixtures(), tmp_path, log=lambda m: None)
    wl = leaf_dir(tmp_path, "warmup", KEY, 0)
    out_keys = ("loss", "loss_by_task", "grad_norm", "grid_token_acc")
    la = [json.loads(x) for x in (wl / "train_log.jsonl").read_text().splitlines()]
    lb = [json.loads(x) for x in (cw_leaf / "train_log.jsonl").read_text().splitlines()]
    n = min(len(la), len(lb))
    assert n >= 1 and [{k: v for k, v in r.items() if k not in out_keys} for r in la[:n]] == \
        [{k: v for k, v in r.items() if k not in out_keys} for r in lb[:n]]
    ia, ib = (yaml.safe_load((d / "config.yaml").read_text()) for d in (wl, cw_leaf))
    assert ia["data"] == ib["data"] and ia["training"]["warmup"] == ib["training"]["warmup"] and "parent" not in ia
    assert (wl / "tokens_init.json").read_text() == (cw_leaf / "tokens_init.json").read_text()  # 网格 token 行的初始值相同
    ra, rb = resolve_run(CFG, supervision="warmup", tiny=True), _cw_rc()
    t_plain = Trainer(_tiny(), CFG, ra, supervision="warmup", seed=0)
    t_plain.setup_fresh()
    parent = resolve_parent(_ckpt(c3_leaf), model_key=KEY, tiny=True, seed=0)
    t_c3 = Trainer(_tiny(), CFG, rb, supervision="warmup", seed=0, parent=parent)
    t_c3.setup_fresh()
    pa = {n: p for n, p in t_plain.ad.model.named_parameters() if "lora_" in n}
    pb = {n: p for n, p in t_c3.ad.model.named_parameters() if "lora_" in n}
    assert set(pa) == set(pb) and all(torch.equal(pa[k], pb[k]) for k in pa)
    assert t_c3.parent_merge["manifest_sha256"] == parent.manifest_sha256
    # base 不同：C3 已合并进去
    wa, wb = _target_weights(t_plain.ad), _target_weights(t_c3.ad)
    assert set(wa) == set(wb) and any(not torch.equal(wa[k], wb[k]) for k in wa)


def test_e5_reads_c3_warmup_as_a_start(cw_leaf):
    """E5 侧（src/e5/stack，只读）：叶目录 config.yaml 的 variant、seed 读得到；merge_start_lora 把秩 64 的 LoRA 当普通起点合并，
    之后的权重指纹与 factory.load_trained(merge=True) 的相同（E5 开训前的第 2 项核对）。"""
    from src.e5.stack import _leaf_seed, _leaf_variant, merge_start_lora, stack_fingerprint

    ck = _ckpt(cw_leaf)
    assert _leaf_variant(ck)[0] == C3_WARMUP and _leaf_seed(ck) == 0
    ad = _tiny()
    info = merge_start_lora(ad, ck)
    assert info["lora"]["r"] == 64 and ad.lora_state == "none" and not ad.grid_token_ids
    ref, _ = factory.load_trained(ck, device="cpu", merge=True)
    assert stack_fingerprint(ad) == stack_fingerprint(ref)


def _altered_c3(c3_leaf: Path, dst: Path) -> Path:
    """另一份合法的 C3 检查点（LoRA 改了一点，manifest sha256 不同），叶目录的 config.yaml 照抄（variant c3、种子 0）。"""
    dst.mkdir(parents=True)
    shutil.copyfile(c3_leaf / "config.yaml", dst / "config.yaml")
    ad, _ = factory.load_trained(_ckpt(c3_leaf), device="cpu", merge=False)
    with torch.no_grad():
        p = next(p for n, p in ad.model.named_parameters() if "lora_B" in n)
        p.add_(0.01)
    ad.save_trained(_ckpt(dst))
    return dst


def test_c3_warmup_resume_identical_and_parent_change_refused(c3_leaf, cw_leaf, tmp_path):
    paths = D.E3Paths.fixtures()
    rc = _cw_rc()
    parent = _ckpt(c3_leaf)
    ad = _tiny()
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        run_training(ad, CFG, _spec(KEY, ad, "warmup", parent=parent), rc, paths, tmp_path / "B", log=lambda m: None,
                     _fail_after_step=3)
    lb = leaf_dir(tmp_path / "B", C3_WARMUP, KEY, 0)
    latest = lb / "ckpt" / "latest" / "model"
    assert read_parent_marker(latest) == file_sha256(parent / "manifest.json")
    assert read_manifest(latest)["extra_keys"] == [PARENT_MARKER] and read_manifest(latest)["lora"]["r"] == 32  # 续跑用的中间状态
    # 起点换成另一份 C3（sha256 不同）：拒绝续跑
    alt = _altered_c3(c3_leaf, tmp_path / "alt_c3")
    ad = _tiny()
    with pytest.raises(ParentError, match="续跑"):
        run_training(ad, CFG, _spec(KEY, ad, "warmup", parent=_ckpt(alt)), rc, paths, tmp_path / "B", resume=True,
                     log=lambda m: None)
    with pytest.raises(CheckpointError, match="sha256"):
        check_rank_concat(_ckpt(cw_leaf), _ckpt(alt))
    ad = _tiny()
    sb = run_training(ad, CFG, _spec(KEY, ad, "warmup", parent=parent), rc, paths, tmp_path / "B", resume=True, log=lambda m: None)
    assert sb["complete"]
    for f in ("train_log.jsonl", "warmup_eval.jsonl", "config.yaml", "README.md", "tokens_init.json"):
        assert (cw_leaf / f).read_bytes() == (lb / f).read_bytes(), f
    sa = json.loads((cw_leaf / "train_summary.json").read_text())
    assert {k: v for k, v in sa.items() if k != "parent"} == {k: v for k, v in sb.items() if k != "parent"}
    assert sa["parent"]["manifest_sha256"] == sb["parent"]["manifest_sha256"]
    for f in ("lora/adapter_model.safetensors", "token_rows.safetensors", "tokens.json", "lora/adapter_config.json"):
        assert (_ckpt(cw_leaf) / f).read_bytes() == (_ckpt(lb) / f).read_bytes(), f
    assert sorted(x.name for x in (lb / "ckpt").iterdir()) == ["final"]


def test_resolve_parent_rejections(c3_leaf, cw_leaf, tmp_path):
    ck = _ckpt(c3_leaf)
    ref = resolve_parent(ck, model_key=KEY, tiny=True, seed=0)
    assert ref.variant == "c3" and ref.seed == 0 and ref.manifest_sha256 == file_sha256(ck / "manifest.json")
    summ = json.loads((c3_leaf / "train_summary.json").read_text())
    assert ref.leaf_identity_sha256 == summ["identity_sha256"]  # 叶目录 config.yaml 的身份 = C3 运行的身份
    with pytest.raises(ParentError, match="种子"):
        resolve_parent(ck, model_key=KEY, tiny=True, seed=1)
    with pytest.raises(ParentError, match="构造方式"):
        resolve_parent(ck, model_key="gemma-4-e2b-it", tiny=True, seed=0)
    with pytest.raises(ParentError, match="新增 token"):  # 带网格 token 的检查点（E3b、预热）不是 C3
        resolve_parent(_ckpt(cw_leaf), model_key=KEY, tiny=True, seed=0)
    loose = tmp_path / "loose"
    shutil.copytree(ck, loose)
    with pytest.raises(ParentError, match="核对不了"):
        resolve_parent(loose, model_key=KEY, tiny=True, seed=0)
    fake = tmp_path / "fake"
    shutil.copytree(c3_leaf, fake)
    doc = yaml.safe_load((fake / "config.yaml").read_text())
    doc["variant"] = "e3a"
    (fake / "config.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    with pytest.raises(ParentError, match="'c3'"):
        resolve_parent(_ckpt(fake), model_key=KEY, tiny=True, seed=0)
    (_ckpt(fake) / "tokens.json").write_text("{}")
    with pytest.raises(ParentError, match="读不了"):
        resolve_parent(_ckpt(fake), model_key=KEY, tiny=True, seed=0)
    rc = resolve_run(CFG, supervision="warmup", tiny=True)  # 基座上的预热的 run_cfg 配起点：不符
    ad = _tiny()
    with pytest.raises(ValueError, match="不符"):
        run_training(ad, CFG, _spec(KEY, ad, "warmup", parent=ck), rc, D.E3Paths.fixtures(), tmp_path / "x", log=lambda m: None)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True, cwd=REPO)


def test_cli_three_families_and_checkpoint_loads(tmp_path):
    out = tmp_path / "o"
    p = run_script("train_e3b", "--tiny", "--model", *FAMILIES, "--supervision", "text", "--seed", "0", "--out", str(out))
    assert p.returncode == 0, p.stderr[-3000:]
    for key in FAMILIES:
        c3 = out / "c3" / key / "seed0" / "ckpt" / "final" / "model"
        p = run_script("train_e3b", "--tiny", "--model", key, "--supervision", "warmup", "--seed", "0", "--start-checkpoint", str(c3),
                       "--out", str(out))
        assert p.returncode == 0, p.stderr[-3000:]
        assert f"{key} c3_warmup seed0 steps=" in p.stdout and "stop_rule_met=" in p.stdout
        ck = out / C3_WARMUP / key / "seed0" / "ckpt" / "final" / "model"
        man = read_manifest(ck)
        assert man["lora"]["r"] == 64 and check_rank_concat(ck, c3)["checked"] == ["parent"]
    ck = out / C3_WARMUP / FAMILIES[2] / "seed0" / "ckpt" / "final" / "model"  # InternVL：lm_head 不共享
    p = run_script("eval_e3b", "--tiny", "--checkpoint", str(ck), "--model", FAMILIES[2], "--variant", C3_WARMUP, "--seed", "0",
                   "--task", "depth", "--limit", "1", "--k-values", "1", "--out", str(tmp_path / "e"))
    assert p.returncode == 0, p.stderr[-3000:]
    mc = json.loads((tmp_path / "e" / "eval" / C3_WARMUP / FAMILIES[2] / "seed0" / "merge_check.json").read_text())
    assert mc["pass"] is True and list(mc["checks"]) == ["merge", "restricted"] and mc["checkpoint"]["lora"]["r"] == 64
    p = run_script("run_e1", "--tiny", "--model", FAMILIES[2], "--task", "grid16", "--limit", "1", "--checkpoint", str(ck),
                   "--checkpoint-label", "t_text/c3_warmup/seed0", "--out", str(tmp_path / "t"))
    assert p.returncode == 0, p.stderr[-3000:]
    assert (tmp_path / "t" / "t_text" / C3_WARMUP / "seed0" / FAMILIES[2] / "grid16" / "metrics.json").exists()


def test_cli_rules(tmp_path):
    base = ["--tiny", "--seed", "0", "--out", str(tmp_path / "x")]
    p = run_script("train_e3b", "--model", KEY, "--supervision", "grid", "--start-checkpoint", str(tmp_path), *base)
    assert p.returncode == 2 and "--start-checkpoint 只配" in p.stderr
    p = run_script("train_e3b", "--model", KEY, "--supervision", "warmup", "--variant", C3_WARMUP, *base)
    assert p.returncode == 2 and "要给 --start-checkpoint" in p.stderr
    p = run_script("train_e3b", "--model", KEY, "gemma-4-e2b-it", "--supervision", "warmup", "--start-checkpoint", str(tmp_path), *base)
    assert p.returncode == 2 and "只配一个 --model" in p.stderr
    p = run_script("train_e3b", "--model", KEY, "--supervision", "warmup", "--variant", "warmup", "--start-checkpoint", str(tmp_path),
                   *base)
    assert p.returncode == 2 and C3_WARMUP in p.stderr
    # 起点不合规：运行时报错（退出码 2，证据写进 errors/）
    p = run_script("train_e3b", "--model", KEY, "--supervision", "warmup", "--start-checkpoint", str(tmp_path), *base)
    assert p.returncode == 2 and "ParentError" in p.stderr
    assert copy.deepcopy(CFG.doc["warmup"]["stop"]) == {"eval_every": 100, "threshold": 0.01, "consecutive": 2, "max_steps": 2000}
