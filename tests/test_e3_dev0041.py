"""DEV-0041：E3b 的指令检查（§1）、网格的结构强制解码（§2）、LoRA 缩放的派生检查点（§3）。CPU fp32 小模型。

- §2 结构强制（src/grid/structured_decode.py，src/e3/evaluate 的 decode = structured）：
  - 形状的规则（逐位置的允许集合、长度 n(n + 1) + 1、逐行参数化、没有停止位置的变体）与 src/e3/targets 的序列格式逐位一致；
  - 处理器在合成 logits 上的贪心必然得到合法网格；一个实例只能用于一次 generate；入口断言抓得住处理器没生效；
  - 三个家族的小模型：不加结构约束时 generate_batch(allowed_token_ids, logits_processor = [不收窄的处理器]) 与 generate_restricted
    逐 token 相同；加上结构约束后，在受限解码第一次违反结构之前两者逐 token 相同，之后结构强制的输出仍是合法网格；
  - 评测：单轮、多轮、匹配计算量的采样都走结构强制（n = 8），全部解析成功；运行身份与 metrics 记 decode，叶目录 __structured；
    不给 --decode（或给 restricted）时输出与 53aefd2 的快照逐字节相同；续跑时解码方式不同拒绝混写。
- §1 指令检查：
  - none 的指令与 E3a 的 no_instruction 是同一个值（从 src/readout/evaluate 读出）；swap 的候选规则（同图、不同 ann_id、按评测总体
    顺序、哈希取第 k 个、没有候选跳过）；
  - 语义（替身适配器：按指令里的指代句输出那个物体的 GT 网格）：正常评测满分；none 崩；swap 的预测跟着换到新物体上（新目标满分、
    原目标下降），配对差的方向与定义一致；正常一侧按 id 配对、与它的 scores.jsonl 逐项核对；
  - 正常一侧的运行身份或样本顺序对不上就拒绝；命令行规则（只用于 refseg、只跑单轮、--limit 取批大小的整数倍、--normal-leaf 只配检查）；
  - --tiny 命令行跑通 none、swap 与结构强制下的检查。
- §3 scripts/scale_lora_ckpt.py：s = 1 合并后与原检查点合并后逐位相同；s = 0 合并后除网格 token 行以外与 base 逐位相同；s = 0.5 抽一个
  模块，合并后的增量是原增量的 0.5 倍（fp32，误差 ≤ 1e-6）；manifest 的 derived 与逐文件 sha256、叶目录的 config.yaml 拷贝；拒绝的情形；
  eval_e3b 的运行身份记 derived，原生检查点不变。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from transformers import LogitsProcessor

from src.adapters import factory, registry
from src.adapters.checkpoint import read_manifest
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.data.records import LoadResult, Sample
from src.e3 import eval_scoring as ES
from src.e3 import evaluate as EV
from src.e3.config import load_config
from src.e3.data import load_rgb
from src.e3.targets import grid_to_ids, parse_grid_ids, parse_target, seg_grid, seq_len
from src.e3.tokens import GridTokens, add_vocab, attach_vocab
from src.grid.structured_decode import (
    GridStructure,
    StructuredDecodeError,
    StructuredGridProcessor,
    check_structured_ids,
)

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()
DETERMINISTIC = ("README.md", "config.yaml", "metrics.json", "scores.jsonl")
SNAP_EV = REPO / "tests" / "fixtures" / "e3" / "evaluate_53aefd2.py"  # 53aefd2 的 src/e3/evaluate.py 原样（DEV-0041 之前）
SNAP_EV_SHA256 = "a9c10a20d4a883648a0b27653ad89dea42b66547f9488d1ff713bbd39aa213af"
SNAP_SCRIPT = REPO / "tests" / "fixtures" / "e3" / "eval_e3b_53aefd2.py"  # 53aefd2 的 scripts/eval_e3b.py 原样
SNAP_SCRIPT_SHA256 = "b1a7721a4fb90076e3314e158d3e00045e51b773acc651630c434d7a2fd1c83e"


def _tiny(key):
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _load_path(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def _load_script(name: str):
    return _load_path(REPO / "scripts" / f"{name}.py", f"_{name}_dev0041")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True, cwd=REPO)


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """训练框架（scripts/train_e3b.py --tiny）产出的小检查点（Qwen），叶目录布局 <叶>/ckpt/final/model。"""
    d = tmp_path_factory.mktemp("tiny_train41")
    p = run_script("train_e3b", "--tiny", "--model", FAMILIES[0], "--supervision", "grid", "--seed", "0", "--out", str(d))
    assert p.returncode == 0, p.stderr[-3000:]
    return d / "e3b" / FAMILIES[0] / "seed0" / "ckpt" / "final" / "model"


def _cli_args(ck: Path) -> list[str]:
    return ["--tiny", "--checkpoint", str(ck), "--model", FAMILIES[0], "--seed", "0"]


def _mdir(out: Path) -> Path:
    return out / "eval" / "e3b" / FAMILIES[0] / "seed0"


# ---------------------------------------------------------------------------
# §2 结构强制：形状的规则
# ---------------------------------------------------------------------------


def test_structure_rules_and_target_format():
    st = GridStructure.uniform([11, 12], row_id=20, stop_id=21, n=3)
    assert st.length == 3 * 4 + 1 == seq_len(3)
    kinds = [st.kind_at(t) for t in range(st.length)]
    assert kinds == [("cell", 0)] * 3 + [("row", 0)] + [("cell", 1)] * 3 + [("row", 1)] + [("cell", 2)] * 3 + [("row", 2)] + \
        [("stop", 3)]
    assert [st.allowed_at(t) for t in (0, 3, 12)] == [(11, 12), (20,), (21,)]
    with pytest.raises(StructuredDecodeError):
        st.kind_at(st.length)
    assert st.all_ids() == [11, 12, 20, 21]
    # 与 src/e3/targets 的序列格式逐位一致：任何网格的目标序列都合结构；任何合结构的序列都能解析
    rng = np.random.default_rng(0)
    for n, k in ((3, 2), (8, 2), (32, 2), (32, 64)):
        vals = list(range(100, 100 + k))
        s = GridStructure.uniform(vals, row_id=7, stop_id=8, n=n)
        g = rng.integers(0, k, (n, n))
        ids = grid_to_ids(g, vals, 7, 8, n=n)
        check_structured_ids(ids, s, stopped=True)
        p = parse_grid_ids(ids, vals, 7, 8, n=n)
        assert p.failure is None and np.array_equal(p.grid, g)
    # 逐行参数化（E5 用）与没有停止位置的变体
    pr = GridStructure.per_row([[1, 2], [3, 4, 5]], row_id=9, stop_id=None)
    assert pr.n == 2 and pr.length == 6 and pr.allowed_at(3) == (3, 4, 5) and pr.allowed_at(5) == (9,)
    assert pr.as_dict()["value_ids_per_row"] is True and st.as_dict()["value_ids"] == [11, 12]
    for bad in (dict(value_ids=[20, 12], row_id=20, stop_id=21, n=2), dict(value_ids=[11, 11], row_id=20, stop_id=21, n=2),
                dict(value_ids=[11], row_id=20, stop_id=20, n=2), dict(value_ids=[11], row_id=20, stop_id=21, n=0)):
        with pytest.raises(ValueError):
            GridStructure.uniform(**bad)
    # 入口断言
    good = grid_to_ids(np.zeros((3, 3), dtype=np.int64), [11, 12], 20, 21, n=3)
    with pytest.raises(StructuredDecodeError, match="恰好"):
        check_structured_ids(good[:-1], st)
    with pytest.raises(StructuredDecodeError, match="不在该位置"):
        check_structured_ids(good[:3] + [21] + good[4:], st)
    with pytest.raises(StructuredDecodeError, match="stopped"):
        check_structured_ids(good, st, stopped=False)


def test_processor_greedy_on_synthetic_scores():
    """合成 logits：每一步随机打分（停止与行分隔的分数故意最高），贪心经过处理器后必然是合法网格；一个实例不能用两次。"""
    vocab, n = 50, 4
    st = GridStructure.uniform([3, 4, 5], row_id=10, stop_id=11, n=n)
    g = torch.Generator().manual_seed(0)
    proc = StructuredGridProcessor(st)
    ids = torch.zeros((2, 7), dtype=torch.long)
    out = []
    for _ in range(st.length):
        scores = torch.randn((2, vocab), generator=g)
        scores[:, 11] = 100.0  # 不加约束时每一步都会停
        scores[:, 10] = 50.0
        nxt = proc(ids, scores).argmax(-1)
        out.append(nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
    seq = torch.stack(out, dim=1)
    for row in seq.tolist():
        check_structured_ids(row, st, stopped=True)
    assert proc.n_calls == st.length
    with pytest.raises(StructuredDecodeError, match="一次 generate"):
        proc(torch.zeros((2, 3), dtype=torch.long), torch.zeros((2, vocab)))
    with pytest.raises(StructuredDecodeError, match="超出"):
        p2 = StructuredGridProcessor(GridStructure.uniform([3], row_id=10, stop_id=11, n=1))
        x = torch.zeros((1, 2), dtype=torch.long)
        for _ in range(4):  # 长度 3，第 4 次越界
            p2(x, torch.zeros((1, vocab)))
            x = torch.cat([x, x[:, :1]], dim=1)


# ---------------------------------------------------------------------------
# §2 结构强制：小模型上与 generate_restricted 的关系
# ---------------------------------------------------------------------------


class _CountingProcessor(LogitsProcessor):  # 不改 scores，只计数（"不收窄"的处理器）
    def __init__(self):
        self.n_calls = 0

    def __call__(self, input_ids, scores):
        self.n_calls += 1
        return scores


def _small_inputs(ad, ctx, k=3):
    root = REPO / "tests" / "fixtures" / "e3" / "eval_root" / "refcoco" / "images"
    imgs = sorted(root.glob("*.jpg"))[:k]
    exprs = ["the person on the left", "dog", "a red bowl behind the others"]
    return [ad.build_inputs([load_rgb(str(p))], ctx.grid_template.format(expression=e, n=ctx.n), ctx.order)
            for p, e in zip(imgs, exprs)]


def _distinct_grid_rows(ad, tokens) -> None:
    """add_vocab 把新行初始化成均值：64 个深度 token 的 logits 恰好平局，两条计算路径的舍入不同就会各取一个（与实现无关）。测试里把
    网格 token 的嵌入与 lm_head 行换成固定种子的随机行（量级同原矩阵），去掉这种人为的平局；训练过的检查点没有这个问题。"""
    g = torch.Generator().manual_seed(41)
    ids = torch.tensor(sorted(tokens.grid_ids), dtype=torch.long)
    ws = {id(w): w for w in (ad.model.get_input_embeddings().weight, ad.lm_head_weight())}
    with torch.no_grad():
        for w in ws.values():
            w[ids] = torch.randn((len(ids), w.shape[1]), generator=g, dtype=w.dtype) * w.std()


@pytest.mark.parametrize("key", FAMILIES)
def test_structured_vs_restricted_tiny(key):
    ad = _tiny(key)
    tokens = add_vocab(ad, CFG.vocab)
    _distinct_grid_rows(ad, tokens)
    ad.eval_mode()
    for task in ("refseg", "depth"):
        ctx = EV.make_context(CFG, tokens, task, n=4)
        inputs = _small_inputs(ad, ctx)
        with torch.no_grad():
            rest = ad.generate_restricted(inputs, list(ctx.allowed), max_new_tokens=ctx.max_new_tokens,
                                          stop_ids=[int(tokens.stop_id)], enable_thinking=False)
            cp = _CountingProcessor()
            plain = ad.generate_batch(inputs, enable_thinking=False, max_new_tokens=ctx.max_new_tokens,
                                      allowed_token_ids=list(ctx.allowed), logits_processor=[cp], do_sample=False)
            struct = EV.structured_generate(ad, ctx, inputs, where=f"{key} {task}")
        assert cp.n_calls >= 1
        # 不加结构约束：两条路径逐 token 相同（CPU fp32）
        assert [list(r.token_ids) for r in plain] == [list(r.token_ids) for r in rest], (key, task)
        assert [r.stopped for r in plain] == [r.stopped for r in rest]
        st = EV.grid_structure(ctx)
        for r_s, r_r in zip(struct, rest):
            check_structured_ids(r_s.token_ids, st, stopped=r_s.stopped)
            assert parse_target(task, r_s.token_ids, tokens, n=ctx.n).failure is None
            rr = [int(t) for t in r_r.token_ids]
            first_bad = next((t for t, i in enumerate(rr) if t >= st.length or i not in st.allowed_at(t)), len(rr))
            # 受限解码第一次违反结构之前，两者逐 token 相同（结构强制只在受限解码会写坏形状的位置上改变选择）
            assert list(r_s.token_ids)[:first_bad] == rr[:first_bad], (key, task)


def test_structured_eval_all_stages_n8(tiny_ckpt, tmp_path):
    """评测（n = 8）：单轮、多轮的每一轮、匹配计算量的每次采样都走结构强制、全部解析成功；运行身份与 metrics 记 decode；续跑同一身份
    照常、不同身份拒绝。"""
    ad, _ = factory.load_trained(tiny_ckpt, device="cpu", merge=True)
    tokens = attach_vocab(ad, CFG.vocab)
    e = registry.get_model(FAMILIES[0])
    spec = EV.EvalSpec(model_key=FAMILIES[0], family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=FAMILIES[0],
                       tiny=True, dtype="fp32", variant="e3b", seed=0,
                       checkpoint={"manifest_sha256": hashlib.sha256((tiny_ckpt / "manifest.json").read_bytes()).hexdigest()})
    s_st = EV.resolve_eval(CFG, tiny=True, decode="structured")
    s_re = EV.resolve_eval(CFG, tiny=True)
    assert EV.resolve_eval(CFG, tiny=True, decode="restricted") == s_re and "decode" not in s_re
    assert {k: v for k, v in s_st.items() if k != "decode"} == s_re and s_st["decode"]["mode"] == "structured"
    with pytest.raises(ValueError, match="decode"):
        EV.resolve_eval(CFG, tiny=True, decode="free")
    assert EV.leaf_name("refseg", s_st) == "refseg__structured" and EV.leaf_name("refseg", s_re) == "refseg"
    assert EV.leaf_name("refseg", s_st, "swap") == "refseg__instr_swap__structured"
    out = tmp_path / "o"
    paths = EV.EvalPaths.fixtures()
    for task in ("refseg", "depth"):
        ctx = EV.make_context(CFG, tokens, task, n=8)
        sel = EV.load_eval_set(task, paths)
        m = EV.run_eval_task(ad, CFG, s_st, spec, ctx, sel, out, warmup=False, log=lambda _m: None)
        leaf = EV.leaf_dir(out, spec, task, s_st)
        assert leaf.name == f"{task}__structured"
        for st_name, d in m["stages"].items():
            assert d["n_valid"] == d["n_called"] == m["n"] and d["parse_failures"] == {}, (task, st_name)
            assert d["generated_tokens"]["min"] == d["generated_tokens"]["max"] == seq_len(8)
        assert m["decode"] == s_st["decode"]
        ident = yaml.safe_load((leaf / "config.yaml").read_text())
        assert ident["eval"]["decode"]["mode"] == "structured"
        assert "解码：structured" in (leaf / "README.md").read_text()
        before = {f: (leaf / f).read_bytes() for f in DETERMINISTIC}
        EV.run_eval_task(ad, CFG, s_st, spec, ctx, sel, out, resume=True, warmup=False, log=lambda _m: None)
        assert {f: (leaf / f).read_bytes() for f in DETERMINISTIC} == before
        # 续跑时解码方式不同：拒绝混写（把结构强制的叶目录放到 restricted 的位置上再续跑）
        alt = tmp_path / f"alt_{task}"
        shutil.copytree(leaf, EV.leaf_dir(alt, spec, task, s_re))
        with pytest.raises(ValueError, match="运行身份"):
            EV.run_eval_task(ad, CFG, s_re, spec, ctx, sel, alt, resume=True, warmup=False, log=lambda _m: None)


def test_default_outputs_identical_to_53aefd2(tiny_ckpt, tmp_path):
    """不给 --decode、--instruction-check：新脚本与改动前（53aefd2）的脚本加模块，确定的输出（README、config.yaml、metrics.json、
    scores.jsonl 与各阶段的 predictions）逐字节相同；--decode restricted 也一样。"""
    assert hashlib.sha256(SNAP_EV.read_bytes()).hexdigest() == SNAP_EV_SHA256
    assert hashlib.sha256(SNAP_SCRIPT.read_bytes()).hexdigest() == SNAP_SCRIPT_SHA256
    old_ev = _load_path(SNAP_EV, "_e3_evaluate_53aefd2")
    for k in ("REPO_ROOT", "FIXTURE_REFSEG_ROOT", "FIXTURE_REFSEG_DEV", "FIXTURE_DEPTH_ROOT"):
        setattr(old_ev, k, getattr(EV, k))
    old_script = _load_path(SNAP_SCRIPT, "_eval_e3b_53aefd2")
    old_script.EV = old_ev
    assert old_ev.resolve_eval(CFG, tiny=True) == EV.resolve_eval(CFG, tiny=True)
    new_script = _load_script("eval_e3b")
    outs = []
    for k, (mod, extra) in enumerate(((old_script, []), (new_script, []), (new_script, ["--decode", "restricted"]))):
        out = tmp_path / f"o{k}"
        assert mod.main(_cli_args(tiny_ckpt) + ["--out", str(out)] + extra) == 0
        outs.append(_mdir(out))
    stages = tuple(EV.stage_file(s) for s in EV.stage_names(EV.resolve_eval(CFG, tiny=True)))
    for task in ("refseg", "depth"):
        assert sorted(p.name for p in outs[1].iterdir() if p.is_dir()) == ["depth", "refseg"]
        for f in DETERMINISTIC + stages:
            for o in outs[1:]:
                assert (outs[0] / task / f).read_bytes() == (o / task / f).read_bytes(), (task, f)


# ---------------------------------------------------------------------------
# §1 指令检查：规则
# ---------------------------------------------------------------------------


def _ref(sid: str, image_id: int, ann_id: int, expr: str, img: str = "x.png") -> Sample:
    return Sample(id=sid, task="refexp", images=(img,), image_keys=(f"coco:{image_id}",),
                  fields={"expression": expr, "image_id": image_id, "ann_id": ann_id, "category": "dog"}, gt={}, source="t",
                  split="val")


def test_swap_candidate_rule_and_none_definition():
    pool = [_ref("r/1", 1, 10, "a"), _ref("r/2", 1, 10, "a2"), _ref("r/3", 1, 11, "b"), _ref("r/4", 1, 12, "c"),
            _ref("r/5", 2, 20, "d"), _ref("r/6", 3, 30, "e"), _ref("r/7", 3, 31, "f")]
    got = EV.swap_candidates(pool, pool)
    # 同一张图、不同 ann_id 的全部句子，按 pool 的顺序；取第 swap_index 个
    want_c = {"r/1": ["r/3", "r/4"], "r/2": ["r/3", "r/4"], "r/3": ["r/1", "r/2", "r/4"], "r/4": ["r/1", "r/2", "r/3"],
              "r/5": [], "r/6": ["r/7"], "r/7": ["r/6"]}
    for s, (c, nc) in zip(pool, got):
        cands = want_c[s.id]
        assert nc == len(cands)
        if not cands:
            assert c is None
        else:
            k = int.from_bytes(hashlib.sha256(f"visint-e3b-instr-swap|{EV.SWAP_SEED}|{s.id}".encode()).digest()[:8], "big")
            assert c.id == cands[k % len(cands)] and c.fields["ann_id"] != s.fields["ann_id"]
            assert c.fields["image_id"] == s.fields["image_id"]
    assert EV.SWAP_SEED == 20261008
    # 只取前 N 条时，候选仍来自整个评测总体
    assert [c.id if c else None for c, _ in EV.swap_candidates(pool[:1], pool)] == [got[0][0].id]
    tokens = GridTokens(vocab=CFG.vocab, ids={n: 1000 + i for i, n in enumerate(CFG.vocab.names)}, stop_id=999)
    ctx = EV.make_context(CFG, tokens, "refseg", n=8)
    plan = EV.instruction_check_plan("swap", ctx, pool, pool, tiny=True)
    assert plan.identity["n_skipped"] == 1 and plan.identity["n_called"] == 6 and [p["skipped"] for p in plan.pairs].count(True) == 1
    assert [s.id for s in plan.called] == ["r/1", "r/2", "r/3", "r/4", "r/6", "r/7"]
    assert plan.ctx.instruction(plan.called[0], correction=False) == ctx.grid_template.format(
        expression=plan.targets["r/1"].fields["expression"], n=8)
    # none：与 E3a 的 no_instruction 同一个值（从 src/readout/evaluate 读出），user 轮里只剩图像
    from src.readout.config import load_config as load_e3a
    from src.readout.evaluate import resolve_eval as e3a_resolve

    ni = e3a_resolve(load_e3a(), tiny=True)["leak_checks"]["no_instruction"]
    pn = EV.instruction_check_plan("none", ctx, pool, pool, tiny=True)
    assert pn.ctx.instruction(pool[0], correction=False) == ni["instruction"] == ""
    assert pn.identity["e3a_config"]["sha256"] == load_e3a().sha256 and pn.identity["n_skipped"] == 0
    with pytest.raises(ValueError):
        EV.instruction_check_plan("none", EV.make_context(CFG, tokens, "depth", n=8), pool, pool, tiny=True)


# ---------------------------------------------------------------------------
# §1 指令检查：语义（替身适配器）
# ---------------------------------------------------------------------------

N8 = 8


class FollowStub:
    """替身：按指令里的指代句输出那个物体的 GT 网格（n = 8）；空指令输出全 0。用来核对检查的方向与配对，不涉及真模型。"""

    def __init__(self, ctx, samples):
        self.ctx = ctx
        self.by_expr = {}
        for s in samples:
            g = seg_grid(ES.read_gt("refseg", s), n=N8)
            self.by_expr[ctx.grid_template.format(expression=str(s.fields["expression"]).strip(), n=N8)] = g
        self.calls = []

    def build_inputs(self, images, instruction, order):
        return SimpleNamespace(instruction=instruction)

    def generate_restricted(self, inputs, allowed, *, max_new_tokens, stop_ids, enable_thinking):
        out = []
        for inp in inputs:
            self.calls.append(inp.instruction)
            g = self.by_expr.get(inp.instruction, np.zeros((N8, N8), dtype=np.int64))
            ids = grid_to_ids(g, self.ctx.tokens.value_ids("refseg"), self.ctx.tokens.row_id, self.ctx.tokens.stop_id, n=N8)
            out.append(SimpleNamespace(token_ids=tuple(ids), stopped=True, n_generated_tokens=len(ids), n_prompt_tokens=10,
                                       n_visual_tokens=4, n_image_pixels=64, wall_time_s=0.0, batch_wall_time_s=0.0))
        return out


def _stub_setup(tmp_path, extra_single=True):
    tokens = GridTokens(vocab=CFG.vocab, ids={n: 1000 + i for i, n in enumerate(CFG.vocab.names)}, stop_id=999)
    ctx = EV.make_context(CFG, tokens, "refseg", n=N8)
    sel = EV.load_eval_set("refseg", EV.EvalPaths.fixtures())
    recs = list(sel.records)
    if extra_single:  # 一张只有一个被指代物体的图：swap 没有候选，跳过并计数
        s0 = recs[0]
        recs.append(Sample(id="refcoco_unc/val/99999", task="refexp", images=s0.images, image_keys=("coco:99999",),
                           fields={**s0.fields, "image_id": 99999, "ann_id": 99999, "expression": "the only thing"}, gt=s0.gt,
                           source=s0.source, split=s0.split))
    sel = LoadResult(recs, sel.stats)
    e = registry.get_model(FAMILIES[0])
    spec = EV.EvalSpec(model_key=FAMILIES[0], family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=FAMILIES[0],
                       tiny=True, dtype="fp32", variant="e3b", seed=0, checkpoint={"manifest_sha256": "stub"})
    settings = EV.resolve_eval(CFG, tiny=True, k_values=[1])
    return tokens, ctx, sel, spec, settings, FollowStub(ctx, recs)


def test_instruction_checks_semantics_with_following_stub(tmp_path):
    tokens, ctx, sel, spec, settings, stub = _stub_setup(tmp_path)
    out = tmp_path / "o"
    q = lambda _m: None  # noqa: E731
    mn = EV.run_eval_task(stub, CFG, settings, spec, ctx, sel, out, warmup=False, log=q)
    leafn = EV.leaf_dir(out, spec, "refseg", settings)
    assert all(r["procs"]["single"]["grid"]["exact"] for r in _jsonl(leafn / "scores.jsonl"))  # 替身跟着指令走：网格逐格正确
    full = mn["headline"]["cIoU"]["value"]
    assert full > 0.5
    n = len(sel.records)
    # none：全 0 → 崩；配对差 = 正常 − 检查 = 正常的 cIoU
    m0 = EV.run_instruction_check(stub, CFG, settings, spec, ctx, sel, out, check="none", warmup=False, log=q)
    leaf0 = EV.leaf_dir(out, spec, "refseg", settings, "none")
    assert leaf0.name == "refseg__instr_none"
    assert all(r["instruction"] == "" for r in _jsonl(leaf0 / "predictions.jsonl"))
    o = m0["targets"]["original"]
    assert set(m0["targets"]) == {"original"} and o["check"]["cIoU"]["value"] == 0.0 and o["normal"]["cIoU"]["value"] == full
    assert o["paired_normal_minus_check"]["cIoU"]["estimate"] == pytest.approx(full)
    assert o["paired_normal_minus_check"]["cIoU"]["n"] == n
    assert m0["normal"]["n_paired"] == n == m0["normal"]["n_verified_against_scores"]
    assert m0["instruction_check"]["n_skipped"] == 0
    # swap：预测换到新物体上（新目标逐格正确、原目标下降），没有候选的那条跳过
    m1 = EV.run_instruction_check(stub, CFG, settings, spec, ctx, sel, out, check="swap", warmup=False, log=q)
    leaf1 = EV.leaf_dir(out, spec, "refseg", settings, "swap")
    ic = m1["instruction_check"]
    assert ic["n_skipped"] == 1 and ic["n_called"] == n - 1 and m1["normal"]["n_paired"] == n - 1
    rows1 = _jsonl(leaf1 / "scores.jsonl")
    assert all(r["targets"]["swapped"]["check"]["grid"]["exact"] and r["targets"]["original"]["normal"]["grid"]["exact"]
               for r in rows1)
    sw, og = m1["targets"]["swapped"], m1["targets"]["original"]
    assert sw["check"]["cIoU"]["value"] > sw["normal"]["cIoU"]["value"]
    assert og["check"]["cIoU"]["value"] < og["normal"]["cIoU"]["value"]
    assert sw["paired_normal_minus_check"]["cIoU"]["estimate"] < 0 < og["paired_normal_minus_check"]["cIoU"]["estimate"]
    pairs = _jsonl(leaf1 / "swap_pairs.jsonl")
    assert [p["id"] for p in pairs] == [s.id for s in sel.records] and sum(p["skipped"] for p in pairs) == 1
    preds = _jsonl(leaf1 / "predictions.jsonl")
    by = {p["id"]: p for p in pairs}
    for r in preds:
        assert r["instruction"] == ctx.grid_template.format(expression=by[r["id"]]["expression"], n=N8)
    rows = _jsonl(leaf1 / "scores.jsonl")
    assert all(set(r["targets"]) == {"original", "swapped"} and r["candidate_id"] == by[r["id"]]["candidate_id"] for r in rows)
    ident = yaml.safe_load((leaf1 / "config.yaml").read_text())
    assert ident["instruction_check"]["kind"] == "swap" and ident["normal"]["predictions_sha256"]
    # 确定：续跑（同一身份）输出不变
    before = {f: (leaf1 / f).read_bytes() for f in DETERMINISTIC + ("swap_pairs.jsonl", "predictions.jsonl")}
    EV.run_instruction_check(stub, CFG, settings, spec, ctx, sel, out, check="swap", resume=True, warmup=False, log=q)
    assert {f: (leaf1 / f).read_bytes() for f in before} == before
    # --limit：只取前 N 条（批大小的整数倍）；正常一侧取已有全量结果的前 N 条
    spec4 = EV.EvalSpec(**{**spec.__dict__, "limit": 4})
    m4 = EV.run_instruction_check(stub, CFG, settings, spec4, ctx, sel, tmp_path / "o4", check="none", warmup=False, log=q,
                                  normal_leaf=EV.leaf_dir(out, spec, "refseg", settings))
    assert m4["n"] == 4 and m4["normal"]["n_paired"] == 4 and m4["normal"]["n"] == n


def test_instruction_check_normal_side_refusals(tmp_path):
    tokens, ctx, sel, spec, settings, stub = _stub_setup(tmp_path)
    q = lambda _m: None  # noqa: E731
    out = tmp_path / "o"
    with pytest.raises(FileNotFoundError):  # 没有正常评测
        EV.run_instruction_check(stub, CFG, settings, spec, ctx, sel, out, check="none", warmup=False, log=q)
    EV.run_eval_task(stub, CFG, settings, spec, ctx, sel, out, warmup=False, log=q)
    normal = EV.leaf_dir(out, spec, "refseg", settings)
    # 检查点不同
    spec_b = EV.EvalSpec(**{**spec.__dict__, "checkpoint": {"manifest_sha256": "other"}})
    with pytest.raises(ValueError, match="checkpoint"):
        EV.run_instruction_check(stub, CFG, settings, spec_b, ctx, sel, tmp_path / "b", check="none", normal_leaf=normal,
                                 warmup=False, log=q)
    # 解码不同（结构强制的检查配 restricted 的正常评测）
    s_st = EV.resolve_eval(CFG, tiny=True, k_values=[1], decode="structured")
    with pytest.raises(ValueError, match="eval.decode"):
        EV.run_instruction_check(stub, CFG, s_st, spec, ctx, sel, tmp_path / "c", check="none", normal_leaf=normal,
                                 warmup=False, log=q)
    # 样本顺序对不上（正常评测的前 N 条不是本次的 N 条）
    rev = LoadResult(list(reversed(sel.records)), sel.stats)
    with pytest.raises(ValueError, match="不能逐条配对"):
        EV.run_instruction_check(stub, CFG, settings, spec, ctx, rev, tmp_path / "d", check="none", normal_leaf=normal,
                                 warmup=False, log=q)
    # 只跑单轮
    with pytest.raises(ValueError, match="单轮"):
        EV.run_instruction_check(stub, CFG, EV.resolve_eval(CFG, tiny=True), spec, ctx, sel, tmp_path / "e", check="none",
                                 normal_leaf=normal, warmup=False, log=q)
    # 正常一侧的 scores.jsonl 与本次规则重算的不同：配对可能错位，拒绝
    bad = tmp_path / "bad"
    shutil.copytree(normal, bad)
    rows = _jsonl(bad / "scores.jsonl")
    rows[0]["procs"]["single"]["guided"]["inter"] += 1
    (bad / "scores.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    with pytest.raises(RuntimeError, match="错位"):
        EV.run_instruction_check(stub, CFG, settings, spec, ctx, sel, tmp_path / "f", check="none", normal_leaf=bad,
                                 warmup=False, log=q)
    # 服务器压缩过的正常一侧（predictions.jsonl.gz、scores.jsonl.gz）照样能读
    import gzip

    gzd = tmp_path / "gz"
    shutil.copytree(normal, gzd)
    for f in ("predictions.jsonl", "scores.jsonl"):
        (gzd / f"{f}.gz").write_bytes(gzip.compress((gzd / f).read_bytes()))
        (gzd / f).unlink()
    m = EV.run_instruction_check(stub, CFG, settings, spec, ctx, sel, tmp_path / "g", check="none", normal_leaf=gzd,
                                 warmup=False, log=q)
    assert m["normal"]["predictions_file"] == "predictions.jsonl.gz" and m["normal"]["n_verified_against_scores"] == len(sel.records)


def test_cli_instruction_check_rules(tmp_path):
    ev = _load_script("eval_e3b")
    args = _cli_args(REPO) + ["--out", str(tmp_path / "x")]  # --checkpoint 只核对是目录
    a = ev.parse_args(args + ["--instruction-check", "swap"])
    assert a.task == ["refseg"] and a.k_values == [1] and a.instruction_check == "swap"
    assert ev.parse_args(args + ["--instruction-check", "none", "--task", "refseg", "--k-values", "1"]).task == ["refseg"]
    a0 = ev.parse_args(args)
    assert a0.task == ["refseg", "depth"] and a0.k_values is None and a0.decode is None and a0.instruction_check is None
    assert ev.parse_args(args + ["--decode", "structured"]).decode == "structured"
    for bad in (["--instruction-check", "none", "--task", "depth"], ["--instruction-check", "none", "--task", "refseg", "depth"],
                ["--instruction-check", "swap", "--k-values", "1", "2"], ["--normal-leaf", "x"], ["--decode", "free"],
                ["--instruction-check", "all"]):
        with pytest.raises(SystemExit) as e:
            ev.parse_args(args + bad)
        assert e.value.code == 2, bad


def test_cli_tiny_instruction_checks_and_structured(tiny_ckpt, tmp_path):
    """--tiny 命令行：正常评测（单轮）→ none、swap；结构强制的正常评测 → 结构强制的 swap；--limit 不是批大小的整数倍报错。"""
    ev = _load_script("eval_e3b")
    out = tmp_path / "e"
    base = _cli_args(tiny_ckpt) + ["--out", str(out)]
    assert ev.main(base + ["--task", "refseg", "--k-values", "1"]) == 0
    for c in ("none", "swap"):
        assert ev.main(base + ["--instruction-check", c]) == 0
        leaf = _mdir(out) / f"refseg__instr_{c}"
        m = json.loads((leaf / "metrics.json").read_text())
        assert m["complete"] and m["instruction_check"]["kind"] == c and m["normal"]["n_verified_against_scores"] == m["n"] == 6
        assert set(m["targets"]) == ({"original"} if c == "none" else {"original", "swapped"})
    assert ev.main(base + ["--instruction-check", "none", "--limit", "3"]) == 2  # 批大小 2：3 不是整数倍
    assert ev.main(base + ["--task", "refseg", "--k-values", "1", "--decode", "structured"]) == 0
    assert ev.main(base + ["--instruction-check", "swap", "--decode", "structured"]) == 0
    leaf = _mdir(out) / "refseg__instr_swap__structured"
    m = json.loads((leaf / "metrics.json").read_text())
    assert m["decode"]["mode"] == "structured" and m["normal"]["decode"] == "structured"
    assert m["stages"]["round1"]["n_valid"] == m["stages"]["round1"]["n_called"] == 6
    sm = json.loads((_mdir(out) / "refseg__structured" / "metrics.json").read_text())
    assert sm["stages"]["round1"]["parse_failures"] == {} and sm["decode"]["mode"] == "structured"


# ---------------------------------------------------------------------------
# §3 LoRA 缩放的派生检查点
# ---------------------------------------------------------------------------


def _scale(ck: Path, s: float, root: Path) -> Path:
    out = root / f"s{s}" / "ckpt" / "final" / "model"
    sc = _load_script("scale_lora_ckpt")
    assert sc.main(["--checkpoint", str(ck), "--scale", str(s), "--out", str(out)]) == 0
    return out


def _merged_state(ck: Path) -> dict[str, torch.Tensor]:
    ad, _ = factory.load_trained(ck, device="cpu", merge=True)
    return {k: v.detach().clone() for k, v in ad.model.state_dict().items()}


@pytest.fixture(scope="module")
def scaled(tiny_ckpt, tmp_path_factory):
    root = tmp_path_factory.mktemp("scaled")
    return {s: _scale(tiny_ckpt, s, root) for s in (1.0, 0.0, 0.5)}


def test_scale_manifest_and_leaf_files(tiny_ckpt, scaled, tmp_path):
    src_man = read_manifest(tiny_ckpt)
    src_sha = hashlib.sha256((tiny_ckpt / "manifest.json").read_bytes()).hexdigest()
    for s, ck in scaled.items():
        man = read_manifest(ck)  # 逐文件 sha256 与字节数核对通过
        assert man["derived"]["from"] == src_sha and man["derived"]["lora_scale"] == s
        assert {k: v for k, v in man.items() if k not in ("derived", "files")} == \
            {k: v for k, v in src_man.items() if k != "files"}
        changed = sorted(k for k in man["files"] if man["files"][k] != src_man["files"][k])
        assert set(man["files"]) == set(src_man["files"])
        assert changed == ([] if s == 1.0 else ["lora/adapter_model.safetensors"]), (s, changed)
        leaf, src_leaf = ck.parents[2], tiny_ckpt.parents[2]
        assert (leaf / "config.yaml").read_bytes() == (src_leaf / "config.yaml").read_bytes()
        assert (leaf / "train_summary.json").read_bytes() == (src_leaf / "train_summary.json").read_bytes()
        d = json.loads((leaf / "derived.json").read_text())
        assert d["from_manifest_sha256"] == src_sha and d["lora_scale"] == s
    # eval_e3b 的运行身份：派生的记 derived，原生的不变
    ev = _load_script("eval_e3b")
    assert ev.checkpoint_block(scaled[0.5], read_manifest(scaled[0.5]))["derived"]["lora_scale"] == 0.5
    assert "derived" not in ev.checkpoint_block(tiny_ckpt, src_man)
    # 拒绝：派生的再派生、输出非空、布局不对、负的倍数
    sc = _load_script("scale_lora_ckpt")
    assert sc.main(["--checkpoint", str(scaled[0.5]), "--scale", "0.5", "--out", str(tmp_path / "a/ckpt/final/model")]) == 2
    assert sc.main(["--checkpoint", str(tiny_ckpt), "--scale", "0.5", "--out", str(scaled[1.0])]) == 2
    assert sc.main(["--checkpoint", str(tiny_ckpt), "--scale", "0.5", "--out", str(tmp_path / "b/model")]) == 2
    assert sc.main(["--checkpoint", str(tiny_ckpt), "--scale", "-1", "--out", str(tmp_path / "c/ckpt/final/model")]) == 2
    assert not (tmp_path / "b").exists() and not (tmp_path / "c").exists()


def test_scale_1_merged_identical(tiny_ckpt, scaled):
    a, b = _merged_state(tiny_ckpt), _merged_state(scaled[1.0])
    assert set(a) == set(b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_scale_0_merged_equals_base_except_grid_rows(tiny_ckpt, scaled):
    ad, _ = factory.load_trained(scaled[0.0], device="cpu", merge=True)
    merged0 = {k: v.detach().clone() for k, v in ad.model.state_dict().items()}
    grid_rows = sorted(int(i) for i in ad.grid_token_ids.values())
    spec = read_manifest(tiny_ckpt)["build_spec"]
    base = make_adapter(spec["key"], device="cpu", dtype=torch.float32, image_policy=parse_policy(spec["image_policy"]),
                        tiny=bool(spec["tiny"]), **dict(spec["family_kw"]))  # 刚构造的 base（没加网格 token、没有 LoRA）
    bs = base.model.state_dict()
    assert set(bs) == set(merged0)
    n_diff = 0
    for k, v in bs.items():
        w = merged0[k]
        if v.shape == w.shape and torch.equal(v, w):
            continue
        # 只有嵌入 / lm_head 可以不同，而且只在网格 token 行上（扩表时多出来的行也只能是网格 token 行）
        assert v.dim() == 2 and w.shape[1] == v.shape[1] and w.shape[0] >= v.shape[0], k
        keep = torch.ones(w.shape[0], dtype=torch.bool)
        keep[grid_rows] = False
        assert torch.equal(w[: v.shape[0]][keep[: v.shape[0]]], v[keep[: v.shape[0]]]), k
        assert not keep[v.shape[0]:].any(), k
        n_diff += 1
    assert n_diff >= 1  # 网格 token 行是训练过的值，与 base 不同


def test_scale_half_delta_is_half(tiny_ckpt, scaled):
    ad_u, _ = factory.load_trained(tiny_ckpt, device="cpu", merge=False)
    mods = dict(ad_u.model.named_modules())
    m1, m5 = _merged_state(tiny_ckpt), _merged_state(scaled[0.5])
    n_checked = 0
    for name in ad_u.lora_target_modules()[:3]:  # 抽前三个模块
        k = f"{name}.weight"
        assert k in m1 and k in m5, k
        w_base = mods[name].base_layer.weight.detach()
        d1, d5 = m1[k] - w_base, m5[k] - w_base
        assert d1.abs().max() > 0, f"{name}：原检查点的增量为 0，测试没有意义"
        err = (d5 - 0.5 * d1).abs().max().item()
        assert err <= 1e-6, (name, err)
        n_checked += 1
    assert n_checked == 3
