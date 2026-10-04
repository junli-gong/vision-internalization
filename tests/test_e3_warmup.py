"""DEV-0035 §2：网格格式预热（src/e3/warmup.py；scripts/train_e3b.py --supervision warmup）。

- 配置：warmup 节的数字（混合 1:1、阈值 0.01、连续 2 次、留出集 200 条、每 100 步、上限 2000 步）与模板的占位符钉住；resolve_run 的
  warmup 一节；e3b / c3 的 resolve_run 不多键；
- 生成器：指令文字由一个独立的解析器读回来，与目标网格逐格相同（分割三种、深度）；确定（同一个种子同一份）、换种子就换；留出集与运行
  种子无关、不进训练（重抽）；目标是 E3b 同一套 token、同一种格式（长度 1057、解析回原网格）；输入没有图像（build_inputs 的视觉 token 为 0）；
- 停止规则：StopRule 的判定；训练循环里（读数用假的，按步给定格式错误率）连续两次 < 1% 就停在那一步、存 final、记下步数；到了上限
  不满足照样存、记 stop_rule_met = false；中途崩溃后续跑与一次跑完逐字节相同；
- 真读数：留出集上的受限解码数格式错误（evaluate_heldout），小模型上能跑通；小模型在 4×4 的预热上训练到停止规则按设计触发；
- 命令行：三个家族 --tiny 跑通，检查点能被 eval_e3b 与 run_e1 --checkpoint 加载。
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.adapters import registry
from src.adapters.checkpoint import policy_spec
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.constants import DEPTH_NUM_BINS
from src.e3 import data as D
from src.e3 import warmup as W
from src.e3.config import ConfigError, load_config, resolve_run
from src.e3.targets import SEQ_LEN, parse_target, seq_len
from src.e3.tokens import add_vocab
from src.e3.train import RunSpec, Trainer, leaf_dir, run_training

REPO = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
CFG = load_config()
N = CFG.grid_n


def _tiny(key):
    fam = registry.get_model(key).family
    t = CFG.doc["tiny"]
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _spec(n: int = N, **over) -> W.WarmupSpec:
    doc = copy.deepcopy(dict(CFG.doc))
    for k, v in over.items():
        sec, key = k.split("__")
        doc["warmup"][sec][key] = v
    return W.WarmupSpec.from_doc(doc, n=n)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_pins():
    w = CFG.doc["warmup"]
    assert w["mix"] == {"refseg": 1, "depth": 1} == CFG.doc["mix"]["weights"]  # 与 E3b 的混合相同（basis 里写的）
    assert w["stop"] == {"eval_every": 100, "threshold": 0.01, "consecutive": 2, "max_steps": 2000}
    assert w["heldout"] == {"n": 200, "seed": 0}
    assert w["refseg"]["kinds"] == {"rect": 1, "rects": 1, "shape": 1}
    # 模板的结尾与 E3b 的网格指令相同
    assert w["refseg"]["template"].endswith(CFG.doc["tasks"]["refseg"]["grid_template"].split(". ")[-1])
    assert w["depth"]["template"].endswith(CFG.doc["tasks"]["depth"]["grid_instruction"].split(". ")[-1])
    rc = resolve_run(CFG, supervision="warmup", tiny=False)
    assert rc["steps"] == 2000 and rc["variant"] == "warmup" and rc["correction_fraction"] is None
    assert rc["warmup"] == {"version": w["version"], "eval_every": 100, "threshold": 0.01, "consecutive": 2, "max_steps": 2000,
                            "heldout_n": 200, "heldout_batch_size": CFG.doc["eval"]["batch_size"]}
    assert rc["warmup_steps"] == 60 and rc["lora"] == resolve_run(CFG, supervision="grid", tiny=False)["lora"]
    assert rc["optim"] == resolve_run(CFG, supervision="grid", tiny=False)["optim"]  # LoRA 与 token 行同 E3b 的网格监督
    t = resolve_run(CFG, supervision="warmup", tiny=True)
    assert (t["steps"], t["warmup"]["eval_every"], t["warmup"]["heldout_n"]) == (6, 2, 4)
    for sup in ("grid", "text"):  # e3b、c3 的运行身份不多键
        assert set(resolve_run(CFG, supervision=sup, tiny=False)) == {
            "supervision", "variant", "steps", "micro_batch_size", "grad_accum_steps", "ckpt_every", "lora", "optim",
            "loss_reduction", "gradient_checkpointing", "enable_thinking", "allowed_scope", "mix", "correction_fraction",
            "data_workers", "batch_size", "warmup_steps"}


OLD_CONFIG = REPO / "tests" / "fixtures" / "e3" / "e3b_train_3bff3604.json"  # D67 的 E3b 配置原样


def test_config_only_adds_c1prime_and_warmup():
    """DEV-0035 只为 C1′ 与预热新增条目，已有条目逐项不变（与 D67 的版本 3bff3604… 比较）。"""
    import hashlib

    assert hashlib.sha256(OLD_CONFIG.read_bytes()).hexdigest() == (
        "3bff36043ace0a63161b106ed92aa9bff8e674062140665ccdd76965faf064b6")
    old = json.loads(OLD_CONFIG.read_text(encoding="utf-8"))
    assert set(CFG.doc) - set(old) == {"c1prime", "warmup"} and not set(old) - set(CFG.doc)
    assert all(CFG.doc[k] == old[k] for k in old)
    assert list(CFG.doc)[-3:] == ["c1prime", "warmup", "tiny"]


@pytest.mark.parametrize("mutate,match", [
    (lambda w: w["refseg"].__setitem__("rect", "Rows {rows} are foreground."), "占位符"),
    (lambda w: w["refseg"]["kinds"].__setitem__("blob", 1), "kinds"),
    (lambda w: w["refseg"].__setitem__("shapes", ["disk", "star"]), "shapes"),
    (lambda w: w["stop"].__setitem__("threshold", 0.0), "stop"),
    (lambda w: w["depth"].__setitem__("template", "{n} {regions}"), "占位符"),
])
def test_config_rejects_bad_warmup(tmp_path, mutate, match):
    doc = copy.deepcopy(dict(CFG.doc))
    mutate(doc["warmup"])
    p = tmp_path / "c.json"
    p.write_text(json.dumps(doc, ensure_ascii=False))
    with pytest.raises(ConfigError, match=match):
        load_config(p)


# ---------------------------------------------------------------------------
# 生成器：文字 ↔ 网格
# ---------------------------------------------------------------------------


def _ranges(s: str) -> list[tuple[int, int]]:
    out = []
    for part in s.split(" and "):
        a, b = part.split("-")
        out.append((int(a), int(b)))
    return out


def read_back(it: W.WarmupItem, n: int) -> np.ndarray:
    """独立于生成器的解析：只按指令文字把网格画出来（措辞见配置的 warmup 节）。"""
    t = it.instruction
    head = f"A {n}x{n} grid has rows and columns numbered 0 to {n - 1}, starting from the top-left cell. "
    if it.task == "refseg":
        assert t.startswith(head) and t.endswith(f" All other cells are background. Output the {n}x{n} mask grid.")
        body = t[len(head): -len(f" All other cells are background. Output the {n}x{n} mask grid.")]
        g = np.zeros((n, n), dtype=np.int64)
        if it.kind == "rect":
            m = re.fullmatch(r"Rows (\d+)-(\d+), columns (\d+)-(\d+) are foreground\.", body)
            assert m, body
            r0, r1, c0, c1 = map(int, m.groups())
            g[r0:r1 + 1, c0:c1 + 1] = 1
        elif it.kind == "rects":
            assert body.startswith("Foreground cells: rows ") and body.endswith(".")
            regs = re.findall(r"rows (\d+)-(\d+), columns (\d+)-(\d+)", body)
            assert len(regs) == body.count("; ") + 1 >= 2
            for r0, r1, c0, c1 in regs:
                g[int(r0):int(r1) + 1, int(c0):int(c1) + 1] = 1
        else:
            assert body.startswith("Foreground cells, row by row: row ") and body.endswith(".")
            for part in body[len("Foreground cells, row by row: "):-1].split("; "):
                m = re.fullmatch(r"row (\d+): columns ([0-9\- and]+)", part)
                assert m, part
                for a, b in _ranges(m.group(2)):
                    g[int(m.group(1)), a:b + 1] = 1
        return g
    dh = (f"A {n}x{n} depth grid has rows and columns numbered 0 to {n - 1}, starting from the top-left cell, and depth levels "
          f"numbered 0 (nearest) to {DEPTH_NUM_BINS - 1} (farthest). Every cell is at level ")
    tail = f". Later regions override earlier ones. Output the {n}x{n} depth grid."
    assert t.startswith(dh) and t.endswith(tail)
    body = t[len(dh): -len(tail)]
    m = re.fullmatch(r"(\d+) except (.*)", body)
    assert m, body
    g = np.full((n, n), int(m.group(1)), dtype=np.int64)
    regs = re.findall(r"rows (\d+)-(\d+), columns (\d+)-(\d+) at level (\d+)", m.group(2))
    assert len(regs) == m.group(2).count("; ") + 1
    for r0, r1, c0, c1, lv in regs:
        g[int(r0):int(r1) + 1, int(c0):int(c1) + 1] = int(lv)
    return g


def test_text_reads_back_to_target_grid():
    spec = _spec()
    seen = set()
    for k in range(600):
        task = "refseg" if k % 2 == 0 else "depth"
        it = W.make_item(spec, task, np.random.default_rng([7, k]))
        seen.add((it.task, it.kind, it.info.get("shape")))
        assert it.grid.shape == (N, N) and it.grid.dtype == np.int64
        assert np.array_equal(read_back(it, N), it.grid), (it.kind, it.instruction)
        if it.task == "refseg":
            assert set(np.unique(it.grid)) <= {0, 1} and it.grid.sum() >= 1
        else:
            assert 0 <= it.grid.min() and it.grid.max() < DEPTH_NUM_BINS
    assert seen == {("refseg", "rect", None), ("refseg", "rects", None), ("depth", "regions", None)} | {
        ("refseg", "shape", s) for s in W.SHAPES}


def test_shapes_and_ranges_are_as_documented():
    g = W._shape_mask("triangle", 8, 2, 3, 2)
    assert g.tolist()[2:5] == [[0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0]] and g.sum() == 9
    assert W._shape_mask("diamond", 8, 0, 0, 1).sum() == 3  # 出界的格子裁掉
    assert W._shape_mask("disk", 9, 4, 4, 2).sum() == 13
    spec = _spec()
    assert W._range(spec, 5, 5) == "5-5" and W._range(spec, 3, 12) == "3-12"
    assert W._pick(np.random.default_rng(0), {"a": 1}) == "a"


def test_generator_deterministic_and_seeded():
    spec = _spec()
    tok = add_vocab(_tiny("qwen3.5-0.8b"), CFG.vocab)
    a = W.WarmupSource(spec, seed=0, tokens=tok, batch_size=4, orders={t: "instruction_first" for t in ("refseg", "depth")})
    b = W.WarmupSource(spec, seed=0, tokens=tok, batch_size=4, orders={t: "instruction_first" for t in ("refseg", "depth")})
    c = W.WarmupSource(spec, seed=1, tokens=tok, batch_size=4, orders={t: "instruction_first" for t in ("refseg", "depth")})
    ea = [a.build(p) for p in a.step_plans(3)]
    eb = [b.build(p) for p in b.step_plans(3)]
    ec = [c.build(p) for p in c.step_plans(3)]
    assert [(e.instruction, e.target_ids) for e in ea] == [(e.instruction, e.target_ids) for e in eb]
    assert [e.instruction for e in ea] != [e.instruction for e in ec]
    assert [e.plan.task for e in ea] == ["refseg", "depth", "refseg", "depth"]  # 1:1 的轮转
    # 留出集与运行种子无关，是配置给的条数，按 mix 各半
    assert W.items_digest(a.heldout) == W.items_digest(c.heldout) and len(a.heldout) == 200
    assert sum(it.task == "refseg" for it in a.heldout) == 100
    for e in ea:
        assert e.images == [] and len(e.target_ids) == SEQ_LEN == seq_len(N) and e.weights == [1] * SEQ_LEN
        assert e.target_ids[-1] == tok.stop_id
        p = parse_target(e.plan.task, e.target_ids, tok)
        assert p.failure is None and np.array_equal(p.grid, a.item(e.plan.k)[0].grid)
        assert set(e.target_ids) <= set(tok.allowed_ids(e.plan.task, CFG.allowed_scope))  # 同一套 token、同一个允许集合


def test_heldout_never_in_training(monkeypatch):
    """训练样本的指令与留出集某条逐字相同时重抽（确定的）。用很窄的随机范围逼出撞车。"""
    spec = _spec(n=N, refseg__kinds={"rect": 1})
    tok = add_vocab(_tiny("qwen3.5-0.8b"), CFG.vocab)
    src = W.WarmupSource(spec, seed=0, tokens=tok, batch_size=4, orders={t: "instruction_first" for t in ("refseg", "depth")})
    held = set(src._heldout_texts)
    # 把留出集的第一条硬塞成训练样本 k 的第一次抽样：重抽之后不在留出集里
    first = src.heldout[0]
    real = W.make_item

    def fake(spec_, task, rng, _calls=[0]):
        _calls[0] += 1
        return first if _calls[0] == 1 else real(spec_, task, rng)

    monkeypatch.setattr(W, "make_item", fake)
    it, attempt = src.item(0)
    assert attempt == 1 and it.instruction not in held
    monkeypatch.setattr(W, "make_item", real)
    assert all(src.item(k)[0].instruction not in held for k in range(400))


def test_inputs_have_no_images():
    ad = _tiny("internvl3.5-2b-hf")
    tok = add_vocab(ad, CFG.vocab)
    src = W.WarmupSource(_spec(), seed=0, tokens=tok, batch_size=2, orders={t: "instruction_first" for t in ("refseg", "depth")})
    for e in [src.build(p) for p in src.step_plans(0)]:
        inp = ad.build_inputs(e.images, e.instruction, e.order)
        assert inp.images == ()
        out = ad.training_forward([inp], [e.target_ids], enable_thinking=False)
        assert list(out.target_lengths) == [SEQ_LEN]


# ---------------------------------------------------------------------------
# 停止规则
# ---------------------------------------------------------------------------


def test_stop_rule_unit():
    r = W.StopRule(threshold=0.01, consecutive=2)
    h = lambda *xs: [{"format_error_rate": x} for x in xs]  # noqa: E731
    assert not r.met(h()) and not r.met(h(0.0)) and not r.met(h(0.0, 0.5)) and not r.met(h(0.0, 0.01))  # 0.01 不算低于
    assert r.met(h(0.5, 0.0, 0.005)) and r.met(h(0.009, 0.0)) and not r.met(h(0.0, 0.0, 0.02))
    assert not r.below(None)
    assert W.StopRule(threshold=0.01, consecutive=1).met(h(0.5, 0.0))


class FakeEval:
    """按调用次序给定格式错误率（续跑时从已做过的读数次数接着数）。"""

    def __init__(self, rates, done: int = 0):
        self.rates, self.k = list(rates), done

    def __call__(self, ad, tokens, items):
        r = self.rates[self.k]
        self.k += 1
        nf = int(round(r * len(items)))
        return {"n": len(items), "n_format_errors": nf, "format_error_rate": r, "failures": {}, "by_task": {},
                "generated_tokens": 0, "note": "fake"}


def _run_spec(key, ad):
    e = registry.get_model(key)
    return RunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                   image_policy=policy_spec(ad.image_policy), model_dtype="fp32", supervision="warmup", seed=0)


def _rc(steps=12, every=2, ckpt_every=4):
    rc = resolve_run(CFG, supervision="warmup", tiny=True)
    rc["steps"], rc["ckpt_every"], rc["data_workers"] = steps, ckpt_every, 0
    rc["warmup"] = {**rc["warmup"], "eval_every": every, "max_steps": steps}
    return rc


def _patched_cfg(tmp_path, steps, every):
    """run_training 按配置的 warmup.tiny 核对 max_steps：测试用一份改了 tiny 节的配置。"""
    doc = copy.deepcopy(dict(CFG.doc))
    doc["warmup"]["tiny"].update(max_steps=steps, eval_every=every)
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(doc, ensure_ascii=False))
    return load_config(p)


def test_stop_rule_stops_training_and_resume_identical(tmp_path):
    cfg = _patched_cfg(tmp_path, 12, 2)
    rc = resolve_run(cfg, supervision="warmup", tiny=True)
    rc["ckpt_every"], rc["data_workers"] = 4, 0
    rates = [0.5, 0.005, 0.25, 0.0, 0.0, 0.0]  # 第 2、4、6、8、10、12 步：第 10 步是连续第二次 < 1%
    paths = D.E3Paths.fixtures()
    key = "qwen3.5-0.8b"
    ad = _tiny(key)
    a = run_training(ad, cfg, _run_spec(key, ad), rc, paths, tmp_path / "A", log=lambda m: None, _warmup_eval=FakeEval(rates))
    la = leaf_dir(tmp_path / "A", "warmup", key, 0)
    w = a["warmup"]
    assert a["complete"] and a["steps"] == 10 and w["stopped_at_step"] == 10 and w["stop_rule_met"] is True
    assert w["reason"] == "format_error_below_threshold" and w["n_evals"] == 5 and w["max_steps"] == 12
    assert [h["step"] for h in w["history"]] == [2, 4, 6, 8, 10]
    log = [json.loads(x) for x in (la / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log] == list(range(10))
    ev = [json.loads(x) for x in (la / "warmup_eval.jsonl").read_text().splitlines()]
    assert [(e["step"], e["below_threshold"], e["stop_rule_met"]) for e in ev] == [
        (2, False, False), (4, True, False), (6, False, False), (8, True, False), (10, True, True)]
    assert (la / "ckpt" / "final" / "model" / "manifest.json").exists() and not (la / "ckpt" / "latest").exists()
    assert "stop_rule_met = True" in (la / "README.md").read_text()
    acc = json.loads((la / "accounting.json").read_text())
    assert acc["warmup_eval_n"] == 5 and acc["warmup_eval_s"] >= 0
    # 中途崩溃（第 6 步之后；最近的检查点在第 4 步）→ 续跑：与一次跑完逐字节相同
    ad = _tiny(key)
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        run_training(ad, cfg, _run_spec(key, ad), rc, paths, tmp_path / "B", log=lambda m: None, _warmup_eval=FakeEval(rates),
                     _fail_after_step=6)
    ad = _tiny(key)
    b = run_training(ad, cfg, _run_spec(key, ad), rc, paths, tmp_path / "B", resume=True, log=lambda m: None,
                     _warmup_eval=FakeEval(rates, done=2))
    lb = leaf_dir(tmp_path / "B", "warmup", key, 0)
    for f in ("train_log.jsonl", "warmup_eval.jsonl", "config.yaml", "README.md", "train_summary.json"):
        assert (la / f).read_bytes() == (lb / f).read_bytes(), f
    assert a == b


def test_max_steps_without_meeting_rule(tmp_path):
    cfg = _patched_cfg(tmp_path, 6, 2)
    rc = resolve_run(cfg, supervision="warmup", tiny=True)
    rc["data_workers"] = 0
    key = "gemma-4-e2b-it"
    ad = _tiny(key)
    s = run_training(ad, cfg, _run_spec(key, ad), rc, D.E3Paths.fixtures(), tmp_path / "o", log=lambda m: None,
                     _warmup_eval=FakeEval([0.0, 0.5, 0.0]))
    assert s["steps"] == 6 and s["warmup"]["stop_rule_met"] is False and s["warmup"]["reason"] == "max_steps"
    assert s["warmup"]["last"] == {"step": 6, "n": 4, "n_format_errors": 0, "format_error_rate": 0.0}
    assert (leaf_dir(tmp_path / "o", "warmup", key, 0) / "ckpt" / "final" / "model" / "lora").is_dir()


def test_run_cfg_must_match_spec(tmp_path):
    key = "qwen3.5-0.8b"
    ad = _tiny(key)
    rc = resolve_run(CFG, supervision="grid", tiny=True)
    with pytest.raises(ValueError, match="不符"):
        run_training(ad, CFG, _run_spec(key, ad), rc, D.E3Paths.fixtures(), tmp_path, log=lambda m: None)


# ---------------------------------------------------------------------------
# 真读数：留出集上的受限解码
# ---------------------------------------------------------------------------


class _Res:
    def __init__(self, ids):
        self.token_ids, self.n_generated_tokens = tuple(ids), len(ids)


class _Scripted:
    """build_inputs 记下文字；generate_restricted 照剧本返回（核对格式错误的计数）。"""

    def __init__(self, outs):
        self.outs = list(outs)
        self.calls = []

    def build_inputs(self, images, text, order):
        assert images == []
        return text

    def generate_restricted(self, inputs, allowed, *, max_new_tokens, stop_ids, enable_thinking):
        self.calls.append((list(inputs), list(allowed), max_new_tokens, list(stop_ids)))
        return [_Res(self.outs.pop(0)) for _ in inputs]


def test_evaluate_heldout_counts_format_errors():
    tok = add_vocab(_tiny("qwen3.5-0.8b"), CFG.vocab)
    spec = _spec()
    items = W.heldout_items(spec)[:6]  # 分割、深度交替
    good = [W.target_ids(it, tok, N) for it in items]
    bad_rows = list(good[2])
    bad_rows[N] = tok.value_ids("refseg")[0]  # 第一行末尾不是行分隔：列数不对
    wrong_cell = list(good[0])
    wrong_cell[0] = tok.value_ids("refseg")[1 - items[0].grid[0, 0]]  # 格式对、内容错一格
    outs = {0: wrong_cell, 1: good[1], 2: bad_rows, 3: good[3][:-1], 4: good[4], 5: good[5][:SEQ_LEN - 34] + [tok.stop_id]}
    # 按任务分组、组内按顺序：refseg 是 0、2、4，depth 是 1、3、5
    ad = _Scripted([outs[i] for i in (0, 2, 4, 1, 3, 5)])
    ev = W.evaluate_heldout(ad, tok, items, n=N, scope="task", orders={"refseg": "instruction_first", "depth": "instruction_first"},
                            enable_thinking=False, batch_size=2)
    assert ev["n"] == 6 and ev["n_format_errors"] == 3 and ev["format_error_rate"] == 0.5
    assert ev["failures"] == {"row_separator": 1, "truncated": 1, "wrong_length": 1}
    assert ev["by_task"]["refseg"]["n_format_errors"] == 1 and ev["by_task"]["depth"]["n_format_errors"] == 2
    assert ev["by_task"]["refseg"]["exact_rate"] == pytest.approx(1 / 3)
    assert ev["by_task"]["refseg"]["cell_accuracy"] == pytest.approx((N * N - 1 + N * N) / (3 * N * N))
    assert [len(c[0]) for c in ad.calls] == [2, 1, 2, 1]  # 批大小 2，按任务分组
    assert all(c[2] == SEQ_LEN and c[3] == [tok.stop_id] for c in ad.calls)
    assert ad.calls[0][1] == tok.allowed_ids("refseg", "task") and ad.calls[2][1] == tok.allowed_ids("depth", "task")


def test_evaluate_heldout_on_tiny_model():
    ad = _tiny("gemma-4-e2b-it")
    tok = add_vocab(ad, CFG.vocab)
    ad.eval_mode()
    items = W.heldout_items(W.WarmupSpec.from_doc(CFG.doc, n=N, tiny=True))
    ev = W.evaluate_heldout(ad, tok, items, n=N, scope="task", orders={"refseg": "instruction_first", "depth": "instruction_first"},
                            enable_thinking=False, batch_size=2)
    assert ev["n"] == 4 and 0 <= ev["format_error_rate"] <= 1 and set(ev["by_task"]) == {"refseg", "depth"}
    assert ev["n_format_errors"] == sum(ev["failures"].values())


SMALL_N = 4


def test_tiny_model_learns_format_and_rule_triggers():
    """小模型（Qwen，4 层、隐藏 64）在 4×4 的预热上训练（lr 放大、批 4），每 20 步在留出集（8 条，与训练不重合）上数格式错误，
    停止规则按设计触发（连续两次 < 1%）。实测：第 180 步起格式错误一直是 0（之后 11 次读数都是 0），第 200 步停；内容并没有学会
    （分割逐格 0.55，深度约 0），说明规则量的确实只是格式。8×8 上同样的设置 400 步内学不会格式（一直截断），所以测试用 4×4；
    32×32 的格式由 test_text_reads_back_to_target_grid 与 targets 的测试钉住。"""
    doc = copy.deepcopy(dict(CFG.doc))
    doc["warmup"]["refseg"].update(rects_max_side=2, shape_size=[1, 1])
    doc["warmup"]["tiny"]["heldout_n"] = 8
    spec = W.WarmupSpec.from_doc(doc, n=SMALL_N, tiny=True)
    key = "qwen3.5-0.8b"
    ad = _tiny(key)
    rc = resolve_run(CFG, supervision="warmup", tiny=True)
    rc["optim"] = copy.deepcopy(rc["optim"])
    rc["optim"]["lora"]["lr"], rc["optim"]["token_rows"]["lr"], rc["optim"]["min_lr_ratio"] = 5e-4, 5e-3, 1.0
    rc["lora"] = {**rc["lora"], "dropout": 0.0}
    rc["steps"], rc["warmup_steps"], rc["micro_batch_size"], rc["grad_accum_steps"], rc["batch_size"] = 400, 0, 4, 1, 4
    tr = Trainer(ad, CFG, rc, supervision="warmup", seed=0)
    tr.setup_fresh()
    orders = {"refseg": "instruction_first", "depth": "instruction_first"}
    src = W.WarmupSource(spec, seed=0, tokens=tr.tokens, batch_size=4, orders=orders)
    rule = W.StopRule(threshold=0.01, consecutive=2)
    hist = []
    ad.train_mode()
    for step in range(400):
        tr.optimize([src.build(p) for p in src.step_plans(step)])
        if (step + 1) % 20 == 0:
            ad.eval_mode()
            hist.append(W.evaluate_heldout(ad, tr.tokens, src.heldout, n=SMALL_N, scope="task", orders=orders,
                                           enable_thinking=False, batch_size=4))
            ad.train_mode()
            if rule.met(hist):
                break
    assert rule.met(hist), [h["format_error_rate"] for h in hist]
    assert hist[0]["format_error_rate"] > 0.01  # 开始时格式不对：规则不是一开始就满足


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / f"{name}.py"), *args], capture_output=True, text=True, cwd=REPO)


def test_cli_three_families_and_checkpoint_loads(tmp_path):
    out = tmp_path / "w"
    p = run_script("train_e3b", "--tiny", "--model", *FAMILIES, "--supervision", "warmup", "--seed", "0", "--out", str(out))
    assert p.returncode == 0, p.stderr[-3000:]
    assert p.stdout.count("stop_rule_met=") == 3
    for key in FAMILIES:
        leaf = out / "warmup" / key / "seed0"
        s = json.loads((leaf / "train_summary.json").read_text())
        assert s["variant"] == "warmup" and s["warmup"]["n_evals"] == 3 and s["warmup"]["max_steps"] == 6
        files = sorted(str(x.relative_to(leaf / "ckpt" / "final" / "model")) for x in (leaf / "ckpt" / "final" / "model").rglob("*")
                       if x.is_file())
        assert files == ["lora/adapter_config.json", "lora/adapter_model.safetensors", "manifest.json", "token_rows.safetensors",
                         "tokens.json"]  # 与 E3b 同一种检查点
        ev = [json.loads(x) for x in (leaf / "warmup_eval.jsonl").read_text().splitlines()]
        assert [e["step"] for e in ev] == [2, 4, 6] and all(e["n"] == 4 for e in ev)
        log = [json.loads(x) for x in (leaf / "train_log.jsonl").read_text().splitlines()]
        assert all(r["n_visual_tokens"] == 0 and r["n_image_pixels"] == 0 for r in log)  # 输入没有图像
        assert set().union(*(r["n_by_kind"] for r in log)) <= {"rect", "rects", "shape", "regions"}
    ck = out / "warmup" / FAMILIES[0] / "seed0" / "ckpt" / "final" / "model"
    p = run_script("eval_e3b", "--tiny", "--checkpoint", str(ck), "--model", FAMILIES[0], "--variant", "warmup", "--seed", "0",
                   "--task", "depth", "--limit", "1", "--out", str(tmp_path / "e"))
    assert p.returncode == 0, p.stderr[-3000:]
    mc = json.loads((tmp_path / "e" / "eval" / "warmup" / FAMILIES[0] / "seed0" / "merge_check.json").read_text())
    assert mc["pass"] is True and list(mc["checks"]) == ["merge", "restricted"]
    p = run_script("run_e1", "--tiny", "--model", FAMILIES[0], "--task", "grid16", "--limit", "1", "--checkpoint", str(ck),
                   "--checkpoint-label", "t_text/warmup/seed0", "--out", str(tmp_path / "t"))
    assert p.returncode == 0, p.stderr[-3000:]
    assert (tmp_path / "t" / "t_text" / "warmup" / "seed0" / FAMILIES[0] / "grid16" / "metrics.json").exists()


def test_cli_rules(tmp_path):
    p = run_script("train_e3b", "--tiny", "--model", FAMILIES[0], "--supervision", "warmup", "--variant", "c1prime", "--seed", "0",
                   "--out", str(tmp_path / "x"))
    assert p.returncode == 2 and "c1prime" in p.stderr
    # 真模型的预热不需要 --data-root（不读训练池），目录名照样要带运行编号
    p = run_script("train_e3b", "--model", FAMILIES[0], "--supervision", "warmup", "--seed", "0", "--out", str(tmp_path / "noid"))
    assert p.returncode == 2 and "运行编号" in p.stderr and "要给 --data-root" not in p.stderr
    p = run_script("train_e3b", "--model", FAMILIES[0], "--supervision", "grid", "--seed", "0", "--out", str(tmp_path / "0040_x"))
    assert p.returncode == 2 and "要给 --data-root" in p.stderr  # 网格监督照旧要
