"""DEV-0038 §1（D77-3）：E5 主比较的小档修正——步数下限、val 选检查点、配种子（CPU fp32 小模型，--tiny）。

- 步数：1K、4K 档 = 1,000 步（按 epoch 是 125、500 步），16K、64K 不变；学习率调度按实际步数排；下限生效时日程按步数排、每步满批，
  前 2 个 epoch 与不设下限时逐条相同；
- val 选择：挑的是 val 上分数最高的一步（同分取早），final/ 与那一步的模型逐字节相同；训练进程里生成器从不生成 test 样本（生成器的
  划分核对处记下全部调用），val 的入口断言拒绝 test；规则 mean_headline_v1 的算法；
- 配种子：起点检查点的种子与 --seed 不同时拒绝训练，--allow-seed-mismatch 放开并记进运行身份；核对不了（不在训练叶目录下）也拒绝；
- 续跑：带 val 选择的训练中断（停在两次评测之间、停在评测那一步、残留一个没被 latest 记下的候选）后 --resume，输出与一次跑完逐字节相同。
DEV-0043 起 val 选择默认用结构强制解码；本文件测的是选择与续跑的流程，带 val 选择的档一律加 --decode restricted（快；受限解码的输出与改动前
逐字节相同）。结构强制下的 val 选择见 tests/test_e5_v3.py。
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import yaml

import src.synth.api as SYNTH_API
from src.e5 import val_select as VS
from src.e5.protocol import E5_TASKS, EvalItem, in_headline
from src.e5.train_config import load_train_config, plan_steps, resolve_run
from src.e5.train_data import Schedule, TrainSet

REPO = Path(__file__).resolve().parents[1]
CFG = load_train_config()
KEY = "qwen3.5-0.8b"
_SCRIPTS: dict[str, object] = {}


def _script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_e5small", REPO / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.path[:] = saved
        _SCRIPTS[name] = mod
    return _SCRIPTS[name]


def _leaf(out: Path, tag: str = "n10", seed: int = 0, init: str = "base") -> Path:
    return out / "train" / init / KEY / tag / f"seed{seed}"


def _train(out: Path, *extra: str, n: int = 10, seed: int = 0, init: str = "base") -> int:
    dec = ("--decode", "restricted") if CFG.selection(n, True) is not None else ()  # DEV-0043（模块说明）；没有 val 选择的档不能给
    return _script("train_e5").main(["--tiny", "--init", init, "--model", KEY, "--n-train", str(n), "--seed", str(seed),
                                     "--out", str(out), *dec, *extra])


# ---------------------------------------------------------------------------
# 步数
# ---------------------------------------------------------------------------


def test_min_steps_floor():
    rc = {n: resolve_run(CFG, n_train=n, n_examples=n, tiny=False) for n in CFG.tiers(False)}
    assert rc[1000]["steps"] == 1000 and rc[1000]["steps_by_epochs"] == 125  # 1K 档 = 1,000 步
    assert rc[4000]["steps"] == 1000 and rc[4000]["steps_by_epochs"] == 500
    assert rc[16000]["steps"] == 2000 == rc[16000]["steps_by_epochs"] and rc[64000]["steps"] == 8000  # 16K、64K 不变（2 个 epoch）
    assert [rc[n]["min_steps_binding"] for n in CFG.tiers(False)] == [True, True, False, False]
    assert rc[1000]["n_samples"] == 16000 and rc[4000]["n_samples"] == 16000  # 每步满批：16、4 个 epoch
    assert rc[16000]["n_samples"] == 32000 and rc[64000]["n_samples"] == 128000
    assert rc[1000]["warmup_steps"] == 30 and rc[16000]["warmup_steps"] == 60  # 学习率调度按实际步数排（warmup 3%）
    assert rc[1000]["selection"]["val_every"] == 200 and rc[4000]["selection"]["n_per_task"] == 64  # D78：100 → 200
    assert rc[16000]["selection"] is None and rc[64000]["selection"] is None  # 只对受下限影响的档
    assert rc[1000]["selection"]["batch_size"] == CFG.doc["eval"]["batch_size"]  # 与 test 评测同一个批大小
    # 下限按数据量档判：a–d 子集的 1K 档（800 条）同样 1,000 步
    assert resolve_run(CFG, n_train=1000, n_examples=800, tiny=False)["steps"] == 1000
    # tiny：10 条的档是 1K 的同构缩小（6 步、带 val 选择），20 条的档照旧固定 epoch
    t10, t20 = (resolve_run(CFG, n_train=n, n_examples=n, tiny=True) for n in (10, 20))
    assert (t10["steps"], t10["steps_by_epochs"], t10["n_samples"]) == (6, 3, 24) and t10["selection"]["val_every"] == 2
    assert (t20["steps"], t20["n_samples"], t20["selection"]) == (5, 20, None)
    assert plan_steps(100, 2, 16, 5) == {"steps_by_epochs": 13, "steps": 13, "n_samples": 200, "floor_binding": 0}  # 下限不起作用


def test_schedule_with_floor():
    ts = TrainSet.build(CFG, 10, None, 0)
    rc = resolve_run(CFG, n_train=10, n_examples=len(ts), tiny=True)
    sch = Schedule(ts, seed=3, epochs=rc["epochs"], batch_size=rc["batch_size"], total=rc["n_samples"])
    assert sch.total == 24 and sch.epochs == 3
    plans = [p for s in range(rc["steps"]) for p in sch.step_plans(s)]
    assert len(plans) == 24 and all(len(sch.step_plans(s)) == rc["batch_size"] for s in range(rc["steps"]))  # 每步满批
    for ep in (0, 1):
        assert sorted(p.sample_id for p in plans if p.epoch == ep) == sorted(it.id for it in ts.items())
    assert len([p for p in plans if p.epoch == 2]) == 4  # 最后一个 epoch 只取置换的前一段
    plain = Schedule(ts, seed=3, epochs=1, batch_size=rc["batch_size"])  # 不设下限：前 1 个 epoch 逐条相同
    assert [plain.plan(k) for k in range(10)] == [sch.plan(k) for k in range(10)]
    with pytest.raises(IndexError):
        sch.plan(24)


# ---------------------------------------------------------------------------
# val 选择：规则、样本、入口断言
# ---------------------------------------------------------------------------


def test_rule_and_best_of():
    by = {"rotation3d": {"h": VS.task_value("rotation3d", {"relief_error": {"strict": 0.8}, "mask_iou_grid": {"strict": 0.1}})},
          "occlusion": {"h": VS.task_value("occlusion", {"mask_iou_grid": {"strict": 0.5}})}}
    assert by["rotation3d"]["h"] == pytest.approx(0.6)  # a：1 − 起伏误差 / 2.0（不用 IoU）
    assert VS.selection_score(by) == pytest.approx(0.55)
    assert VS.task_value("rotation3d", {"relief_error": {"strict": 2.0}, "mask_iou_grid": {"strict": 1.0}}) == 0.0  # 失败 = 2.0 → 0
    assert VS.HEADLINE_METRIC == {"rotation3d": "relief", "viewpoint": "iou", "occlusion": "iou", "physics": "iou", "imagery": "iou"}
    hist = [{"step": 2, "score": 0.3}, {"step": 4, "score": 0.7}, {"step": 6, "score": 0.7}, {"step": 8, "score": 0.1}]
    assert VS.best_of(hist)["step"] == 4  # 最高分；同分取较早的一步
    assert VS.best_of([]) is None
    with pytest.raises(ValueError):
        VS.selection_score({"occlusion": {"h": None}})


def test_selection_items_val_headline_only():
    items = VS.selection_items(("rotation3d", "viewpoint", "occlusion", "imagery"), 3, tiny=True)
    for t, its in items.items():
        assert len(its) == 3 and all(it.split == "val" for it in its)
        assert [it.index for it in its] == sorted(it.index for it in its)  # val 序号升序的前 3 条合格的
        for it in its:
            s = VS.val_sample(it)
            f = VS.SC.sample_fields(s)
            assert f["headline"] and in_headline(t, f["bin"], f["groups"])  # 都在标题口径里（与 test 同一个函数）
            if t == "rotation3d":
                assert f["bin"] in (1, 2, 3) and not VS.target_relief_flat(s)
    assert items["occlusion"][0].index == 0 and items["viewpoint"][0].index > 0  # b 只要环绕的 [20,180]
    assert VS.selection_items(("viewpoint",), 3, tiny=True) == {"viewpoint": items["viewpoint"]}  # 确定的
    with pytest.raises(VS.SplitGuardError):
        VS.val_sample(EvalItem("occlusion", "test", 0, 0))  # 入口断言：test 一条都不生成
    with pytest.raises(VS.SplitGuardError):
        VS.ValSelector.build({**resolve_run(CFG, n_train=10, n_examples=10, tiny=True)["selection"], "split": "test"},
                             ("occlusion",), tiny=True)


# ---------------------------------------------------------------------------
# val 选择：训练流程
# ---------------------------------------------------------------------------


def _files(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def test_selection_picks_val_best(tmp_path, monkeypatch):
    """假的 val 评测（第 2、4、6 步依次 0.3、0.7、0.5）：选中第 4 步，final/model 与停在第 4 步的 latest/model 逐字节相同。"""
    scores = [0.3, 0.7, 0.5]
    calls = itertools.count()

    def fake(self, ad, e5tok, prompt, n, log=None):
        k = next(calls)
        return {"score": scores[k], "by_task": {t: {"h": scores[k]} for t in self.items}}

    monkeypatch.setattr(VS.ValSelector, "evaluate", fake)
    a = tmp_path / "a"
    assert _train(a) == 0
    leaf = _leaf(a)
    s = json.loads((leaf / "train_summary.json").read_text())
    assert s["steps"] == 6 and s["selected_step"] == 4 and s["selection"]["selected_score"] == 0.7
    assert [h["step"] for h in s["selection"]["history"]] == [2, 4, 6]
    stack = json.loads((leaf / "ckpt" / "final" / "e5_stack.json").read_text())
    assert stack["selection"]["selected_step"] == 4 and stack["steps"] == 6
    val = [json.loads(l) for l in (leaf / VS.VAL_LOG).read_text().splitlines()]
    assert [(v["step"], v["score"], v["best_step"]) for v in val] == [(2, 0.3, 2), (4, 0.7, 4), (6, 0.5, 4)]
    assert not list((leaf / "ckpt").glob("best_step*")) and not (leaf / "ckpt" / "latest").exists()  # 候选与 latest 都清掉
    # 停在第 4 步（带 latest）的另一次运行：它的 latest/model 就是第 4 步的模型
    calls = itertools.count()
    b = tmp_path / "b"
    assert _train(b, "--stop-after-steps", "4") == 0
    fa, fb = _files(leaf / "ckpt" / "final" / "model"), _files(_leaf(b) / "ckpt" / "latest" / "model")
    assert fa == fb
    # 第 4 步的模型确实不同于第 2 步（选择有区分力）
    calls = itertools.count()
    c = tmp_path / "c"
    assert _train(c, "--stop-after-steps", "2") == 0
    assert _files(_leaf(c) / "ckpt" / "latest" / "model")["lora/adapter_model.safetensors"] != fa["lora/adapter_model.safetensors"]
    # 运行身份与 README 记下选择
    cfg = yaml.safe_load((leaf / "config.yaml").read_text())
    assert cfg["training"]["min_steps"] == 6 and cfg["training"]["selection"]["rule"] == VS.RULE == "mean_headline_v1"
    assert cfg["selection"]["split"] == "val" and set(cfg["selection"]["items"]) == set(E5_TASKS)
    assert "选中第 4 步" in (leaf / "README.md").read_text()


def test_real_val_eval_never_generates_test(tmp_path, monkeypatch):
    """真的 val 评测（小模型、受限解码）：训练进程里生成器收到的划分只有 train 与 val。"""
    seen: list[str] = []
    orig = SYNTH_API.check_split

    def record(split):
        seen.append(split)
        return orig(split)

    monkeypatch.setattr(SYNTH_API, "check_split", record)
    assert _train(tmp_path / "o") == 0
    assert "test" not in seen and {"train", "val"} <= set(seen)
    leaf = _leaf(tmp_path / "o")
    val = [json.loads(l) for l in (leaf / VS.VAL_LOG).read_text().splitlines()]
    assert [v["step"] for v in val] == [2, 4, 6] and all(set(v["by_task"]) == set(E5_TASKS) for v in val)
    for v in val:
        for t, row in v["by_task"].items():
            assert row["n"] == 1 and row["metric"] == VS.HEADLINE_METRIC[t] and 0.0 <= row["h"] <= 1.0
            assert ("relief_strict" in row) == (t in ("rotation3d", "viewpoint"))
    run = json.loads(sorted((leaf / "logs").glob("run_*.json"))[-1].read_text())
    assert [e["step"] for e in run["val_evals"]] == [2, 4, 6]  # 墙钟只进运行记录


def test_resume_bytewise_with_selection(tmp_path):
    det = ("train_log.jsonl", VS.VAL_LOG, "config.yaml", "train_summary.json", "README.md")
    a = tmp_path / "a0"  # 与 b3、b4 同长：e5_stack.json 记训练叶目录的路径，final_checkpoint_bytes 随路径长度变
    assert _train(a) == 0
    ref = {f: (_leaf(a) / f).read_bytes() for f in det}
    ref_model = _files(_leaf(a) / "ckpt" / "final" / "model")
    for stop in ("3", "4"):  # 停在两次评测之间、停在评测那一步
        b = tmp_path / f"b{stop}"
        assert _train(b, "--stop-after-steps", stop) == 0
        lb = _leaf(b)
        if stop == "3":  # 残留一个没被 latest 记下的候选（中断在存候选与存 latest 之间）：续跑时作废
            shutil.copytree(lb / "ckpt" / "best_step000002", lb / "ckpt" / "best_step000004")
        assert _train(b) == 2  # 非空目录不加 --resume：拒绝
        assert _train(b, "--resume") == 0
        assert {f: (lb / f).read_bytes() for f in det} == ref, stop
        assert _files(lb / "ckpt" / "final" / "model") == ref_model
        assert not list((lb / "ckpt").glob("best_step*"))


# ---------------------------------------------------------------------------
# 配种子
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def e3b_seed0(tmp_path_factory):
    root = tmp_path_factory.mktemp("e5small_start")
    assert _script("train_e3b").main(["--tiny", "--model", KEY, "--supervision", "grid", "--seed", "0", "--out", str(root)]) == 0
    return root / "e3b" / KEY / "seed0" / "ckpt" / "final" / "model"


def test_seed_pairing(e3b_seed0, tmp_path):
    ck = ["--start-checkpoint", str(e3b_seed0)]
    out = tmp_path / "o"
    assert _train(out, *ck, n=20, seed=1, init="e3b") == 2  # E3 的种子 0 ≠ E5 的种子 1：拒绝
    (err,) = (out / "train" / "e3b" / KEY / "errors").glob("error_*.json")
    assert "配种子" in json.loads(err.read_text())["error"]
    assert not _leaf(out, "n20", 1, "e3b").exists()
    assert _train(out, *ck, "--allow-seed-mismatch", n=20, seed=1, init="e3b") == 0  # 放开，记进运行身份
    pair = yaml.safe_load((_leaf(out, "n20", 1, "e3b") / "config.yaml").read_text())["seed_pairing"]
    assert (pair["status"], pair["start_seed"], pair["seed"], pair["allow_mismatch"]) == ("mismatch", 0, 1, True)
    # 核对不了（检查点不在训练叶目录下）：也拒绝
    moved = tmp_path / "moved" / "model"
    shutil.copytree(e3b_seed0, moved)
    assert _train(tmp_path / "p", "--start-checkpoint", str(moved), n=20, seed=0, init="e3b") == 2
    from src.e5 import stack as STK

    st = STK.resolve_start(CFG, "e3b", moved, model_key=KEY, tiny=True)
    assert st.leaf_seed is None and STK.check_seed_pairing(st, 0, allow_mismatch=True)["status"] == "unverifiable"
    ok = STK.resolve_start(CFG, "e3b", e3b_seed0, model_key=KEY, tiny=True)
    assert ok.leaf_seed == 0 and STK.check_seed_pairing(ok, 0, allow_mismatch=False)["status"] == "match"
    with pytest.raises(STK.StartError, match="配种子"):
        STK.check_seed_pairing(ok, 2, allow_mismatch=False)
    base = STK.resolve_start(CFG, "base", None, model_key=KEY, tiny=True)
    assert STK.check_seed_pairing(base, 2, allow_mismatch=False)["status"] == "not_applicable"
    assert math.isfinite(json.loads((_leaf(out, "n20", 1, "e3b") / "train_summary.json").read_text())["final_loss"])
