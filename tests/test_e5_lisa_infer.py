"""DEV-0019：scripts/external/lisa_e5_infer.py（LISA-7B-v1 在 E5 上的推理，外部对照系统那一侧）。

开发机上跑不了 LISA，用替身（FakeRunner，实现脚本的推理接口：给定图像与提示，返回固定的文本、[SEG] 个数与 mask）。
- 约定：脚本里另写的常量（标记文件名、格式版本、字段名、a–d 的任务名、导出文件名）与 src/e5/external.py、src/e5/protocol.py
  一致；系统身份（LISA commit、权重 revision）与 RUN-0015 的记录一致。
- 静态：脚本不 import src/（AST），顶层只 import 标准库与 numpy；在仓库外的 cwd 里不带 PYTHONPATH 也能 --help。
- 未压缩 RLE：只用 numpy，与 pycocotools 的编码逐位相同。
- 端到端：run_e5.py --export-inputs（--tiny 彩排同一套开发集）→ 替身脚本 → run_e5.py --system external：mask 逐像素相同、
  failure 原因保留、标记文件通过 read_manifest、所选 id 的指纹等于评分 config.yaml 的 sample_ids_sha256。
- 续跑：被打断（含写到一半的行）后 --resume 只补没写出的 id，结果与不打断的一次运行相同；重复 id、身份不同都报错。
- 连续异常：连续 MAX_CONSECUTIVE_EXCEPTIONS 条异常即停止且这一串不写出；零星的异常照常写成 failure 行。
- 隔离：--out 与读的目录重叠、落在 src/ 或 $PROJ/data 下都在写任何东西之前报错；动态探针核对只写 --out、只读 --inputs 与 --out。
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

from src.e5 import external as EXT
from src.e5.protocol import E5_TASKS, TASK_LETTER
from src.executors.raster import rle_to_mask

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "external" / "lisa_e5_infer.py"
RUN_E5 = REPO / "scripts" / "run_e5.py"
SMOKE = REPO / "experiments" / "P0" / "0015_lisa_smoke"
N_PER_BIN, LIMIT = 6, 6  # 每个任务取序号 0..5：替身按序号 % 6 覆盖每一种结果
SUBSET_ARGS = ["--subset", "dev", "--n-per-bin", str(N_PER_BIN), "--limit", str(LIMIT)]
ABCD = tuple(t for t in E5_TASKS if TASK_LETTER[t] in "abcd")


def _load(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass 要在 sys.modules 里找到模块
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


LI = _load(SCRIPT, "_lisa_e5_infer_under_test")


@pytest.fixture(scope="module")
def run_e5():
    return _load(RUN_E5, "_run_e5_for_lisa_test")


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------


def _index(image_path: str) -> tuple[str, int]:
    stem = Path(image_path).stem  # <任务>_<划分>_<序号:09d>（src/e5/external.py 的 export_inputs）
    task, _split, idx = stem.rsplit("_", 2)
    return task, int(idx)


def expected_failure(idx: int) -> str | None:
    return {1: "no_seg", 2: "empty_mask", 3: "ValueError"}.get(idx % 6)


class FakeRunner:
    """序号 % 6：1 → 没有 [SEG]；2 → 全空 mask；3 → 抛 ValueError；4 → 两个 [SEG]（取第一个）；其余 → 一个随序号移动的矩形。
    interrupt_at：第几次调用时抛 KeyboardInterrupt；raise_idx：这些序号抛 RuntimeError；raise_all：每条都抛。"""

    def __init__(self, fixes=(LI.FIX_VISION_TOWER,), interrupt_at=None, raise_idx=(), raise_all=False):
        self.info = {"fixes_applied": list(fixes), "load_s": 0.0, "fake": True}
        self.calls: list[tuple[str, int]] = []
        self.prompts: list[str] = []
        self.masks: dict[tuple[str, int], np.ndarray] = {}
        self.interrupt_at, self.raise_idx, self.raise_all = interrupt_at, set(raise_idx), raise_all

    def infer(self, image_path, prompt):
        key = _index(image_path)
        self.calls.append(key)
        self.prompts.append(prompt)
        if self.interrupt_at is not None and len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt
        if self.raise_all or key[1] in self.raise_idx:
            raise RuntimeError(f"fake CUDA error at {key}")
        with Image.open(image_path) as im:  # 真的读导出的图像
            h, w = im.height, im.width
        r = key[1] % 6
        if r == 1:
            return LI.InferOutput(text="I cannot find it.</s>", n_seg=0, mask=None, n_generated_tokens=6)
        if r == 2:
            return LI.InferOutput(text="Sure, [SEG] .</s>", n_seg=1, mask=np.zeros((h, w), bool), n_generated_tokens=5)
        if r == 3:
            raise ValueError("fake failure")
        m = np.zeros((h, w), bool)
        i = key[1]
        m[10 + 7 * i: 60 + 9 * i, 20 + 5 * i: 95 + 11 * i] = True
        m[h - 1, w - 1] = True  # 列优先顺序的最后一个像素
        self.masks[key] = m
        return LI.InferOutput(text="Sure, the segmentation result is [SEG] .</s>", n_seg=2 if r == 4 else 1, mask=m,
                              n_generated_tokens=9)

    def stats(self):
        return {"fake": True}


def good_source(_p):
    return {"commit": LI.LISA_COMMIT, "dirty_files": []}


@pytest.fixture(scope="module")
def env(tmp_path_factory, run_e5):
    """导出开发集的输入（a–e，各 6 条）+ 假的 LISA 源码目录与两个 HF 快照目录。"""
    root = tmp_path_factory.mktemp("lisa_env")
    inputs = root / "inputs"
    assert run_e5.main(["--export-inputs", str(inputs), "--task", "all", *SUBSET_ARGS]) == 0
    lisa = root / "LISA"
    lisa.mkdir()
    snaps = {}
    for key, (repo, rev) in LI.WEIGHTS.items():
        snaps[f"{key}_snap"] = root / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / rev
        snaps[f"{key}_snap"].mkdir(parents=True)
    return {"root": root, "inputs": inputs, "lisa": lisa, **snaps}


def argv(env, out, *extra, lisa_w=None, clip_w=None, inputs=None):
    return ["--lisa-src", str(env["lisa"]), "--inputs", str(inputs or env["inputs"]), "--out", str(out),
            "--lisa-weights", str(lisa_w or env["lisa_snap"]), "--clip-weights", str(clip_w or env["clip_snap"]), *extra]


def run_lisa(env, out, *extra, runner=None, source=good_source, **kw):
    runner = runner if runner is not None else FakeRunner()
    rc = LI.main(argv(env, out, *extra, **kw), make_runner=lambda a: runner, source_info=source, log=lambda m: None)
    return rc, runner


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def input_rows(env, task):
    return lines(env["inputs"] / task / "inputs.jsonl")


# ---------------------------------------------------------------------------
# 约定与静态检查
# ---------------------------------------------------------------------------


def test_constants_match_src_and_run_0015():
    assert LI.EXTERNAL_MARKER == EXT.EXTERNAL_MARKER and LI.EXTERNAL_FORMAT_VERSION == EXT.EXTERNAL_FORMAT_VERSION
    src = (REPO / "src" / "e5" / "external.py").read_text(encoding="utf-8")
    for field in (LI.FIELD_ID, LI.FIELD_MASK_RLE, LI.FIELD_FAILURE, LI.INPUTS_FILE):
        assert f'"{field}"' in src, field  # 读取方（load_task）与导出方（export_inputs）用的同一个字面量
    assert LI.LISA_TASKS == ABCD == ("rotation3d", "viewpoint", "occlusion", "physics")
    assert tuple(LI.SKIPPED_TASKS) == tuple(t for t in E5_TASKS if TASK_LETTER[t] == "e") == ("imagery",)
    assert EXT.MISSING not in (LI.FAIL_NO_SEG, LI.FAIL_EMPTY_MASK)
    assert LI.SYSTEM == "lisa-7b-v1" and LI.PROMPT_TEMPLATE == "{instruction} Please output segmentation mask."
    assert LI.MAX_NEW_TOKENS == 512 and LI.CHAT_DEFAULTS["precision"] == "bf16" and LI.CHAT_DEFAULTS["image_size"] == 1024
    assert set(LI.FIXES) == {LI.FIX_VISION_TOWER, LI.FIX_ADDEDTOKEN}
    # 系统身份 = RUN-0015 核对过的那一份
    assert (SMOKE / "lisa_commit.txt").read_text().split()[0] == LI.LISA_COMMIT
    wd = json.loads((SMOKE / "weights_download.json").read_text())
    assert {k: (repo, wd[repo]["revision"]) for k, (repo, _rev) in LI.WEIGHTS.items()} == LI.WEIGHTS


def test_script_does_not_import_src():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "不用相对 import"
            mods.append(node.module or "")
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert name not in ("import_module", "__import__", "run_path", "run_module"), "不做动态 import"
    assert mods and not [m for m in mods if m == "src" or m.startswith("src.")]
    top = [m for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
           for m in ([a.name for a in node.names] if isinstance(node, ast.Import) else [node.module])]
    assert {m.split(".")[0] for m in top} <= set(sys.stdlib_module_names) | {"numpy"}, top
    # 只把 --lisa-src 加进 sys.path（不加我们的仓库根）
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.count("sys.path.insert") == 1 and "sys.path.insert(0, str(lisa_src))" in text


def test_runs_outside_the_repo_without_src(tmp_path):
    envv = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    p = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=tmp_path, env=envv, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0 and "--lisa-src" in p.stdout, p.stderr


@pytest.mark.parametrize("shape,kind", [((7, 5), "zeros"), ((7, 5), "ones"), ((1, 1), "ones"), ((30, 40), "random"),
                                        ((30, 40), "corners"), ((512, 512), "rect")])
def test_uncompressed_rle_matches_coco(shape, kind):
    rng = np.random.default_rng(0)
    m = {"zeros": lambda: np.zeros(shape, bool), "ones": lambda: np.ones(shape, bool),
         "random": lambda: rng.random(shape) < 0.3}.get(kind, lambda: np.zeros(shape, bool))()
    if kind == "corners":
        m[0, 0] = m[-1, -1] = m[0, -1] = True
    if kind == "rect":
        m[100:300, 50:400] = True
    rle = LI.mask_to_uncompressed_rle(m)
    assert rle["size"] == list(shape) and sum(rle["counts"]) == m.size and all(type(c) is int for c in rle["counts"])
    assert json.loads(json.dumps(rle)) == rle
    assert np.array_equal(rle_to_mask(rle), m)  # src.executors.raster（pycocotools）读回逐像素相同
    coco = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))["counts"]
    assert mask_utils.frPyObjects(rle, *shape)["counts"] == coco  # 与 pycocotools 自己的编码逐位相同
    with pytest.raises(ValueError):
        LI.mask_to_uncompressed_rle(np.zeros((2, 2, 2), bool))


# ---------------------------------------------------------------------------
# 端到端：导出 → 替身 → 评分
# ---------------------------------------------------------------------------


def test_end_to_end_export_fake_lisa_score(env, tmp_path, run_e5):
    out = tmp_path / "lisa_predictions"
    rc, fake = run_lisa(env, out)
    assert rc == 0
    marker = EXT.read_manifest(out)  # 标记文件通过校验
    assert marker["system"] == "lisa-7b-v1" and marker["lisa_commit"] == LI.LISA_COMMIT
    assert marker["weights"] == {k: {"repo": r, "revision": v} for k, (r, v) in LI.WEIGHTS.items()}
    assert marker["prompt_template"] == LI.PROMPT_TEMPLATE and marker["fixes_applied"] == [LI.FIX_VISION_TOWER]
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["tasks"] == list(ABCD) and "imagery" in cfg["skipped_tasks"] and cfg["limit"] is None
    assert not (out / "imagery.jsonl").exists() and all(t != "imagery" for t, _ in fake.calls)
    assert len(fake.calls) == 4 * LIMIT
    # 提示 = 模板 + 导出的指令原文
    inst = {r["id"]: r["instruction"] for t in ABCD for r in input_rows(env, t)}
    assert fake.prompts[0] == f"{inst['rotation3d/val/0']} Please output segmentation mask."
    for t in ABCD:
        rows = input_rows(env, t)
        ids = [r["id"] for r in rows]
        got, st = EXT.load_task(out, t, ids, tuple(rows[0]["canvas_hw"]))
        assert st["n_missing"] == 0 and st["n_extra_ignored"] == 0 and st["system"] == "lisa-7b-v1"
        recs = {r["id"]: r for r in lines(out / f"{t}.jsonl")}
        for sid in ids:
            idx = int(sid.split("/")[-1])
            want = expected_failure(idx)
            if want is None:
                assert np.array_equal(got[sid], fake.masks[(t, idx)])  # 逐像素相同
                assert "failure" not in recs[sid] and recs[sid]["n_mask_pixels"] == int(fake.masks[(t, idx)].sum())
                assert recs[sid]["n_seg"] == (2 if idx % 6 == 4 else 1)
            else:
                assert got[sid] == want  # failure 原因保留
                assert "mask_rle" not in recs[sid]
            assert recs[sid]["wall_s"] >= 0 and recs[sid]["prompt"].endswith("Please output segmentation mask.")
        assert recs[f"{t}/val/1"]["text"] == "I cannot find it.</s>" and recs[f"{t}/val/1"]["n_seg"] == 0
        assert recs[f"{t}/val/3"]["error"] == "fake failure"
    exc = lines(out / "logs" / "exceptions.jsonl")
    assert [e["class"] for e in exc] == ["ValueError"] * 4 and "fake failure" in exc[0]["traceback"]
    log = json.loads((out / "logs" / "run_001.json").read_text())
    assert log["status"] == "done" and log["fixes_applied"] == [LI.FIX_VISION_TOWER]
    assert [s["counts"] for s in log["tasks"]] == [{"ok": 3, "no_seg": 1, "empty_mask": 1, "exception": 1}] * 4

    # run_e5.py --system external 评分（--tiny：d 在非规范平台上只允许彩排）
    score = tmp_path / "score"
    assert run_e5.main(["--tiny", "--system", "external", "--model", "lisa-7b-v1", "--external-dir", str(out), "--task",
                        "a", "b", "c", "d", *SUBSET_ARGS, "--out", str(score)]) == 0
    for t in ABCD:
        leaf = score / "external" / "lisa-7b-v1" / f"{t}__dev"
        preds = lines(leaf / "predictions.jsonl")
        assert [p["external_failure"] for p in preds] == [expected_failure(int(p["id"].split("/")[-1])) for p in preds]
        for p in preds:
            if p["valid"]:
                assert np.array_equal(rle_to_mask(p["pred"]["rle"]), fake.masks[(t, p["index"])])
        m = json.loads((leaf / "metrics.json").read_text())
        assert m["parse"]["failures"] == {"external_failure": 3} and m["n"] == LIMIT
        conf = yaml.safe_load((leaf / "config.yaml").read_text())
        assert conf["data"]["sample_ids_sha256"] == cfg["inputs"][t]["selected_ids_sha256"]  # 导出与评分是同一批 id
        assert conf["external"]["marker_sha256"] == hashlib.sha256((out / LI.EXTERNAL_MARKER).read_bytes()).hexdigest()


def test_limit_takes_prefix_per_task(env, tmp_path):
    rc, fake = run_lisa(env, tmp_path / "o", "--limit", "2")
    assert rc == 0 and len(fake.calls) == 2 * 4
    cfg = json.loads((tmp_path / "o" / "run_config.json").read_text())
    assert cfg["limit"] == 2 and all(cfg["inputs"][t]["n_selected"] == 2 and cfg["inputs"][t]["n_rows"] == LIMIT for t in ABCD)
    assert [r["id"] for r in lines(tmp_path / "o" / "rotation3d.jsonl")] == ["rotation3d/val/0", "rotation3d/val/1"]


# ---------------------------------------------------------------------------
# 续跑
# ---------------------------------------------------------------------------


def _by_id(path: Path) -> dict[str, dict]:
    return {r["id"]: {k: v for k, v in r.items() if k != "wall_s"} for r in lines(path)}


def test_resume_after_interrupt_matches_uninterrupted(env, tmp_path):
    full = tmp_path / "full"
    assert run_lisa(env, full)[0] == 0
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        run_lisa(env, out, runner=FakeRunner(interrupt_at=4))  # 第 4 次调用（rotation3d/val/3）时被打断
    assert [r["id"] for r in lines(out / "rotation3d.jsonl")] == [f"rotation3d/val/{i}" for i in range(3)]
    assert json.loads((out / "logs" / "run_001.json").read_text())["status"] == "interrupted"
    with open(out / "rotation3d.jsonl", "a") as f:
        f.write('{"id": "rotation3d/val/3", "mask_rle": {"si')  # 写到一半的行
    rc, _ = run_lisa(env, out)
    assert rc == 2  # 非空目录不加 --resume：拒绝
    rc, fake = run_lisa(env, out, "--resume")
    assert rc == 0
    assert len(fake.calls) == 4 * LIMIT - 3 and ("rotation3d", 0) not in fake.calls and fake.calls[0] == ("rotation3d", 3)
    for t in ABCD:
        assert _by_id(out / f"{t}.jsonl") == _by_id(full / f"{t}.jsonl"), t
    log2 = json.loads((out / "logs" / "run_002.json").read_text())
    assert log2["status"] == "done" and log2["tasks"][0]["n_done_before"] == 3
    rc, fake = run_lisa(env, out, "--resume")  # 全部完成后再续跑：什么都不做
    assert rc == 0 and fake.calls == []


def test_resume_rejects_duplicates_and_changed_identity(env, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    assert run_lisa(env, out)[0] == 0
    path = out / "occlusion.jsonl"
    orig = path.read_text()
    path.write_text(orig + orig.splitlines()[0] + "\n")
    rc, fake = run_lisa(env, out, "--resume")
    assert rc == 2 and "重复" in capsys.readouterr().err and fake.calls == []
    path.write_text(orig + json.dumps({"id": "occlusion/val/999", "failure": "x"}) + "\n")
    assert run_lisa(env, out, "--resume")[0] == 2  # 不在本次输入里的 id
    path.write_text(orig)
    assert run_lisa(env, out, "--resume", runner=FakeRunner(fixes=(LI.FIX_ADDEDTOKEN, LI.FIX_VISION_TOWER)))[0] == 2
    assert "fixes_applied" in capsys.readouterr().err  # 用到的处理不同 = 身份不同
    assert run_lisa(env, out, "--resume", "--limit", "3")[0] == 2
    monkeypatch.setattr(LI, "PROMPT_TEMPLATE", "{instruction} Segment it.")
    assert run_lisa(env, out, "--resume")[0] == 2
    monkeypatch.undo()
    rc, fake = run_lisa(env, out, "--resume")
    assert rc == 0 and fake.calls == []


def test_duplicate_or_bad_input_rows_rejected(env, tmp_path, capsys):
    inputs = tmp_path / "inputs"
    (inputs / "occlusion").mkdir(parents=True)
    rows = input_rows(env, "occlusion")
    (inputs / "occlusion" / "inputs.jsonl").write_text("\n".join(json.dumps(r) for r in rows + rows[:1]) + "\n")
    rc, fake = run_lisa(env, tmp_path / "o", inputs=inputs)
    assert rc == 2 and "重复" in capsys.readouterr().err and fake.calls == [] and not (tmp_path / "o").exists()
    bad = dict(rows[0], image="../../escape.png")
    (inputs / "occlusion" / "inputs.jsonl").write_text(json.dumps(bad) + "\n")
    assert run_lisa(env, tmp_path / "o", inputs=inputs)[0] == 2
    empty = tmp_path / "empty_inputs"
    (empty / "imagery").mkdir(parents=True)
    assert run_lisa(env, tmp_path / "o", inputs=empty)[0] == 2  # 没有 a–d 的任何任务


# ---------------------------------------------------------------------------
# 连续异常
# ---------------------------------------------------------------------------


def test_consecutive_exceptions_abort_without_writing_the_streak(env, tmp_path, capsys):
    out = tmp_path / "out"
    rc, fake = run_lisa(env, out, runner=FakeRunner(raise_all=True))
    assert rc == 2 and "连续" in capsys.readouterr().err
    assert len(fake.calls) == LI.MAX_CONSECUTIVE_EXCEPTIONS
    assert not (out / "rotation3d.jsonl").exists() or lines(out / "rotation3d.jsonl") == []
    exc = lines(out / "logs" / "exceptions.jsonl")
    assert len(exc) == LI.MAX_CONSECUTIVE_EXCEPTIONS and all(e["written"] is False for e in exc)
    log = json.loads((out / "logs" / "run_001.json").read_text())
    assert log["status"] == "aborted" and log["tasks"][0]["counts"] == {}
    rc, fake = run_lisa(env, out, "--resume")  # 环境修好后续跑：这一串被重试
    assert rc == 0 and fake.calls[0] == ("rotation3d", 0) and len(fake.calls) == 4 * LIMIT


def test_short_exception_streak_is_written_in_order(env, tmp_path):
    out = tmp_path / "out"
    rc, _ = run_lisa(env, out, runner=FakeRunner(raise_idx=(0, 1)))  # 连续 2 条（< 3），随后是 empty_mask
    assert rc == 0
    recs = lines(out / "rotation3d.jsonl")
    assert [(r["id"], r.get("failure")) for r in recs[:4]] == [("rotation3d/val/0", "RuntimeError"),
                                                              ("rotation3d/val/1", "RuntimeError"),
                                                              ("rotation3d/val/2", "empty_mask"),
                                                              ("rotation3d/val/3", "ValueError")]


# ---------------------------------------------------------------------------
# 隔离与身份
# ---------------------------------------------------------------------------


def _never(_a):
    raise AssertionError("身份或隔离不合格时不应加载模型")


def test_git_source_info_parsing(monkeypatch, tmp_path):
    """git_source_info（服务器上用它核对 LISA 的 commit）：只发两条只读命令，HEAD 去空白，status 的每一行是一个改动；
    命令失败报 LisaInferError。subprocess.run 换成桩，不在开发机上建仓库。"""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        out = {"rev-parse": LI.LISA_COMMIT + "\n", "status": " M model/LISA.py\n\n"}[cmd[3]]
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(LI.subprocess, "run", fake_run)
    info = LI.git_source_info(tmp_path)
    assert info == {"commit": LI.LISA_COMMIT, "dirty_files": [" M model/LISA.py"]}
    assert calls == [["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
                     ["git", "-C", str(tmp_path), "status", "--porcelain", "--untracked-files=no"]]

    def failing(cmd, **kw):
        raise subprocess.CalledProcessError(128, cmd, stderr="not a git repository")

    monkeypatch.setattr(LI.subprocess, "run", failing)
    with pytest.raises(LI.LisaInferError):
        LI.git_source_info(tmp_path)


@pytest.mark.parametrize("where", ["inside_inputs", "equals_lisa_src", "contains_inputs", "repo_src", "proj_data"])
def test_out_dir_isolation(env, tmp_path, where):
    out = {"inside_inputs": env["inputs"] / "rotation3d" / "lisa_out", "equals_lisa_src": env["lisa"],
           "contains_inputs": env["root"], "repo_src": REPO / "src" / "_lisa_out_must_not_exist",
           "proj_data": REPO.parent / "data" / "_lisa_out_must_not_exist"}[where]
    before = sorted(p.as_posix() for p in env["root"].rglob("*"))
    rc = LI.main(argv(env, out), make_runner=_never, source_info=good_source, log=lambda m: None)
    assert rc == 2
    assert sorted(p.as_posix() for p in env["root"].rglob("*")) == before
    if where in ("repo_src", "proj_data"):
        assert not out.exists()


def test_weights_and_source_identity(env, tmp_path, capsys):
    out = tmp_path / "o"
    wrong_rev = env["clip_snap"].parent / ("0" * 40)
    wrong_rev.mkdir()
    not_snap = tmp_path / "weights"
    not_snap.mkdir()
    wrong_repo = env["root"] / "hub" / "models--xinlai--LISA-13B-llama2-v1" / "snapshots" / LI.WEIGHTS["lisa"][1]
    wrong_repo.mkdir(parents=True)
    for kw in ({"clip_w": wrong_rev}, {"lisa_w": not_snap}, {"lisa_w": wrong_repo}, {"lisa_w": env["clip_snap"]}):
        assert LI.main(argv(env, out, **kw), make_runner=_never, source_info=good_source, log=lambda m: None) == 2, kw
    for src in ({"commit": "f" * 40, "dirty_files": []}, {"commit": LI.LISA_COMMIT, "dirty_files": [" M model/LISA.py"]}):
        assert LI.main(argv(env, out), make_runner=_never, source_info=lambda _p, s=src: s, log=lambda m: None) == 2
    assert "LISA 源码必须是固定 commit" in capsys.readouterr().err and not out.exists()


class _IOProbe:
    """记下 open / os.open / mkdir 的路径与是否写。"""

    def __init__(self, monkeypatch):
        self.reads: list[str] = []
        self.writes: list[str] = []
        for mod, name in ((builtins, "open"), (io, "open")):
            monkeypatch.setattr(mod, name, self._wrap_open(getattr(mod, name)))
        monkeypatch.setattr(os, "open", self._wrap_osopen(os.open))
        monkeypatch.setattr(os, "mkdir", self._wrap_mkdir(os.mkdir))

    def _rec(self, path, write):
        if isinstance(path, (str, bytes, os.PathLike)):
            (self.writes if write else self.reads).append(os.path.abspath(os.fsdecode(path)))

    def _wrap_open(self, orig):
        def f(file, mode="r", *a, **k):
            self._rec(file, any(c in mode for c in "wax+"))
            return orig(file, mode, *a, **k)
        return f

    def _wrap_osopen(self, orig):
        def f(path, flags, *a, **k):
            self._rec(path, bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC)))
            return orig(path, flags, *a, **k)
        return f

    def _wrap_mkdir(self, orig):
        def f(path, *a, **k):
            self._rec(path, True)
            return orig(path, *a, **k)
        return f


def test_dynamic_only_writes_out_and_reads_inputs(env, tmp_path, monkeypatch):
    out = (tmp_path / "out").resolve()
    inputs = env["inputs"].resolve()
    runner = FakeRunner()
    probe = _IOProbe(monkeypatch)
    rc = LI.main(argv(env, out), make_runner=lambda a: runner, source_info=good_source, log=lambda m: None)
    monkeypatch.undo()
    assert rc == 0 and probe.writes and probe.reads

    def under(p, root):
        return p == str(root) or p.startswith(str(root) + os.sep)

    assert [p for p in probe.writes if not under(p, out)] == []
    assert [p for p in probe.reads if not (under(p, out) or under(p, inputs))] == []
    assert any(under(p, inputs) and p.endswith(".png") for p in probe.reads)  # 正对照：探针确实记到了读图
