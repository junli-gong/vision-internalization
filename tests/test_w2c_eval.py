"""DEV-0061：W2-c 的评测接入（scripts/run_w2.py --setting w2c --checkpoint；src/w2/w2c.py）、判定（scripts/w2_gate.py --main-name W2-c）、
健康检查（scripts/health_check.py --checkpoint）与定位、计数的 T 文本读数（scripts/run_e1.py --checkpoint 原样）。命令行彩排（--tiny，子进程，
与服务器同一流程）：

- scripts/train_w2c.py --tiny 训出一个 W2-c 检查点（假数据，W2 配置用 4 个假生成器）；run_w2 --setting w2c 加载它：评测前的合并核对
  （t_text 口径）通过、写进 <out>/<模型>/w2c_merge_check.json；训练过的模型按 E1 协议重新出框，放在 refexp_box/propose_w2c（与 base 的
  baseline_e1 分开）；候选放在 refseg/candidates_w2c；挑选关思考（generate_batch）；config.yaml 记 checkpoint（路径、manifest 的 sha256）。
  随机模型给不出合法的框，所以先跑一遍，再把提出的记录换成合法的框（第一条保留失败），重跑候选与挑选（同 tests/test_w2_cli.py 的做法）。
- w2_gate --main-name W2-c 与同一 devcheck2 上 base 的 w1f 配对能读（彩排，不作数）；--main-name W2-a 读 w2c 的目录报错。
- 拒绝：w2c 不给 --checkpoint、别的设置给 --checkpoint、检查点不是 W2-c（C3 的检查点）、检查点训练时的 W2 配置与本次不同、同一个 --out
  换了检查点（提出不混用）。
- 健康检查与 T 文本读数：现有脚本加 --checkpoint 就能跑（health_check 的 --checkpoint 是本任务加的最小改动；run_e1 原样）。
- DEV-0062：临时的 W2 配置从草稿夹具（tests/_w2_draft，冻结前的 configs/w2.json 原样）改生成器得到，不读仓库配置，所以
  configs/w2.json 冻结成任何一个臂之后照样成立。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests import _w2c_fakes as FK
from tests._w2_draft import w2_draft_doc

REPO = Path(__file__).resolve().parents[1]
RUN_W2 = REPO / "scripts" / "run_w2.py"
TRAIN = REPO / "scripts" / "train_w2c.py"
GATE = REPO / "scripts" / "w2_gate.py"
HEALTH = REPO / "scripts" / "health_check.py"
RUN_E1 = REPO / "scripts" / "run_e1.py"
TRAIN_E3B = REPO / "scripts" / "train_e3b.py"
QWEN = "qwen3.5-0.8b"
FAKES = ["--candidates-module", "tests._w2_fakes", "--render-module", "tests._w2_fakes"]
GENS = [{"type": "grabcut_default"}, {"type": "box_scale", "percent": 20}, {"type": "box_scale", "percent": -20}, {"type": "box"}]


def run(script: Path, args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    env.pop("VISINT_DATA_ROOT", None)
    p = subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def leaf(out: Path, task: str, setting: str, split: str = "devcheck2") -> Path:
    return out / QWEN / task / setting / split


def w2(setting, out, cfg, *extra, task="refseg", split="devcheck2"):
    return ["--tiny", "--model", QWEN, "--task", task, "--split", split, "--setting", setting, "--out", str(out), "--w2-config", str(cfg),
            *extra]


def inject_boxes(box_leaf: Path, split: str) -> None:
    """把 E1 refexp_box 格式的记录换成合法的框（GT 框内缩 1 像素；第一条保留原来的解析失败），只改 failure 与 parsed。"""
    from src.w2.data import tiny_paths, tiny_splits, w2_items

    gt = {s.id: s.gt["box"] for s in w2_items("refexp_box", split, tiny_paths(), splits=tiny_splits())}
    recs = jsonl(box_leaf / "predictions.jsonl")
    assert recs[0]["failure"] is not None  # 随机模型给不出框
    for r in recs[1:]:
        x0, y0, x1, y1 = gt[r["id"]]
        r.update(failure=None, parsed={"box": [x0 + 1, y0 + 1, x1 - 1, y1]})
    (box_leaf / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


def train_ckpt(tmp: Path, name: str, w2cfg: Path, *extra) -> Path:
    data = FK.write_fake_w2c_data(tmp / f"{name}_data", w2_config=w2cfg)
    out = tmp / name
    run(TRAIN, ["--tiny", "--model", QWEN, "--w2c-data", str(data), "--out", str(out), "--w2-config", str(w2cfg), *extra], tmp)
    seed = extra[extra.index("--seed") + 1] if "--seed" in extra else "0"
    return out / "w2c" / QWEN / f"seed{seed}" / "ckpt" / "final" / "model"


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w2c_eval")
    cfg = tmp / "w2_k4.json"
    d = w2_draft_doc()  # 草稿夹具（DEV-0062：不随仓库配置冻结而变）
    d["candidates"]["generators"] = GENS
    cfg.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    ck = train_ckpt(tmp, "train", cfg)
    out = tmp / "out"
    logs = {"first": run(RUN_W2, w2("w2c", out, cfg, "--checkpoint", str(ck), *FAKES), tmp)}
    # 随机模型给不出框：换成合法的框，删掉候选与 w2c 的结果，重跑（提出已写完，直接读）
    inject_boxes(leaf(out, "refexp_box", "propose_w2c"), "devcheck2")
    shutil.rmtree(leaf(out, "refseg", "candidates_w2c"))
    shutil.rmtree(leaf(out, "refseg", "w2c"))
    logs["w2c"] = run(RUN_W2, w2("w2c", out, cfg, "--checkpoint", str(ck), *FAKES), tmp)
    # base 9B 的对照（这里是同一个小模型的 base）：E1 refexp_box → 合法的框 → w1f
    run(RUN_W2, w2("baseline_e1", out, cfg, task="refexp_box"), tmp)
    inject_boxes(leaf(out, "refexp_box", "baseline_e1"), "devcheck2")
    run(RUN_W2, w2("w1f", out, cfg, *FAKES), tmp)
    return {"tmp": tmp, "cfg": cfg, "ck": ck, "out": out, "logs": logs}


def test_w2c_outputs_and_identity(env):
    out, ck = env["out"], env["ck"]
    man_sha = hashlib.sha256((ck / "manifest.json").read_bytes()).hexdigest()
    mc = json.loads((out / QWEN / "w2c_merge_check.json").read_text())
    assert mc["pass"] is True and mc["checkpoint"]["manifest_sha256"] == man_sha and sorted(mc["checks"]) == ["merge"]  # t_text 口径
    pl = leaf(out, "refexp_box", "propose_w2c")
    assert pl != leaf(out, "refexp_box", "baseline_e1")
    pc = yaml.safe_load((pl / "config.yaml").read_text())
    assert pc["checkpoint"]["manifest_sha256"] == man_sha and pc["checkpoint"]["path"] == str(ck.resolve())
    assert pc["data"]["selection_stats"]["w2"]["setting"] == "propose_w2c" and pc["task"] == "refexp_box"
    assert pc["generation"]["enable_thinking"] is False
    assert "先跑" not in env["logs"]["w2c"].stderr and "出框" not in env["logs"]["w2c"].stderr  # 提出已写完：直接读
    assert "出框" in env["logs"]["first"].stderr
    lx = leaf(out, "refseg", "w2c")
    c = yaml.safe_load((lx / "config.yaml").read_text())
    assert c["w2"]["setting"] == "w2c" and c["checkpoint"]["manifest_sha256"] == man_sha
    assert c["checkpoint"]["training"]["variant"] == "w2c" and c["checkpoint"]["training"]["w2_config_sha256"] == c["w2_config"]["config_sha256"]
    assert c["checkpoint"]["merge_check"]["checks"] == ["merge"] and c["checkpoint"]["lora"]["r"] == 32
    assert c["propose"]["mode"] == "box" and c["propose"]["run"]["leaf"].endswith("refexp_box/propose_w2c/devcheck2")
    assert c["candidates"]["leaf"].endswith("refseg/candidates_w2c/devcheck2") and c["candidates"]["k"] == 4
    assert c["select"]["thinking"] is False and c["select"]["method"] == "generate_batch" and c["select"]["system"] is None
    recs = jsonl(lx / "predictions.jsonl")
    assert len(recs) == 3 and recs[0]["n_candidates"] == 0 and recs[0]["failure"] is not None
    for r in recs[1:]:
        assert r["n_candidates"] == 4 and r["n_model_calls"] == 2 and r["select"]["method"] == "generate_batch"
        assert r["fallback"] in (None, "none", "parse_failure") and r["final_index"] == (r["pick"] or 1)
    m = json.loads((lx / "metrics.json").read_text())
    assert m["complete"] is True and m["n"] == 3 and "select" in m and m["n_with_candidates"] == 2
    assert "w2c" in (lx / "README.md").read_text() and "检查点" in (lx / "README.md").read_text()
    # base 的候选缓存与 w2c 的分开
    assert leaf(out, "refseg", "candidates").exists() and leaf(out, "refseg", "candidates_w2c").exists()
    from src.w2.runner import read_identity

    ident = read_identity(lx)
    assert ident["setting"] == "w2c" and ident["checkpoint"]["manifest_sha256"] == man_sha and ident["complete"] is True
    # 续跑：已完成的 --resume 逐字节不变；不加 --resume 拒绝
    before = (lx / "predictions.jsonl").read_bytes()
    run(RUN_W2, w2("w2c", out, env["cfg"], "--checkpoint", str(env["ck"]), "--resume", *FAKES), env["tmp"])
    assert (lx / "predictions.jsonl").read_bytes() == before
    bad = run(RUN_W2, w2("w2c", out, env["cfg"], "--checkpoint", str(env["ck"]), *FAKES), env["tmp"], ok=False)
    assert bad.returncode == 2 and "已存在且非空" in bad.stderr


def test_gate_reads_w2c(env, tmp_path):
    out = env["out"]
    args = ["--main-name", "W2-c", "--main", str(leaf(out, "refseg", "w2c")), "--w1f", str(leaf(out, "refseg", "w1f")),
            "--out-dir", str(tmp_path / "g")]
    p = run(GATE, args, env["tmp"])
    doc = json.loads((tmp_path / "g" / "gate.json").read_text())
    assert doc["main_name"] == "W2-c" and doc["runs"]["main"]["setting"] == "w2c" and doc["runs"]["main"]["checkpoint"]["label"] == "w2c"
    assert doc["n"] == 3 and doc["verdict"].startswith("彩排，不作数") and "W2-c（w2c）" in p.stdout
    assert not doc["gate_point"]["checks"]["not_tiny"] and not doc["gate_point"]["checks"]["w2_config_frozen"]
    bad = run(GATE, ["--main-name", "W2-a", "--main", args[3], "--w1f", args[5]], env["tmp"], ok=False)
    assert bad.returncode == 2 and "w2a" in bad.stderr


def test_w2c_rejections(env, tmp_path):
    cfg, ck, tmp = env["cfg"], env["ck"], env["tmp"]
    p = run(RUN_W2, w2("w2c", tmp_path / "o1", cfg, *FAKES), tmp, ok=False)
    assert p.returncode == 2 and "必须给 --checkpoint" in p.stderr
    p = run(RUN_W2, w2("w2a", tmp_path / "o2", cfg, "--checkpoint", str(ck), *FAKES), tmp, ok=False)
    assert p.returncode == 2 and "必须给 --checkpoint" in p.stderr
    p = run(RUN_W2, w2("w1f", tmp_path / "o3", cfg, "--merge-check-n", "2", *FAKES), tmp, ok=False)
    assert p.returncode == 2 and "只配 --checkpoint" in p.stderr
    p = run(RUN_W2, w2("w2c", tmp_path / "o4", cfg, "--checkpoint", str(ck), "--task", "count", "--split", "test_countbench"), tmp,
            ok=False)
    assert p.returncode == 2 and "只用于" in p.stderr
    # 训练时的 W2 配置与本次不同（默认的 configs/w2.json）
    p = run(RUN_W2, ["--tiny", "--model", QWEN, "--task", "refseg", "--split", "devcheck2", "--setting", "w2c", "--out", str(tmp_path / "o5"),
                     "--checkpoint", str(ck), *FAKES], tmp, ok=False)
    assert p.returncode == 2 and "W2 配置" in p.stderr
    # C3 的检查点不是 W2-c
    run(TRAIN_E3B, ["--tiny", "--model", QWEN, "--supervision", "text", "--seed", "0", "--out", str(tmp_path / "c3")], tmp)
    c3 = tmp_path / "c3" / "c3" / QWEN / "seed0" / "ckpt" / "final" / "model"
    p = run(RUN_W2, w2("w2c", tmp_path / "o6", cfg, "--checkpoint", str(c3), *FAKES), tmp, ok=False)
    assert p.returncode == 2 and "变体" in p.stderr
    # 同一个 --out 换了检查点：提出不混用
    ck2 = train_ckpt(tmp_path, "train2", cfg, "--steps", "2")
    p = run(RUN_W2, w2("w2c", env["out"], cfg, "--checkpoint", str(ck2), "--resume", *FAKES), tmp, ok=False)
    assert p.returncode == 2 and "另一个检查点" in p.stderr
    assert json.loads((env["out"] / QWEN / "w2c_merge_check.json").read_text())["checkpoint"]["manifest_sha256"] == \
        hashlib.sha256((ck / "manifest.json").read_bytes()).hexdigest()  # 拒绝在合并核对之前：不覆盖


def test_health_check_and_t_text_with_checkpoint(env, tmp_path):
    """健康检查：scripts/health_check.py --checkpoint（本任务加的最小改动）；定位与计数的 T 文本读数：scripts/run_e1.py --checkpoint 原样。"""
    ck, tmp = env["ck"], env["tmp"]
    man_sha = hashlib.sha256((ck / "manifest.json").read_bytes()).hexdigest()
    out = tmp_path / "health_w2c.json"
    run(HEALTH, ["--tiny", "--model", QWEN, "--checkpoint", str(ck), "--limit", "1", "--skip-ifeval-scoring", "--out", str(out)], tmp)
    rep = json.loads(out.read_text())
    assert rep["ok"] is True and rep["checkpoint"]["manifest_sha256"] == man_sha and rep["checkpoint"]["merge_check"]["pass"] is True
    assert rep["run"]["config"]["checkpoint"]["manifest_sha256"] == man_sha and rep["model"]["key"] == QWEN
    assert json.loads((tmp_path / "health_w2c_merge_check.json").read_text())["pass"] is True
    cmp_out = tmp_path / "cmp.json"
    run(HEALTH, ["--compare", str(out), str(out), "--out", str(cmp_out)], tmp)
    assert json.loads(cmp_out.read_text())["any_flag"] is False
    bad = run(HEALTH, ["--tiny", "--model", QWEN, "--merge-check-n", "2", "--out", str(tmp_path / "x.json")], tmp, ok=False)
    assert bad.returncode == 2 and "只配 --checkpoint" in bad.stderr
    bad = run(HEALTH, ["--tiny", "--model", "gemma-4-e2b-it", "--checkpoint", str(ck), "--limit", "1", "--skip-ifeval-scoring", "--out",
                       str(tmp_path / "y.json")], tmp, ok=False)
    assert bad.returncode == 2 and "构造方式" in json.loads((tmp_path / "y.json").read_text())["error"]
    # T 文本读数（口径同 E3 的 C3 对照表）
    tt = tmp_path / "ttext"
    run(RUN_E1, ["--tiny", "--model", QWEN, "--task", "refexp_box", "count", "--subset", "eval", "--limit", "2", "--checkpoint", str(ck),
                 "--checkpoint-label", "t_text/w2c/seed0", "--out", str(tt)], tmp)
    for task in ("refexp_box", "count"):
        d = tt / "t_text" / "w2c" / "seed0" / QWEN / task
        assert json.loads((d / "metrics.json").read_text())["n"] == 2
        assert yaml.safe_load((d / "config.yaml").read_text())["checkpoint"]["manifest_sha256"] == man_sha
