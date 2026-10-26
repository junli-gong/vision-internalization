"""DEV-0065：W2-c 加回放（D125-2 (b)）——回放数据（src/w2/w2c_replay.py、scripts/build_w2c_replay.py）与训练的回放流（src/w2/w2c_train.py、
scripts/train_w2c.py --replay-data、configs/w2c_replay_train.json）。CPU fp32 小型随机 Qwen，测试造的数据。

- 回放数据的规则：规范化与重叠（全文相同或相同的连续 13 词；短题只比全文）；dolly 的四种剔除与 hash_rank 顺序；健康检查 200 题的 sha256
  核对、MMLU-Pro parquet 的读法；计数 = W2-c 的训练图 × 训练池里这张图上的类别，CountBench 的图（同键或感知哈希近重复）整张剔除；复数表
  覆盖 COCO 80 类；manifest 的自洽核对。
- tiny 端到端（命令行，子进程）：输出齐全、行的格式、生成口径（通用 = 健康检查的、计数 = E1 计数的）；中途中断（半行、半批、缺一类）后
  续跑与一次跑完逐字节相同；分阶段（plan → general → count）与一次跑完相同；换了抽样就拒绝混用；用法错误。
- 训练：configs/w2c_replay_train.json 除 replay 一节外与 configs/w2c_train.json 逐项相同，回放是两条流（通用 3、计数 2，权重各 0.5；
  管理员裁定）；配置核对；回放数据的核对（同一个 base、跑完、截断的不进池、目标以停止 token 结尾）；四条流的日程与损失（每条流各自除以
  自己的监督 token 数再乘权重；合成一条流的写法里微批不满也照样）；**默认配置逐字节不变**
  （7f53f240——RUN-0069 训出 W2-c 的那一版——的 src/w2/w2c_train.py 与 scripts/train_w2c.py 原样存在 tests/fixtures/w2c/，同一进程、同一组
  参数下输出逐字节相同）；带回放的 tiny 训练中途崩溃后续跑、分段运行与一次跑完逐位相同；检查点能被 run_w2.py --setting w2c、
  health_check.py、run_e1.py 直接评测。
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from src.adapters import registry
from src.adapters.checkpoint import policy_spec
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.data.subsets import hash_rank
from src.e1.prompts import PROMPTS_PATH, load_prompt_config
from src.e1.tasks import DataPaths
from src.e3.train import leaf_dir
from src.health import protocol as HP
from src.w2 import runner as R
from src.w2 import w2c_data as WD
from src.w2 import w2c_replay as RP
from src.w2 import w2c_train as T
from src.w2.config import load_w2_config
from tests import _w2c_fakes as FK
from tests._w2_draft import w2_draft_doc

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / "scripts" / "build_w2c_replay.py"
TRAIN = REPO / "scripts" / "train_w2c.py"
RUN_W2 = REPO / "scripts" / "run_w2.py"
HEALTH = REPO / "scripts" / "health_check.py"
RUN_E1 = REPO / "scripts" / "run_e1.py"
QWEN = "qwen3.5-0.8b"
CFG = T.load_w2c_config()
CFG_R = T.load_w2c_config(T.W2C_REPLAY_CONFIG_PATH)
W2 = load_w2_config()
IMPL = R.load_impl(R.DEFAULT_CANDIDATES_MODULE, R.DEFAULT_RENDER_MODULE)
PROMPTS = load_prompt_config(PROMPTS_PATH)
SNAP = REPO / "tests" / "fixtures" / "w2c"
SNAP_SHA256 = {  # 7f53f240（DEV-0065 之前、RUN-0069 训出 W2-c 的那一版）的文件原样
    "w2c_train_7f53f240.py": "ce3450d240e5bf4ae822ccea15cd1a9efffce743e5f969211df7ded3f49a3d12",
    "train_w2c_7f53f240.py": "51d015b76f243ec2acd6e23fca8cc2f9712d71407035cc63ef8aa2e57af0b256",
}
FAKES = ["--candidates-module", "tests._w2_fakes", "--render-module", "tests._w2_fakes"]
GENS = [{"type": "grabcut_default"}, {"type": "box_scale", "percent": 20}, {"type": "box_scale", "percent": -20}, {"type": "box"}]
DET = ("train_log.jsonl", "config.yaml", "README.md", "train_summary.json")


def run(script: Path, args, tmp: Path, *, ok: bool = True) -> subprocess.CompletedProcess:
    env = {**os.environ, "TMPDIR": str(tmp), "HF_HUB_OFFLINE": "1"}
    env.pop("VISINT_DATA_ROOT", None)
    p = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, cwd=REPO, env=env, timeout=1800)
    if ok:
        assert p.returncode == 0, p.stderr[-3000:]
    return p


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(p: Path, rows) -> None:
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def _tiny(cfg=CFG):
    fam = registry.get_model(QWEN).family
    t = cfg.doc["tiny"]
    return make_adapter(QWEN, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _spec(ad, seed=0):
    e = registry.get_model(QWEN)
    return T.W2CRunSpec(model_key=e.key, family=e.family, repo=e.repo, revision=e.revision, adapter_model_key=ad.model_key, tiny=True,
                        image_policy=policy_spec(ad.image_policy), model_dtype="fp32", seed=seed)


def _tensors(d: Path) -> dict:
    from safetensors.torch import load_file

    out = {}
    for f in sorted(d.rglob("*.safetensors")):
        for k, v in load_file(str(f)).items():
            out[f"{f.relative_to(d)}:{k}"] = v
    return out


def _streams(rc) -> dict:
    return {s: v["kinds"] for s, v in rc["replay"]["streams"].items()}


def load_rdata(run_dir: Path, rc=None, **kw):
    rc = rc or T.resolve_w2c_run(CFG_R, tiny=True)
    e = registry.get_model(QWEN)
    args = {"streams": _streams(rc), "model_key": e.key, "revision": e.revision, "tiny": True, "data_root": FK.ROOT, **kw}
    return RP.load_replay_data(run_dir, **args)


def inject_stops(run_dir: Path, *, keep_truncated: dict | None = None) -> None:
    """随机模型不会停（同 tests/test_w2c_eval.py 换框的做法）：把回答换成以停止 token 结尾的短回答（keep_truncated 里的 id 保持截断）。"""
    ad = _tiny()
    stop = ad.end_of_turn_token_id()
    keep = keep_truncated or {}
    for kind in RP.KINDS:
        p = RP.data_file(run_dir, kind)
        rows = jsonl(p)
        for i, r in enumerate(rows):
            if r["id"] in keep.get(kind, ()):
                continue
            text = f"Fixture answer {i}." if kind == RP.GENERAL else str(i % 4)
            ids = list(ad.tokenizer(text, add_special_tokens=False)["input_ids"]) + [stop]
            r.update(answer=text, raw_text=text + "<|im_end|>", token_ids=ids, n_generated_tokens=len(ids), stopped=True, truncated=False)
        write_jsonl(p, rows)


@pytest.fixture(scope="module")
def w2c_dir(tmp_path_factory):
    return FK.write_fake_w2c_data(tmp_path_factory.mktemp("w2cr") / "w2c_data")


@pytest.fixture(scope="module")
def data(w2c_dir):
    return T.load_w2c_data(w2c_dir, w2=W2, sources=CFG.doc["data"]["sources"], data_root=FK.ROOT, require_manifest=True)


@pytest.fixture(scope="module")
def built(w2c_dir, tmp_path_factory):
    """tiny 端到端的回放数据（命令行）：随机模型的回答全被截断。"""
    tmp = tmp_path_factory.mktemp("rp")
    out = tmp / "rp"
    p = run(BUILD, ["--tiny", "--w2c-data", w2c_dir, "--out", out], tmp)
    return {"out": out, "tmp": tmp, "stdout": p.stdout}


@pytest.fixture(scope="module")
def replay_run(built, tmp_path_factory):
    """回放数据换成以停止 token 结尾的回答；通用的最后一条保持截断（不进池）。"""
    dst = tmp_path_factory.mktemp("rpi") / "rp"
    shutil.copytree(built["out"], dst)
    last = jsonl(RP.data_file(dst, RP.GENERAL))[-1]["id"]
    inject_stops(dst, keep_truncated={RP.GENERAL: [last]})
    return dst


# ---------------------------------------------------------------------------
# 回放数据的规则
# ---------------------------------------------------------------------------


def test_overlap_rules():
    opt = "because of rayleigh scattering of sunlight in the upper atmosphere by small molecules of air and dust"
    refs = {"health_check": [RP.RefItem("health_check", "gsm8k/1", "A farmer has 12 cows and buys 7 more. How many cows does he have?")],
            "mmlu_pro": [RP.RefItem("mmlu_pro", "5", "Why is the sky blue?", (opt,))]}
    idx = RP.OverlapIndex(refs)
    assert idx.find("why is the SKY blue ?!") == {"rule": "exact", "set": "mmlu_pro", "item": "5"}
    assert idx.find("Ｗｈｙ ｉｓ ｔｈｅ ｓｋｙ ｂｌｕｅ？")["rule"] == "exact"  # NFKC：全角
    assert idx.find("Why is the sky blue today?") is None  # 短题只比全文
    w = opt.split()
    hit = idx.find("Explain: " + " ".join(w[2:15]) + ", please.")  # 选项里连续 13 个词
    assert hit["rule"] == "ngram13" and hit["set"] == "mmlu_pro" and hit["ngram"] == " ".join(w[2:15])
    assert idx.find("Explain: " + " ".join(w[2:14])) is None  # 12 个词不算
    assert idx.find("a farmer has 12 cows and buys 7 more how many cows does he have")["rule"] == "exact"
    assert RP.norm_words("It's 3.5 km—far!") == ["it", "s", "3", "5", "km", "far"]
    assert RP.NGRAM == 13


def _health_items() -> dict:
    return json.loads((REPO / HP.ITEMS_PATH).read_text(encoding="utf-8"))["items"]


def test_select_general_rules():
    rows = RP.read_jsonl(RP.raw_path(RP.TINY_RAW_DIR, RP.DOLLY))
    q = _health_items()["gsm8k"][0]["question"]
    rows.append({"instruction": q, "context": "", "response": "x", "category": "open_qa"})  # 与健康检查的一道题全文相同
    raw = RP.resolve_raw(RP.TINY_RAW_DIR, tiny=True)
    assert raw["mmlu_pro"] is None and raw["gsm8k"] is None and raw["ifeval"] is not None  # tiny：缺的来源只记 missing
    refs, info = RP.reference_sets(raw, items_path=REPO / HP.ITEMS_PATH, tiny=True)
    assert info["health_check"]["n_items"] == 200 and info["mmlu_pro"]["missing"] is True and info["ifeval"]["n_items"] == 2
    sel = RP.select_general(rows, RP.OverlapIndex(refs), n=100, allow_fewer=True)
    st = sel["steps"]
    assert {k: v["n"] for k, v in st["removed"].items()} == {"too_long": 1, "structural_marker": 1, "eval_overlap": 2, "duplicate": 1}
    hits = {h["id"]: h for h in st["eval_overlap"]["hits"]}
    assert hits["dolly/7"]["rule"] == "ngram13" and hits["dolly/7"]["set"] == "ifeval"
    assert hits["dolly/12"] == {"id": "dolly/12", "rule": "exact", "set": "health_check", "item": f"gsm8k/{_health_items()['gsm8k'][0]['id']}"}
    ids = [it["id"] for it in sel["items"]]
    kept = [f"dolly/{i}" for i in (0, 1, 2, 3, 8, 9, 10, 11)]
    assert sorted(ids) == sorted(kept) and ids == hash_rank(kept, RP.PURPOSE_GENERAL, RP.SALT) and st["kept"]["n"] == 8
    by = {it["id"]: it for it in sel["items"]}
    assert by["dolly/1"]["prompt"] == rows[1]["instruction"] + "\n\n" + rows[1]["context"]  # 有 context：接在空行之后
    assert by["dolly/0"]["prompt"] == "Name three primary colors." and by["dolly/0"]["source"] == {
        "dataset": "databricks-dolly-15k", "row": 0, "category": "open_qa"}
    assert [it["id"] for it in RP.select_general(rows, RP.OverlapIndex(refs), n=3)["items"]] == ids[:3]  # 前 k 条就是 n = k 的抽样
    with pytest.raises(RP.ReplayDataError, match="不够"):
        RP.select_general(rows, RP.OverlapIndex(refs), n=100)


def test_sources_and_pins(tmp_path):
    assert (RP.DOLLY.repo, RP.DOLLY.revision, RP.DOLLY.file, RP.DOLLY.license, RP.DOLLY.n_rows) == (
        "databricks/databricks-dolly-15k", "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a", "databricks-dolly-15k.jsonl", "CC-BY-SA-3.0", 15_011)
    assert RP.DOLLY.file_sha256 == "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
    assert set(RP.all_sources()) == {"dolly", "mmlu_pro", "gsm8k", "ifeval"} and RP.all_sources()["mmlu_pro"] is HP.SOURCES["mmlu_pro"]
    h = RP.health_refs(REPO / HP.ITEMS_PATH)
    assert len(h) == 200 and sum(1 for r in h if r.id.startswith("mmlu_pro/") and len(r.extra) >= 3) == 100
    bad = tmp_path / "items.json"
    bad.write_bytes((REPO / HP.ITEMS_PATH).read_bytes() + b" ")
    with pytest.raises(RP.ReplayDataError, match="ITEMS_SHA256"):
        RP.health_refs(bad)
    raw = tmp_path / "raw"
    mp = RP.raw_path(raw, HP.SOURCES["mmlu_pro"])
    mp.parent.mkdir(parents=True)
    shutil.copy(REPO / "tests" / "fixtures" / "health" / "mmlu_pro_validation.parquet", mp)  # 同一写出器、同一 schema 的 validation 文件
    refs = RP.source_refs("mmlu_pro", mp)
    assert refs and all(r.set == "mmlu_pro" and r.main and len(r.extra) >= 3 for r in refs)
    assert RP.source_refs("ifeval", RP.raw_path(RP.TINY_RAW_DIR, HP.SOURCES["ifeval"]))[0].id == "9001"
    with pytest.raises(RP.ReplayDataError, match="不在"):
        RP.resolve_raw(raw, tiny=False)  # 非 tiny：dolly 与三个来源都必须在
    with pytest.raises(RP.ReplayDataError, match="sha256"):
        RP.fetch(HP.SOURCES["mmlu_pro"], raw, download=False)  # 不是固定 revision 的文件
    with pytest.raises(RP.ReplayDataError, match="sha256"):
        RP.fetch(RP.DOLLY, RP.TINY_RAW_DIR, download=False)  # fixture 是测试造的，不是真数据


def test_coco_plurals_and_count_prompt():
    assert len(RP.COCO_PLURALS) == 80 and all(v for v in RP.COCO_PLURALS.values())
    assert RP.count_object("person") == ("people", "coco_plural") and RP.count_object("sheep") == ("sheep", "coco_plural")
    assert RP.count_object("unicorn") == ("unicorn", "fallback_name")
    tpl = PROMPTS.entry("count", "qwen3.5")["template"]
    assert RP.count_prompt(tpl, "dogs") == "How many dogs are there in the image? Answer with a single number."
    assert all(PROMPTS.entry("count", f)["template"] == tpl for f in ("gemma4", "internvl3.5"))  # 三个家族逐字相同


def _pool():
    paths = WD.W2CPaths(train_root=FK.ROOT, eval=DataPaths(root=RP.TINY_EVAL_ROOT))
    return WD.load_train_pool(paths, real_model=False)[0]


def test_select_count_rules(w2c_dir):
    rows, shards = RP.read_w2c_rows(w2c_dir)
    pool = _pool()
    cb, cb_info = RP.countbench_hashes(RP.TINY_EVAL_ROOT)
    assert cb_info["n_images"] == len(cb) >= 3 and all(k.startswith("countbench:") for k in cb)
    tpl = PROMPTS.entry("count", "qwen3.5")["template"]
    sel = RP.select_count(rows, pool, cb, train_root=FK.ROOT, template=tpl, n=100, allow_fewer=True)
    by_id = {s.id: s for s in pool}
    keys = {by_id[r["id"]].image_keys[0] for r in rows}
    cats = {k: {s.fields["category"] for s in pool if s.image_keys[0] == k} for k in keys}
    want = sorted(f"{k}|{c}" for k in keys for c in cats[k])
    ids = [it["id"] for it in sel["items"]]
    assert sorted(ids) == want and ids == hash_rank(want, RP.PURPOSE_COUNT, RP.SALT)
    st = sel["steps"]
    assert st["w2c_images"]["n_images"] == len(keys) and st["population"]["n_pairs"] == len(want)
    assert st["countbench"]["n_images_removed"] == 0
    for it in sel["items"]:
        k, c = it["id"].split("|")
        s = next(x for x in pool if x.image_keys[0] == k)
        assert it["image"] == str(Path(s.images[0]).relative_to(FK.ROOT)) and (FK.ROOT / it["image"]).is_file()
        assert it["image_size"] == list(s.fields["image_size"]) and it["source"]["category"] == c
        assert it["prompt"] == tpl.format(object=RP.COCO_PLURALS[c]).strip() and it["source"]["object_rule"] == "coco_plural"
    # 与 CountBench 的图重合（感知哈希相同、或同键）：整张图剔除
    k0 = sorted(keys)[0]
    ph = next(s.phash[0] for s in pool if s.image_keys[0] == k0)
    for cb2 in ({**cb, "countbench:999": ph}, {**cb, k0: None}):
        s2 = RP.select_count(rows, pool, cb2, train_root=FK.ROOT, template=tpl, n=100, allow_fewer=True)
        assert s2["steps"]["countbench"]["images_removed"] == [k0] and all(it["source"]["image_key"] != k0 for it in s2["items"])
    with pytest.raises(RP.ReplayDataError, match="不够"):
        RP.select_count(rows, pool, cb, train_root=FK.ROOT, template=tpl, n=len(want) + 1)
    with pytest.raises(RP.ReplayDataError, match="manifest 的图"):
        RP.select_count(rows, pool, cb, train_root=FK.ROOT, template=tpl, n=3, image_of={r["id"]: "coco:1" for r in rows})
    with pytest.raises(RP.ReplayDataError, match="不在 E3 的训练池"):
        RP.select_count([{**rows[0], "id": "refcoco_unc/train/x"}], pool, cb, train_root=FK.ROOT, template=tpl, n=1)


def test_check_w2c_manifest_on_real_dev0060_output():
    """DEV-0060 在 06 上的真实输出（RUN-0067，入库）：manifest 核对通过、选中的 id 恰是 16 片的行；fake 数据没有真 manifest 就拒绝。"""
    d = REPO / "experiments" / "W2" / "0067_w2c_data_hpcgpu06" / "w2c_data"
    if not (d / "manifest.json").is_file():
        pytest.skip("RUN-0067 的数据不在这个检出里")
    rows, shards = RP.read_w2c_rows(d)
    chk = RP.check_w2c_manifest(d, rows)
    assert len(rows) == 16_000 and len(shards) == 16 and chk["checks"]["selected_on_w2_eval_images"] == 0
    assert len(set(chk["image_of"].values())) == 10_438
    with pytest.raises(RP.ReplayDataError, match="选中的 id"):
        RP.check_w2c_manifest(d, rows[:-1])


# ---------------------------------------------------------------------------
# tiny 端到端（命令行）
# ---------------------------------------------------------------------------


def test_build_tiny_outputs(built, w2c_dir):
    out = built["out"]
    man, man_sha = RP.load_manifest(out)
    assert RP.check_manifest(man) == [] and man["tiny"] is True and man["family"] == "qwen3.5"
    ident = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert ident["kind"] == "w2c_replay_data" and ident["manifest_sha256"] == man_sha and ident["model"]["key"] == QWEN
    g, c = ident["generation"]["general"], ident["generation"]["count"]
    assert (g["order"], g["enable_thinking"], g["decoding"], g["max_new_tokens"]) == (HP.ORDER, False, dict(HP.GREEDY_KWARGS), 8)
    assert (c["order"], c["enable_thinking"], c["decoding"], c["max_new_tokens"]) == (PROMPTS.order, False, PROMPTS.decoding(), 4)
    assert g["batch_size"] == c["batch_size"] == 4 and g["method"] == "generate_batch"
    assert RP.MAX_NEW_TOKENS == {"general": 512, "count": 16} and RP.DEFAULT_N == {"general": 2000, "count": 2000}  # 任务书的正式取值
    for kind in RP.KINDS:
        rows = jsonl(RP.data_file(out, kind))
        assert [r["id"] for r in rows] == [it["id"] for it in man["items"][kind]] and len(rows) == RP.TINY_N[kind]
        for r, it in zip(rows, man["items"][kind]):
            assert tuple(r) == RP.ROW_KEYS and r["prompt"] == it["prompt"] and r["truncated"] is (not r["stopped"])
            assert r["truncated"] and r["n_generated_tokens"] == len(r["token_ids"]) == r["max_new_tokens"]  # 随机模型不会停：照实记
            assert r["enable_thinking"] is False and r["kind"] == kind
            if kind == RP.COUNT:
                assert r["images"] == [it["image"]] and r["n_visual_tokens"] > 0 and r["order"] == "image_first"
            else:
                assert r["images"] == [] and r["image_size"] is None and r["n_visual_tokens"] == 0 and r["order"] == HP.ORDER
    m = json.loads((out / "metrics.json").read_text())
    assert m["complete"] is True and m["general"]["n_truncated"] == 6 and m["count"]["n_usable"] == 0
    comp = json.loads((out / "compute.json").read_text())
    assert comp["n_samples"] == 12 and comp["generated_tokens"] == 6 * 8 + 6 * 4
    assert "plan：通用 6 条" in built["stdout"] and "complete = True" in built["stdout"]
    # 选中的提示与参照集不重叠（重新核对一遍）；计数的图不在 CountBench 上
    refs, _ = RP.reference_sets(RP.resolve_raw(RP.TINY_RAW_DIR, tiny=True), items_path=REPO / HP.ITEMS_PATH, tiny=True)
    idx = RP.OverlapIndex(refs)
    assert all(idx.find(it["prompt"]) is None for it in man["items"]["general"])
    assert man["count"]["countbench"]["n_images_removed"] == 0 and man["count"]["train_pool"]["e3_exclusions"]["eval_sets"]
    assert "countbench/eval" in man["count"]["train_pool"]["e3_exclusions"]["eval_sets"]


def test_build_resume_and_stages_are_bytewise_identical(built, w2c_dir, tmp_path):
    a = built["out"]
    files = ["replay/manifest.json", "replay/general.jsonl", "replay/count.jsonl", "config.yaml", "metrics.json", "README.md"]
    # 中断：通用写了 1 条（半批）加半行，计数一条没写 → 同一条命令续跑
    b = tmp_path / "b"
    shutil.copytree(a, b)
    g = RP.data_file(b, RP.GENERAL)
    lines = g.read_text(encoding="utf-8").splitlines(keepends=True)
    g.write_text(lines[0] + lines[1][:40], encoding="utf-8")
    RP.data_file(b, RP.COUNT).unlink()
    run(BUILD, ["--tiny", "--w2c-data", w2c_dir, "--out", b], tmp_path)
    for f in files:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    assert len(list((b / "logs").glob("run_*.json"))) == 2
    # 分阶段：plan → general → count
    c = tmp_path / "c"
    for st in ("plan", "general", "count"):
        run(BUILD, ["--tiny", "--w2c-data", w2c_dir, "--out", c, "--stage", st], tmp_path)
        if st == "plan":
            assert sorted(p.name for p in c.rglob("*") if p.is_file()) == ["manifest.json"]
    for f in files:
        assert (a / f).read_bytes() == (c / f).read_bytes(), f
    # 换了抽样（--n-general）：manifest 不同，拒绝混用
    p = run(BUILD, ["--tiny", "--w2c-data", w2c_dir, "--out", c, "--n-general", "3"], tmp_path, ok=False)
    assert p.returncode == 2 and "拒绝混用" in p.stderr and list((c / "errors").glob("error_*.json"))
    # 已写的行不是 manifest 的前缀：拒绝续跑
    d = tmp_path / "d"
    shutil.copytree(a, d)
    rows = jsonl(RP.data_file(d, RP.GENERAL))
    write_jsonl(RP.data_file(d, RP.GENERAL), rows[1:2])
    p = run(BUILD, ["--tiny", "--w2c-data", w2c_dir, "--out", d], tmp_path, ok=False)
    assert p.returncode == 2 and "前缀" in p.stderr


def test_build_cli_usage(tmp_path, w2c_dir):
    p = run(BUILD, ["--out", tmp_path / "0001_x"], tmp_path, ok=False)
    assert p.returncode == 2 and "--raw-dir" in p.stderr
    p = run(BUILD, ["--raw-dir", "r", "--w2c-data", w2c_dir, "--train-data-root", "t", "--eval-data-root", "e", "--out", tmp_path / "x"],
            tmp_path, ok=False)
    assert p.returncode == 2 and "运行编号" in p.stderr
    p = run(BUILD, ["--stage", "fetch"], tmp_path, ok=False)
    assert p.returncode == 2 and "--raw-dir" in p.stderr
    p = run(BUILD, ["--tiny", "--out", tmp_path / "y"], tmp_path, ok=False)
    assert p.returncode == 2 and "--w2c-data" in p.stderr
    p = run(BUILD, ["--tiny", "--w2c-data", w2c_dir, "--out", tmp_path / "z", "--stage", "general"], tmp_path, ok=False)
    assert p.returncode == 2 and "先跑 --stage plan" in p.stderr


# ---------------------------------------------------------------------------
# 训练：配置与回放数据
# ---------------------------------------------------------------------------


def test_replay_config_is_w2c_plus_replay_section():
    w, r = json.loads((REPO / "configs" / "w2c_train.json").read_text()), json.loads(T.W2C_REPLAY_CONFIG_PATH.read_text())
    assert list(r)[-1] == "replay" and {k: v for k, v in r.items() if k != "replay"} == w and list(r)[:-1] == list(w)
    rp = r["replay"]
    assert rp["data"]["path"] is None
    assert rp["streams"] == [  # 管理员裁定（DEV-0065 待裁第 1 条）：两条回放流，每步 3 + 2，权重各 0.5
        {"name": "replay_general", "kinds": ["general"], "per_step": 3, "tiny_per_step": 2, "loss_weight": 0.5},
        {"name": "replay_count", "kinds": ["count"], "per_step": 2, "tiny_per_step": 1, "loss_weight": 0.5}]
    for tiny in (False, True):
        a, b = T.resolve_w2c_run(CFG, tiny=tiny), T.resolve_w2c_run(CFG_R, tiny=tiny)
        assert "replay" not in a and {k: v for k, v in b.items() if k != "replay"} == a
    rc = T.resolve_w2c_run(CFG_R, tiny=False)
    assert rc["streams"]["per_step"] == {"propose": 8, "select": 8} and rc["batch_size"] == 16
    assert rc["replay"] == {"streams": {
        "replay_general": {"kinds": ["general"], "per_step": 3, "loss_weight": 0.5, "micro_batches": 1, "index": 2},
        "replay_count": {"kinds": ["count"], "per_step": 2, "loss_weight": 0.5, "micro_batches": 1, "index": 3}},
        "order": ["replay_general", "replay_count"], "per_step_total": 5}
    assert round(5 / (16 + 5), 3) == 0.238  # 回放约 24%（每步 8 + 8 + 3 + 2）
    names = ("propose", "select", "replay_general", "replay_count")
    assert T.stream_names(rc) == names and T.stream_order(rc) == list(names)
    w = T.stream_weights(rc)
    assert w == {"propose": 1.0, "select": 1.0, "replay_general": 0.5, "replay_count": 0.5}
    assert (w["replay_general"] + w["replay_count"]) / sum(w.values()) == pytest.approx(1 / 3)  # 回放的总权重同任务书的 1 : 1 : 1
    rt = T.resolve_w2c_run(CFG_R, tiny=True)
    assert {s: (v["per_step"], v["micro_batches"]) for s, v in rt["replay"]["streams"].items()} == {
        "replay_general": (2, 1), "replay_count": (1, 1)} and rt["replay"]["per_step_total"] == 3
    assert T.stream_names(T.resolve_w2c_run(CFG, tiny=True)) == T.STREAMS


@pytest.mark.parametrize("edit, msg", [
    (lambda r: r["streams"][0].update(name="extra"), "开头"),
    (lambda r: r["streams"].append(dict(r["streams"][0])), "不重名"),
    (lambda r: r["streams"][0].update(kinds=["general", "poems"]), "子集"),
    (lambda r: r["streams"].append({"name": "replay2", "kinds": ["count"], "per_step": 1, "tiny_per_step": 1, "loss_weight": 1}), "不重叠"),
    (lambda r: r["streams"][0].update(per_step=0), "正整数"),
    (lambda r: r["streams"][0].update(tiny_per_step=1.5), "正整数"),
    (lambda r: r["streams"][0].update(loss_weight=0), "正数"),
    (lambda r: r["streams"][0].update(loss_weight=True), "正数"),
    (lambda r: r.update(streams=[]), "非空"),
    (lambda r: r["data"].update(path=""), "path"),
])
def test_replay_config_validation(tmp_path, edit, msg):
    d = copy.deepcopy(dict(CFG_R.doc))
    edit(d["replay"])
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d, ensure_ascii=False))
    with pytest.raises(T.W2CConfigError, match=msg):
        T.load_w2c_config(p)


def test_load_replay_data_checks(replay_run, built, tmp_path):
    rd = load_rdata(replay_run)
    assert {s: len(p) for s, p in rd.pools.items()} == {"replay_general": 5, "replay_count": 6}  # 通用的最后一条截断：不进池
    trunc = [i for i, r in enumerate(rd.rows) if r["truncated"]]
    assert len(trunc) == 1 and all(trunc[0] not in p for p in rd.pools.values()) and rd.rows[trunc[0]]["kind"] == "general"
    assert all(rd.rows[i]["kind"] == "general" for i in rd.pool("replay_general"))
    assert all(rd.rows[i]["kind"] == "count" for i in rd.pool("replay_count"))
    ident = rd.identity()
    assert ident["pools"]["replay_count"]["n"] == 6 and ident["by_kind"]["general"]["n_truncated"] == 1
    assert ident["by_kind"]["count"]["n_usable"] == 6 and ident["model"]["key"] == QWEN
    assert str(replay_run) not in json.dumps(ident)  # 运行身份不含路径
    merged = load_rdata(replay_run, streams={"replay": ["general", "count"]})  # 合成一条流的写法（未采用）
    assert {s: len(p) for s, p in merged.pools.items()} == {"replay": 11}
    e = registry.get_model("qwen3.5-2b")
    with pytest.raises(RP.ReplayDataError, match="同一个 base"):
        load_rdata(replay_run, model_key=e.key, revision=e.revision)
    with pytest.raises(RP.ReplayDataError, match="同一个 base"):
        load_rdata(replay_run, tiny=False)
    with pytest.raises(RP.ReplayDataError, match="池是空的"):
        load_rdata(built["out"])  # 全部截断
    with pytest.raises(RP.ReplayDataError, match="不重叠"):
        load_rdata(replay_run, streams={"replay": ["general", "count"], "replay_b": ["count"]})
    inc = tmp_path / "inc"
    shutil.copytree(replay_run, inc)
    p = RP.data_file(inc, RP.COUNT)
    write_jsonl(p, jsonl(p)[:-1])
    with pytest.raises(RP.ReplayDataError, match="没有跑完"):
        load_rdata(inc)
    bad = tmp_path / "bad"
    shutil.copytree(replay_run, bad)
    p = RP.data_file(bad, RP.GENERAL)
    rows = jsonl(p)
    rows[0]["enable_thinking"] = True
    write_jsonl(p, rows)
    with pytest.raises(RP.ReplayDataError, match="关思考"):
        load_rdata(bad)
    tam = tmp_path / "tam"
    shutil.copytree(replay_run, tam)
    mp = RP.manifest_path(tam)
    mp.write_text(mp.read_text(encoding="utf-8").replace("\"n_requested\": 6", "\"n_requested\": 7", 1), encoding="utf-8")
    with pytest.raises(RP.ReplayDataError, match="sha256"):
        load_rdata(tam)
    # 目标不以停止 token 结尾：开训前拒绝
    nostop = tmp_path / "nostop"
    shutil.copytree(replay_run, nostop)
    p = RP.data_file(nostop, RP.COUNT)
    rows = jsonl(p)
    rows[0]["token_ids"] = rows[0]["token_ids"][:-1] + [rows[0]["token_ids"][0]]
    write_jsonl(p, rows)
    with pytest.raises(RP.ReplayDataError, match="停止 token"):
        T.ReplayBuilder(load_rdata(nostop), data_root=FK.ROOT, stop_ids=_tiny().stop_token_ids())


# ---------------------------------------------------------------------------
# 训练：日程与损失
# ---------------------------------------------------------------------------


def _one_step(data, rdata, rc, monkeypatch):
    ad = _tiny()
    tr = T.W2CTrainer(ad, CFG_R, rc, seed=0, number_ids=None)
    tr.setup_fresh()
    from src.e3.tokens import family_stop_id

    b = T.W2CBuilder(CFG_R, data, w2=W2, impl=IMPL, data_root=FK.ROOT, stop_id=family_stop_id(ad),
                     encode=lambda s: list(ad.tokenizer(s, add_special_tokens=False)["input_ids"]), family="qwen3.5",
                     format_boxes=ad.format_boxes)
    rb = T.ReplayBuilder(rdata, data_root=FK.ROOT, stop_ids=ad.stop_token_ids())
    total = rc["replay"]["per_step_total"]
    sched = T.W2CSchedule(data, seed=0, per_step=rc["streams"]["per_step"], order=rc["streams"]["order"], extra_per_step=total)
    rs = T.ReplaySchedule(rdata, seed=0, spec=rc["replay"], base_batch=sched.batch_size)
    plans = sched.step_plans(1) + rs.step_plans(1)
    assert sorted(p.k for p in plans) == list(range(sched.step_size, 2 * sched.step_size)) and sched.step_size == 4 + total
    ex = [rb.build(p) if p.stream in rc["replay"]["streams"] else b.build(p) for p in plans]
    sums, scaled = [], []
    orig_loss, orig_bwd = ad.chunked_lm_loss, torch.Tensor.backward

    def loss(*a, **k):
        out = orig_loss(*a, **k)
        sums.append(float(out.detach()))
        return out

    def bwd(self, *a, **k):
        scaled.append(float(self.detach()))
        return orig_bwd(self, *a, **k)

    monkeypatch.setattr(ad, "chunked_lm_loss", loss)
    monkeypatch.setattr(torch.Tensor, "backward", bwd)
    rec, times, comps = tr.optimize(ex)
    monkeypatch.undo()
    return rec, ex, sums, scaled, comps


def test_replay_streams_schedule_and_loss(data, replay_run, monkeypatch):
    """仓库配置（两条回放流，权重 0.5、0.5）与合成一条流的写法（权重 1，3 = 2 + 1 走到不满的微批）。"""
    rc = T.resolve_w2c_run(CFG_R, tiny=True)
    rdata = load_rdata(replay_run)
    rec, ex, sums, scaled, comps = _one_step(data, rdata, rc, monkeypatch)
    names = ["propose", "select", "replay_general", "replay_count"]
    assert rec["n_by_stream"] == {"propose": 2, "select": 2, "replay_general": 2, "replay_count": 1} and len(comps) == 7
    assert list(rec["loss_by_stream"]) == names and set(rec["by_stream"]) == set(names)
    # 微批：提出 2、挑选 2、通用 2、计数 1；每个微批的损失除以本流这一步的监督 token 数，再乘本流的权重
    assert len(sums) == len(scaled) == 4
    n = rec["n_supervised_by_stream"]
    w = {"propose": 1.0, "select": 1.0, "replay_general": 0.5, "replay_count": 0.5}
    for st, v, sc in zip(names, sums, scaled):
        assert sc == pytest.approx(w[st] * v / n[st]) and rec["loss_by_stream"][st] == pytest.approx(v / n[st])
    assert rec["loss"] == pytest.approx(sum(w[st] * rec["loss_by_stream"][st] for st in names))  # 1 : 1 : 0.5 : 0.5
    for st, kind in (("replay_general", "general"), ("replay_count", "count")):
        rep = [e for e in ex if e.plan.stream == st]
        assert n[st] == sum(len(e.target_ids) for e in rep)
        for e in rep:
            row = rdata.rows[e.plan.row]
            assert row["kind"] == kind and not row["truncated"]
            assert e.target_ids == row["token_ids"] and e.instruction == row["prompt"] and e.order == row["order"]
            assert len(e.images) == (1 if kind == "count" else 0)
    assert n["replay_count"] == 2  # 一个数字 + 回合结束：单独归一，不被几百个 token 的通用回答淹没
    # 合成一条流（权重 1）：回放 3 条 = 2 + 1，最后一个微批不满，两个微批都除以整条流的 token 数
    d = copy.deepcopy(dict(CFG_R.doc))
    d["replay"]["streams"] = [{"name": "replay", "kinds": ["general", "count"], "per_step": 5, "tiny_per_step": 3, "loss_weight": 1}]
    rc1 = T.resolve_w2c_run(T.W2CConfig(path="x", sha256="0" * 64, doc=d), tiny=True)
    rd1 = load_rdata(replay_run, rc=rc1)
    rec1, ex1, sums1, scaled1, _ = _one_step(data, rd1, rc1, monkeypatch)
    assert rec1["n_by_stream"] == {"propose": 2, "select": 2, "replay": 3} and len(sums1) == 4
    n1 = rec1["n_supervised_by_stream"]
    for st, v, sc in zip(["propose", "select", "replay", "replay"], sums1, scaled1):
        assert sc == pytest.approx(v / n1[st])
    assert rec1["loss_by_stream"]["replay"] == pytest.approx((sums1[2] + sums1[3]) / n1["replay"])
    assert rec1["loss"] == pytest.approx(sum(rec1["loss_by_stream"].values()))


def test_replay_schedule_is_stateless_and_covers_pool(replay_run):
    rc = T.resolve_w2c_run(CFG_R, tiny=False)
    rdata = load_rdata(replay_run, rc=rc)
    rs = T.ReplaySchedule(rdata, seed=0, spec=rc["replay"], base_batch=16)
    seen = {}
    for st in ("replay_general", "replay_count"):
        n = len(rdata.pool(st))
        seen[st] = [rs.plan(st, j).row for j in range(n)]
        assert sorted(seen[st]) == sorted(rdata.pool(st))  # 一个 epoch 恰好覆盖本流的池一次
    again = T.ReplaySchedule(rdata, seed=0, spec=rc["replay"], base_batch=16)
    assert [p.sample_id for p in again.step_plans(7)] == [p.sample_id for p in rs.step_plans(7)]
    plans = rs.step_plans(2)
    assert [p.stream for p in plans] == ["replay_general"] * 3 + ["replay_count"] * 2
    assert [p.k for p in plans] == [2 * 21 + 16 + i for i in range(5)]  # 每步 8 + 8 + 3 + 2
    other = T.ReplaySchedule(rdata, seed=1, spec=rc["replay"], base_batch=16)
    assert [other.plan("replay_count", j).row for j in range(len(seen["replay_count"]))] != seen["replay_count"]
    assert T.pool_position(3, 10, seed=0, stream="propose") == T.pool_position(3, 10, seed=0, stream="x", index=0)


# ---------------------------------------------------------------------------
# 默认配置逐字节不变（与 7f53f240 的原文件对照）
# ---------------------------------------------------------------------------


def _load_path(path: Path, name: str):
    saved = list(sys.path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    sys.path[:] = saved
    return mod


def _old_modules():
    for name, sha in SNAP_SHA256.items():
        assert hashlib.sha256((SNAP / name).read_bytes()).hexdigest() == sha, name
    old_t = _load_path(SNAP / "w2c_train_7f53f240.py", "_w2c_train_7f53f240")
    old_t.REPO_ROOT, old_t.W2C_CONFIG_PATH = T.REPO_ROOT, T.W2C_CONFIG_PATH  # 快照放在 tests/fixtures/w2c/ 下，按自己的位置推出的仓库根不对
    old_s = _load_path(SNAP / "train_w2c_7f53f240.py", "_train_w2c_7f53f240")
    old_s.T = old_t
    new_s = _load_path(TRAIN, "_train_w2c_dev0065")
    return old_t, old_s, new_s


def test_default_config_bytewise_identical_to_7f53f240(w2c_dir, tmp_path, capsys):
    old_t, old_s, new_s = _old_modules()
    oc = old_t.load_w2c_config(T.W2C_CONFIG_PATH)
    assert (oc.sha256, oc.path, oc.doc) == (CFG.sha256, CFG.path, CFG.doc)
    for tiny in (False, True):
        assert old_t.resolve_w2c_run(oc, tiny=tiny) == T.resolve_w2c_run(CFG, tiny=tiny)
        assert old_t.resolve_w2c_run(oc, tiny=tiny, steps=7) == T.resolve_w2c_run(CFG, tiny=tiny, steps=7)
    outs, prints = [], []
    for k, mod in enumerate((old_s, new_s)):
        out = tmp_path / f"o{k}"
        assert mod.main(["--tiny", "--model", QWEN, "--w2c-data", str(w2c_dir), "--out", str(out), "--stop-after-steps", "3"]) == 0
        assert mod.main(["--tiny", "--model", QWEN, "--w2c-data", str(w2c_dir), "--out", str(out), "--resume"]) == 0
        prints.append(capsys.readouterr().out)
        outs.append(leaf_dir(out, "w2c", QWEN, 0))
    assert prints[0] == prints[1] and "steps=4" in prints[1]
    a, b = outs
    for f in DET:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    for f in ("manifest.json", "tokens.json"):
        assert (a / "ckpt/final/model" / f).read_bytes() == (b / "ckpt/final/model" / f).read_bytes(), f
    ta, tb = _tensors(a / "ckpt/final/model"), _tensors(b / "ckpt/final/model")
    assert set(ta) == set(tb) and all(torch.equal(ta[x], tb[x]) for x in ta)
    acc = [json.loads((x / "accounting.json").read_text()) for x in outs]
    assert list(acc[0]) == list(acc[1]) and acc[0]["by_stream"] == acc[1]["by_stream"] and set(acc[1]["by_stream"]) == {"propose", "select"}
    for x in ("run_001.json", "run_002.json"):
        ra, rb = (json.loads((o / "logs" / x).read_text()) for o in outs)
        assert list(ra) == list(rb) and ra["paths"].keys() == rb["paths"].keys() and ra["audit"] == rb["audit"]
    assert sorted(p.relative_to(a).as_posix() for p in a.rglob("*")) == sorted(p.relative_to(b).as_posix() for p in b.rglob("*"))
    assert "replay" not in yaml.safe_load((b / "config.yaml").read_text())


# ---------------------------------------------------------------------------
# 带回放的 tiny 训练
# ---------------------------------------------------------------------------


def _train_r(ad, data, rdata, out, *, rc=None, **kw):
    rc = rc or T.resolve_w2c_run(CFG_R, tiny=True)
    w2_info = T.resolve_w2_config(CFG_R, tiny=True)[1]
    return T.run_w2c_training(ad, CFG_R, _spec(ad), rc, data, w2=W2, w2_info=w2_info, impl=IMPL, data_root=FK.ROOT, out_dir=out,
                              log=lambda m: None, replay=rdata, **kw)


def test_replay_training_resume_segments_identity(data, replay_run, tmp_path):
    rdata = load_rdata(replay_run)
    a, b, c = tmp_path / "A", tmp_path / "B", tmp_path / "C"
    sa = _train_r(_tiny(), data, rdata, a)
    assert sa["complete"] and sa["steps"] == 4 and sa["replay"]["pools"] == {"replay_general": 5, "replay_count": 6}
    with pytest.raises(RuntimeError, match="测试注入的中断"):
        _train_r(_tiny(), data, rdata, b, _fail_after_step=3)
    sb = _train_r(_tiny(), data, rdata, b, resume=True)
    assert _train_r(_tiny(), data, rdata, c, stop_after_steps=1) is None
    assert _train_r(_tiny(), data, rdata, c, stop_after_steps=2, resume=True) is None
    sc = _train_r(_tiny(), data, rdata, c, resume=True)
    la, lb, lc = (leaf_dir(x, "w2c", QWEN, 0) for x in (a, b, c))
    for other in (lb, lc):
        for f in DET:
            assert (la / f).read_bytes() == (other / f).read_bytes(), f
        ta, tb = _tensors(la / "ckpt" / "final" / "model"), _tensors(other / "ckpt" / "final" / "model")
        assert set(ta) == set(tb) and all(torch.equal(ta[k], tb[k]) for k in ta)
    assert sa == sb == sc
    log = jsonl(la / "train_log.jsonl")
    assert [r["step"] for r in log] == [0, 1, 2, 3] and all(math.isfinite(r["loss"]) for r in log)
    assert all(r["n_by_stream"] == {"propose": 2, "select": 2, "replay_general": 2, "replay_count": 1} for r in log)
    lb = lambda r, s: r["loss_by_stream"][s]  # noqa: E731
    assert all(r["loss"] == pytest.approx(lb(r, "propose") + lb(r, "select") + 0.5 * lb(r, "replay_general") + 0.5 * lb(r, "replay_count"))
               for r in log)
    ident = yaml.safe_load((la / "config.yaml").read_text())
    assert ident["variant"] == "w2c" and ident["config"]["sha256"] == CFG_R.sha256  # run_w2 --setting w2c 认 variant
    assert ident["replay"]["manifest_sha256"] == rdata.manifest_sha256 and ident["replay"]["pools"]["replay_general"]["n"] == 5
    assert ident["training"]["replay"]["per_step_total"] == 3 and ident["replay"]["model"]["key"] == QWEN
    assert str(replay_run) not in (la / "config.yaml").read_text()
    run1 = json.loads((la / "logs" / "run_001.json").read_text())
    assert {e["stream"] for e in run1["audit"]["first_step_examples"]} == {"propose", "select", "replay_general", "replay_count"}
    assert run1["paths"]["replay_data"] == str(replay_run)
    acc = json.loads((lc / "accounting.json").read_text())
    assert acc["by_stream"]["replay_general"]["n_samples"] == 8 and acc["by_stream"]["replay_count"]["n_samples"] == 4
    assert acc["n_runs"] == 3
    readme = (la / "README.md").read_text()
    assert "回放" in readme and "replay_count 1 条（count，损失权重 0.5" in readme and "回放的权重合计 1" in readme
    man = json.loads((la / "ckpt" / "final" / "model" / "manifest.json").read_text())
    assert man["format"] == "visint-e3-checkpoint/1" and man["trainable_token_rows"] is None  # 与 W2-c、C3 同格式
    # 回放数据变了：运行身份不同，拒绝续跑
    alt = tmp_path / "alt"
    shutil.copytree(replay_run, alt)
    inject_stops(alt)  # 通用的最后一条也不截断了：池变大
    _train_r(_tiny(), data, rdata, tmp_path / "D", stop_after_steps=1)
    with pytest.raises(ValueError, match="运行身份"):
        _train_r(_tiny(), data, load_rdata(alt), tmp_path / "D", resume=True)
    # 回放数据与 run_cfg 的 replay 一节必须同时有
    with pytest.raises(ValueError, match="同时有"):
        _train_r(_tiny(), data, None, tmp_path / "E")
    with pytest.raises(ValueError, match="同时有"):
        T.run_w2c_training(_tiny(), CFG, _spec(_tiny()), T.resolve_w2c_run(CFG, tiny=True), data, w2=W2,
                           w2_info=T.resolve_w2_config(CFG, tiny=True)[1], impl=IMPL, data_root=FK.ROOT, out_dir=tmp_path / "F",
                           replay=rdata)


# ---------------------------------------------------------------------------
# 命令行：训练与评测（检查点与 W2-c 同格式）
# ---------------------------------------------------------------------------


def test_cli_train_with_replay_and_evaluate(replay_run, tmp_path):
    tmp = tmp_path
    cfg = tmp / "w2_k4.json"
    d = w2_draft_doc()
    d["candidates"]["generators"] = GENS
    cfg.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n")
    w2c = FK.write_fake_w2c_data(tmp / "w2c_data", w2_config=cfg)
    base = ["--tiny", "--model", QWEN, "--w2c-data", w2c, "--w2-config", cfg]
    # 用法：回放配置不给 --replay-data（path 是占位）、默认配置给 --replay-data，都拒绝
    p = run(TRAIN, base + ["--config", T.W2C_REPLAY_CONFIG_PATH, "--out", tmp / "n1"], tmp, ok=False)
    assert p.returncode == 2 and "占位" in p.stderr
    p = run(TRAIN, base + ["--replay-data", replay_run, "--out", tmp / "n2"], tmp, ok=False)
    assert p.returncode == 2 and "不收 --replay-data" in p.stderr
    out = tmp / "train"
    p = run(TRAIN, base + ["--config", T.W2C_REPLAY_CONFIG_PATH, "--replay-data", replay_run, "--out", out], tmp)
    assert "steps=4" in p.stdout and "replay_count" in p.stdout
    ck = out / "w2c" / QWEN / "seed0" / "ckpt" / "final" / "model"
    man_sha = hashlib.sha256((ck / "manifest.json").read_bytes()).hexdigest()
    # run_w2 --setting w2c：合并核对通过、按 E1 协议重新出框、候选、挑选（随机模型给不出框：提出失败的句子交 1 号）
    ev = tmp / "eval"
    run(RUN_W2, ["--tiny", "--model", QWEN, "--task", "refseg", "--split", "devcheck2", "--setting", "w2c", "--out", ev, "--w2-config",
                 cfg, "--checkpoint", ck, *FAKES], tmp)
    mc = json.loads((ev / QWEN / "w2c_merge_check.json").read_text())
    assert mc["pass"] is True and mc["checkpoint"]["manifest_sha256"] == man_sha
    c = yaml.safe_load((ev / QWEN / "refseg" / "w2c" / "devcheck2" / "config.yaml").read_text())
    assert c["checkpoint"]["manifest_sha256"] == man_sha and c["checkpoint"]["training"]["variant"] == "w2c"
    assert c["checkpoint"]["training"]["config_sha256"] == CFG_R.sha256 and c["checkpoint"]["training"]["complete"] is True
    assert json.loads((ev / QWEN / "refseg" / "w2c" / "devcheck2" / "metrics.json").read_text())["complete"] is True
    # 健康检查与 T 文本读数：现有脚本加 --checkpoint
    hj = tmp / "health_replay.json"
    run(HEALTH, ["--tiny", "--model", QWEN, "--checkpoint", ck, "--limit", "1", "--skip-ifeval-scoring", "--out", hj], tmp)
    rep = json.loads(hj.read_text())
    assert rep["ok"] is True and rep["checkpoint"]["manifest_sha256"] == man_sha
    tt = tmp / "ttext"
    run(RUN_E1, ["--tiny", "--model", QWEN, "--task", "count", "--subset", "eval", "--limit", "2", "--checkpoint", ck,
                 "--checkpoint-label", "t_text/w2c_replay/seed0", "--out", tt], tmp)
    m = json.loads((tt / "t_text" / "w2c_replay" / "seed0" / QWEN / "count" / "metrics.json").read_text())
    assert m["n"] == 2
