"""DEV-0043（D83、D84、D85；E5 预注册 §8a v3）：E5 的新起点（E3b+C3、C3+预热）与结构强制解码（CPU fp32 小模型，--tiny）。

- 配置：版本 e5-train-v3；起点多 e3b_c3、c3_warmup，v3 主比较 = base_warmup、e3b_c3、c3_warmup（in_v3），e3b、e3a、base 保留；decode 一节
  的正式值是 structured，--decode 可另给；坏的 decode / in_v3 被拒；只评起点的起点 = 网格 token 行训练过的起点（含新的两个，另保留 e3b）；
- 秩 64（C3+预热）：setup_start 之后与 factory.load_trained(merge=True) 全部参数逐位相同、权重指纹相同（开训前第 2 项核对的前提）；与
  "先合并 C3、再合并预热"两次合并只差末位舍入（D85-2：不主张逐位相同）；
- 结构强制：输出长度恒为 n(n + 1) + 1（32 时 1057），逐位置都在该位置的允许集合里（二值 2 个、起伏 33 个值 token）；处理器不生效时报错；
  同一个调用去掉结构的收窄（挂"不收窄"的处理器）时与 generate_restricted 逐 token 相同（三个家族）；
- 只评起点与训练后评测：默认结构强制，叶目录 <任务>__structured，运行身份、metrics、README 记解码方式；--decode restricted 写进 <任务>/；
  续跑时解码方式不同就拒绝混写；结构强制下的续跑逐字节相同；结构强制下"只评起点" = "E5 LoRA 为恒等"的训练后评测；
- val 选择用结构强制：选出的步与分数可复现（一次跑完与中断续跑逐字节相同）；训练续跑时解码方式不同就拒绝；
- decode = restricted：训练（带 val 选择）、训练后评测、只评起点的确定输出与改动前（36bcbbf 的脚本与模块，原样存在 tests/fixtures/e5/）
  逐字节相同；
- 新起点端到端：train_e3b --tiny 的 E3b+C3、C3 → C3+预热，按 scripts/server/train_queue.sh 的 finalize 搬到别处；只评起点、E5 训练
  （带 val 选择）、训练后评测都跑通，变体与配种子的核对照常。

结构强制每条要生成满 1057 个 token，小模型在 CPU 上一批约 3–10 秒（InternVL 最快）；端到端的测试用 InternVL 的小模型。
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
from transformers import LogitsProcessor

from src.adapters import factory, registry
from src.adapters.checkpoint import read_manifest
from src.adapters.factory import make_adapter
from src.adapters.image_policy import parse_policy
from src.e3.config import C3_WARMUP, VARIANT_OF
from src.e3.tokens import add_vocab
from src.e5 import stack as STK
from src.e5 import token_grid as TG
from src.e5 import trained_eval as TE
from src.e5 import val_select as VS
from src.e5.protocol import E5_TASKS, eval_items
from src.e5.runner import make_sample
from src.e5.train import E5Trainer
from src.e5.train_config import STARTS, E5ConfigError, load_train_config, resolve_run
from src.e5.train_data import TrainSet
from src.grid.codec import RELIEF_NUM_LEVELS
from src.grid.structured_decode import StructuredDecodeError, StructuredGridProcessor

REPO = Path(__file__).resolve().parents[1]
CFG = load_train_config()
FAMILIES = ("qwen3.5-0.8b", "gemma-4-e2b-it", "internvl3.5-2b-hf")
QWEN = "qwen3.5-0.8b"
FAST = "internvl3.5-2b-hf"  # 结构强制的端到端用它（小模型里 1057 步最快）
DET = ("predictions.jsonl", "metrics.json", "README.md", "config.yaml")  # 确定的输出（compute.json、meta.json、logs/ 含墙钟与主机）
TRAIN_DET = ("train_log.jsonl", VS.VAL_LOG, "config.yaml", "train_summary.json", "README.md")
V3 = ("base_warmup", "e3b_c3", "c3_warmup")
SNAP = REPO / "tests" / "fixtures" / "e5"
SNAP_SHA256 = {  # 36bcbbf（DEV-0043 之前）的文件原样
    "eval_e5_trained_36bcbbf.py": "a3ce277104dcdf5103a18ed1b542c60411d1cabd61e15c30900eff08ab3d0417",
    "trained_eval_36bcbbf.py": "6820189c133f52f00c541e22b7360eb2ac05a2bfdee23159615bf17a851cd4e7",
    "train_e5_36bcbbf.py": "6722c88a555f74bfe379048defeac8c53752070ffbeba54c1b00565e70b9ece0",
    "train_36bcbbf.py": "5a93e10e052b18f99d9a85737e8cc702885d08eb850a5adef559cd3a40c3c8c1",
    "val_select_36bcbbf.py": "5d6a09499c303448ca0db7218a52b3b531dc15cb323074910098394d342344a8",
}
RESTRICTED = ("--decode", "restricted")
_SCRIPTS: dict[str, object] = {}


def _script(name: str):
    if name not in _SCRIPTS:
        saved = list(sys.path)
        spec = importlib.util.spec_from_file_location(f"_{name}_under_test_e5v3", REPO / "scripts" / f"{name}.py")
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


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tiny(key: str):
    t = CFG.e3b.doc["tiny"]
    fam = registry.get_model(key).family
    return make_adapter(key, device="cpu", dtype=torch.float32, image_policy=parse_policy(t["image_policies"][fam]), tiny=True,
                        init_std=float(t["init_std"]))


def _sharp_start(root: Path, key: str) -> Path:
    """合成的 e3b 结构起点（同 tests/test_e5_start_only._sharp_start）：网格 token 行与 LoRA 都随机，输出长短不一、token 各异。按训练叶目录
    的布局存，旁边放 config.yaml（variant e3b、seed 0）。"""
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


@pytest.fixture(scope="module")
def sharp(tmp_path_factory):
    root = tmp_path_factory.mktemp("e5v3_sharp")
    return {key: _sharp_start(root / key, key) for key in FAMILIES}


def _mdir(out: Path, init: str, key: str, tag: str, seed: int = 0) -> Path:
    return out / "trained" / init / key / tag / f"seed{seed}"


def _leaf(out: Path, init: str, key: str, tag: str = "n10", seed: int = 0) -> Path:
    return out / "train" / init / key / tag / f"seed{seed}"


def _det(leaf: Path, out: Path) -> dict[str, bytes]:
    """训练叶目录的确定输出。train_summary.json 里起点合并核对的摘要带着核对文件的路径、sha256（文件里有墙钟）、是否复用与退出码（续跑时
    复用、没有退出码），去掉这几项；其余逐字节。"""
    res = {f: (leaf / f).read_bytes() for f in TRAIN_DET}
    s = json.loads(res["train_summary.json"])
    mc = s.get("start_merge_check") or {}
    assert str(mc.get("file", str(out))).startswith(str(out))
    for k in ("file", "file_sha256", "reused", "exit_code"):
        mc.pop(k, None)
    res["train_summary.json"] = json.dumps(s, indent=1, ensure_ascii=False).encode()
    return res


def _files(d: Path) -> dict[str, bytes]:
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def _check_positions(ids, value_ids, row_id: int, stop_id: int, n: int) -> None:
    """逐位置的允许集合（独立于 GridStructure 的写法）：第 r 行的格子只许值 token，行末只许行分隔，最后只许停止 token，长度 n(n + 1) + 1。"""
    ids = [int(i) for i in ids]
    assert len(ids) == n * (n + 1) + 1
    vals = {int(v) for v in value_ids}
    for t, i in enumerate(ids):
        r, c = divmod(t, n + 1)
        want = {int(stop_id)} if r == n else ({int(row_id)} if c == n else vals)
        assert i in want, (t, i)


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


def test_config_v3():
    d = CFG.doc
    assert d["version"] == "e5-train-v3" and d["schema"] == 1
    assert STARTS == ("base_warmup", "e3b_c3", "c3_warmup", "e3b", "e3a", "base")
    assert tuple(k for k in STARTS if CFG.in_v3(k)) == V3  # D83-5：v3 主比较的三个起点；e3b、e3a、base 保留在配置里
    assert set(k for k in d["starts"] if k != "basis") == set(STARTS)
    # 新起点的变体就是 train_e3b 写进叶目录 config.yaml 的变体（src/e3/config 的名字，不在这里另写）
    assert CFG.start("e3b_c3")["variant"] == VARIANT_OF["grid+text"] == "e3b_c3"
    assert CFG.start("c3_warmup")["variant"] == C3_WARMUP == "c3_warmup"
    assert all(CFG.start(k)["checkpoint"] and CFG.start(k)["grid_tokens"] == "from_checkpoint" for k in ("e3b_c3", "c3_warmup"))
    # 只评起点的起点 = 带检查点、网格 token 行训练过的起点：v3 的三个 + e3b（不属于 v3 主比较，保留）
    assert TE.START_ONLY_KINDS == tuple(k for k in STARTS if CFG.start(k)["checkpoint"] and
                                        CFG.start(k)["grid_tokens"] == "from_checkpoint") == V3 + ("e3b",)
    # 解码：正式值 structured（D83-3），--decode 可另给
    assert d["decode"]["mode"] == "structured" == CFG.decode_mode() and CFG.decode_mode("restricted") == "restricted"
    assert TG.DECODE_MODES == ("restricted", "structured")
    with pytest.raises(TG.TokenGridError):
        CFG.decode_mode("greedy")
    # val 选择：structured 时 selection 另记 decode；restricted 时与改动前相同（没有这个键）
    s = resolve_run(CFG, n_train=1000, n_examples=1000, tiny=False)["selection"]
    r = resolve_run(CFG, n_train=1000, n_examples=1000, tiny=False, decode="restricted")["selection"]
    assert s == {**r, "decode": "structured"} and "decode" not in r
    assert resolve_run(CFG, n_train=16000, n_examples=16000, tiny=False)["selection"] is None
    # 结构的数：二值每格 2 个值 token，起伏 33 个（背景 + 32 档），长度 1057
    tok = _e5tok(_tiny(QWEN))
    for fmt, k in (("binary", 2), ("relief", 1 + RELIEF_NUM_LEVELS)):
        st = tok.structure_of(fmt, CFG.grid_n)
        assert len(st.value_ids[0]) == k == len(tok.value_ids_of(fmt)) and st.length == 1057 == TG.max_new_tokens(CFG.grid_n)
        assert set(st.all_ids()) <= set(tok.allowed_ids_of(fmt))
    assert 1 + RELIEF_NUM_LEVELS == 33


def _e5tok(ad):
    return TG.E5Tokens(add_vocab(ad, CFG.e3b.vocab), CFG.e3b.allowed_scope)


def test_config_rejects_bad_decode_and_in_v3(tmp_path):
    d = json.loads((REPO / "configs" / "e5_train.json").read_text())
    for mutate in (lambda x: x.pop("decode"), lambda x: x["decode"].update(mode="greedy"),
                   lambda x: x["starts"]["e3b_c3"].pop("in_v3"), lambda x: x["starts"]["base"].update(in_v3="no"),
                   lambda x: x["starts"].pop("c3_warmup")):
        y = json.loads(json.dumps(d))
        mutate(y)
        p = tmp_path / "c.json"
        p.write_text(json.dumps(y))
        with pytest.raises(E5ConfigError):
            load_train_config(p)
    y = json.loads(json.dumps(d))
    y["decode"]["mode"] = "restricted"
    p = tmp_path / "r.json"
    p.write_text(json.dumps(y))
    assert load_train_config(p).decode_mode() == "restricted"


def test_cli_usage(tmp_path, capsys):
    tr, ev = _script("train_e5"), _script("eval_e5_trained")
    for argv in (["--tiny", "--init", "base", "--model", QWEN, "--n-train", "10", "--seed", "0", "--out", str(tmp_path / "o"),
                  "--decode", "greedy"],):
        with pytest.raises(SystemExit) as e:
            tr.parse_args(argv)
        assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        ev.parse_args(["--tiny", "--checkpoint", str(tmp_path), "--model", QWEN, "--out", str(tmp_path / "o"), "--decode", "x"])
    assert e.value.code == 2
    # 没有 val 选择的档不解码：给 --decode 是用法错误（退出码 2），不建任何目录
    assert tr.main(["--tiny", "--init", "base", "--model", QWEN, "--n-train", "20", "--seed", "0", "--out", str(tmp_path / "o"),
                    *RESTRICTED]) == 2
    assert not (tmp_path / "o").exists()
    assert "--decode 只用于有 val 选择的档" in capsys.readouterr().err
    # 不属于 v3 主比较的起点：照常能用，stderr 提示
    a = ev.parse_args(["--tiny", "--start-only", "--init", "e3b", "--start-checkpoint", str(tmp_path), "--model", QWEN, "--seed", "0",
                       "--out", str(tmp_path / "o")])
    assert a.init == "e3b" and a.decode is None
    for init in ("e3b_c3", "c3_warmup", "base_warmup"):
        assert ev.parse_args(["--tiny", "--start-only", "--init", init, "--start-checkpoint", str(tmp_path), "--model", QWEN,
                              "--seed", "0", "--out", str(tmp_path / "o")]).init == init
    ev.v3_note(CFG, "e3a")
    ev.v3_note(CFG, "e3b_c3")
    err = capsys.readouterr().err
    assert "起点 e3a 不属于 E5 v3 主比较" in err and "e3b_c3 不属于" not in err


# ---------------------------------------------------------------------------
# 结构强制：逐位置的允许集合；不收窄时与受限解码相同
# ---------------------------------------------------------------------------


class _Passthrough(LogitsProcessor):  # 不改 scores，只计数（"不收窄"的处理器）
    def __init__(self):
        self.n_calls = 0

    def __call__(self, input_ids, scores):
        self.n_calls += 1
        return scores


def _inputs(ad, task: str, n: int, k: int = 2):
    pf = CFG.prompt_format()
    samples = [make_sample(it) for it in eval_items(task, "dev", k)[:k]]
    return [ad.build_inputs(TG.sample_images(s), pf.render(s, n), pf.order) for s in samples]


@pytest.mark.parametrize("key", FAMILIES)
def test_structured_positions_and_passthrough(sharp, key):
    """8×8 网格（73 个 token，省时间；32×32 的 1057 见端到端）：结构强制的每一行逐位置都在允许集合里、解析成功，值 token 用到不止一种；
    decode_grids 传给 generate_batch 的就是受限解码的允许集合、贪心、关思考；同一个调用把处理器换成不收窄的，与 generate_restricted 逐 token
    相同（CPU fp32）。"""
    n = 8
    start = STK.resolve_start(CFG, "e3b", sharp[key], model_key=key, tiny=True)
    ad, tokens, _ = STK.load_start_only(start, CFG, device="cpu", dtype=torch.float32)
    ad.eval_mode()
    e5tok = TG.E5Tokens(tokens, CFG.e3b.allowed_scope)
    lens = []
    for task in ("rotation3d", "occlusion"):
        fmt = TG.task_format(task)
        inputs = _inputs(ad, task, n)
        calls = []
        orig = ad.generate_batch

        def spy(inp, **kw):
            calls.append(kw)
            return orig(inp, **kw)

        ad.generate_batch = spy
        try:
            with torch.no_grad():
                res = TG.decode_grids(ad, e5tok, inputs, fmt, n, decode="structured", where=key)
        finally:
            del ad.generate_batch
        (kw,) = calls
        assert kw["allowed_token_ids"] == e5tok.allowed_ids_of(fmt) and kw["do_sample"] is False
        assert kw["enable_thinking"] is False and kw["max_new_tokens"] == n * (n + 1) + 1
        (proc,) = kw["logits_processor"]
        assert isinstance(proc, StructuredGridProcessor) and proc.n_calls == n * (n + 1) + 1
        vals = e5tok.value_ids_of(fmt)
        assert len(vals) == (33 if fmt == "relief" else 2)
        cells = []
        for r in res:
            assert r.stopped is True
            _check_positions(r.token_ids, vals, e5tok.row_id, e5tok.stop_id, n)
            assert e5tok.parse(task, r.token_ids, n)[1] is None  # 结构强制下必然解析成功
            cells += [int(i) for t, i in enumerate(r.token_ids) if t < n * (n + 1) and t % (n + 1) < n]
        assert len(set(cells)) >= 2, (key, task)
        with torch.no_grad():
            cp = _Passthrough()
            plain = ad.generate_batch(list(inputs), enable_thinking=False, max_new_tokens=n * (n + 1) + 1,
                                      allowed_token_ids=list(e5tok.allowed_ids_of(fmt)), logits_processor=[cp], do_sample=False)
            rest = TG.decode_grids(ad, e5tok, inputs, fmt, n, decode="restricted")
        assert cp.n_calls >= 1
        assert [list(r.token_ids) for r in plain] == [list(r.token_ids) for r in rest], (key, task)
        assert [r.stopped for r in plain] == [r.stopped for r in rest]
        lens += [len(r.token_ids) for r in rest]
    assert max(lens) > 1  # 受限解码的输出不是一开口就停：逐 token 相同有区分力


def test_structured_refuses_when_processor_inert(sharp, monkeypatch):
    """入口断言：处理器没生效（换成不收窄的）就报错，不当成模型的输出。"""
    n = 4
    start = STK.resolve_start(CFG, "e3b", sharp[QWEN], model_key=QWEN, tiny=True)
    ad, tokens, _ = STK.load_start_only(start, CFG, device="cpu", dtype=torch.float32)
    ad.eval_mode()
    e5tok = TG.E5Tokens(tokens, CFG.e3b.allowed_scope)

    class Inert(StructuredGridProcessor):
        def __call__(self, input_ids, scores):
            self.n_calls += 1
            return scores

    monkeypatch.setattr(TG, "StructuredGridProcessor", Inert)
    with pytest.raises(RuntimeError) as e, torch.no_grad():
        TG.decode_grids(ad, e5tok, _inputs(ad, "occlusion", n), "binary", n, decode="structured", where="inert")
    assert isinstance(e.value, StructuredDecodeError) or "处理器被调用了" in str(e.value)
    with pytest.raises(TG.TokenGridError):
        TG.decode_grids(ad, e5tok, [], "binary", n, decode="beam")


# ---------------------------------------------------------------------------
# 只评起点（N = 0）与训练后评测：默认结构强制
# ---------------------------------------------------------------------------


def test_start_only_structured_default(sharp, tmp_path, monkeypatch):
    seen: list[tuple[list[int], tuple[int, ...], int, int, int]] = []
    orig = TG.check_structured_ids

    def spy(ids, st, **kw):
        seen.append(([int(i) for i in ids], tuple(st.value_ids[0]), int(st.row_id), int(st.stop_id), int(st.n)))
        return orig(ids, st, **kw)

    monkeypatch.setattr(TG, "check_structured_ids", spy)
    ev = _script("eval_e5_trained")
    out = tmp_path / "o"
    base = ["--tiny", "--start-only", "--init", "e3b", "--start-checkpoint", str(sharp[FAST]), "--model", FAST, "--seed", "0",
            "--out", str(out)]
    args = base + ["--task", "a", "c", "--limit", "2"]
    assert ev.main(args) == 0  # 不给 --decode：取配置的 decode.mode（structured）
    mdir = _mdir(out, "e3b", FAST, "n0")
    assert sorted(p.name for p in mdir.iterdir()) == ["occlusion__structured", "rotation3d__structured"]
    value_sets = {}
    for task in ("rotation3d", "occlusion"):
        leaf = mdir / f"{task}__structured"
        recs = [json.loads(x) for x in (leaf / "predictions.jsonl").read_text().splitlines()]
        assert len(recs) == 2 and all(r["n_generated_tokens"] == 1057 and r["stopped"] is True and r["valid"] is True and
                                      r["failure"] is None for r in recs)
        d = yaml.safe_load((leaf / "config.yaml").read_text())["decoding"]
        assert d["mode"] == "structured" and d["path"] == "generate_batch" and d["logits_processor"] == "StructuredGridProcessor"
        st = d["structure"]
        assert st["length"] == 1057 == d["max_new_tokens"] and st["n"] == 32 and st["stop_id"] == d["stop_ids"][0]
        assert len(st["value_ids"]) == (33 if task == "rotation3d" else 2) and st["value_ids"] == d["value_ids"]
        assert d["allowed_ids"] == sorted(set(d["value_ids"]) | {st["row_id"], st["stop_id"]})
        value_sets[task] = (tuple(d["value_ids"]), st["row_id"], st["stop_id"])
        m = json.loads((leaf / "metrics.json").read_text())
        assert m["decode"]["mode"] == "structured" and m["parse"]["failure_rate"] == 0.0 and m["start_only"] is True
        assert "结构强制" in (leaf / "README.md").read_text()
    # 每一行生成（32×32：1057 个 token）逐位置都在允许集合里（独立于 GridStructure 的写法）；值 token 的集合就是本任务的
    assert len(seen) == 4
    for (ids, vals, row, stop, n), task in zip(seen, ("rotation3d",) * 2 + ("occlusion",) * 2):
        assert (vals, row, stop, n) == (*value_sets[task], 32)
        _check_positions(ids, vals, row, stop, 32)
    rel_cells = {i for ids, *_ in seen[:2] for t, i in enumerate(ids) if t < 1056 and t % 33 < 32}
    assert len(rel_cells) > 2  # 起伏任务确实用到了 2 个以上的值 token（33 个的集合）
    # --decode restricted（诊断）：同一个 --out 里写进 <任务>/，与结构强制的结果并列、互不影响
    before = {t: _files(mdir / f"{t}__structured") for t in ("rotation3d", "occlusion")}
    assert ev.main(args + list(RESTRICTED)) == 0
    assert sorted(p.name for p in mdir.iterdir()) == ["occlusion", "occlusion__structured", "rotation3d", "rotation3d__structured"]
    ident = yaml.safe_load((mdir / "rotation3d" / "config.yaml").read_text())
    assert ident["decoding"]["path"] == "generate_restricted" and "mode" not in ident["decoding"]
    assert "decode" not in json.loads((mdir / "rotation3d" / "metrics.json").read_text())
    assert {t: _files(mdir / f"{t}__structured") for t in before} == before
    # 续跑时解码方式不同：拒绝混写（把 restricted 的结果放到结构强制的叶目录上冒充）
    shutil.rmtree(mdir / "occlusion__structured")
    shutil.copytree(mdir / "occlusion", mdir / "occlusion__structured")
    assert ev.main(base + ["--task", "c", "--limit", "2", "--resume"]) == 2
    (err,) = (out / "trained" / "errors").glob("error_*.json")
    assert "拒绝混写" in json.loads(err.read_text())["error"]


def test_structured_resume_bytewise(sharp, tmp_path):
    ev = _script("eval_e5_trained")
    out = tmp_path / "o"
    args = ["--tiny", "--start-only", "--init", "e3b", "--start-checkpoint", str(sharp[FAST]), "--model", FAST, "--seed", "0",
            "--out", str(out), "--task", "c", "--limit", "3", "--batch-size", "2"]
    assert ev.main(args) == 0
    leaf = _mdir(out, "e3b", FAST, "n0") / "occlusion__structured"
    before = {f: (leaf / f).read_bytes() for f in DET}
    lines = (leaf / "predictions.jsonl").read_bytes().splitlines(keepends=True)
    (leaf / "predictions.jsonl").write_bytes(lines[0] + lines[1][:7])  # 中断在一批中间、末行不完整
    (leaf / "metrics.json").unlink()
    assert ev.main(args + ["--resume"]) == 0
    assert {f: (leaf / f).read_bytes() for f in DET} == before


def _identity_e5_checkpoint(start_ck: Path, key: str, final: Path) -> None:
    """刚初始化就保存的 E5 检查点（E5 LoRA 为恒等；同 tests/test_e5_start_only._identity_e5_checkpoint）。"""
    start = STK.resolve_start(CFG, "e3b", start_ck, model_key=key, tiny=True)
    ad = STK.make_base(start.manifest["build_spec"], device="cpu", dtype=torch.float32)
    ts = TrainSet.build(CFG, 20, None, 0)
    tr = E5Trainer(ad, CFG, resolve_run(CFG, n_train=20, n_examples=len(ts), tiny=True), seed=0, start=start)
    tr.setup_fresh()
    ad.save_trained(final / "model", extra=STK.marker_extra(start))
    STK.write_stack_file(final, {"model_dir": "model", "start": start.record(),
                                 "fingerprint": {"value": tr.stack_info["fingerprint"], "device_key": STK.device_key(ad)},
                                 "model": {"key": key, "tiny": True}, "seed": 0, "tag": ts.tag, "data": ts.identity(),
                                 "identity_sha256": "identity-e5-lora（DEV-0043 测试：训练 0 步）", "config": CFG.identity(),
                                 "train_leaf": str(final.parent), "tokens_e5": tr.e5.layout(), "steps": 0, "selection": None,
                                 "seed_pairing": STK.check_seed_pairing(start, 0, allow_mismatch=False)})


def test_structured_start_only_equals_identity_e5_lora(sharp, tmp_path):
    """结构强制下（同 DEV-0040 的关键一条）：同一个起点，"只评起点"与"E5 LoRA 为恒等"的训练后评测 predictions 逐字节相同。"""
    final = tmp_path / "e5" / "ckpt" / "final"
    _identity_e5_checkpoint(sharp[FAST], FAST, final)
    ev = _script("eval_e5_trained")
    common = ["--task", "a", "c", "--limit", "2"]
    assert ev.main(["--tiny", "--checkpoint", str(final), "--model", FAST, "--out", str(tmp_path / "a"), *common]) == 0
    assert ev.main(["--tiny", "--start-only", "--init", "e3b", "--start-checkpoint", str(sharp[FAST]), "--model", FAST, "--seed", "0",
                    "--out", str(tmp_path / "b"), *common]) == 0
    for task in ("rotation3d", "occlusion"):
        pa = (_mdir(tmp_path / "a", "e3b", FAST, "n20") / f"{task}__structured" / "predictions.jsonl").read_bytes()
        pb = (_mdir(tmp_path / "b", "e3b", FAST, "n0") / f"{task}__structured" / "predictions.jsonl").read_bytes()
        assert pa == pb and len(set(pa.splitlines())) == 2


# ---------------------------------------------------------------------------
# val 选择用结构强制：可复现；续跑时解码方式不同就拒绝
# ---------------------------------------------------------------------------


def test_val_selection_structured_reproducible(sharp, tmp_path, capsys):
    tr = _script("train_e5")
    ck = str(sharp[FAST])

    def train(out: Path, *extra: str) -> int:
        return tr.main(["--tiny", "--init", "e3b", "--start-checkpoint", ck, "--model", FAST, "--n-train", "10", "--seed", "0",
                        "--out", str(out), *extra])

    a, b = tmp_path / "a0", tmp_path / "b0"  # 同长：e5_stack.json 记训练叶目录的路径
    assert train(a) == 0
    la = _leaf(a, "e3b", FAST)
    val = [json.loads(x) for x in (la / VS.VAL_LOG).read_text().splitlines()]
    assert [v["step"] for v in val] == [2, 4, 6] and all(set(v["by_task"]) == set(E5_TASKS) for v in val)
    for v in val:
        for t, row in v["by_task"].items():
            assert row["parse_failure_rate"] == 0.0 and row["mean_generated_tokens"] == 1057 and row["iou_grid_valid_rate"] == 1.0
    ident = yaml.safe_load((la / "config.yaml").read_text())
    assert ident["training"]["selection"]["decode"] == "structured" and ident["selection"]["decode"] == "structured"
    s = json.loads((la / "train_summary.json").read_text())
    assert s["selection"]["decode"]["mode"] == "structured" and s["selected_step"] in (2, 4, 6)
    best = max(val, key=lambda v: (v["score"], -v["step"]))
    assert s["selected_step"] == best["step"] and s["selection"]["selected_score"] == best["score"]
    stack = json.loads((la / "ckpt" / "final" / "e5_stack.json").read_text())
    assert stack["selection"]["decode"]["mode"] == "structured"
    assert "结构强制解码" in (la / "README.md").read_text()
    # 中断（停在两次评测之间）后续跑：与一次跑完逐字节相同（选出的步、分数、全部 val 读数、最终检查点）
    assert train(b, "--stop-after-steps", "3") == 0
    assert train(b, "--resume") == 0
    lb = _leaf(b, "e3b", FAST)
    assert _det(lb, b) == _det(la, a)
    assert _files(lb / "ckpt" / "final" / "model") == _files(la / "ckpt" / "final" / "model")
    # 已完成的叶目录：换解码方式就拒绝混写
    assert train(a, "--resume", *RESTRICTED) == 2
    # 训练中断后换解码方式续跑：拒绝混写
    c = tmp_path / "c0"
    assert train(c, "--stop-after-steps", "2", *RESTRICTED) == 0
    capsys.readouterr()
    assert train(c, "--resume") == 2
    (err,) = (c / "train" / "e3b" / FAST / "errors").glob("error_*.json")
    assert "拒绝混写" in json.loads(err.read_text())["error"]
    # 训练后评测：checkpoint.selection 记下 val 选择的解码方式
    ev = _script("eval_e5_trained")
    assert ev.main(["--tiny", "--checkpoint", str(la / "ckpt" / "final"), "--model", FAST, "--out", str(a), "--task", "c",
                    "--limit", "1"]) == 0
    ck_ident = yaml.safe_load((_mdir(a, "e3b", FAST, "n10") / "occlusion__structured" / "config.yaml").read_text())["checkpoint"]
    assert ck_ident["selection"]["decode"]["mode"] == "structured" and ck_ident["selection"]["selected_step"] == s["selected_step"]
    assert json.loads((_mdir(a, "e3b", FAST, "n10") / "merge_check.json").read_text())["pass"] is True


# ---------------------------------------------------------------------------
# decode = restricted：与改动前（36bcbbf）逐字节相同
# ---------------------------------------------------------------------------


def test_restricted_identical_to_36bcbbf(sharp, tmp_path, capsys):
    """同一进程、同一组参数：改动前（36bcbbf）的脚本加模块对改动后的脚本加 --decode restricted。
    - 训练（10 条的档，带 val 选择）：两边用同一份配置（正式配置的 decode.mode 改成 restricted：旧脚本调用的是新的 resolve_run，它按配置
      取解码方式），训练日志、val 读数、运行身份、摘要、README、最终检查点、e5_stack.json（去掉训练叶目录的路径）逐字节相同；
    - 训练后评测、只评起点：正式配置，确定的输出、打印的结果行逐字节相同；合并核对（去掉墙钟）：DEV-0046 起判据是 e5-v2，改动前就有的
      字段逐个相同、v1 的判定照旧（v1_report_only）。"""
    for name, sha in SNAP_SHA256.items():
        assert _sha(SNAP / name) == sha, name
    old_vs = _load_path(SNAP / "val_select_36bcbbf.py", "_e5_val_select_36bcbbf")
    old_train = _load_path(SNAP / "train_36bcbbf.py", "_e5_train_36bcbbf")
    old_train.VS = old_vs
    old_train.REPO_ROOT = REPO  # 快照按自己的位置推出的仓库根不对（只影响 meta.json 里配置文件的路径）
    old_te = _load_path(SNAP / "trained_eval_36bcbbf.py", "_e5_trained_eval_36bcbbf")
    old_te.REPO_ROOT = TE.REPO_ROOT
    old_tr_script = _load_path(SNAP / "train_e5_36bcbbf.py", "_train_e5_36bcbbf")
    old_tr_script.run_e5_training, old_tr_script.RunSpec = old_train.run_e5_training, old_train.RunSpec
    old_ev_script = _load_path(SNAP / "eval_e5_trained_36bcbbf.py", "_eval_e5_trained_36bcbbf")
    old_ev_script.TE = old_te
    doc = json.loads((REPO / "configs" / "e5_train.json").read_text())
    doc["decode"]["mode"] = "restricted"
    rcfg = tmp_path / "e5_train_restricted.json"
    rcfg.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    ck = str(sharp[QWEN])
    # 训练
    outs, roots = [], []
    for k, (mod, extra) in enumerate(((old_tr_script, ()), (_script("train_e5"), RESTRICTED))):
        o = tmp_path / f"t{k}"
        assert mod.main(["--tiny", "--init", "e3b", "--start-checkpoint", ck, "--model", QWEN, "--n-train", "10", "--seed", "0",
                         "--out", str(o), "--config", str(rcfg), *extra]) == 0
        outs.append(_leaf(o, "e3b", QWEN))
        roots.append(o)
    d0, d1 = _det(outs[0], roots[0]), _det(outs[1], roots[1])
    for f in TRAIN_DET:
        assert d0[f] == d1[f], f
    assert _files(outs[0] / "ckpt" / "final" / "model") == _files(outs[1] / "ckpt" / "final" / "model")
    s0, s1 = (json.loads((d / "ckpt" / "final" / "e5_stack.json").read_text()) for d in outs)
    assert s0.pop("train_leaf") != s1.pop("train_leaf") and s0 == s1
    val = [json.loads(x) for x in (outs[1] / VS.VAL_LOG).read_text().splitlines()]
    assert any(row["mean_generated_tokens"] not in (1.0, 1057.0) for v in val for row in v["by_task"].values())  # 输出长短不一
    assert "decode" not in yaml.safe_load((outs[1] / "config.yaml").read_text())["training"]["selection"]
    # 训练后评测与只评起点
    final = outs[1] / "ckpt" / "final"
    capsys.readouterr()
    for mode, base_args in (("checkpoint", ["--tiny", "--checkpoint", str(final), "--model", QWEN]),
                            ("start_only", ["--tiny", "--start-only", "--init", "e3b", "--start-checkpoint", ck, "--model", QWEN,
                                            "--seed", "0"])):
        res, printed = [], []
        for k, (mod, extra) in enumerate(((old_ev_script, ()), (_script("eval_e5_trained"), RESTRICTED))):
            o = tmp_path / f"{mode}{k}"
            assert mod.main([*base_args, "--out", str(o), "--limit", "2", *extra]) == 0, mode
            printed.append(capsys.readouterr().out)
            res.append(_mdir(o, "e3b", QWEN, "n10" if mode == "checkpoint" else "n0"))
        assert printed[0] == printed[1] and printed[0].count("\n") == len(E5_TASKS), mode
        lens = []
        for task in E5_TASKS:
            for f in DET:
                assert (res[0] / task / f).read_bytes() == (res[1] / task / f).read_bytes(), (mode, task, f)
            r0, r1 = (json.loads((d / task / "logs" / "run_001.json").read_text()) for d in res)
            assert list(r0) == list(r1)
            lens += [json.loads(x)["n_generated_tokens"] for x in (res[1] / task / "predictions.jsonl").read_text().splitlines()]
        assert max(lens) > 1 and len(set(lens)) > 1  # 输出不是一开口就停
        if mode == "checkpoint":
            m0, m1 = (json.loads((d / "merge_check.json").read_text()) for d in res)
            for m in (m0, m1):
                m.pop("timing_s")
                m.pop("written_utc")
            # DEV-0046：判据改为 e5-v2（另有 v1_report_only、metric_level、previous）；改动前就有的字段照旧，v1 的判定照旧
            assert m1["criteria_version"] == "e5-v2" and m0["pass"] is m1["pass"] is True
            v1 = m1["v1_report_only"]
            assert (v1["criteria_version"], v1["failures"], v1["pass"], v1["criteria"]) == \
                (m0["criteria_version"], m0["failures"], m0["pass"], m0["criteria"])
            same = set(m0) - {"criteria_version", "criteria", "failures", "pass", "rule"}
            assert {k: m0[k] for k in same} == {k: m1[k] for k in same} and "checks" in same


# ---------------------------------------------------------------------------
# 新起点：E3b+C3、C3+预热（秩 64）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def new_starts(tmp_path_factory):
    """scripts/train_e3b.py --tiny 的 E3b+C3（grid+text）、C3（text）与 C3+预热（warmup --start-checkpoint <C3>），种子 0，InternVL；
    另按 scripts/server/train_queue.sh 的 finalize 搬一份：<搬到的根>/<叶目录相对 out 的路径>/ckpt/final + config.yaml + train_summary.json。"""
    root = tmp_path_factory.mktemp("e5v3_new")
    e3 = _script("train_e3b")
    out = root / "e3"
    base = ["--tiny", "--model", FAST, "--seed", "0", "--out", str(out)]
    assert e3.main([*base, "--supervision", "grid+text"]) == 0
    assert e3.main([*base, "--supervision", "text"]) == 0
    c3 = out / "c3" / FAST / "seed0" / "ckpt" / "final" / "model"
    assert e3.main([*base, "--supervision", "warmup", "--start-checkpoint", str(c3)]) == 0
    res = {"c3": c3}
    for init in ("e3b_c3", "c3_warmup"):
        rel = Path(init) / FAST / "seed0"
        leaf = out / rel
        dst = root / "moved" / rel
        shutil.copytree(leaf / "ckpt" / "final", dst / "ckpt" / "final")
        shutil.copy(leaf / "config.yaml", dst / "config.yaml")
        shutil.copy(leaf / "train_summary.json", dst / "train_summary.json")
        res[init] = leaf / "ckpt" / "final" / "model"
        res[f"{init}_moved"] = dst / "ckpt" / "final" / "model"
    return res


def _peft_merge(ad, lora_dir: Path) -> None:
    """与 load_trained(merge=True)、src/e5/stack.merge_start_lora 同一串 peft 调用（只合并 LoRA）。"""
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    pm = get_peft_model(ad.model, LoraConfig.from_pretrained(str(lora_dir / "lora")))
    set_peft_model_state_dict(pm, load_file(str(lora_dir / "lora" / "adapter_model.safetensors")))
    assert pm.merge_and_unload() is ad.model


def _split_child(ck: Path, out: Path) -> Path:
    """从拼接的检查点里取出后一段秩（预热的 LoRA），存成单独的 LoRA 目录（同 tests/test_e3_c3_warmup._split_child）。"""
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


def _target_weights(ad) -> dict[str, torch.Tensor]:
    mods = dict(ad.model.named_modules())
    return {n: getattr(mods[n], "base_layer", mods[n]).weight.detach().clone() for n in sorted(ad.lora_target_modules())}


MAX_ABS_DIFF_SEQUENTIAL = 1e-6  # 拼接后一次合并 vs 依次合并：只差末位舍入（D85-2：tiny fp32 实测最大约 1.5e-8）


def test_rank64_start_merge(new_starts, tmp_path):
    """C3+预热的起点（秩 64 的拼接 LoRA）：E5 的 setup_start（merge_start_lora + restore_tokens）之后，全部参数（含网格 token 行）与
    factory.load_trained(起点, merge=True) 逐位相同、权重指纹相同——E5 开训前第 2 项核对（reference_fingerprint）比较的就是这个；
    与"先合并 C3、再合并预热"两次合并只差末位舍入（不逐位相同，D85-2）。E3b+C3（秩 32）同样逐位相同。"""
    for init in ("c3_warmup", "e3b_c3"):
        ck = new_starts[init]
        start = STK.resolve_start(CFG, init, ck, model_key=FAST, tiny=True)
        r = 64 if init == "c3_warmup" else 32
        assert start.manifest["lora"]["r"] == r and start.leaf_variant == init and start.leaf_seed == 0 and start.n_grid_tokens == 67
        ad = STK.make_base(start.manifest["build_spec"], device="cpu", dtype=torch.float32)
        _, info = STK.setup_start(ad, start, CFG)
        assert info["merged"]["lora"]["r"] == r and ad.lora_state == "none"
        ref, _ = factory.load_trained(ck, device="cpu", dtype=torch.float32, merge=True)
        pa, pb = dict(ad.model.named_parameters()), dict(ref.model.named_parameters())
        assert pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa), init
        assert ad.token_table()["row_tables"] == ref.token_table()["row_tables"]
        assert STK.stack_fingerprint(ad) == STK.stack_fingerprint(ref) == info["fingerprint"]
        ref_doc = STK.reference_fingerprint(start, device="cpu", dtype=torch.float32, out_dir=tmp_path / init, log=lambda m: None)
        assert ref_doc["fingerprint"] == info["fingerprint"]
        if init != "c3_warmup":
            continue
        seq = STK.make_base(start.manifest["build_spec"], device="cpu", dtype=torch.float32)
        base = _target_weights(seq)
        _peft_merge(seq, new_starts["c3"])
        _peft_merge(seq, _split_child(ck, tmp_path / "child_part"))
        a, b = _target_weights(ad), _target_weights(seq)
        diff = max(float((a[k] - b[k]).abs().max()) for k in a)
        delta = max(float((a[k] - base[k]).abs().max()) for k in a)
        assert diff <= MAX_ABS_DIFF_SEQUENTIAL and delta > 1e3 * MAX_ABS_DIFF_SEQUENTIAL, (diff, delta)


def test_new_starts_end_to_end(new_starts, tmp_path):
    """E3b+C3、C3+预热两个起点（按 finalize 搬过的布局）：只评起点、E5 训练（10 条的档，带 val 选择）、训练后评测（结构强制）都跑通；
    运行身份记下起点的变体、种子与 LoRA；变体弄混被拒。val 选择这里用 --decode restricted 省时间（结构强制下的 val 选择见
    test_val_selection_structured_reproducible；完整的默认口径见完成总结里的彩排）。"""
    tr, ev = _script("train_e5"), _script("eval_e5_trained")
    out = tmp_path / "o"
    for init in ("e3b_c3", "c3_warmup"):
        ck = new_starts[f"{init}_moved"]
        r = 64 if init == "c3_warmup" else 32
        assert ev.main(["--tiny", "--start-only", "--init", init, "--start-checkpoint", str(ck), "--model", FAST, "--seed", "0",
                        "--out", str(out), "--task", "a", "c", "--limit", "1"]) == 0, init
        m0 = _mdir(out, init, FAST, "n0")
        assert sorted(p.name for p in m0.iterdir()) == ["occlusion__structured", "rotation3d__structured"]
        ident = yaml.safe_load((m0 / "rotation3d__structured" / "config.yaml").read_text())
        c = ident["checkpoint"]
        assert c["start"]["kind"] == init and c["start"]["leaf_variant"] == init and c["start"]["leaf_seed"] == 0
        assert c["start"]["lora"]["r"] == r and c["seed_pairing"]["status"] == "match"
        assert c["start"]["manifest_sha256"] == _sha(ck / "manifest.json") == _sha(new_starts[init] / "manifest.json")
        assert ident["decoding"]["mode"] == "structured"
        assert tr.main(["--tiny", "--init", init, "--start-checkpoint", str(ck), "--model", FAST, "--n-train", "10", "--seed", "0",
                        "--out", str(out), *RESTRICTED]) == 0, init
        leaf = _leaf(out, init, FAST)
        s = json.loads((leaf / "train_summary.json").read_text())
        assert s["complete"] and s["steps"] == 6 and s["selection"]["selected_step"] in (2, 4, 6) and s["start"] == init
        assert "decode" not in s["selection"] and s["lora"]["r"] == 32  # 新 LoRA 照旧 r 32（起点的秩 64 已合并）
        sha = _sha(ck / "manifest.json")
        checks = STK.start_checks_dir(out, init, FAST)  # 与只评起点同一个位置：已通过的复用
        mc = json.loads((checks / f"merge_check_{sha[:16]}.json").read_text())
        assert mc["pass"] is True and sorted(mc["checks"]) == ["merge", "restricted"]
        stack = json.loads((leaf / "ckpt" / "final" / "e5_stack.json").read_text())
        ref = json.loads((checks / f"reference_{sha[:16]}.json").read_text())
        assert ref["fingerprint"] == stack["fingerprint"]["value"] and stack["start"]["kind"] == init
        assert yaml.safe_load((leaf / "config.yaml").read_text())["seed_pairing"]["status"] == "match"
        assert ev.main(["--tiny", "--checkpoint", str(leaf / "ckpt" / "final"), "--model", FAST, "--out", str(out),
                        "--task", "a", "c", "--limit", "1"]) == 0, init
        m1 = _mdir(out, init, FAST, "n10")
        assert json.loads((m1 / "merge_check.json").read_text())["pass"] is True
        for task in ("rotation3d", "occlusion"):
            rec = json.loads((m1 / f"{task}__structured" / "predictions.jsonl").read_text().splitlines()[0])
            assert rec["n_generated_tokens"] == 1057 and rec["valid"] is True
    # 变体弄混：拒绝
    for kind, ck in (("e3b", "e3b_c3"), ("e3b_c3", "c3_warmup"), ("c3_warmup", "e3b_c3"), ("base_warmup", "c3_warmup")):
        with pytest.raises(STK.StartError, match="变体"):
            STK.resolve_start(CFG, kind, new_starts[ck], model_key=FAST, tiny=True)
    with pytest.raises(STK.StartError):
        STK.resolve_start(CFG, "e3a", new_starts["c3_warmup"], model_key=FAST, tiny=True)  # e3a 不该带网格 token
