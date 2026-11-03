"""X-M 的 COCO 训练样本（DEV-0081；预注册 XM-4；D165 第三节；D167）：从 DEV-0079 的训练目标（src/xm/targets.py：COCO train2017 去重
之后，每个"图像 × 类别"一行：阅读顺序的框列表 + 每实例 16 控制点 B 样条）按 hash 抽对，写成两条训练流的样本行。在 CPU 上做，不跑模型、
不看图（给了 --data-root 才核对图像在）。

**输入**：DEV-0079 的 scripts/xm/build_targets.py 的 --out（<targets>/xm_targets/targets_<k>.jsonl 与 manifest.json）。逐片读（不把三十多万
对一次读进内存）：每片的 sha256 等于 manifest 记的；参数等于 targets.check_params 的；每行过 targets.row_problems（框列表与样条的文本
等于由 boxes_q / ctrl_q 重写的、阅读顺序、控制点在框内）；id 不重复；split = train、source = coco2017、id 以 coco2017/train2017/ 开头
（只读训练集）。

**抽对**（select_pairs）：
- 先剔除**这一类在这张图里带 crowd 标注的对**（n_crowd > 0；D167-4：只列非 crowd 实例会教模型漏掉人群里的个体，SFT 表达不了 COCO 的
  "忽略"），两条流都不用，记数；
- 其余的 id 按 src/data/subsets.hash_rank(ids, "xm_train_pairs", salt = xm_data.pair_salt) 排序，依次取，直到两条流的样本条数都够 need
  （sizing：ceil(steps × 每步条数 × (1 + margin))，steps 与每步条数取训练配置；缺省 2,000 × 6 × 1.2 = 14,400 条）。一个对给 1 条列框
  样本；样条目标可用（spline_ok）时给 ceil(框数 / inst_max_boxes) 条样条样本，不可用时（真值为空或拟合失败）只进列框流（D167-4），记数。
  非 --tiny 时不够就报错。

**样本行**（两条流；目标文本由 src/xm/prompts 写出，并核对等于 DEV-0079 记的 box_text / spline_text，两边写法不一致就报错）：
- 列框 coco_det（det_row）：id = <对的 id>#det；提示 = call1_prompt(category)（训练时现写）；目标 = format_boxes(boxes_q)（全部框，至多 30 个）。
- 每实例样条 coco_inst（inst_rows）：每条至多 inst_max_boxes（缺省 12）个框，超过的按阅读顺序切成几条（[0, 12)、[12, 24)、[24, n)），
  id = <对的 id>#inst<k>；每条的提示只列它那几个框（call2_prompt(category, 这几个框)），目标是它们的样条 format_splines(…)。切开是为了
  控制训练时的显存（A100 40 GB；一条 16 点样条约 157 个 token，12 个框约 1,900 个目标 token）；推理时仍一次最多 30 个。

**输出**（<out>/）：xm_data/coco_det_<分片名>.jsonl、xm_data/coco_inst_<分片名>.jsonl（一个输入分片对应一对输出分片，按输入的行序，
只含选中的对；先写 .tmp、fsync 之后改名，两个都改名之后这一片才算写完）、xm_data/manifest.json（参数、抽样、剔除的计数、输入的指纹、
各文件的条数与 sha256、分布；只由输入与输出的行算，逐字节可复现）、config.yaml（运行身份）、logs/run_<k>.json。
续跑：config.yaml 相同时，已写完的片与重算的逐字节相同才跳过，没写完的片重算；除 logs/ 外与一次跑完逐字节相同。

**读取**（load_xm_data，训练用）：manifest 与每个文件的 sha256 对得上；参数等于训练配置的；目标的写法（src/xm/prompts 的版本、上限、点数）
等于本次的；每行的目标文本等于按 src/xm/prompts 重写的、提示写得出来；给了 need 时每条流的池不少于它（非 --tiny 的训练：一个 epoch 之内，
见 src/w2/w2c_train.py）。

只 import 不含学习的模块。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from src.data.subsets import hash_rank
from src.xm import data as XMD
from src.xm import prompts as XMP
from src.xm import targets as XT

SCHEMA = 1
FRAMEWORK = "src/xm/train_data.py（DEV-0081；X-M 的 COCO 两条训练流）"
DATA_DIR = "xm_data"
MANIFEST = "manifest.json"
CONFIG = "config.yaml"
KINDS: tuple[str, ...] = ("coco_det", "coco_inst")
PAIR_PURPOSE = "xm_train_pairs"
TRAIN_PREFIX = f"{XMD.SOURCE_COCO}/train2017/"
DEFAULT_PARAMS: Mapping[str, Any] = {"pair_salt": "xm_train_pairs_v1", "margin": 0.2, "inst_max_boxes": 12}


class XMDataError(ValueError):
    """X-M 的训练样本不能生成、不合格，或与训练配置对不上。"""


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


def _posint(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 1


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def check_params(doc: Mapping[str, Any]) -> dict[str, Any]:
    """数据参数 {pair_salt, margin, inst_max_boxes} → 规范化的一份；缺的键报错（不补缺省），多余的键忽略。"""
    bad: list[str] = []
    out: dict[str, Any] = {}
    if not isinstance(doc.get("pair_salt"), str) or not doc["pair_salt"]:
        bad.append(f"pair_salt 必须是非空字符串，收到 {doc.get('pair_salt')!r}")
    else:
        out["pair_salt"] = doc["pair_salt"]
    if not _num(doc.get("margin")) or float(doc["margin"]) < 0:
        bad.append(f"margin 必须是非负数，收到 {doc.get('margin')!r}")
    else:
        out["margin"] = float(doc["margin"])
    mb = doc.get("inst_max_boxes")
    if not _posint(mb) or mb > XT.MAX_BOXES:
        bad.append(f"inst_max_boxes 必须是 1..{XT.MAX_BOXES} 的整数，收到 {mb!r}")
    else:
        out["inst_max_boxes"] = int(mb)
    if bad:
        raise XMDataError("X-M 的数据参数不合格：\n- " + "\n- ".join(bad))
    return out


def params_from_config(doc: Mapping[str, Any]) -> dict[str, Any]:
    """训练配置（configs/xm_train.json）→ 数据参数：xm_data 一节的 pair_salt、margin 与 coco_inst 一节的 max_boxes（= inst_max_boxes）。
    训练（src/w2/w2c_train.xm_spec）与生成（scripts/build_xm_data.py）都用这一个函数。"""
    xd, ci = doc.get("xm_data"), doc.get("coco_inst")
    if not isinstance(xd, Mapping) or not isinstance(ci, Mapping):
        raise XMDataError("配置要有 xm_data 与 coco_inst 两节")
    return check_params({"pair_salt": xd.get("pair_salt"), "margin": xd.get("margin"), "inst_max_boxes": ci.get("max_boxes")})


def sizing(steps: int, per_step: Mapping[str, int], margin: float) -> dict[str, Any]:
    """两条流各要多少条：ceil(steps × 每步条数 × (1 + margin))（按十进制精确算，2,000 × 6 × 1.2 = 14,400）。"""
    if not _posint(steps) or set(per_step) != set(KINDS) or any(not _posint(per_step[k]) for k in KINDS):
        raise XMDataError(f"sizing 要正整数的 steps 与 {list(KINDS)} 各自的每步条数，收到 steps={steps!r}、per_step={dict(per_step)}")
    m = Fraction(str(float(margin)))
    need = {k: int(math.ceil(Fraction(int(steps) * int(per_step[k])) * (1 + m))) for k in KINDS}
    return {"steps": int(steps), "per_step": {k: int(per_step[k]) for k in KINDS}, "margin": float(margin), "need": need}


def sizing_from_config(doc: Mapping[str, Any], *, tiny: bool, steps: int | None = None) -> dict[str, Any]:
    """训练配置 → sizing：步数取 steps（给了）或配置的 steps（--tiny 取 tiny.steps），每步条数取 coco_det、coco_inst 两节的 per_step
    （--tiny 取 tiny_per_step）。"""
    st = int(steps) if steps is not None else int(doc["tiny"]["steps"] if tiny else doc["steps"])
    key = "tiny_per_step" if tiny else "per_step"
    return sizing(st, {k: doc[k][key] for k in KINDS}, params_from_config(doc)["margin"])


def format_identity() -> dict[str, Any]:
    """目标文本的写法（进 manifest 与运行身份）：src/xm/prompts 的版本、是否冻结、两段模板的 sha256（prompts.identity()），与 DEV-0079 的
    上限与点数（src/xm/targets）。读取时要求与本次的逐项相同：提示换了版本就得重新生成。"""
    pi = XMP.identity()
    return {"prompt_version": XMP.PROMPT_VERSION, "prompts_frozen": bool(pi["frozen"]), "templates_sha256": pi["templates_sha256"],
            "max_boxes": XT.MAX_BOXES, "n_ctrl": XT.SPLINE_N}


def require_frozen_prompts(*, tiny: bool) -> None:
    """非 --tiny 时提示必须是冻结的那一份（src/xm/prompts.FROZEN；D168：X-M 训练用冻结的 xm-prompts-v1）。"""
    if not tiny and XMP.FROZEN is not True:
        raise XMDataError(f"src/xm/prompts 的 {XMP.PROMPT_VERSION} 没有冻结（FROZEN = {XMP.FROZEN!r}）：X-M 的训练样本只用冻结的提示（D168）")


# ---------------------------------------------------------------------------
# 规则（纯函数）
# ---------------------------------------------------------------------------


def chunk_ranges(n: int, max_boxes: int) -> list[tuple[int, int]]:
    """n 个框按阅读顺序切成每段至多 max_boxes 个：[(0, 12), (12, 24), (24, n)]。"""
    if not _posint(n) or not _posint(max_boxes):
        raise ValueError(f"n 与 max_boxes 必须是正整数，收到 {n!r}、{max_boxes!r}")
    return [(s, min(s + max_boxes, n)) for s in range(0, n, max_boxes)]


def train_problems(t: Mapping[str, Any]) -> list[str]:
    """DEV-0079 的一行在 targets.row_problems 之外、本任务要的核对：只读训练集、至少一个框。"""
    bad = []
    if t.get("split") != XMD.TRAIN or t.get("source") != XMD.SOURCE_COCO or not str(t.get("id")).startswith(TRAIN_PREFIX):
        bad.append(f"只读训练集：split = {t.get('split')!r}、source = {t.get('source')!r}、id = {t.get('id')!r}")
    if not t.get("boxes_q"):
        bad.append("没有框")
    return bad


def _base(t: Mapping[str, Any]) -> dict[str, Any]:
    w, h = t["image_size"]
    return {"pair_id": t["id"], "image": t["image"], "width": int(w), "height": int(h), "category": t["category"],
            "category_id": int(t["category_id"])}


def _int_boxes(boxes: Sequence[Sequence[Any]]) -> list[list[int]]:
    """网格框 → [[x1, y1, x2, y2]] 整数（DEV-0079 的 row_problems 已核对过范围与顺序）。"""
    return [[int(v) for v in b] for b in boxes]


def det_row(t: Mapping[str, Any]) -> dict[str, Any]:
    """列框的一条样本：全部框（阅读顺序），目标 format_boxes；与 DEV-0079 的 box_text 不同就报错。"""
    boxes = _int_boxes(t["boxes_q"])
    text = XMP.format_boxes(boxes)
    if text != t["box_text"]:
        raise XMDataError(f"{t['id']}：src/xm/prompts.format_boxes 写出的目标与 DEV-0079 的 box_text 不同（两边的写法要一致）")
    return {"id": f"{t['id']}#det", "kind": "coco_det", **_base(t), "n_boxes": len(boxes), "boxes": boxes, "target_text": text}


def inst_rows(t: Mapping[str, Any], max_boxes: int) -> list[dict[str, Any]]:
    """每实例样条的样本（只给 spline_ok 的对）：至多 max_boxes 个框一条，按阅读顺序切；每条的框与样条是它那一段，目标 format_splines；
    整对的 format_splines 与 DEV-0079 的 spline_text 不同就报错。"""
    if not t["spline_ok"]:
        raise ValueError(f"{t['id']}：样条目标不可用（spline_ok = false），只进列框流")
    boxes = _int_boxes(t["boxes_q"])
    splines = [[[int(x), int(y)] for x, y in s] for s in t["ctrl_q"]]
    if XMP.format_splines(splines) != t["spline_text"]:
        raise XMDataError(f"{t['id']}：src/xm/prompts.format_splines 写出的目标与 DEV-0079 的 spline_text 不同（两边的写法要一致）")
    fit = list(t["fit_iou"])
    ranges = chunk_ranges(len(boxes), max_boxes)
    return [{"id": f"{t['id']}#inst{k}", "kind": "coco_inst", **_base(t), "chunk": k, "n_chunks": len(ranges), "box_start": s,
             "n_boxes_pair": len(boxes), "boxes": boxes[s:e], "splines": splines[s:e], "fit_iou": fit[s:e],
             "target_text": XMP.format_splines(splines[s:e])} for k, (s, e) in enumerate(ranges)]


# ---------------------------------------------------------------------------
# 输入（DEV-0079 的训练目标，逐片读）
# ---------------------------------------------------------------------------


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def targets_manifest(targets_dir: str | Path) -> tuple[dict[str, Any], str]:
    mp = Path(targets_dir) / XT.DATA_DIR / XT.MANIFEST
    if not mp.is_file():
        raise XMDataError(f"{mp} 不在（DEV-0079 的训练目标没生成完；--targets 是 scripts/xm/build_targets.py 的 --out）")
    raw = mp.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def source_identity(targets_dir: str | Path) -> dict[str, Any]:
    """来源（DEV-0079 的训练目标）的指纹：manifest 的 sha256、参数、来源（划分与去重清单的 sha256）、各分片的条数与 sha256（不含路径）。"""
    man, sha = targets_manifest(targets_dir)
    return {"format": "DEV-0079 的训练目标（src/xm/targets.py）", "manifest_sha256": sha, "params": man["params"],
            "source": man.get("source"), "files": [{"file": f["file"], "n": f["n"], "sha256": f["sha256"]} for f in man["files"]]}


def iter_target_shards(targets_dir: str | Path, *, check_rows: bool = True) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """逐片读 DEV-0079 的训练目标：(分片名, 行)。每片的 sha256 等于 manifest 记的；check_rows 时每行过 targets.row_problems 与
    train_problems、id 不重复（不合格就抛 XMDataError，列出前几条）。"""
    man, _ = targets_manifest(targets_dir)
    params = XT.check_params(man["params"])
    d = Path(targets_dir) / XT.DATA_DIR
    seen: set[str] = set()
    for f in man["files"]:
        p = d / f["file"]
        if not p.is_file() or _sha(p) != f["sha256"]:
            raise XMDataError(f"{p}：文件不在或 sha256 与 DEV-0079 的 manifest 不符")
        rows = read_jsonl(p)
        if check_rows:
            problems = []
            for r in rows:
                bad = XT.row_problems(r, params) + train_problems(r)
                if r.get("id") in seen:
                    bad.append("id 重复")
                if bad:
                    problems.append(f"{f['file']}（{r.get('id')}）：{'；'.join(bad)}")
                seen.add(r.get("id"))
            if problems:
                raise XMDataError(f"训练目标有 {len(problems)} 行不合格：\n- " + "\n- ".join(problems[:10]))
        yield f["file"], rows


# ---------------------------------------------------------------------------
# 抽对与生成
# ---------------------------------------------------------------------------


def _ids_digest(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def select_pairs(info: Mapping[str, Mapping[str, Any]], *, need: Mapping[str, int], salt: str, max_boxes: int,
                 tiny: bool) -> tuple[set[str], dict[str, Any]]:
    """info：对的 id → {n_kept, spline_ok, n_crowd}。剔除带 crowd 的对，其余按 hash 依次取，直到两条流都够 need（模块说明）。
    返回 (选中的 id, 抽样记录)。"""
    crowd = sorted(i for i, v in info.items() if int(v["n_crowd"]) > 0)
    ok_ids = [i for i in info if int(info[i]["n_crowd"]) == 0]
    ranked = hash_rank(ok_ids, PAIR_PURPOSE, salt)
    got = {k: 0 for k in KINDS}
    chosen: list[str] = []
    for pid in ranked:
        if all(got[k] >= need[k] for k in KINDS):
            break
        chosen.append(pid)
        got["coco_det"] += 1
        if info[pid]["spline_ok"]:
            got["coco_inst"] += len(chunk_ranges(int(info[pid]["n_kept"]), max_boxes))
    ok = all(got[k] >= need[k] for k in KINDS)
    if not ok and not tiny:
        raise XMDataError(f"训练目标里去掉带 crowd 的对后只有 {len(ok_ids)} 个对，抽完也不够 {dict(need)}（得 {got}）")
    rec = {"purpose": PAIR_PURPOSE, "salt": salt, "n_available_pairs": len(info), "n_crowd_dropped": len(crowd),
           "crowd_dropped_ids_sha256": _ids_digest(crowd), "n_eligible_pairs": len(ok_ids), "n_selected_pairs": len(chosen),
           "n_selected_spline_unusable": sum(1 for p in chosen if not info[p]["spline_ok"]), "rows": got, "need": dict(need),
           "satisfied": ok, "selected_ids_sha256": _ids_digest(sorted(chosen))}
    return set(chosen), rec


def build_identity(params: Mapping[str, Any], size: Mapping[str, Any], source: Mapping[str, Any], *, tiny: bool) -> dict[str, Any]:
    return {"experiment": "X-M", "framework": FRAMEWORK, "schema": SCHEMA, "tiny": bool(tiny), "params": json.loads(json.dumps(params)),
            "sizing": json.loads(json.dumps(size)), "format": format_identity(), "source": json.loads(json.dumps(source))}


def _dump_yaml(doc: Mapping[str, Any]) -> str:
    import yaml

    return yaml.safe_dump(json.loads(json.dumps(doc)), allow_unicode=True, sort_keys=False)


def _prepare(out: Path, ident: Mapping[str, Any], *, resume: bool) -> None:
    cfg = out / CONFIG
    if cfg.exists():
        import yaml

        if yaml.safe_load(cfg.read_text(encoding="utf-8")) != json.loads(json.dumps(ident)):
            raise XMDataError(f"{out}：已有的 {CONFIG} 与本次的参数或输入不同，拒绝混写；换一个 --out")
        if not resume:
            raise XMDataError(f"{out} 已存在；续跑请加 --resume")
    else:
        if out.exists() and any(out.iterdir()):
            raise XMDataError(f"{out} 非空、却没有 {CONFIG}，拒绝写入")
        out.mkdir(parents=True, exist_ok=True)
        (out / CONFIG).write_text(_dump_yaml(ident), encoding="utf-8")
    (out / DATA_DIR).mkdir(exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)


def out_name(kind: str, shard: str) -> str:
    return f"{kind}_{Path(shard).stem}.jsonl"


def _line(doc: Mapping[str, Any]) -> str:
    return json.dumps(doc, ensure_ascii=False) + "\n"


def _write_pair(d: Path, shard: str, lines: Mapping[str, Sequence[str]]) -> None:
    tmps = {}
    for kind in KINDS:
        tmp = d / (out_name(kind, shard) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("".join(lines[kind]))
            f.flush()
            os.fsync(f.fileno())
        tmps[kind] = tmp
    for kind in KINDS:  # 两个都改名之后这一片才算写完
        os.replace(tmps[kind], d / out_name(kind, shard))


def shard_done(d: Path, shard: str) -> bool:
    return all((d / out_name(kind, shard)).is_file() for kind in KINDS)


def _expected_lines(rows: Sequence[Mapping[str, Any]], max_boxes: int) -> dict[str, list[str]]:
    lines: dict[str, list[str]] = {k: [] for k in KINDS}
    for r in rows:
        lines["coco_det"].append(_line(det_row(r)))
        if r["spline_ok"]:
            lines["coco_inst"].extend(_line(x) for x in inst_rows(r, max_boxes))
    return lines


def build(targets_dir: str | Path, out: str | Path, *, params: Mapping[str, Any], size: Mapping[str, Any], tiny: bool = False,
          resume: bool = False, data_root: str | Path | None = None, argv: Sequence[str] = (),
          log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """生成 X-M 的两条训练流的样本（规则见模块说明），返回 manifest。data_root 给了就核对选中的对的图像都在。"""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    out = Path(out)
    require_frozen_prompts(tiny=tiny)
    params = check_params(params)
    source = source_identity(targets_dir)
    ident = build_identity(params, size, source, tiny=tiny)
    _prepare(out, ident, resume=resume)
    d = out / DATA_DIR
    names = {out_name(k, f["file"]) for f in source["files"] for k in KINDS} | {MANIFEST}
    extra = sorted(p.name for p in d.iterdir() if p.name not in names and not p.name.endswith(".tmp"))
    if extra:
        raise XMDataError(f"{d} 里有不属于这份输入的文件：{extra[:5]}")
    run_k = len(list((out / "logs").glob("run_*.json"))) + 1
    run_path = out / "logs" / f"run_{run_k:03d}.json"
    run_log: dict[str, Any] = {"argv": list(argv), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                               "shards_total": len(source["files"]), "finished_utc": None}
    t0 = time.perf_counter()
    info: dict[str, dict[str, Any]] = {}
    for _, rows in iter_target_shards(targets_dir, check_rows=True):  # 第一遍：逐行核对，只留抽对要的几项
        for r in rows:
            info[r["id"]] = {"n_kept": int(r["n_kept"]), "spline_ok": bool(r["spline_ok"]), "n_crowd": int(r["n_crowd"])}
    selected, sel = select_pairs(info, need=size["need"], salt=params["pair_salt"], max_boxes=params["inst_max_boxes"], tiny=tiny)
    del info
    skipped = []
    for name, rows in iter_target_shards(targets_dir, check_rows=False):  # 第二遍：写选中的对（sha256 已逐片核对）
        chosen = [r for r in rows if r["id"] in selected]
        lines = _expected_lines(chosen, params["inst_max_boxes"])
        if shard_done(d, name):
            for kind in KINDS:
                if (d / out_name(kind, name)).read_text(encoding="utf-8") != "".join(lines[kind]):
                    raise XMDataError(f"{d / out_name(kind, name)}：已写完的片与本次重算的不同，拒绝续跑")
            skipped.append(name)
            continue
        if data_root is not None:
            miss = [r["image"] for r in chosen if not (Path(data_root) / r["image"]).is_file()]
            if miss:
                raise XMDataError(f"{name}：{len(miss)} 张图不在 {data_root} 下，例如 {miss[:3]}")
        for t in d.glob(out_name("*", name) + ".tmp"):
            t.unlink()
        _write_pair(d, name, lines)
        log(f"[X-M 数据] {name}：{len(rows)} 对，选中 {len(chosen)} → 列框 {len(lines['coco_det'])} 条、样条 {len(lines['coco_inst'])} 条")
    man = finalize(out, source, params, size, sel)
    run_log.update(finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), wall_s=round(time.perf_counter() - t0, 3),
                   shards_skipped=skipped, complete=True)
    run_path.write_text(json.dumps(run_log, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return man


def _dist(values: Sequence[float]) -> dict[str, Any] | None:
    v = [float(x) for x in values if x is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=np.float64)
    q = np.percentile(a, [50, 90, 99])
    return {"n": len(v), "mean": float(a.mean()), "min": float(a.min()), "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(a.max())}


def _counts(values) -> dict[str, int]:
    c = Counter(str(v) for v in values)
    return {k: c[k] for k in sorted(c, key=lambda x: (len(x), x))}


def finalize(out: Path, source: Mapping[str, Any], params: Mapping[str, Any], size: Mapping[str, Any],
             selection: Mapping[str, Any]) -> dict[str, Any]:
    """manifest.json：只由输入与输出的行算（不含耗时与路径），逐字节可复现。"""
    d = out / DATA_DIR
    files, det, inst = [], [], []
    for f in source["files"]:
        for kind in KINDS:
            p = d / out_name(kind, f["file"])
            rows = read_jsonl(p)
            files.append({"file": p.name, "kind": kind, "n": len(rows), "sha256": _sha(p)})
            (det if kind == "coco_det" else inst).extend(rows)
    man = {
        "schema": SCHEMA, "framework": FRAMEWORK, "params": dict(params), "sizing": dict(size), "format": format_identity(),
        "source": dict(source), "selection": dict(selection),
        "counts": {"coco_det": len(det), "coco_inst": len(inst), "pairs": len(det), "images": len({r["image"] for r in det}),
                   "pairs_split": sum(1 for r in det if r["n_boxes"] > params["inst_max_boxes"]),
                   "pairs_det_only": selection["n_selected_spline_unusable"], "crowd_pairs_dropped": selection["n_crowd_dropped"]},
        "files": files,
        "distributions": {"boxes_per_pair": _dist([r["n_boxes"] for r in det]), "boxes_per_inst_row": _dist([len(r["boxes"]) for r in inst]),
                          "chunks_per_pair": _counts(r["n_chunks"] for r in inst if r["chunk"] == 0),
                          "fit_iou": _dist([x for r in inst for x in r["fit_iou"]]),
                          "det_target_chars": _dist([len(r["target_text"]) for r in det]),
                          "inst_target_chars": _dist([len(r["target_text"]) for r in inst])},
    }
    (d / MANIFEST).write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return man


# ---------------------------------------------------------------------------
# 读取（训练用）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class XMData:
    """X-M 的训练样本：rows（两类的行，按 manifest 的文件顺序）；pools：kind → rows 的下标。"""

    run_dir: Path
    rows: tuple[Mapping[str, Any], ...]
    pools: Mapping[str, tuple[int, ...]]
    manifest_sha256: str
    manifest: Mapping[str, Any]

    def pool(self, kind: str) -> tuple[int, ...]:
        return self.pools[kind]

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(self.pools)

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.manifest["params"])

    def file_paths(self) -> list[Path]:
        d = self.run_dir / DATA_DIR
        return [d / f["file"] for f in self.manifest["files"]] + [d / MANIFEST]

    def identity(self) -> dict[str, Any]:
        """运行身份里的 X-M 数据一节：manifest 的 sha256、参数、抽样、来源、池的条数与 id 指纹；不含路径。"""
        sel = self.manifest["selection"]
        return {"format": FRAMEWORK, "manifest_sha256": self.manifest_sha256, "params": self.params,
                "sizing": dict(self.manifest["sizing"]), "target_format": dict(self.manifest["format"]),
                "selection": {k: sel[k] for k in ("salt", "n_available_pairs", "n_crowd_dropped", "n_eligible_pairs", "n_selected_pairs",
                                                   "n_selected_spline_unusable", "selected_ids_sha256")},
                "source_manifest_sha256": self.manifest["source"]["manifest_sha256"],
                "pools": {k: {"n": len(p), "ids_sha256": _ids_digest([str(self.rows[i]["id"]) for i in p])} for k, p in self.pools.items()},
                "counts": dict(self.manifest["counts"])}


def row_problems(r: Mapping[str, Any], kind: str, params: Mapping[str, Any]) -> list[str]:
    """一行样本的问题（空 = 合格）：目标文本等于重写的、提示写得出来、样条行至多 inst_max_boxes 个框。"""
    bad: list[str] = []
    try:
        if not _posint(r["width"]) or not _posint(r["height"]) or not str(r["pair_id"]).startswith(TRAIN_PREFIX):
            bad.append("width / height / pair_id 不对")
        if kind == "coco_det":
            if r["id"] != f"{r['pair_id']}#det" or r["target_text"] != XMP.format_boxes(r["boxes"]) or r["n_boxes"] != len(r["boxes"]) \
                    or not 1 <= len(r["boxes"]) <= XT.MAX_BOXES:
                bad.append("列框的 id、框数或目标文本对不上")
            XMP.call1_prompt(r["category"])
        else:
            if r["id"] != f"{r['pair_id']}#inst{r['chunk']}" or len(r["boxes"]) != len(r["splines"]) \
                    or not 1 <= len(r["boxes"]) <= params["inst_max_boxes"] or r["target_text"] != XMP.format_splines(r["splines"]):
                bad.append("样条的 id、框数（至多 inst_max_boxes）或目标文本对不上")
            XMP.call2_prompt(r["category"], r["boxes"])
    except (KeyError, TypeError, ValueError) as e:
        bad.append(f"结构不对：{type(e).__name__}: {e}")
    return bad


def load_xm_data(run_dir: str | Path, *, params: Mapping[str, Any], data_root: str | Path | None = None,
                 need: Mapping[str, int] | None = None) -> XMData:
    """读 build 的输出并核对（模块说明）。params：训练配置里的数据参数；data_root 给了就核对图像在；need 给了就要求每条流的池不少于它。"""
    rd = Path(run_dir)
    d = rd / DATA_DIR
    mp = d / MANIFEST
    if not mp.is_file():
        raise XMDataError(f"{mp} 不在（数据没生成完）")
    man = json.loads(mp.read_text(encoding="utf-8"))
    want = check_params(params)
    if man.get("params") != want:
        raise XMDataError(f"X-M 数据的参数 {man.get('params')} 与训练配置的 {want} 不同")
    if man.get("format") != format_identity():
        raise XMDataError(f"X-M 数据的目标写法 {man.get('format')} 与本次的 src/xm/prompts {format_identity()} 不同：重新生成")
    rows: list[Mapping[str, Any]] = []
    pools: dict[str, list[int]] = {k: [] for k in KINDS}
    problems: list[str] = []
    seen: set[str] = set()
    for f in man["files"]:
        p = d / f["file"]
        if f["kind"] not in KINDS or not p.is_file() or _sha(p) != f["sha256"]:
            raise XMDataError(f"{p}：文件不在、类不对或 sha256 与 manifest 不符")
        for r in read_jsonl(p):
            if r.get("kind") != f["kind"] or r.get("id") in seen:
                problems.append(f"{f['file']}（{r.get('id')}）：kind 不对或 id 重复")
                continue
            bad = row_problems(r, f["kind"], want)
            if not bad and data_root is not None and not (Path(data_root) / r["image"]).is_file():
                bad.append(f"图像不在：{Path(data_root) / r['image']}")
            if bad:
                problems.append(f"{f['file']}（{r['id']}）：{'；'.join(bad)}")
                continue
            seen.add(r["id"])
            pools[f["kind"]].append(len(rows))
            rows.append(r)
    if problems:
        raise XMDataError(f"X-M 的数据有 {len(problems)} 行不合格：\n- " + "\n- ".join(problems[:10]))
    for k in KINDS:
        if not pools[k]:
            raise XMDataError(f"{rd}：{k} 的池是空的")
        if need is not None and len(pools[k]) < int(need[k]):
            raise XMDataError(f"{rd}：{k} 的池 {len(pools[k])} 条，不够这次训练的 {int(need[k])} 条（步数 × 每步条数，一个 epoch 之内）；"
                              "按本次的配置与步数重新生成（scripts/build_xm_data.py）")
    return XMData(run_dir=rd, rows=tuple(rows), pools={k: tuple(pools[k]) for k in KINDS}, manifest_sha256=_sha(mp), manifest=man)


__all__ = [
    "DATA_DIR", "DEFAULT_PARAMS", "KINDS", "MANIFEST", "PAIR_PURPOSE", "TRAIN_PREFIX", "XMData", "XMDataError", "build", "check_params",
    "chunk_ranges", "det_row", "finalize", "format_identity", "inst_rows", "iter_target_shards", "load_xm_data", "out_name",
    "params_from_config", "require_frozen_prompts", "row_problems", "select_pairs", "sizing", "sizing_from_config", "source_identity",
    "targets_manifest", "train_problems",
]
