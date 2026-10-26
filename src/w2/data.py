"""W2 的数据划分（DEV-0059；D113；规则在 configs/w2.json 的 splits，清单在 configs/w2_splits.json）。

划分（名字 → 样本）：
- devtune800（开发-调参，可以随便用）：W1 的 devtune 300 句加 devcheck 500 句，从 configs/w1_splits.json 原样取（两段的条数与 sha256
  先与 W1 的清单核对），按 id 自然顺序列出。用于 refseg 与 refexp_box。
- devcheck2（新的开发-检查集，只用于去留判定）：
  1. 总体 = load_refcoco('refcoco', 'val', subset='eval') 的全部句子（E1 的 refcoco_unc_val；加载器已按 E1 的去重规则剔除开发子集与
     D20 探针图）。条数与 sha256 必须等于 configs/e1_eval_subsets.json 的 populations.refcoco_unc_val（10,634 句，1783dd7f…）；
  2. 按图像排除：devtune800 那 800 句所在的图（Sample.image_keys，coco:<image_id>）上的全部句子都去掉；
  3. 剩下的句子按 sha256(salt|purpose|sent_id) 升序（src/data/subsets.hash_rank，salt = w2_dev_v1，purpose = w2_devcheck）取前 500 句，
     按 id 自然顺序列出。
  各步的条数与 sha256、排除的图、每句所在的图都写进清单（construction、image_of），所以不要数据也能核对"与 W1 不重叠"（按句、按图）。
- test_<数据集_划分方式_划分>：同 W1（E1 的 poly 子集原样，每个 1,000 句）；devtune_countbench、test_countbench：同 W1（计数按输出类型
  路由到 E1 单轮）。都从 configs/w1_splits.json 原样复制。

开发机上没有 RefCOCO 的数据，所以 devcheck2 由服务器生成（CPU 即可）：
    python -m src.w2.data --build --data-root $PROJ/data --out <路径>/w2_splits.json
生成的文件交回后由管理员放到 configs/w2_splits.json。开发机上先用 --write-skeleton 写出 devcheck2 待生成（status = pending）的清单，
其余各划分可以先用。核对：python -m src.w2.data --check [--splits 路径] [--data-root …]（不给数据时核对不需要数据的部分与 devcheck2 的
自洽；给了就整份重算、逐字节比较）。--tiny：在 tests/fixtures 上用同一套规则现算（W1 的 tiny 划分 + tests/fixtures/w2 的 RefCOCO）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data import loaders as L
from src.data.eval_subsets import EVAL_SUBSETS_PATH, load_eval_subsets
from src.data.records import Sample
from src.data.subsets import DEV_SUBSETS_PATH, dump_config, hash_rank, population_digest, sort_ids
from src.e1.tasks import FIXTURE_DIR, DataPaths, _sent_id
from src.w1.config import load_w1_config
from src.w1.data import W1_SPLITS_PATH, load_splits as load_w1_splits, poly_test_split, tiny_splits as w1_tiny_splits, w1_items
from src.w2.config import W2Config, load_w2_config

REPO_ROOT = Path(__file__).resolve().parents[2]
W2_SPLITS_PATH = REPO_ROOT / "configs" / "w2_splits.json"
W2_SPLITS_VERSION = 1
W2_FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "w2" / "root"  # tests/fixtures/e1_posthoc 的 RefCOCO 加两张 val 图（905、906）
REFCOCO_TASKS: tuple[str, ...] = ("refseg", "refexp_box")
DEVTUNE = "devtune800"
DEVCHECK2 = "devcheck2"
STATUS_PENDING = "pending"
STATUS_BUILT = "built"
BUILD_COMMAND = "python -m src.w2.data --build --data-root $PROJ/data --out <路径>/w2_splits.json"


class SplitsError(ValueError):
    """划分不能生成或核对不过。"""


def _rel(p: str | Path) -> str:
    q = Path(p).resolve()
    return str(q.relative_to(REPO_ROOT)) if q.is_relative_to(REPO_ROOT) else str(p)


def _file(path: str | Path) -> dict[str, str]:
    return {"path": _rel(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def needs_frozen(split: str) -> bool:
    """devcheck2 与 test_*（判定集与 test）只能用冻结的配置跑（同 D108）；devtune800 与 devtune_countbench 不要求。"""
    return split == DEVCHECK2 or split.startswith("test")


# ---------------------------------------------------------------------------
# 规则（纯函数）
# ---------------------------------------------------------------------------


def devtune800_ids(w1_doc: Mapping[str, Any], names: Sequence[str]) -> list[str]:
    """W1 清单里 names（devtune、devcheck）的 id 并集（先核对各段的条数与 sha256 与 id 一致、两段不重叠），按 id 自然顺序。"""
    out: list[str] = []
    for nm in names:
        e = w1_doc["splits"][nm]
        ids = [str(i) for i in e["ids"]]
        if len(ids) != e["n"] or population_digest(ids) != e["sha256"]:
            raise SplitsError(f"W1 清单的 {nm}：{len(ids)} 条、指纹 {population_digest(ids)[:12]}… 与记的 {e['n']} 条 {e['sha256'][:12]}… 不符")
        out += ids
    if len(set(out)) != len(out):
        raise SplitsError(f"W1 的 {list(names)} 有重叠的句子")
    return sort_ids(out)


def select_devcheck2(population: Sequence[tuple[str, str]], exclude_ids: Iterable[str], *, n: int, salt: str,
                     purpose: str) -> dict[str, Any]:
    """devcheck2 的规则（模块说明第 2、3 步）。population = 总体的 (sent_id, 图像键)；exclude_ids = 要按图像排除的句子（必须都在总体里）。
    返回 ids（自然顺序）、image_of（选中句子 → 图像键）、排除的图与各步的条数与 sha256。"""
    sids = [str(s) for s, _ in population]
    if len(set(sids)) != len(sids):
        raise SplitsError("总体里的 sent_id 有重复")
    img = {str(s): str(k) for s, k in population}
    excl = sorted({str(i) for i in exclude_ids})
    missing = [i for i in excl if i not in img]
    if missing:
        raise SplitsError(f"要排除的 {len(missing)} 句不在总体里，例如 {missing[:5]}")
    excl_images = sorted({img[i] for i in excl})
    bad = set(excl_images)
    pool = [s for s in sids if img[s] not in bad]
    if len(pool) < n:
        raise SplitsError(f"按图像排除之后只剩 {len(pool)} 句，不够 {n} 句")
    chosen = sort_ids(hash_rank(pool, purpose, salt)[:n])
    return {
        "ids": chosen,
        "image_of": {s: img[s] for s in chosen},
        "steps": {
            "population": {"n": len(sids), "sha256": population_digest(sids)},
            "exclude": {"n_sentences": len(excl), "sentences_sha256": population_digest(excl), "n_images": len(excl_images),
                        "images_sha256": population_digest(excl_images),
                        "n_sentences_removed": len(sids) - len(pool)},
            "after_exclusion": {"n": len(pool), "sha256": population_digest(pool)},
            "selected": {"n": len(chosen), "sha256": population_digest(chosen)},
        },
        "excluded_images": excl_images,
    }


def _refcoco_entry(ids: Sequence[str], rule: str, *, dataset: str = "refcoco", split_by: str = "unc", split: str = "val",
                   **extra: Any) -> dict[str, Any]:
    ids = sort_ids(ids)
    return {"kind": "refcoco", "tasks": list(REFCOCO_TASKS), "dataset": dataset, "split_by": split_by, "split": split,
            "unit": "sent_id", "rule": rule, **extra, "n": len(ids), "sha256": population_digest(ids), "ids": ids}


def devcheck2_rule(cfg: W2Config, n: int) -> str:
    d = cfg.splits[DEVCHECK2]
    return (f"load_refcoco('refcoco', 'val', subset='eval') 的全部句子（E1 的 {d['population']}），按图像排除 {d['exclude_images_of']} 那 "
            f"的句子所在的全部图，剩下的按 sha256({d['salt']}|{d['purpose']}|sent_id) 升序取前 {n} 句")


def pending_devcheck2(cfg: W2Config) -> dict[str, Any]:
    d = cfg.splits[DEVCHECK2]
    return {"kind": "refcoco", "tasks": list(REFCOCO_TASKS), "dataset": d["dataset"], "split_by": d["split_by"], "split": d["split"],
            "unit": "sent_id", "rule": devcheck2_rule(cfg, int(d["n"])), "status": STATUS_PENDING,
            "build_command": BUILD_COMMAND, "n": None, "sha256": None, "ids": None}


def devcheck2_entry(cfg: W2Config, sel: Mapping[str, Any], *, n: int, population_check: Mapping[str, Any]) -> dict[str, Any]:
    d = cfg.splits[DEVCHECK2]
    steps = dict(sel["steps"])
    steps["population"] = {**steps["population"], "name": d["population"],
                           "provenance": "load_refcoco('refcoco', 'val', subset='eval')", "check": dict(population_check)}
    steps["exclude"] = {"of": d["exclude_images_of"], **steps["exclude"], "images": list(sel["excluded_images"])}
    steps["selected"] = {"salt": d["salt"], "purpose": d["purpose"], **steps["selected"]}
    e = _refcoco_entry(sel["ids"], devcheck2_rule(cfg, n), dataset=d["dataset"], split_by=d["split_by"], split=d["split"],
                       status=STATUS_BUILT, construction=steps)
    e["image_of"] = {s: sel["image_of"][s] for s in e["ids"]}
    return e


def build_splits(cfg: W2Config, w1_doc: Mapping[str, Any], *, devcheck2: Mapping[str, Any] | None, tiny: bool,
                 sources: Mapping[str, Any]) -> dict[str, Any]:
    """整份清单（键序固定，可直接 dump_config）。devcheck2 为 None 时写待生成。"""
    sp = cfg.splits
    names = list(sp[DEVTUNE]["w1_splits"])
    splits: dict[str, Any] = {DEVTUNE: _refcoco_entry(devtune800_ids(w1_doc, names), f"W1 的 {' + '.join(names)}（configs/w1_splits.json 原样）")}
    splits[DEVCHECK2] = dict(devcheck2) if devcheck2 is not None else pending_devcheck2(cfg)
    for key in sp["test"]:
        name = poly_test_split(key)
        if name not in w1_doc["splits"]:
            raise SplitsError(f"W1 的清单里没有 {name}")
        splits[name] = dict(w1_doc["splits"][name])
    for name in sp["count"]:
        splits[name] = dict(w1_doc["splits"][name])
    d2 = sp[DEVCHECK2]
    return {"version": W2_SPLITS_VERSION, "salt": d2["salt"], "purpose": d2["purpose"],
            "method": "devtune800 = W1 的 devtune + devcheck 原样；devcheck2 = refcoco_unc_val 的 eval 总体按图像排除 devtune800 所在的图后，"
                      "按 sha256(salt|purpose|sent_id) 升序（src/data/subsets.hash_rank）取前 n 句；test 与计数同 W1。各划分按 id 自然顺序列出；"
                      "sha256 = 自然顺序、换行连接后的 sha256（src/data/subsets.population_digest）",
            "tiny": tiny, "sources": dict(sources), "splits": splits}


def splits_sha256(doc: Mapping[str, Any]) -> str:
    """清单的指纹 = 规范文本（dump_config）的 sha256；入库文件就是这份文本，所以等于文件的 sha256。"""
    return hashlib.sha256(dump_config(doc).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 生成（要数据）
# ---------------------------------------------------------------------------


def _w1_doc(tiny: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """W1 的清单与它的出处：真运行读 configs/w1_splits.json；--tiny 用 W1 的 tiny 划分（tests/fixtures/e1 上现算）。"""
    if tiny:
        return w1_tiny_splits(load_w1_config()), {"path": "tiny（src/w1/data.tiny_splits，tests/fixtures/e1 上现算）"}
    return load_w1_splits(W1_SPLITS_PATH), _file(W1_SPLITS_PATH)


def tiny_paths() -> DataPaths:
    """--tiny 的 RefCOCO：tests/fixtures/w2/root（e1_posthoc 的 48×36 图加 905、906 两张 val 图），清单用 tests/fixtures/e1 的。"""
    return DataPaths(root=W2_FIXTURE_ROOT, dev_subsets=FIXTURE_DIR / "dev_subsets.json", eval_subsets=FIXTURE_DIR / "eval_subsets.json",
                     nyu_pairs=FIXTURE_DIR / "nyu_pairs.json")


def population_records(paths: DataPaths) -> list[Sample]:
    return L.load_refcoco(paths.root, "refcoco", "val", subset="eval", dev_config=paths.dev_subsets, strict=paths.strict).records


def build_with_data(cfg: W2Config, paths: DataPaths, *, tiny: bool = False) -> dict[str, Any]:
    """在数据上生成整份清单（devcheck2 也生成）。真运行时总体的条数与 sha256 必须等于 E1 评测清单记的那一份。"""
    w1_doc, w1_src = _w1_doc(tiny)
    sp = cfg.splits
    d2 = sp[DEVCHECK2]
    n = int(sp["tiny"]["devcheck2_n"] if tiny else d2["n"])
    recs = population_records(paths)
    pop = []
    for s in recs:
        if len(s.image_keys) != 1:
            raise SplitsError(f"{s.id}：RefCOCO 应是单图，收到 {len(s.image_keys)} 个图像键")
        pop.append((_sent_id(s), s.image_keys[0]))
    sids = [p[0] for p in pop]
    if tiny:
        check: dict[str, Any] = {"status": "跳过（--tiny：fixture 的总体与 E1 的清单无关）"}
    else:
        want = load_eval_subsets(paths.eval_subsets)["populations"][d2["population"]]
        got = {"n": len(sids), "sha256": population_digest(sids)}
        if (got["n"], got["sha256"]) != (int(want["n"]), want["sha256"]):
            raise SplitsError(f"总体 {got} 与 {paths.eval_subsets} 记的 {d2['population']}（n = {want['n']}，sha256 {want['sha256'][:12]}…）不同")
        check = {"status": "一致", "against": f"{_rel(paths.eval_subsets)} 的 populations.{d2['population']}", "n": int(want["n"]),
                 "sha256": want["sha256"]}
    excl = devtune800_ids(w1_doc, list(sp[DEVTUNE]["w1_splits"]))
    sel = select_devcheck2(pop, excl, n=n, salt=d2["salt"], purpose=d2["purpose"])
    entry = devcheck2_entry(cfg, sel, n=n, population_check=check)
    sources = {"w1_splits": w1_src, "e1_eval_subsets": _file(paths.eval_subsets), "e1_dev_subsets": _file(paths.dev_subsets),
               "w2_config_splits": {"path": _rel(cfg.path), "splits_sha256": _section_sha(cfg)}}
    return build_splits(cfg, w1_doc, devcheck2=entry, tiny=tiny, sources=sources)


def _section_sha(cfg: W2Config) -> str:
    """configs/w2.json 的 splits 一节的 sha256（规范 JSON）：只认这一节，配置别处的改动不影响划分。"""
    return hashlib.sha256(json.dumps(cfg.splits, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_skeleton(cfg: W2Config) -> dict[str, Any]:
    """不要数据：devcheck2 待生成，其余各划分齐全。"""
    w1_doc, w1_src = _w1_doc(False)
    sources = {"w1_splits": w1_src, "e1_eval_subsets": _file(EVAL_SUBSETS_PATH), "e1_dev_subsets": _file(DEV_SUBSETS_PATH),
               "w2_config_splits": {"path": _rel(cfg.path), "splits_sha256": _section_sha(cfg)}}
    return build_splits(cfg, w1_doc, devcheck2=None, tiny=False, sources=sources)


def tiny_splits(cfg: W2Config | None = None) -> dict[str, Any]:
    """--tiny：fixture 上按同一套规则现算（devcheck2 的大小取 splits.tiny.devcheck2_n）。"""
    return build_with_data(cfg or load_w2_config(), tiny_paths(), tiny=True)


# ---------------------------------------------------------------------------
# 核对（不要数据）与取样本
# ---------------------------------------------------------------------------


def check_devcheck2(doc: Mapping[str, Any], cfg: W2Config) -> list[str]:
    """devcheck2 的自洽（不要数据）：条数、sha256、与 devtune800 按句与按图都不重叠、每句的图都在、选中的句子不在排除的图上、各步的数一致。
    返回问题列表（空 = 通过）。待生成时返回 ["pending"]。"""
    e = doc["splits"][DEVCHECK2]
    if e.get("status") == STATUS_PENDING:
        return ["pending"]
    bad: list[str] = []
    ids = [str(i) for i in e["ids"]]
    n_want = int(cfg.splits["tiny"]["devcheck2_n"] if doc.get("tiny") else cfg.splits[DEVCHECK2]["n"])
    if len(ids) != e["n"] or e["n"] != n_want or population_digest(ids) != e["sha256"]:
        bad.append(f"条数或 sha256 不对：{len(ids)} 条（记 {e['n']}，应为 {n_want}），指纹 {population_digest(ids)[:12]}…（记 {str(e['sha256'])[:12]}…）")
    if ids != sort_ids(ids):
        bad.append("id 不是自然顺序")
    dt = set(doc["splits"][DEVTUNE]["ids"])
    if dt & set(ids):
        bad.append(f"与 devtune800 有 {len(dt & set(ids))} 句重叠")
    img = e.get("image_of") or {}
    if set(img) != set(ids):
        bad.append("image_of 与 ids 对不上")
    c = e["construction"]
    excl = set(c["exclude"]["images"])
    on_excluded = sorted(s for s in ids if img.get(s) in excl)
    if on_excluded:
        bad.append(f"{len(on_excluded)} 句在排除的图上，例如 {on_excluded[:3]}")
    if population_digest(sorted(dt)) != c["exclude"]["sentences_sha256"] or len(dt) != c["exclude"]["n_sentences"]:
        bad.append("construction.exclude 记的句子不是 devtune800")
    if len(excl) != c["exclude"]["n_images"] or population_digest(sorted(excl)) != c["exclude"]["images_sha256"]:
        bad.append("construction.exclude 的图与它记的条数或 sha256 不符")
    if c["selected"]["n"] != len(ids) or c["selected"]["sha256"] != e["sha256"]:
        bad.append("construction.selected 与 ids 不符")
    if c["population"]["n"] - c["exclude"]["n_sentences_removed"] != c["after_exclusion"]["n"]:
        bad.append("construction 各步的条数对不上")
    return bad


def load_splits(path: str | Path = W2_SPLITS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class SplitsDoc(dict):
    """另一份 W2 格式的划分清单（scripts/run_w2.py --splits-file，DEV-0067；如 configs/explore_splits.json）：内容同 dict，另记来源文件
    path，运行身份记它（src/w2/runner.w2_record）。默认的 configs/w2_splits.json 照旧用 load_splits 读成普通 dict，输出不变。"""

    path: Path


def load_splits_file(path: str | Path) -> SplitsDoc:
    doc = SplitsDoc(load_splits(path))
    doc.path = Path(path)
    return doc


def splits_path(doc: Mapping[str, Any]) -> Path:
    """清单的来源文件：load_splits_file 读出的记它自己的 path，其余（普通 dict）是 configs/w2_splits.json。"""
    return Path(getattr(doc, "path", W2_SPLITS_PATH))


def split_names(task: str, doc: Mapping[str, Any]) -> list[str]:
    return [k for k, v in doc["splits"].items() if task in v["tasks"]]


def w2_items(task: str, split: str, paths: DataPaths, *, splits: Mapping[str, Any] | None = None) -> list[Sample]:
    """task × 划分 → 样本（加载器的固定顺序；清单里的 id 找不到、取出的 id 指纹与清单不符都报错，同 src/w1/data.w1_items）。"""
    doc = splits if splits is not None else load_splits()
    if split not in doc["splits"]:
        raise KeyError(f"没有划分 {split!r}；可用：{list(doc['splits'])}")
    e = doc["splits"][split]
    if e.get("status") == STATUS_PENDING:
        raise SplitsError(f"划分 {split} 还没生成（status = pending）：先在服务器上 {e.get('build_command', BUILD_COMMAND)}")
    return w1_items(task, split, paths, splits=doc)


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成或核对 W2 的划分清单（configs/w2_splits.json）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="在数据上生成整份清单（要 --data-root 与 --out；--tiny 用 fixture）")
    g.add_argument("--write-skeleton", action="store_true", help="不要数据：写 configs/w2_splits.json，devcheck2 待生成")
    g.add_argument("--check", action="store_true", help="核对清单（给 --data-root 时整份重算比较）")
    ap.add_argument("--data-root", help="落地目录（服务器上 $PROJ/data）")
    ap.add_argument("--out", help="--build 的输出文件")
    ap.add_argument("--splits", default=str(W2_SPLITS_PATH), help="--check 的清单（默认 configs/w2_splits.json）")
    ap.add_argument("--w2-config", default=None, help="默认 configs/w2.json")
    ap.add_argument("--tiny", action="store_true", help="用 tests/fixtures 的小数据（--build、--check）")
    a = ap.parse_args(argv)
    cfg = load_w2_config(a.w2_config) if a.w2_config else load_w2_config()
    paths = tiny_paths() if a.tiny else (DataPaths(root=Path(a.data_root)) if a.data_root else None)
    try:
        if a.build:
            if paths is None or not a.out:
                ap.error("--build 要 --data-root（或 --tiny）与 --out")
            doc = build_with_data(cfg, paths, tiny=a.tiny)
            text = dump_config(doc)
            out = Path(a.out)
            if out.exists() and out.read_text(encoding="utf-8") != text:
                print(f"{out} 已存在且内容不同，不覆盖", file=sys.stderr)
                return 2
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            c = doc["splits"][DEVCHECK2]["construction"]
            print(f"写出 {out}（sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}）")
            print(f"总体 {c['population']['n']} 句（{c['population']['sha256']}；核对：{c['population']['check']['status']}）；"
                  f"排除 devtune800 的 {c['exclude']['n_sentences']} 句所在的 {c['exclude']['n_images']} 张图，去掉 "
                  f"{c['exclude']['n_sentences_removed']} 句；剩 {c['after_exclusion']['n']} 句（{c['after_exclusion']['sha256']}）；"
                  f"取 {c['selected']['n']} 句（{c['selected']['sha256']}）")
            return 0
        if a.write_skeleton:
            text = dump_config(build_skeleton(cfg))
            if W2_SPLITS_PATH.exists():
                old = load_splits(W2_SPLITS_PATH)
                if old["splits"].get(DEVCHECK2, {}).get("status") == STATUS_BUILT:
                    print(f"{W2_SPLITS_PATH} 里的 devcheck2 已生成，不用骨架覆盖", file=sys.stderr)
                    return 2
            W2_SPLITS_PATH.write_text(text, encoding="utf-8")
            print(f"写出 {W2_SPLITS_PATH}（devcheck2 待生成；sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}）")
            return 0
        doc = load_splits(a.splits)
        text = Path(a.splits).read_text(encoding="utf-8")
        if paths is not None:
            same = dump_config(build_with_data(cfg, paths, tiny=a.tiny)) == text
            print("一致（整份重算）" if same else f"{a.splits} 与重算的结果不同")
            return 0 if same else 1
        ref = build_splits(cfg, *_w1_doc(bool(doc.get("tiny")))[:1], devcheck2=doc["splits"][DEVCHECK2], tiny=bool(doc.get("tiny")),
                           sources=doc["sources"])
        problems = [] if dump_config(ref) == text else ["devcheck2 以外的部分与按 W1 清单重算的不同"]
        d2 = check_devcheck2(doc, cfg)
        if d2 == ["pending"]:
            print(("一致；" if not problems else "") + "devcheck2 待生成" + ("" if not problems else "；" + "；".join(problems)))
            return 0 if not problems else 1
        problems += d2
        print("一致（devcheck2 自洽：与 devtune800 按句、按图都不重叠）" if not problems else "；".join(problems))
        return 0 if not problems else 1
    except SplitsError as e:
        print(f"失败：{e}", file=sys.stderr)
        return 2


__all__ = [
    "BUILD_COMMAND",
    "DEVCHECK2",
    "DEVTUNE",
    "REFCOCO_TASKS",
    "STATUS_BUILT",
    "STATUS_PENDING",
    "SplitsDoc",
    "SplitsError",
    "W2_FIXTURE_ROOT",
    "W2_SPLITS_PATH",
    "build_skeleton",
    "build_splits",
    "build_with_data",
    "check_devcheck2",
    "devtune800_ids",
    "load_splits",
    "load_splits_file",
    "needs_frozen",
    "pending_devcheck2",
    "select_devcheck2",
    "split_names",
    "splits_path",
    "splits_sha256",
    "tiny_paths",
    "tiny_splits",
    "w2_items",
]


if __name__ == "__main__":
    sys.exit(main())
