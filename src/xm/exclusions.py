"""X-M 训练的去重清单（DEV-0079；D165 指令第二节"去重，最要紧"）：configs/xm_train_exclusions.json。只读真值与已有清单，不跑模型。

COCO train2017 里要剔除的图（exclude_train2017_image_ids，按 COCO image_id；COCO 2014 与 2017 共用 id），三类：
1. **refcoco_eval**：RefCOCO（unc）、RefCOCO+（unc）的 val、testA、testB 与 RefCOCOg（umd）的 val、test 里出现过的全部图，取自 E1 落地的
   <refcoco 根>/refcoco/<数据集>_<splitBy>.json（refs 只含评测划分；src/data/land/refcoco）。真运行时每个划分的图数核对
   src/data/land/refcoco.SPEC 的 P0 计数，三者的并集核对 D11-1 钉住的 n_eval_union（src/data/land/refcoco_train.SPEC，6,549 张）。
   另核对这些子集的图都在并集里（subsets_covered）：W2 的 devtune800、devcheck2、5 个 test 子集（configs/w2_splits.json 的句子经 refs 换成图；
   devcheck2 另与清单的 image_of 逐句相同；各划分的图数与 sha256 与 configs/explore_splits.json 的 exclusion.images_of_w2_splits 记的相同），
   探索的 devcheck3 与 explore_val 的 6 个划分（configs/explore_splits.json 的句子与 image_of，两者逐句相同）。
2. **reasonseg_val**：ReasonSeg val 的图不来自 COCO，按两种方式核对与 train2017 的重合：文件名相同；图像字节的 sha256 相同（train2017 的
   118,287 张逐张算 sha256）。重合的 train2017 图进剔除清单（预期 0）。
3. **coco_eval_splits**：xm_tune、xm_decision、xm_final 与 lvis_sub 的图（configs/xm_splits.json）：都在 val2017，与 train2017 按 image_id
   不相交；另按图像字节的 sha256 核对（val2017 的这几张图与 train2017 的全部图），重合的 train2017 图进剔除清单（预期 0）。
每一类记条数、清单的 sha256（src/data/subsets.population_digest）与和 train2017 的交集大小；整份清单记划分文件 xm_splits.json 的 sha256，
load_pairs("train") 核对它与本次读的划分文件相同。

--tiny：RefCOCO 用 tests/fixtures/explore/root，W2 与探索的划分用各自的 tiny_splits（同一套规则在 fixture 上现算），不核对 P0 计数。
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.data.subsets import population_digest, sort_ids
from src.xm import data as D

KIND = "xm_train_exclusions"
VERSION = 1
FRAMEWORK = "src/xm/exclusions.py（DEV-0079；D165）"
REFCOCO_EVAL: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("refcoco", "unc", ("val", "testA", "testB")),
    ("refcoco+", "unc", ("val", "testA", "testB")),
    ("refcocog", "umd", ("val", "test")),
)
CATEGORIES: tuple[str, ...] = ("refcoco_eval", "reasonseg_val", "coco_eval_splits")
TINY_REFCOCO_ROOT = D.REPO_ROOT / "tests" / "fixtures" / "explore" / "root"
RULE = ("COCO train2017 里剔除：RefCOCO / RefCOCO+ / RefCOCOg 全部评测划分（unc 的 val、testA、testB；umd 的 val、test）的图；"
        "与 ReasonSeg val 按文件名或图像 sha256 相同的图；与 xm_tune、xm_decision、xm_final、lvis_sub 按 image_id 或图像 sha256 相同的图。"
        "规则见 src/xm/exclusions.py 的模块说明")


def _digest(ids: Iterable[Any]) -> str:
    return population_digest(str(i) for i in ids)


def _key_id(k: str) -> int:
    if not str(k).startswith("coco:"):
        raise D.XMDataError(f"认不出的图像键 {k!r}（要 coco:<id>）")
    return int(str(k).split(":", 1)[1])


# ---------------------------------------------------------------------------
# RefCOCO 系列
# ---------------------------------------------------------------------------


def _refcoco_doc(root: Path, ds: str) -> dict[str, Any]:
    from src.data.loaders import REFCOCO_FILES

    p = root / "refcoco" / REFCOCO_FILES[ds]
    if not p.is_file():
        raise D.XMDataError(f"{p} 不在（E1 的 RefCOCO 落地；HPC 上 $VI/b200/data）")
    return D.read_json(p)


def refcoco_eval_images(refcoco_root: str | Path, *, tiny: bool) -> tuple[dict[str, list[int]], dict[str, dict[str, int]], dict[str, Any]]:
    """(划分名 → 图（升序）, 数据集 → {sent_id → image_id}, 来源文件的 sha256)。划分名形如 refcoco_unc/val。"""
    from src.data.land import refcoco as RL
    from src.data.loaders import REFCOCO_FILES

    root = Path(refcoco_root)
    out: dict[str, list[int]] = {}
    sents: dict[str, dict[str, int]] = {}
    files: dict[str, Any] = {}
    for ds, sb, splits in REFCOCO_EVAL:
        doc = _refcoco_doc(root, ds)
        if doc.get("splitBy") != sb or list(doc.get("eval_splits") or []) != list(splits):
            raise D.XMDataError(f"{ds}：splitBy {doc.get('splitBy')!r}、eval_splits {doc.get('eval_splits')} 不是 {sb}、{list(splits)}")
        files[ds] = {"file": f"refcoco/{REFCOCO_FILES[ds]}", "sha256": D.file_sha256(root / "refcoco" / REFCOCO_FILES[ds])}
        sents[ds] = {}
        for sp in splits:
            ids = sorted({int(r["image_id"]) for r in doc["refs"] if r["split"] == sp})
            if not tiny:
                want = RL.SPEC["datasets"][ds]["expect"][sp][2]
                if len(ids) != want:
                    raise D.XMDataError(f"{ds}/{sp}：{len(ids)} 张图，P0 的计数是 {want}（src/data/land/refcoco.SPEC）")
            out[f"{ds}_{sb}/{sp}"] = ids
        for r in doc["refs"]:
            for s in r["sentences"]:
                sents[ds][str(s["sent_id"])] = int(r["image_id"])
    return out, sents, files


def _w2_and_explore_docs(*, tiny: bool) -> tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
    from src.data import explore_splits as XS
    from src.w2 import data as W2D
    from src.w2.config import load_w2_config

    if tiny:
        return W2D.tiny_splits(load_w2_config()), XS.tiny_splits(), {"w2_splits": "tiny（src/w2/data.tiny_splits）",
                                                                    "explore_splits": "tiny（src/data/explore_splits.tiny_splits）"}
    w2 = W2D.load_splits(W2D.W2_SPLITS_PATH)
    ex = XS.load_explore_splits(XS.EXPLORE_SPLITS_PATH)
    src = {"w2_splits": {"path": "configs/w2_splits.json", "sha256": D.file_sha256(W2D.W2_SPLITS_PATH)},
           "explore_splits": {"path": "configs/explore_splits.json", "sha256": D.file_sha256(XS.EXPLORE_SPLITS_PATH)}}
    return w2, ex, src


def subsets_covered(w2_doc: Mapping[str, Any], explore_doc: Mapping[str, Any], sents: Mapping[str, Mapping[str, int]],
                    union: set[int]) -> dict[str, Any]:
    """W2 与探索用过的 RefCOCO 子集的图：每个子集的句数、图数、图的 sha256、不在并集里的图数（必须是 0），并与清单记的 image_of /
    images_sha256 核对（不符就报错）。"""
    out: dict[str, Any] = {}
    w2_rec = ((explore_doc.get("exclusion") or {}).get("images_of_w2_splits") or {}).get("splits") or {}
    todo: list[tuple[str, Mapping[str, Any], str]] = []
    for name, e in w2_doc["splits"].items():
        if e.get("kind") == "refcoco":
            todo.append((name, e, "w2_splits"))
    for name, e in explore_doc["splits"].items():
        todo.append((name, e, "explore_splits"))
    for name, e, where in todo:
        ds = str(e["dataset"])
        ids = [str(i) for i in (e.get("ids") or [])]
        if not ids:
            raise D.XMDataError(f"{where} 的 {name} 没有句子（还没生成？）")
        miss = [i for i in ids if i not in sents[ds]]
        if miss:
            raise D.XMDataError(f"{where} 的 {name}：{len(miss)} 句不在 {ds} 的评测 refs 里，例如 {miss[:3]}")
        img_of = {i: sents[ds][i] for i in ids}
        rec_of = e.get("image_of")
        if rec_of is not None and {str(k): _key_id(v) for k, v in rec_of.items()} != img_of:
            raise D.XMDataError(f"{where} 的 {name}：句子所在的图与清单记的 image_of 不同")
        imgs = sorted(set(img_of.values()))
        keys = sort_ids(f"coco:{i}" for i in imgs)
        if name in w2_rec and (w2_rec[name].get("n_images"), w2_rec[name].get("images_sha256")) != (len(keys), population_digest(keys)):
            raise D.XMDataError(f"W2 的 {name}：现算的图与 configs/explore_splits.json 的 exclusion.images_of_w2_splits 记的不同")
        out[name] = {"from": where, "dataset": ds, "n_sentences": len(ids), "n_images": len(imgs), "images_sha256": _digest(imgs),
                     "n_outside_eval_union": len(set(imgs) - union), "image_of_checked": rec_of is not None,
                     "explore_record_checked": name in w2_rec}
    bad = {k: v["n_outside_eval_union"] for k, v in out.items() if v["n_outside_eval_union"]}
    if bad:
        raise D.XMDataError(f"这些子集有图不在 RefCOCO 评测划分的并集里：{bad}")
    return out


# ---------------------------------------------------------------------------
# 图像字节的 sha256
# ---------------------------------------------------------------------------


def hash_files(root: str | Path, rels: Sequence[str], *, workers: int = 8) -> list[str]:
    root = Path(root)

    def one(r: str) -> str:
        return D.file_sha256(root / r, chunk=1 << 20)

    if int(workers) <= 1:
        return [one(r) for r in rels]
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(one, rels))


# ---------------------------------------------------------------------------
# 整份清单
# ---------------------------------------------------------------------------


def build_exclusions(data_root: str | Path, *, splits_path: str | Path, refcoco_root: str | Path | None = None, tiny: bool = False,
                     workers: int = 8, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """在数据上生成 configs/xm_train_exclusions.json 的内容（模块说明）。"""
    from src.data.land.refcoco_train import SPEC as TRAIN_SPEC

    log = log or (lambda m: None)
    root = Path(data_root)
    rc_root = Path(refcoco_root) if refcoco_root is not None else (TINY_REFCOCO_ROOT if tiny else None)
    if rc_root is None:
        raise D.XMDataError("要给 E1 的 RefCOCO 落地根（--refcoco-root，HPC 上 $VI/b200/data）")
    sp = Path(splits_path)
    splits = D.load_splits_doc(sp)
    tr_rec = D.file_record(root, D.COCO_ANN["train2017"])
    tr = D.read_json(root / D.COCO_ANN["train2017"])
    tr_images = sorted(tr["images"], key=lambda im: int(im["id"]))
    tr_ids = {int(im["id"]) for im in tr_images}
    log(f"[X-M 去重] instances_train2017：{len(tr_ids)} 张图")

    # 1. RefCOCO 系列
    per_split, sents, rc_files = refcoco_eval_images(rc_root, tiny=tiny)
    union = set().union(*map(set, per_split.values()))
    if not tiny and len(union) != TRAIN_SPEC["expect_d11_1"]["n_eval_union"]:
        raise D.XMDataError(f"RefCOCO 评测划分的并集 {len(union)} 张，D11-1 钉住的是 {TRAIN_SPEC['expect_d11_1']['n_eval_union']} 张")
    w2_doc, ex_doc, cfg_src = _w2_and_explore_docs(tiny=tiny)
    covered = subsets_covered(w2_doc, ex_doc, sents, union)
    rc_in = sorted(union & tr_ids)
    log(f"[X-M 去重] RefCOCO 评测划分：{len(union)} 张图，其中 {len(rc_in)} 张在 train2017")
    refcoco_sec = {"rule": "三者全部评测划分的 refs 的 image_id（E1 落地的 refcoco JSON）", "files": rc_files,
                   "splits": {k: {"n_images": len(v), "sha256": _digest(v)} for k, v in per_split.items()},
                   "n_images": len(union), "sha256": _digest(sorted(union)),
                   "pin": None if tiny else {"n_eval_union": TRAIN_SPEC["expect_d11_1"]["n_eval_union"], "source": "src/data/land/refcoco_train.SPEC（D11-1）"},
                   "n_in_train2017": len(rc_in), "n_not_in_train2017": len(union) - len(rc_in), "subsets_covered": covered,
                   "image_ids": sorted(union)}

    # 2 与 3 都要 train2017 的图像 sha256
    rels = [f"{D.COCO_IMAGE_DIR['train2017']}/{im['file_name']}" for im in tr_images]
    log(f"[X-M 去重] 计算 train2017 的 {len(rels)} 张图的 sha256（{workers} 个线程）…")
    tr_sha = hash_files(root, rels, workers=workers)
    by_sha: dict[str, list[int]] = {}
    for im, h in zip(tr_images, tr_sha):
        by_sha.setdefault(h, []).append(int(im["id"]))
    names = {str(im["file_name"]): int(im["id"]) for im in tr_images}
    hashes_rec = {"n_images": len(tr_sha), "sha256_of_list": hashlib.sha256("\n".join(f"{int(im['id'])} {h}" for im, h in zip(tr_images, tr_sha))
                                                                            .encode("utf-8")).hexdigest(),
                  "n_duplicate_bytes_groups": sum(1 for v in by_sha.values() if len(v) > 1)}

    # 2. ReasonSeg val
    rs = splits["splits"][D.REASONSEG_VAL]
    stems = sorted(rs["files"])
    rs_now = D.reasonseg_file_records(root, stems)
    if rs_now != rs["files"]:
        raise D.XMDataError("ReasonSeg val 的文件与 xm_splits.json 记的 sha256 不同")
    hit_name = sorted({names[f"{s}.jpg"] for s in stems if f"{s}.jpg" in names})
    hit_sha = sorted({i for s in stems for i in by_sha.get(rs_now[s]["jpg"], [])})
    reason_sec = {"rule": "ReasonSeg val 的 .jpg 与 train2017 按文件名、按图像字节的 sha256 两种方式核对", "n_images": len(stems),
                  "sha256": population_digest(stems), "files_sha256": hashlib.sha256(json.dumps(rs_now, sort_keys=True).encode()).hexdigest(),
                  "train2017_overlap": {"by_file_name": {"n": len(hit_name), "train_image_ids": hit_name},
                                        "by_sha256": {"n": len(hit_sha), "train_image_ids": hit_sha}},
                  "n_in_train2017": len(set(hit_name) | set(hit_sha))}
    log(f"[X-M 去重] ReasonSeg val：{len(stems)} 张图；与 train2017 同名 {len(hit_name)}、同 sha256 {len(hit_sha)}")

    # 3. 三个 COCO 划分与 LVIS 子集
    val = {int(im["id"]): im for im in D.read_json(root / D.COCO_ANN["val2017"])["images"]}
    coco_sec: dict[str, Any] = {}
    coco_hits: set[int] = set()
    for s in (*D.COCO_EVAL_SPLITS, D.LVIS_SUB):
        ids = [int(i) for i in splits["splits"][s]["image_ids"]]
        by_id = sorted(set(ids) & tr_ids)
        vh = hash_files(root, [f"{D.COCO_IMAGE_DIR['val2017']}/{val[i]['file_name']}" for i in ids], workers=workers)
        sha_hit = sorted({t for h in vh for t in by_sha.get(h, [])})
        coco_hits |= set(by_id) | set(sha_hit)
        coco_sec[s] = {"n_images": len(ids), "sha256": _digest(ids), "n_in_train2017_by_id": len(by_id), "train_image_ids_by_id": by_id,
                       "n_in_train2017_by_sha256": len(sha_hit), "train_image_ids_by_sha256": sha_hit}
        log(f"[X-M 去重] {s}：{len(ids)} 张图；与 train2017 同 id {len(by_id)}、同 sha256 {len(sha_hit)}")

    cats = {"refcoco_eval": set(rc_in), "reasonseg_val": set(hit_name) | set(hit_sha), "coco_eval_splits": coco_hits}
    excl = sorted(set().union(*cats.values()))
    doc = {
        "version": VERSION, "kind": KIND, "framework": FRAMEWORK, "tiny": bool(tiny), "rule": RULE,
        "sources": {"coco_train2017": tr_rec, "xm_splits": {"file": sp.name, "sha256": D.file_sha256(sp)},
                    "refcoco_root": "tests/fixtures/explore/root" if tiny else "E1 的落地（--refcoco-root）", **cfg_src},
        "train2017_image_hashes": hashes_rec,
        "categories": {"refcoco_eval": refcoco_sec, "reasonseg_val": reason_sec, "coco_eval_splits": {"rule": "按 image_id 与图像 sha256 两种方式",
                                                                                                      "splits": coco_sec, "n_in_train2017": len(coco_hits)}},
        "by_category": {k: {"n_train2017_images": len(v), "sha256": _digest(sorted(v))} for k, v in cats.items()},
        "n": len(excl), "sha256": _digest(excl), "exclude_train2017_image_ids": excl,
    }
    return doc


def check_exclusions_doc(doc: Mapping[str, Any]) -> list[str]:
    if doc.get("kind") != KIND or doc.get("version") != VERSION:
        return [f"不是 X-M 的去重清单（kind {doc.get('kind')!r}，version {doc.get('version')!r}）"]
    bad = []
    ids = [int(i) for i in doc["exclude_train2017_image_ids"]]
    if ids != sorted(set(ids)) or len(ids) != doc["n"] or _digest(ids) != doc["sha256"]:
        bad.append("exclude_train2017_image_ids 的顺序、条数或 sha256 不对")
    if set(doc["by_category"]) != set(CATEGORIES):
        bad.append(f"by_category 的类不对：{sorted(doc['by_category'])}")
    rc = doc["categories"]["refcoco_eval"]
    if len(set(int(i) for i in rc["image_ids"]) - set(ids)) != rc["n_not_in_train2017"]:
        bad.append("refcoco_eval 里在 train2017 的图没有全部进剔除清单")
    if not doc.get("tiny") and (rc.get("pin") or {}).get("n_eval_union") != rc["n_images"]:
        bad.append("refcoco_eval 的图数不等于 D11-1 钉住的 n_eval_union")
    return bad


def load_exclusions(path: str | Path | None = None, *, splits_path: str | Path | None = None) -> dict[str, Any]:
    """读去重清单并核对：格式、清单的 sha256、记的 xm_splits.json 的 sha256 等于本次读的划分文件（splits_path，缺省入库的）。"""
    p = Path(path) if path is not None else D.XM_EXCLUSIONS_PATH
    if not p.is_file():
        raise D.XMDataError(f"{p} 不在：先在服务器上生成（scripts/xm/build_configs.py），管理员审过再入库")
    doc = D.read_json(p)
    bad = check_exclusions_doc(doc)
    if bad:
        raise D.XMDataError(f"{p} 不合格：" + "；".join(bad))
    sp = Path(splits_path) if splits_path is not None else D.XM_SPLITS_PATH
    if doc["sources"]["xm_splits"]["sha256"] != D.file_sha256(sp):
        raise D.XMDataError(f"{p} 是按另一份划分文件生成的（记的 sha256 与 {sp} 不同）")
    return doc


__all__ = ["CATEGORIES", "KIND", "build_exclusions", "check_exclusions_doc", "hash_files", "load_exclusions", "refcoco_eval_images",
           "subsets_covered"]
