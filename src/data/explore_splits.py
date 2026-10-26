"""探索用的两组划分：密封的 devcheck3 与探索用的 explore_val（DEV-0067；D126-4、D126-5、D128）。清单在 configs/explore_splits.json。

用途：
- devcheck3（**密封**）：探索出来的方法进论文主结果只有这一条通道——在它上面与 W2-c 配对判定（scripts/gate_explore.py，门名 G-EX，
  11-02），最多 2 个候选。判定之前，任何人不看任何系统在它上面的结果；入库与写进预注册的只有 id 与 sha256。
- explore_val：探索期间随便用；与 devcheck3 按图不相交。
configs/w2_splits.json 不动（它的 sha256 是 W2 冻结配置身份的一部分）；这两组另放一份清单，条目与 W2 的 test 划分同一种格式，
src/w2/data.w2_items 直接取样本（scripts/run_w2.py --splits-file configs/explore_splits.json）。

构造（规则写死在本模块；两组同一个总体、同样的排除）：
1. **总体**：E1 的三个 val 评测总体 refcoco_unc_val、refcoco+_unc_val、refcocog_umd_val，即 load_refcoco(根, 数据集, 'val',
   subset='eval')（加载器已剔除 E1 的开发子集与 D20 探针图）。条数与 sha256 必须等于 E1 评测清单（configs/e1_eval_subsets.json）的
   populations。样本 id 带数据集前缀才唯一（Sample.id，如 refcoco+_unc/val/123；同 W2-c 训练池的约定）：哈希排序与组的指纹都用它；
   划分里的 ids 是单数据集内的 sent_id（与 W2 的 test 划分同一种条目）。
2. **按图排除**（COCO 图像 id，Sample.image_keys 的 coco:<id>；refcoco 与 refcoco+ 共用图，所以跨数据集生效）：W1、W2 用过的全部划分
   所在的图，即 W2 清单里 refseg 的全部划分——devtune800（= W1 的 devtune + devcheck）、devcheck2、5 个 test 子集（W1、W2 的 test
   与之相同）。取法与 W2-c 训练数据排除 W2 评测图的同一个函数（src/w2/w2c_data.eval_images：清单的 id 与 sha256 都核对）。
   **test 子集也在内**，与任务书"即 devtune800 与 devcheck2 所在的图"的字面不同（上报待裁）：真实数据里 refcoco 的 val 与 refcocog
   的 test、refcocog 的 val 与 refcoco 的 testA/testB 共用一些图（RUN-0067 的 W2-c 数据清单：devtune800 有 17 张图、devcheck2 有 14 张
   在 test_refcocog_umd_test 里），只排除 devtune800 与 devcheck2，"与 5 个 test 划分按图不相交"的核对必然不过。
3. **按句排除**：E1 评测清单的 subsets 里落在这个总体上的全部句子——直接引用它的子集（poly、grid16、think、extra……），以及经
   population 链间接引用的（如 think8192:refexp_poly → think:refexp_poly → poly:refcoco_unc_val）。D128：E1 的子集是基座零样本的
   读数，按句不重叠就够（按图会把图耗尽）。
4. **抽样**：每层（每个数据集）剩下的句子按 sha256(salt|purpose|样本 id)（src/data/subsets.hash_rank）升序取前 n 句，按 sent_id 的
   自然顺序列出。
   - devcheck3：refcoco 200、refcoco+ 200、refcocog 100（与 test 的 2:2:1 一致）；salt = devcheck3_v1，purpose = devcheck3；
     不够 n 句就报错。
   - explore_val：再按图排除 devcheck3 三层的全部图；400、400、200；salt = explore_val_v1，purpose = explore_val；不够就取全部，
     清单记实数（selected.all_taken）。
   --tiny（tests/fixtures/explore）同一套规则，大小取 GROUPS 的 tiny_n。

清单（键序固定，dump_config 的文本即入库文件）：每个划分的 ids（自然顺序）、image_of、n 与 sha256，构造的每一步（总体、按图排除、
按句排除、抽样）剩下的句数、图数与各自的 sha256；按图排除的全部图（exclusion.images_of_w2_splits.images）与各 W2 划分的图的指纹；
按句排除的 E1 子集与句子的指纹；两个组（devcheck3 = 三个 devcheck3 划分，explore_val 同理）的样本 id、n 与 sha256；
checks：组间、与 W2 各划分、与 E1 子集、与 W2-c 训练池的重叠计数（都必须是 0）。

**密封守卫**：require_unsealed(划分, 门名)。划分属于 devcheck3（组名或任何一层，或条目标了 sealed）而门名不是 G-EX 时拒绝（D126）；
放行时返回门名与时间，运行器把门名记进运行身份（seal_identity；时间记进运行日志，见 scripts/run_w2.py）。devcheck3 只能用冻结的
配置跑（needs_frozen）。**以后的探索运行器取 devcheck3 必须经过这个守卫**（scripts/run_w2.py、scripts/gate_explore.py 都调它）。
生成清单本身只写 id，不跑任何系统，不受守卫限制。

开发机上没有 RefCOCO 的数据：先用 --write-skeleton 写出六个划分待生成（status = pending）的清单；服务器在 CPU 上生成：
    python -m src.data.explore_splits --build --data-root $PROJ/data --train-data-root <W2-c 的训练数据根> --out <路径>/explore_splits.json
（--train-data-root 语法上可省：给了才核对"与 W2-c 训练池按图不相交"并记数，即 src/w2/w2c_data.load_train_pool，与 W2-c 引用同一份
剔除清单 <训练数据根>/dedupe/train_exclusions.json；HPC 上是 $VI/data/train_data（E1 的落地在 $VI/b200/data），B200 上是
/dev/shm/visint/train_data。入库的清单要求给了：tests/test_explore_splits.py 核对 checks.train_pool 是"核对过"。）预计 2–10 分钟
（单进程 CPU：三个 val 总体与 W2 的 7 个划分各读一次 RefCOCO 标注，训练池再读三份 train 标注，约 29 万句）。生成的文件交回后由管理员
放到 configs/explore_splits.json。核对：python -m src.data.explore_splits --check [--splits 路径] [--data-root … [--train-data-root …]]
（不给数据时核对不需要数据的部分；给了就整份重算、逐字节比较，参数要与生成时相同）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data import loaders as L
from src.data.eval_subsets import EVAL_SUBSETS_PATH, POLY_SPLITS, load_eval_subsets
from src.data.subsets import DEV_SUBSETS_PATH, dump_config, hash_rank, population_digest, sort_ids
from src.e1.tasks import FIXTURE_DIR, DataPaths
from src.w2 import data as W2D
from src.w2.config import load_w2_config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPLORE_SPLITS_PATH = REPO_ROOT / "configs" / "explore_splits.json"
EXPLORE_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "explore"  # tests/fixtures/explore/build_fixture.py
VERSION = 1
KIND = "explore_splits"
DEVCHECK3 = "devcheck3"
EXPLORE_VAL = "explore_val"
GATE = "G-EX"  # 唯一能打开 devcheck3 的门名（D126-4、D128）
SEAL_DECISION = "D126-4、D126-5、D128"
POPULATIONS: tuple[tuple[str, str, str], ...] = tuple(t for t in POLY_SPLITS if t[2] == "val")  # E1 的三个 val 评测总体
GROUPS: dict[str, dict[str, Any]] = {
    DEVCHECK3: {"salt": "devcheck3_v1", "purpose": "devcheck3", "n": (200, 200, 100), "tiny_n": (3, 3, 2), "allow_fewer": False,
                "sealed": True, "exclude_images_of_groups": ()},
    EXPLORE_VAL: {"salt": "explore_val_v1", "purpose": "explore_val", "n": (400, 400, 200), "tiny_n": (6, 6, 50), "allow_fewer": True,
                  "sealed": False, "exclude_images_of_groups": (DEVCHECK3,)},
}
BUILD_COMMAND = ("python -m src.data.explore_splits --build --data-root $PROJ/data [--train-data-root <W2-c 的训练数据根>] "
                 "--out <路径>/explore_splits.json")
W2_IMAGES_RULE = ("W1、W2 用过的全部划分所在的 COCO 图：configs/w2_splits.json 里 refseg 的全部划分（devtune800、devcheck2、5 个 test 子集；"
                  "src/w2/w2c_data.eval_images，与 W2-c 训练数据排除 W2 评测图的同一个函数）")
E1_SENTENCES_RULE = ("configs/e1_eval_subsets.json 的 subsets 里落在这个总体上的全部句子：直接引用它的子集，以及经 population 链间接引用的")
TRAIN_POOL_RULE = ("W2-c 与 E3 的训练池（src/w2/w2c_data.load_train_pool：三个数据集 train 的全部句子，训练加载器按图剔除 E3 的剔除清单，"
                   "即 refcoco/eval 等评测集的图；与 W2-c 引用同一份 <训练数据根>/dedupe/train_exclusions.json）")
SAMPLE_ID_HASHED = "样本 id（Sample.id，带数据集前缀，如 refcoco+_unc/val/123）"
STATUS_PENDING, STATUS_BUILT = W2D.STATUS_PENDING, W2D.STATUS_BUILT
SplitsError = W2D.SplitsError


class SealedSplitError(PermissionError):
    """密封的 devcheck3 没有给对门名（D126-4）。"""


# ---------------------------------------------------------------------------
# 名字
# ---------------------------------------------------------------------------


def population_name(dataset: str, split_by: str, split: str) -> str:
    """E1 的总体名（同 src/data/eval_subsets.eval_populations），如 refcoco+_unc_val。"""
    return f"{dataset}_{split_by}_{split}"


def sample_id(dataset: str, split_by: str, split: str, sent_id: str | int) -> str:
    """样本 id（同 src/data/loaders.load_refcoco 的 Sample.id；生成时逐条核对两者相同）。"""
    return f"{dataset}_{split_by}/{split}/{sent_id}"


def split_name(group: str, population: str) -> str:
    return f"{group}_{population}"


def group_split_names(group: str) -> list[str]:
    if group not in GROUPS:
        raise KeyError(f"没有组 {group!r}；可用：{list(GROUPS)}")
    return [split_name(group, population_name(*p)) for p in POPULATIONS]


def group_target_n(group: str, *, tiny: bool = False) -> int:
    return sum(GROUPS[group]["tiny_n" if tiny else "n"])


# ---------------------------------------------------------------------------
# 密封守卫
# ---------------------------------------------------------------------------


def is_sealed(split: str, entry: Mapping[str, Any] | None = None) -> bool:
    """devcheck3 的组名、任何一层（devcheck3_<总体>），或清单条目标了 sealed = true。"""
    return split == DEVCHECK3 or split.startswith(DEVCHECK3 + "_") or bool(entry is not None and entry.get("sealed") is True)


def needs_frozen(split: str, entry: Mapping[str, Any] | None = None) -> bool:
    """devcheck3 只能用冻结的配置跑（同 src/w2/data.needs_frozen 对 devcheck2 与 test 的要求）；explore_val 不要求。"""
    return is_sealed(split, entry)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def require_unsealed(split: str, unseal: str | None, *, entry: Mapping[str, Any] | None = None,
                     now: str | None = None) -> dict[str, Any] | None:
    """密封守卫（模块说明）。不是密封划分：返回 None（unseal 不看）。密封划分：unseal 必须恰是 G-EX，否则 SealedSplitError；
    放行时返回 {gate, decision, unsealed_utc}（门名与时间；运行身份记 seal_identity 的部分）。"""
    if not is_sealed(split, entry):
        return None
    if unseal != GATE:
        got = "没有给" if unseal is None else f"给的是 {unseal!r}"
        raise SealedSplitError(f"划分 {split} 属于密封的 devcheck3（D126-4：判定之前任何人不看任何系统在它上面的结果；D128）。"
                               f"只有判定 {GATE} 能用它：显式给 --unseal-devcheck3 {GATE}（{got}）")
    return {"gate": GATE, "decision": SEAL_DECISION, "unsealed_utc": now or _utc_now()}


def seal_identity(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """放行记录里进运行身份的部分（门名与依据；不含时间，否则续跑与共用的中间结果对不上身份）。"""
    return None if record is None else {"gate": record["gate"], "decision": record["decision"]}


# ---------------------------------------------------------------------------
# 规则（纯函数）
# ---------------------------------------------------------------------------


def _count(ids: Sequence[str], img: Mapping[str, str]) -> dict[str, Any]:
    images = sorted({img[i] for i in ids})
    return {"n": len(ids), "sha256": population_digest(ids), "n_images": len(images), "images_sha256": population_digest(images)}


def e1_subsets_on(eval_doc: Mapping[str, Any], population: str) -> list[str]:
    """E1 评测清单里落在 population 上的子集名（直接引用，或经 population 链间接引用），按清单里的顺序。"""
    subs, pops = eval_doc["subsets"], eval_doc["populations"]

    def root(name: str, chain: tuple[str, ...]) -> str:
        if name in pops:
            return name
        if name not in subs:
            raise SplitsError(f"E1 评测清单：{chain[-1]} 的 population {name!r} 既不是总体也不是子集")
        if name in chain:
            raise SplitsError(f"E1 评测清单：population 链有环 {chain + (name,)}")
        return root(str(subs[name]["population"]), chain + (name,))

    return [k for k, e in subs.items() if root(str(e["population"]), (k,)) == population]


def e1_sentences_on(eval_doc: Mapping[str, Any], population: str) -> dict[str, Any]:
    """按句排除的那一份：子集名与它们的句子（并集，自然顺序）。"""
    names = e1_subsets_on(eval_doc, population)
    ids = sort_ids({str(i) for k in names for i in eval_doc["subsets"][k]["ids"]})
    return {"subsets": names, "n": len(ids), "sha256": population_digest(ids), "ids": ids}


def select_layer(population: Sequence[tuple[str, str]], *, prefix: str, exclude_images: Iterable[str], exclude_sentences: Iterable[str],
                 n: int, salt: str, purpose: str, allow_fewer: bool, what: str = "") -> dict[str, Any]:
    """一层的规则（模块说明第 2–4 步）。population = 总体的 (sent_id, 图像键)；prefix = 样本 id 的前缀（如 refcoco_unc/val/），
    哈希排序用 prefix + sent_id。返回 ids（sent_id，自然顺序）、image_of 与各步的条数、图数与 sha256。"""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n 必须是正整数，收到 {n!r}")
    sids = [str(s) for s, _ in population]
    if len(set(sids)) != len(sids):
        raise SplitsError(f"{what}：总体里的 sent_id 有重复")
    img = {str(s): str(k) for s, k in population}
    bad_images = {str(k) for k in exclude_images}
    after_img = [s for s in sids if img[s] not in bad_images]
    bad_sents = {str(s) for s in exclude_sentences}
    after = [s for s in after_img if s not in bad_sents]
    if len(after) < n and not allow_fewer:
        raise SplitsError(f"{what}：排除之后只剩 {len(after)} 句，不够 {n} 句")
    ranked = hash_rank([prefix + s for s in after], purpose, salt)[:n]
    chosen = sort_ids(r[len(prefix):] for r in ranked)
    pop_images = set(img.values())
    return {
        "ids": chosen,
        "image_of": {s: img[s] for s in chosen},
        "steps": {
            "population": _count(sids, img),
            "exclude_images": {"n_images_listed": len(bad_images), "n_images_hit": len(pop_images & bad_images),
                               "n_sentences_removed": len(sids) - len(after_img), "after": _count(after_img, img)},
            "exclude_sentences": {"n_sentences_listed": len(bad_sents), "n_in_population": len(bad_sents & set(sids)),
                                  "n_sentences_removed": len(after_img) - len(after), "after": _count(after, img)},
            "selected": {"salt": salt, "purpose": purpose, "hashed": SAMPLE_ID_HASHED, "n_requested": n, "all_taken": len(after) <= n,
                         **_count(chosen, img)},
        },
    }


def _layer_rule(group: str, population: str, n: int) -> str:
    g = GROUPS[group]
    more = "、devcheck3 三层" if g["exclude_images_of_groups"] else ""
    tail = "（不够就取全部）" if g["allow_fewer"] else "（不够就报错）"
    return (f"load_refcoco 的 {population} eval 总体（E1 的评测总体），按图排除 W2 清单里 refseg 的全部划分{more}所在的图，按句排除 E1 "
            f"落在这个总体上的评测子集，剩下的按 sha256({g['salt']}|{g['purpose']}|样本 id) 升序取前 {n} 句{tail}")


def _entry(group: str, pop: tuple[str, str, str], n: int, *, sel: Mapping[str, Any] | None,
           construction: Mapping[str, Any] | None) -> dict[str, Any]:
    """划分的条目：与 W2 的 test 划分同一种格式（src/w2/data._refcoco_entry），另记组、密封与构造。"""
    ds, sb, sp = pop
    name = population_name(*pop)
    extra = {"group": group, "sealed": bool(GROUPS[group]["sealed"])}
    if sel is None:
        return {"kind": "refcoco", "tasks": list(W2D.REFCOCO_TASKS), "dataset": ds, "split_by": sb, "split": sp, "unit": "sent_id",
                "rule": _layer_rule(group, name, n), **extra, "status": STATUS_PENDING, "build_command": BUILD_COMMAND,
                "n": None, "sha256": None, "ids": None}
    e = W2D._refcoco_entry(sel["ids"], _layer_rule(group, name, n), dataset=ds, split_by=sb, split=sp, **extra, status=STATUS_BUILT,
                           construction=dict(construction or {}))
    e["image_of"] = {s: sel["image_of"][s] for s in e["ids"]}
    return e


def build_groups(pops: Mapping[str, Sequence[tuple[str, str]]], w2_images: Iterable[str], e1_excl: Mapping[str, Mapping[str, Any]],
                 *, tiny: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """两组六个划分（devcheck3 先、explore_val 后：它要排除 devcheck3 的图）。pops = 总体名 → (sent_id, 图像键)；
    w2_images = 按图排除的 W2 各划分的图；e1_excl = 总体名 → e1_sentences_on 的结果。返回 (splits, groups)。"""
    w2_set = {str(k) for k in w2_images}
    splits: dict[str, Any] = {}
    groups: dict[str, Any] = {}
    group_images: dict[str, set[str]] = {}
    for group, g in GROUPS.items():
        extra_images = set().union(*(group_images[o] for o in g["exclude_images_of_groups"]))
        excl_images = w2_set | extra_images
        ids_prefixed: list[str] = []
        images: set[str] = set()
        for pop, n in zip(POPULATIONS, g["tiny_n" if tiny else "n"]):
            name = population_name(*pop)
            prefix = sample_id(*pop, "")
            sel = select_layer(pops[name], prefix=prefix, exclude_images=excl_images, exclude_sentences=e1_excl[name]["ids"], n=n,
                               salt=g["salt"], purpose=g["purpose"], allow_fewer=g["allow_fewer"], what=split_name(group, name))
            st = sel["steps"]
            ex_img = {"of": ["W2 清单里 refseg 的全部划分（exclusion.images_of_w2_splits）", *g["exclude_images_of_groups"]],
                      "images_sha256": population_digest(sorted(excl_images)), **st["exclude_images"]}
            if g["exclude_images_of_groups"]:
                ex_img["n_images_of_groups"] = len(extra_images)
                ex_img["images_of_groups_sha256"] = population_digest(sorted(extra_images))
            ex_sent = {"of": f"exclusion.sentences_of_e1_subsets.{name}", "sentences_sha256": e1_excl[name]["sha256"],
                       **st["exclude_sentences"]}
            construction = {"population": {"name": name, **st["population"]}, "exclude_images": ex_img, "exclude_sentences": ex_sent,
                            "selected": st["selected"]}
            splits[split_name(group, name)] = _entry(group, pop, n, sel=sel, construction=construction)
            ids_prefixed += [sample_id(*pop, s) for s in sel["ids"]]
            images |= set(sel["image_of"].values())
        group_images[group] = images
        groups[group] = _group_entry(group, ids_prefixed, images, tiny=tiny)
    return splits, groups


def _group_entry(group: str, ids_prefixed: Sequence[str] | None, images: Iterable[str] | None, *, tiny: bool) -> dict[str, Any]:
    g = GROUPS[group]
    head = {"splits": group_split_names(group), "sealed": bool(g["sealed"]), "gate": GATE if g["sealed"] else None,
            "unit": SAMPLE_ID_HASHED, "n_target": group_target_n(group, tiny=tiny)}
    if ids_prefixed is None:
        return {**head, "status": STATUS_PENDING, "n": None, "sha256": None, "n_images": None, "images_sha256": None, "ids": None}
    ids = sort_ids(ids_prefixed)
    imgs = sorted(set(images or ()))
    return {**head, "status": STATUS_BUILT, "n": len(ids), "sha256": population_digest(ids), "n_images": len(imgs),
            "images_sha256": population_digest(imgs), "ids": ids}


def overlap_checks(splits: Mapping[str, Any], w2_split_images: Mapping[str, Iterable[str]],
                   e1_excl: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """组间（按图、按样本 id）、与 W2 各划分（按图）、与 E1 子集（按句）的重叠计数；都应是 0。"""
    g_img: dict[str, set[str]] = {g: set() for g in GROUPS}
    g_ids: dict[str, set[str]] = {g: set() for g in GROUPS}
    e1_hits = {g: 0 for g in GROUPS}
    for e in splits.values():
        g = e["group"]
        g_img[g] |= set(e["image_of"].values())
        g_ids[g] |= {sample_id(e["dataset"], e["split_by"], e["split"], s) for s in e["ids"]}
        e1_hits[g] += len(set(e["ids"]) & set(e1_excl[population_name(e["dataset"], e["split_by"], e["split"])]["ids"]))
    return {"devcheck3_vs_explore_val": {"n_images_shared": len(g_img[DEVCHECK3] & g_img[EXPLORE_VAL]),
                                         "n_samples_shared": len(g_ids[DEVCHECK3] & g_ids[EXPLORE_VAL])},
            "vs_w2_splits_by_image": {g: {name: len(g_img[g] & set(imgs)) for name, imgs in w2_split_images.items()} for g in GROUPS},
            "vs_e1_subsets_by_sentence": e1_hits}


def _nonzero(checks: Mapping[str, Any]) -> list[str]:
    out = []
    for k, v in checks["devcheck3_vs_explore_val"].items():
        if v:
            out.append(f"devcheck3 与 explore_val 的 {k} = {v}")
    for g, per in checks["vs_w2_splits_by_image"].items():
        out += [f"{g} 与 W2 的 {name} 共用 {v} 张图" for name, v in per.items() if v]
    out += [f"{g} 与 E1 评测子集共用 {v} 句" for g, v in checks["vs_e1_subsets_by_sentence"].items() if v]
    tp = checks.get("train_pool") or {}
    for g, v in (tp.get("overlap") or {}).items():
        if v.get("n_images"):
            out.append(f"{g} 与 W2-c 训练池共用 {v['n_images']} 张图")
    return out


# ---------------------------------------------------------------------------
# 整份清单
# ---------------------------------------------------------------------------


METHOD = ("总体 = E1 的三个 val 评测总体（load_refcoco(…, 'val', subset='eval')）；按图排除 W2 清单里 refseg 的全部划分（devtune800、"
          "devcheck2、5 个 test 子集）所在的 COCO 图；按句排除 E1 评测清单里落在这个总体上的全部子集的句子；每层剩下的按 "
          "sha256(salt|purpose|样本 id) 升序（src/data/subsets.hash_rank）取前 n 句。devcheck3 先取（200/200/100，不够报错）；explore_val "
          "再按图排除 devcheck3 的全部图（400/400/200，不够取全部）。各划分按 sent_id 自然顺序列出；组按样本 id 自然顺序；sha256 = 自然顺序、"
          "换行连接后的 sha256（src/data/subsets.population_digest）。规则见 src/data/explore_splits.py 的模块说明")
SEAL = {"group": DEVCHECK3, "gate": GATE, "decision": SEAL_DECISION,
        "rule": (f"判定 {GATE} 之前，任何人不看任何系统在 devcheck3 上的结果；运行器取 devcheck3 必须经 "
                 f"src/data/explore_splits.require_unsealed（--unseal-devcheck3 {GATE}），只能用冻结的配置跑；入库与预注册只有 id 与 sha256")}


def _sources(*, tiny: bool, eval_subsets: Path, dev_subsets: Path) -> dict[str, Any]:
    w2 = {"path": "tiny（src/w2/data.tiny_splits，tests/fixtures 上现算）"} if tiny else W2D._file(W2D.W2_SPLITS_PATH)
    return {"w2_splits": w2, "e1_eval_subsets": W2D._file(eval_subsets), "e1_dev_subsets": W2D._file(dev_subsets)}


def _e1_section(eval_doc: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for pop in POPULATIONS:
        name = population_name(*pop)
        x = e1_sentences_on(eval_doc, name)
        out[name] = {"rule": E1_SENTENCES_RULE, "subsets": x["subsets"], "n": x["n"], "sha256": x["sha256"]}
    return out


def assemble(*, tiny: bool, sources: Mapping[str, Any], eval_doc: Mapping[str, Any], w2_images: Mapping[str, Any] | None,
             populations: Mapping[str, Any] | None, splits: Mapping[str, Any] | None, groups: Mapping[str, Any] | None,
             checks: Mapping[str, Any] | None) -> dict[str, Any]:
    """整份清单（键序固定）。w2_images 等为 None 时写待生成。"""
    if splits is None:
        splits = {split_name(g, population_name(*p)): _entry(g, p, n, sel=None, construction=None)
                  for g, spec in GROUPS.items() for p, n in zip(POPULATIONS, spec["tiny_n" if tiny else "n"])}
        groups = {g: _group_entry(g, None, None, tiny=tiny) for g in GROUPS}
    w2_sec = ({"rule": W2_IMAGES_RULE, "status": STATUS_PENDING} if w2_images is None else
              {"rule": W2_IMAGES_RULE, "splits": {k: {kk: v[kk] for kk in ("n_sentences", "sentences_sha256", "n_images", "images_sha256")}
                                                   for k, v in w2_images["splits"].items()},
               "n_images": w2_images["n_images"], "images_sha256": w2_images["images_sha256"], "images": list(w2_images["images"])})
    return {"version": VERSION, "kind": KIND, "method": METHOD, "seal": dict(SEAL), "tiny": tiny, "build_command": BUILD_COMMAND,
            "sources": dict(sources), "exclusion": {"images_of_w2_splits": w2_sec, "sentences_of_e1_subsets": _e1_section(eval_doc)},
            "populations": None if populations is None else dict(populations), "splits": dict(splits), "groups": dict(groups or {}),
            "checks": None if checks is None else dict(checks)}


def splits_sha256(doc: Mapping[str, Any]) -> str:
    return W2D.splits_sha256(doc)


# ---------------------------------------------------------------------------
# 生成（要数据）
# ---------------------------------------------------------------------------


def tiny_paths() -> DataPaths:
    """--tiny 的数据：tests/fixtures/explore（W2 fixture 加 16 张 val 图，见 build_fixture.py）；开发子集用 tests/fixtures/e1 的。"""
    return DataPaths(root=EXPLORE_FIXTURE_DIR / "root", dev_subsets=FIXTURE_DIR / "dev_subsets.json",
                     eval_subsets=EXPLORE_FIXTURE_DIR / "eval_subsets.json", nyu_pairs=FIXTURE_DIR / "nyu_pairs.json")


def load_population(paths: DataPaths, pop: tuple[str, str, str], eval_doc: Mapping[str, Any], *,
                    eval_path: Path) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """一个总体：(sent_id, 图像键) 与它的记录（条数与 sha256 核对 E1 评测清单）。"""
    ds, sb, sp = pop
    name = population_name(*pop)
    res = L.load_refcoco(paths.root, ds, sp, subset="eval", dev_config=paths.dev_subsets, strict=paths.strict)
    pairs = []
    for s in res.records:
        sid = str(s.fields["sent_id"])
        if len(s.image_keys) != 1 or not str(s.image_keys[0]).startswith("coco:"):
            raise SplitsError(f"{s.id}：RefCOCO 应恰有一个 coco: 图像键，收到 {s.image_keys}")
        if s.id != sample_id(ds, sb, sp, sid):
            raise SplitsError(f"样本 id 的约定变了：加载器给 {s.id!r}，sample_id 给 {sample_id(ds, sb, sp, sid)!r}")
        pairs.append((sid, str(s.image_keys[0])))
    sids = [p[0] for p in pairs]
    want = eval_doc["populations"].get(name)
    got = {"n": len(sids), "sha256": population_digest(sids)}
    if want is None or (got["n"], got["sha256"]) != (int(want["n"]), want["sha256"]):
        raise SplitsError(f"总体 {name} {got} 与 {eval_path} 记的 {None if want is None else (want['n'], want['sha256'])} 不同")
    img = dict(pairs)
    rec = {"dataset": ds, "split_by": sb, "split": sp, "provenance": f"load_refcoco({ds!r}, {sp!r}, subset='eval')",
           **_count(sids, img), "check": {"status": "一致", "against": f"{W2D._rel(eval_path)} 的 populations.{name}"}}
    return pairs, rec


def train_pool_check(train_root: str | Path, eval_paths: DataPaths, group_images: Mapping[str, Iterable[str]], *,
                     tiny: bool) -> dict[str, Any]:
    """与 W2-c 训练池按图不相交（有训练数据时）：训练池的图与各组的图的重叠，记数；重叠不为 0 就报错。"""
    from src.w2.w2c_data import W2CPaths, load_train_pool  # 按需 import（W2-c 的模块较重）

    pool, steps = load_train_pool(W2CPaths(train_root=Path(train_root), eval=eval_paths), real_model=not tiny)
    by_image: dict[str, int] = {}
    for s in pool:
        k = str(s.image_keys[0])
        by_image[k] = by_image.get(k, 0) + 1
    ex = steps["e3_exclusions"]
    out = {"status": "核对过", "rule": TRAIN_POOL_RULE,
           "exclusions": {k: ex.get(k) for k in ("file", "sha256", "schema", "max_hamming", "eval_sets", "required_eval_sets",
                                                  "n_sentences_removed")},
           "pool": {"n_sentences": steps["population"]["n"], "sha256": steps["population"]["sha256"], "n_images": len(by_image)},
           "overlap": {}}
    for g, imgs in group_images.items():
        hit = sorted(set(imgs) & set(by_image))
        out["overlap"][g] = {"n_images": len(hit), "n_pool_sentences": sum(by_image[k] for k in hit)}
    bad = {g: v for g, v in out["overlap"].items() if v["n_images"]}
    if bad:
        raise SplitsError(f"与 W2-c 训练池按图有重叠：{bad}")
    return out


def build_with_data(paths: DataPaths, *, tiny: bool = False, train_root: str | Path | None = None) -> dict[str, Any]:
    """在数据上生成整份清单。真运行时三个总体的条数与 sha256 必须等于 E1 评测清单记的那一份（--tiny 对 fixture 的清单）。"""
    from src.w2.w2c_data import eval_images  # 按需 import（W2-c 的模块较重）

    cfg = load_w2_config()
    w2_doc = W2D.tiny_splits(cfg) if tiny else W2D.load_splits(W2D.W2_SPLITS_PATH)
    eval_path = Path(paths.eval_subsets)
    eval_doc = load_eval_subsets(eval_path)
    pops, pop_recs = {}, {}
    for pop in POPULATIONS:
        name = population_name(*pop)
        pops[name], pop_recs[name] = load_population(paths, pop, eval_doc, eval_path=eval_path)
    w2_img = eval_images(cfg, w2_doc, paths)
    e1_excl = {population_name(*p): e1_sentences_on(eval_doc, population_name(*p)) for p in POPULATIONS}
    splits, groups = build_groups(pops, w2_img["images"], e1_excl, tiny=tiny)
    checks = overlap_checks(splits, {k: v["images"] for k, v in w2_img["splits"].items()}, e1_excl)
    group_images = {g: {img for e in splits.values() if e["group"] == g for img in e["image_of"].values()} for g in GROUPS}
    checks["train_pool"] = (train_pool_check(train_root, paths, group_images, tiny=tiny) if train_root is not None else
                            {"status": "未核对（生成时没给 --train-data-root）", "rule": TRAIN_POOL_RULE})
    bad = _nonzero(checks)
    if bad:
        raise SplitsError("重叠核对不过：" + "；".join(bad))
    return assemble(tiny=tiny, sources=_sources(tiny=tiny, eval_subsets=eval_path, dev_subsets=Path(paths.dev_subsets)),
                    eval_doc=eval_doc, w2_images=w2_img, populations=pop_recs, splits=splits, groups=groups, checks=checks)


def build_skeleton() -> dict[str, Any]:
    """不要数据：六个划分待生成，按句排除的 E1 子集（入库的清单就能算）照写。"""
    return assemble(tiny=False, sources=_sources(tiny=False, eval_subsets=EVAL_SUBSETS_PATH, dev_subsets=DEV_SUBSETS_PATH),
                    eval_doc=load_eval_subsets(EVAL_SUBSETS_PATH), w2_images=None, populations=None, splits=None, groups=None,
                    checks=None)


def tiny_splits() -> W2D.SplitsDoc:
    """--tiny：fixture 上按同一套规则现算（scripts/run_w2.py --tiny --splits-file configs/explore_splits.json 用它）。"""
    doc = W2D.SplitsDoc(build_with_data(tiny_paths(), tiny=True))
    doc.path = EXPLORE_SPLITS_PATH
    return doc


# ---------------------------------------------------------------------------
# 核对（不要数据）与取用
# ---------------------------------------------------------------------------


def status(doc: Mapping[str, Any]) -> str:
    """六个划分都待生成 → pending；都已生成 → built；混杂 → 报错。"""
    st = {e.get("status") for e in doc["splits"].values()}
    if st == {STATUS_PENDING}:
        return STATUS_PENDING
    if st == {STATUS_BUILT}:
        return STATUS_BUILT
    raise SplitsError(f"清单里各划分的状态不一致：{sorted(map(str, st))}")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_doc(doc: Mapping[str, Any], *, w2_doc: Mapping[str, Any] | None = None,
              eval_doc: Mapping[str, Any] | None = None) -> list[str]:
    """已生成的清单的核对（不要数据）。返回问题列表（空 = 通过）；待生成时返回 ["pending"]。
    w2_doc、eval_doc 默认读入库的 configs/w2_splits.json 与 configs/e1_eval_subsets.json（--tiny 的清单在 fixture 上现算）。"""
    if doc.get("kind") != KIND or doc.get("version") != VERSION:
        return [f"不是探索划分的清单（kind {doc.get('kind')!r}，version {doc.get('version')!r}）"]
    if set(doc["splits"]) != {split_name(g, population_name(*p)) for g in GROUPS for p in POPULATIONS} or set(doc["groups"]) != set(GROUPS):
        return ["划分或组的名字不对"]
    if status(doc) == STATUS_PENDING:
        return ["pending"]
    tiny = bool(doc.get("tiny"))
    bad: list[str] = []
    if eval_doc is None:
        eval_path = tiny_paths().eval_subsets if tiny else EVAL_SUBSETS_PATH
        if not tiny and doc["sources"]["e1_eval_subsets"]["sha256"] != _file_sha(eval_path):
            bad.append("sources.e1_eval_subsets 的 sha256 与入库的 configs/e1_eval_subsets.json 不同")
        eval_doc = load_eval_subsets(eval_path)
    if w2_doc is None:
        if tiny:
            w2_doc = W2D.tiny_splits(load_w2_config())
        else:
            if doc["sources"]["w2_splits"]["sha256"] != _file_sha(W2D.W2_SPLITS_PATH):
                bad.append("sources.w2_splits 的 sha256 与入库的 configs/w2_splits.json 不同")
            w2_doc = W2D.load_splits(W2D.W2_SPLITS_PATH)
    # 按句排除的 E1 子集：现算与记的相同
    e1 = {population_name(*p): e1_sentences_on(eval_doc, population_name(*p)) for p in POPULATIONS}
    rec_e1 = doc["exclusion"]["sentences_of_e1_subsets"]
    for name, x in e1.items():
        r = rec_e1.get(name) or {}
        if (r.get("subsets"), r.get("n"), r.get("sha256")) != (x["subsets"], x["n"], x["sha256"]):
            bad.append(f"exclusion.sentences_of_e1_subsets.{name} 与按 E1 评测清单现算的不同")
    # 按图排除的 W2 各划分：记的图与它记的条数、sha256 相符；devtune800、devcheck2 的图（W2 清单里记着）现算相符且都在里面
    w2_sec = doc["exclusion"]["images_of_w2_splits"]
    excl = set(w2_sec["images"])
    if len(excl) != w2_sec["n_images"] or population_digest(sorted(excl)) != w2_sec["images_sha256"]:
        bad.append("exclusion.images_of_w2_splits 的图与它记的条数或 sha256 不符")
    d2 = w2_doc["splits"][W2D.DEVCHECK2]
    if d2.get("status") != STATUS_BUILT:
        bad.append("W2 清单的 devcheck2 还没生成")
    else:
        known = {W2D.DEVTUNE: sorted(set(d2["construction"]["exclude"]["images"])), W2D.DEVCHECK2: sorted(set(d2["image_of"].values()))}
        for name, imgs in known.items():
            r = w2_sec["splits"].get(name) or {}
            if (r.get("n_images"), r.get("images_sha256")) != (len(imgs), population_digest(imgs)) or not set(imgs) <= excl:
                bad.append(f"exclusion.images_of_w2_splits 里 {name} 的图与 W2 清单记的不同")
    want_w2 = sorted(k for k, v in w2_doc["splits"].items() if "refseg" in v["tasks"])
    if sorted(w2_sec["splits"]) != want_w2:
        bad.append(f"exclusion.images_of_w2_splits 的划分 {sorted(w2_sec['splits'])} 不是 W2 清单里 refseg 的全部划分 {want_w2}")
    for name, r in w2_sec["splits"].items():
        e = w2_doc["splits"].get(name) or {}
        if (r["n_sentences"], r["sentences_sha256"]) != (e.get("n"), e.get("sha256")):
            bad.append(f"exclusion.images_of_w2_splits 里 {name} 的句子与 W2 清单的不同")
    w2_refcoco_val_ids = {str(i) for k in (W2D.DEVTUNE, W2D.DEVCHECK2) for i in (w2_doc["splits"][k].get("ids") or [])}
    # 各划分
    group_images: dict[str, set[str]] = {g: set() for g in GROUPS}
    group_ids: dict[str, list[str]] = {g: [] for g in GROUPS}
    for name, e in doc["splits"].items():
        g = e["group"]
        spec = GROUPS[g]
        k = [population_name(*p) for p in POPULATIONS].index(population_name(e["dataset"], e["split_by"], e["split"]))
        n_req = spec["tiny_n" if tiny else "n"][k]
        ids = [str(i) for i in e["ids"]]
        c = e["construction"]
        if e["sealed"] is not bool(spec["sealed"]) or name != split_name(g, c["population"]["name"]):
            bad.append(f"{name}：组、密封或总体名不对")
        if len(ids) != e["n"] or population_digest(ids) != e["sha256"] or ids != sort_ids(ids) or len(set(ids)) != len(ids):
            bad.append(f"{name}：条数、sha256 或顺序不对")
        sel = c["selected"]
        after = c["exclude_sentences"]["after"]
        if (sel["n"], sel["sha256"], sel["n_requested"], sel["salt"], sel["purpose"]) != (len(ids), e["sha256"], n_req, spec["salt"],
                                                                                          spec["purpose"]):
            bad.append(f"{name}：construction.selected 与 ids 或规则不符")
        if len(ids) < n_req and not (spec["allow_fewer"] and sel["all_taken"] and len(ids) == after["n"]):
            bad.append(f"{name}：{len(ids)} 句，少于 {n_req} 句")
        if bool(sel["all_taken"]) != (after["n"] <= n_req):
            bad.append(f"{name}：all_taken 与剩下的句数不符")
        pop_n = c["population"]["n"]
        if (pop_n - c["exclude_images"]["n_sentences_removed"] != c["exclude_images"]["after"]["n"]
                or c["exclude_images"]["after"]["n"] - c["exclude_sentences"]["n_sentences_removed"] != after["n"]):
            bad.append(f"{name}：construction 各步的条数对不上")
        img = {str(s): str(v) for s, v in (e.get("image_of") or {}).items()}
        if set(img) != set(ids):
            bad.append(f"{name}：image_of 与 ids 对不上")
            continue
        chosen_images = set(img.values())
        if (sel["n_images"], sel["images_sha256"]) != (len(chosen_images), population_digest(sorted(chosen_images))):
            bad.append(f"{name}：construction.selected 的图与 image_of 不符")
        on_excl = sorted(s for s in ids if img[s] in excl)
        if on_excl:
            bad.append(f"{name}：{len(on_excl)} 句在按图排除的图上，例如 {on_excl[:3]}")
        hit_e1 = sorted(set(ids) & set(e1[c["population"]["name"]]["ids"]))
        if hit_e1:
            bad.append(f"{name}：{len(hit_e1)} 句在 E1 的评测子集里，例如 {hit_e1[:3]}")
        if e["dataset"] == "refcoco" and set(ids) & w2_refcoco_val_ids:
            bad.append(f"{name}：与 W2 的 devtune800 / devcheck2 有相同的句子")
        group_images[g] |= chosen_images
        group_ids[g] += [sample_id(e["dataset"], e["split_by"], e["split"], s) for s in ids]
    if group_images[DEVCHECK3] & group_images[EXPLORE_VAL]:
        bad.append(f"devcheck3 与 explore_val 共用 {len(group_images[DEVCHECK3] & group_images[EXPLORE_VAL])} 张图")
    for name, e in doc["splits"].items():  # explore_val 记的"按图排除 devcheck3 的图"就是 devcheck3 的图
        if e["group"] == EXPLORE_VAL:
            x = e["construction"]["exclude_images"]
            dc3 = sorted(group_images[DEVCHECK3])
            if (x.get("n_images_of_groups"), x.get("images_of_groups_sha256")) != (len(dc3), population_digest(dc3)):
                bad.append(f"{name}：construction.exclude_images 记的 devcheck3 的图与 devcheck3 的不同")
            if x["images_sha256"] != population_digest(sorted(excl | group_images[DEVCHECK3])):
                bad.append(f"{name}：construction.exclude_images 的图不是 W2 各划分的图并 devcheck3 的图")
        elif e["construction"]["exclude_images"]["images_sha256"] != w2_sec["images_sha256"]:
            bad.append(f"{name}：construction.exclude_images 的图不是 W2 各划分的图")
    # 组
    for g, ge in doc["groups"].items():
        ids = sort_ids(group_ids[g])
        if ge["splits"] != group_split_names(g) or ge["sealed"] is not bool(GROUPS[g]["sealed"]):
            bad.append(f"组 {g}：划分名或密封不对")
        if list(ge["ids"]) != ids or ge["n"] != len(ids) or ge["sha256"] != population_digest(ids):
            bad.append(f"组 {g}：样本 id、条数或 sha256 与各划分不符")
        if (ge["n_images"], ge["images_sha256"]) != (len(group_images[g]), population_digest(sorted(group_images[g]))):
            bad.append(f"组 {g}：图的条数或 sha256 与各划分不符")
    if set(doc["groups"][DEVCHECK3]["ids"]) & set(doc["groups"][EXPLORE_VAL]["ids"]):
        bad.append("devcheck3 与 explore_val 有相同的样本")
    if group_target_n(DEVCHECK3, tiny=tiny) != doc["groups"][DEVCHECK3]["n"]:
        bad.append(f"devcheck3 不是 {group_target_n(DEVCHECK3, tiny=tiny)} 句")
    # 生成时记的重叠计数
    checks = doc.get("checks") or {}
    try:
        nz = _nonzero(checks)
    except (KeyError, TypeError, AttributeError):
        nz = ["checks 的格式不对"]
    bad += [f"checks：{x}" for x in nz]
    return bad


def load_explore_splits(path: str | Path = EXPLORE_SPLITS_PATH) -> W2D.SplitsDoc:
    return W2D.load_splits_file(path)


def group_ids(doc: Mapping[str, Any], group: str) -> list[str]:
    """组的样本 id（自然顺序）；清单待生成、或组记的与各划分不符时报错。"""
    if status(doc) != STATUS_BUILT:
        raise SplitsError(f"清单还没生成（status = pending）：先在服务器上 {BUILD_COMMAND}")
    ge = doc["groups"][group]
    ids = sort_ids(sample_id(e["dataset"], e["split_by"], e["split"], s) for nm in ge["splits"] for e in [doc["splits"][nm]]
                   for s in e["ids"])
    if ids != list(ge["ids"]) or population_digest(ids) != ge["sha256"] or len(ids) != ge["n"]:
        raise SplitsError(f"组 {group} 记的样本 id 与各划分不符")
    return ids


def split_of_sample(doc: Mapping[str, Any], group: str) -> dict[str, str]:
    """组里每个样本 id → 它所在的划分名。"""
    group_ids(doc, group)
    out = {}
    for nm in doc["groups"][group]["splits"]:
        e = doc["splits"][nm]
        for s in e["ids"]:
            out[sample_id(e["dataset"], e["split_by"], e["split"], s)] = nm
    return out


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def _summary(doc: Mapping[str, Any]) -> list[str]:
    out = []
    for name, e in doc["splits"].items():
        c = e["construction"]
        out.append(f"{name}：总体 {c['population']['n']} 句 / {c['population']['n_images']} 张图 → 按图排除后 "
                   f"{c['exclude_images']['after']['n']} / {c['exclude_images']['after']['n_images']} → 按句排除后 "
                   f"{c['exclude_sentences']['after']['n']} / {c['exclude_sentences']['after']['n_images']} → 取 {e['n']} 句"
                   f"{'（全部）' if c['selected']['all_taken'] else ''}（{e['sha256']}）")
    for g, ge in doc["groups"].items():
        out.append(f"组 {g}：{ge['n']} 句、{ge['n_images']} 张图（{ge['sha256']}）")
    tp = doc["checks"]["train_pool"]
    out.append(f"W2-c 训练池：{tp['status']}" + (f"（{tp['pool']['n_sentences']} 句、{tp['pool']['n_images']} 张图；重叠 "
                                               f"{ {g: v['n_images'] for g, v in tp['overlap'].items()} }）" if tp.get("pool") else ""))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成或核对探索用的划分清单（configs/explore_splits.json：devcheck3 与 explore_val）")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true", help="在数据上生成整份清单（要 --data-root 与 --out；--tiny 用 fixture）")
    g.add_argument("--write-skeleton", action="store_true", help="不要数据：写 configs/explore_splits.json，六个划分待生成")
    g.add_argument("--check", action="store_true", help="核对清单（给 --data-root 或 --tiny 时整份重算比较）")
    ap.add_argument("--data-root", help="E1 的落地目录（服务器上 $PROJ/data）")
    ap.add_argument("--train-data-root", help="可选：W2-c 的训练数据根；给了才核对与 W2-c 训练池按图不相交并记数")
    ap.add_argument("--out", help="--build 的输出文件")
    ap.add_argument("--splits", default=str(EXPLORE_SPLITS_PATH), help="--check 的清单（默认 configs/explore_splits.json）")
    ap.add_argument("--tiny", action="store_true", help="用 tests/fixtures/explore 的小数据（--build、--check）")
    a = ap.parse_args(argv)
    paths = tiny_paths() if a.tiny else (DataPaths(root=Path(a.data_root)) if a.data_root else None)
    try:
        if a.build:
            if paths is None or not a.out:
                ap.error("--build 要 --data-root（或 --tiny）与 --out")
            doc = build_with_data(paths, tiny=a.tiny, train_root=a.train_data_root)
            text = dump_config(doc)
            out = Path(a.out)
            if out.exists() and out.read_text(encoding="utf-8") != text:
                print(f"{out} 已存在且内容不同，不覆盖", file=sys.stderr)
                return 2
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"写出 {out}（sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}）")
            print("\n".join(_summary(doc)))
            return 0
        if a.write_skeleton:
            text = dump_config(build_skeleton())
            if EXPLORE_SPLITS_PATH.exists() and status(load_explore_splits(EXPLORE_SPLITS_PATH)) == STATUS_BUILT:
                print(f"{EXPLORE_SPLITS_PATH} 已生成，不用骨架覆盖", file=sys.stderr)
                return 2
            EXPLORE_SPLITS_PATH.write_text(text, encoding="utf-8")
            print(f"写出 {EXPLORE_SPLITS_PATH}（六个划分待生成；sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}）")
            return 0
        doc = load_explore_splits(a.splits)
        text = Path(a.splits).read_text(encoding="utf-8")
        if paths is not None:
            same = dump_config(build_with_data(paths, tiny=a.tiny, train_root=a.train_data_root)) == text
            print("一致（整份重算）" if same else f"{a.splits} 与重算的结果不同")
            return 0 if same else 1
        problems = check_doc(doc)
        if problems == ["pending"]:
            same = dump_config(build_skeleton()) == text
            print("一致；六个划分待生成" if same else f"{a.splits} 待生成，但与按入库的清单重算的骨架不同")
            return 0 if same else 1
        print("一致（不要数据的部分：组间按图与按样本不相交，与 W2 各划分按图、与 E1 评测子集按句不相交，各步的数自洽）" if not problems
              else "；".join(problems))
        return 0 if not problems else 1
    except SplitsError as e:
        print(f"失败：{e}", file=sys.stderr)
        return 2


__all__ = [
    "BUILD_COMMAND",
    "DEVCHECK3",
    "EXPLORE_SPLITS_PATH",
    "EXPLORE_VAL",
    "GATE",
    "GROUPS",
    "KIND",
    "POPULATIONS",
    "SealedSplitError",
    "SplitsError",
    "build_groups",
    "build_skeleton",
    "build_with_data",
    "check_doc",
    "e1_sentences_on",
    "e1_subsets_on",
    "group_ids",
    "group_split_names",
    "group_target_n",
    "is_sealed",
    "load_explore_splits",
    "needs_frozen",
    "overlap_checks",
    "population_name",
    "require_unsealed",
    "sample_id",
    "seal_identity",
    "select_layer",
    "split_name",
    "split_of_sample",
    "status",
    "tiny_paths",
    "tiny_splits",
]


if __name__ == "__main__":
    sys.exit(main())
