"""训练集与评测集之间的去重（硬约束 7；D11-7；DEV-0022 任务书 §2）。

跨机器的做法：训练数据在 A800，评测数据在 B200。两边各自用 export_root 把"图像键 + 感知哈希"导出成一个小 JSON
（每张图一条：[image_key, phash, 相对源目录的路径]），拷到任意一台机器上用 compare 比较。也可以直接读数据根或单个
MANIFEST.json（load_input 自动识别）。

- 集合：每个源按划分分成集合，名字 "<源>/<划分>"。DEV-0022 的源在清单里逐文件记 split；E1 的源没有，按 E1_SET_NAMES
  命名（nyuv2 按 index.json 分成 test 与开发子集 dev_train）。TRAIN_SETS 里的是训练集，其余一律当评测集（保守）。
  另有两个内置的评测集：D20 的探针用图（键 + 感知哈希，src/data/constants）与 COCO val2017 全部 5,000 个 id（只有键，
  P0 存档的 URL 清单，sha256 钉住），用来核对"COCO train2014 对 COCO val2017"而不依赖 E1 只落了 500 张。
- 比较：每个训练集对每个评测集，(1) 同键（image_key 相同，COCO 2014/2017 共用 id）；(2) 感知哈希汉明距离 ≤ 阈值。
  阈值 = src/data/constants.PHASH_MAX_HAMMING；另报 REPORT_THRESHOLDS 各档的计数，看阈值附近的灵敏度。
- 标定：已知重合（不同集合里同键、都有感知哈希的图，例如 CV-Bench 的 COCO 图对 E1 的 COCO 子集、CV-Bench 的 NYU 图
  对 E1 的 NYU test、法向包的 RGB 对 E1 的 NYU test）的汉明距离分布，与随机不同键图对的距离分布，一起写进报告。
- 输出：dedupe_report.json、dedupe_report.md、train_exclusions.json（被剔除的训练样本，按源目录内的相对路径；
  src/data/train_loaders.py 默认读 <训练数据根>/dedupe/train_exclusions.json，自动排除）。
- 剔除清单的格式（EXCLUSIONS_SCHEMA = 2，DEV-0024；D53-3 的小问题 A、B）：
  {schema, created_utc, max_hamming, rule,
   train_sets: 参与比较的训练集名单，
   eval_sets: 参与比较的评测集名单（含内置的两个），
   inputs: 各输入清单的摘要 [{what, host, root, created_utc, git_commit, sha256（输入是文件时为文件的 sha256，数据根为 null）,
            sets_sha256（集合内容的 sha256，与读文件还是读数据根无关）, sets: {集合: 张数}}]，
   sets: {训练集: [被剔除的样本]}}——每个参与比较的训练集都有键（没有被剔除的样本时是空列表），所以"缺键"就是"没比较过"。
  旧格式（schema 1，没有 eval_sets / train_sets / inputs）照样能读（load_exclusions、exclusions_meta 里 legacy = True）；
  训练加载器在 require_dedupe 时照样要求训练集有键，E2 用真模型时拒绝旧格式（src/e2/data.check_dedupe_coverage）。
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.constants import PHASH_MAX_HAMMING, PROBE_IMAGE_KEYS, PROBE_IMAGE_PHASH
from src.data.land.manifest import MANIFEST_NAME, REPO_ROOT
from src.runlog import git_info

EXPORT_KIND = "visint-image-hashes"
EXPORT_SCHEMA = 1
REPORT_SCHEMA = 1
EXCLUSIONS_SCHEMA = 2  # 剔除清单（格式见模块说明）；1 是旧格式（DEV-0022），照样能读
EXCLUSIONS_SCHEMAS_READABLE = (1, 2)
EXCLUSIONS_REL = "dedupe/train_exclusions.json"  # 相对训练数据根
TRAIN_SETS = ("refcoco_train/train", "nyu_depth_train/train", "ade20k/train", "nyu_normals/train")
REPORT_THRESHOLDS = (0, 2, 4, 6, 8, 10, 12)
N_EXAMPLES = 10
N_RANDOM_PAIRS = 200_000
# E1 的源：清单里的文件不带 split，整个源是一个评测集（nyuv2 另按 index.json 分）
E1_SET_NAMES = {
    "refcoco": "refcoco/eval",
    "coco_val2017": "coco_val2017/e1sub500",
    "odinw13": "odinw13/eval",
    "countbench": "countbench/eval",
    "blink_reldepth": "blink_reldepth/val",
    "cvbench": "cvbench/test",
    "embspatial": "embspatial/test",
}
PROBE_SET = "probe/D20"
COCO_VAL2017_SET = "coco_val2017_full/val2017"
COCO_VAL2017_URLS = "experiments/P0/0001_env_probe/raw/p6/A/coco_val2017/coco_val2017_coco_urls.txt"
COCO_VAL2017_URLS_SHA256 = "60267210d97f81450fafcf340317b7c6723fcd8fe340aa0d141bb0ba28bb49c2"
COCO_VAL2017_N = 5000


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def role_of(set_name: str) -> str:
    return "train" if set_name in TRAIN_SETS else "eval"


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def _nyuv2_split_map(src_dir: Path) -> dict[str, str]:
    idx = src_dir / "index.json"
    if not idx.exists():
        return {}
    out = {}
    for r in json.loads(idx.read_text())["images"]:
        out[r["rgb"]] = "test" if r["split"] == "test" else "dev_train"
    return out


def set_name_for(source: str, rel: str, rec: Mapping[str, Any], nyuv2_map: Mapping[str, str] | None = None) -> str:
    """一个图像文件属于哪个集合。"""
    if rec.get("split"):
        return f"{source}/{rec['split']}"
    if source == "nyuv2":
        sp = (nyuv2_map or {}).get(rel)
        return f"nyuv2/{sp}" if sp else "nyuv2/unknown"
    return E1_SET_NAMES.get(source, f"{source}/all")


def export_manifest(manifest_path: str | Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """一个源的清单 → ({集合名: {source, role, items}}, 该源的统计与去重事实)。只导出 kind == image 的文件。"""
    mp = Path(manifest_path)
    man = json.loads(mp.read_text())
    source = man["source"]
    nyu_map = _nyuv2_split_map(mp.parent) if source == "nyuv2" else None
    sets: dict[str, dict[str, Any]] = {}
    for rel, rec in man["files"].items():
        if rec.get("kind") != "image":
            continue
        name = set_name_for(source, rel, rec, nyu_map)
        s = sets.setdefault(name, {"source": source, "role": role_of(name), "items": []})
        s["items"].append([rec.get("image_key") or f"{source}:{rel}", rec.get("phash"), rel])
    for s in sets.values():
        s["items"].sort(key=lambda it: it[2])
    facts = {"stats": man.get("stats", {}), "selection_rule": (man.get("selection") or {}).get("rule"), "dedupe_facts": man.get("dedupe_facts", {}), "n_files": man.get("n_files"), "landed_at_utc": man.get("landed_at_utc"), "git": (man.get("script") or {}).get("git")}
    return sets, facts


def builtin_sets() -> dict[str, dict[str, Any]]:
    """D20 探针用图（键 + 感知哈希）；COCO val2017 全部 id（只有键，从 P0 存档的 URL 清单读，sha256 钉住）。"""
    out = {PROBE_SET: {"source": "constants", "role": "eval", "items": [[k, PROBE_IMAGE_PHASH.get(k), None] for k in PROBE_IMAGE_KEYS]}}
    p = REPO_ROOT / COCO_VAL2017_URLS
    if p.exists():
        data = p.read_bytes()
        if hashlib.sha256(data).hexdigest() != COCO_VAL2017_URLS_SHA256:
            raise ValueError(f"{p} 的 sha256 与钉住的不符")
        ids = [int(Path(u).stem) for u in data.decode().split()]
        if len(ids) != COCO_VAL2017_N or len(set(ids)) != COCO_VAL2017_N:
            raise ValueError(f"{p}：应有 {COCO_VAL2017_N} 个不同的 id")
        out[COCO_VAL2017_SET] = {"source": COCO_VAL2017_URLS, "role": "eval", "items": [[f"coco:{i}", None, None] for i in sorted(ids)]}
    return out


def export_root(root: str | Path) -> dict[str, Any]:
    """数据根下每个 <源>/MANIFEST.json 都导出。"""
    root = Path(root)
    sets: dict[str, dict[str, Any]] = {}
    facts: dict[str, Any] = {}
    for mp in sorted(root.glob(f"*/{MANIFEST_NAME}")):
        s, f = export_manifest(mp)
        sets.update(s)
        facts[mp.parent.name] = f
    return {
        "kind": EXPORT_KIND,
        "schema": EXPORT_SCHEMA,
        "created_utc": utc_now(),
        "host": platform.node(),
        "root": str(root.resolve()),
        "git": git_info(REPO_ROOT),
        "sets": dict(sorted(sets.items())),
        "facts": facts,
    }


def load_input(path: str | Path) -> dict[str, Any]:
    """导出的 JSON、单个 MANIFEST.json 或数据根目录 → 导出格式。另加 input_source = {path, sha256}（只在内存里，给剔除清单的输入
    摘要用；输入是文件时 sha256 是文件本身的，数据根为 None）。"""
    p = Path(path)
    if p.is_dir():
        return {**export_root(p), "input_source": {"path": str(p), "sha256": None}}
    raw = p.read_bytes()
    src = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()}
    doc = json.loads(raw)
    if doc.get("kind") == EXPORT_KIND:
        if doc.get("schema") != EXPORT_SCHEMA:
            raise ValueError(f"{p}：导出清单的 schema {doc.get('schema')} ≠ {EXPORT_SCHEMA}")
        return {**doc, "input_source": src}
    if "files" in doc and "source" in doc:
        s, f = export_manifest(p)
        return {"kind": EXPORT_KIND, "schema": EXPORT_SCHEMA, "created_utc": utc_now(), "host": platform.node(), "root": str(p.parent.parent.resolve()), "sets": s, "facts": {doc["source"]: f}, "input_source": src}
    raise ValueError(f"{p}：既不是导出的哈希清单，也不是 MANIFEST.json")


def sets_sha256(sets: Mapping[str, Mapping[str, Any]]) -> str:
    """集合内容（集合名 → [键, 感知哈希, 路径] 列表）的 sha256，规范 JSON；与读导出文件还是读数据根无关。"""
    canon = {n: [list(it) for it in s["items"]] for n, s in sorted(sets.items())}
    return hashlib.sha256(json.dumps(canon, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def input_summary(what: str, doc: Mapping[str, Any]) -> dict[str, Any]:
    """一份输入清单的摘要（剔除清单与报告的 inputs）：来源、主机、sha256、集合与张数。"""
    sets = doc.get("sets", {})
    return {"what": what, "host": doc.get("host"), "root": doc.get("root"), "created_utc": doc.get("created_utc"),
            "git_commit": (doc.get("git") or {}).get("commit"), "sha256": (doc.get("input_source") or {}).get("sha256"),
            "sets_sha256": sets_sha256(sets), "sets": {n: len(s["items"]) for n, s in sorted(sets.items())}}


def merge_inputs(docs: Sequence[dict[str, Any]], *, with_builtin: bool = True) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """合并多份导出。同名集合按 (键, 路径) 去重合并（两台机器都落了同一个源时）。"""
    sets: dict[str, dict[str, Any]] = {}
    facts: dict[str, Any] = {}
    for d in docs:
        for name, s in d["sets"].items():
            cur = sets.setdefault(name, {"source": s["source"], "role": role_of(name), "items": []})
            seen = {(it[0], it[2]) for it in cur["items"]}
            cur["items"].extend(it for it in s["items"] if (it[0], it[2]) not in seen)
        facts.update(d.get("facts", {}))
    if with_builtin:
        for name, s in builtin_sets().items():
            sets.setdefault(name, s)
    return dict(sorted(sets.items())), facts


# ---------------------------------------------------------------------------
# 比较
# ---------------------------------------------------------------------------


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(axis=-1, dtype=np.uint8)


def _hashes(items: Sequence[Sequence[Any]]) -> tuple[np.ndarray, list[int]]:
    rows = [k for k, it in enumerate(items) if it[1]]
    return np.array([int(items[k][1], 16) for k in rows], dtype=np.uint64), rows


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n,) 与 b (m,) 两组 64 位哈希的两两汉明距离 (n, m)，uint8。"""
    return _popcount(a[:, None] ^ b[None, :]).astype(np.uint8)


def compare(sets: Mapping[str, Mapping[str, Any]], *, max_hamming: int = PHASH_MAX_HAMMING, thresholds: Sequence[int] = REPORT_THRESHOLDS, chunk: int = 512) -> dict[str, Any]:
    """每个训练集对每个评测集：同键与感知哈希近邻。返回 {"pairs": [...], "exclusions": {训练集: [...]}}。"""
    tmax = max(max(thresholds), max_hamming)
    train = {n: s for n, s in sets.items() if s["role"] == "train"}
    evals = {n: s for n, s in sets.items() if s["role"] == "eval"}
    ev_names = list(evals)
    ev_h, ev_rows, ev_set = [], [], []
    for j, n in enumerate(ev_names):
        h, rows = _hashes(evals[n]["items"])
        ev_h.append(h)
        ev_rows.extend(rows)
        ev_set.extend([j] * len(rows))
    ev_all = np.concatenate(ev_h) if ev_h else np.zeros(0, np.uint64)
    ev_set_arr = np.array(ev_set, dtype=np.int64)
    bounds = np.cumsum([0] + [len(h) for h in ev_h])
    pairs: list[dict[str, Any]] = []
    exclusions: dict[str, list[dict[str, Any]]] = {}
    for tn, ts in train.items():
        titems = ts["items"]
        hits: dict[int, list[dict[str, Any]]] = {}
        # (1) 同键
        tkeys: dict[str, list[int]] = {}
        for k, it in enumerate(titems):
            tkeys.setdefault(it[0], []).append(k)
        per_eval: dict[str, dict[str, Any]] = {}
        for en in ev_names:
            ekeys = {it[0] for it in evals[en]["items"]}
            same = sorted(set(tkeys) & ekeys)
            for key in same:
                for k in tkeys[key]:
                    hits.setdefault(k, []).append({"eval_set": en, "by": "image_key", "matched": key})
            per_eval[en] = {"same_key": same, "pair_counts": np.zeros(tmax + 1, np.int64), "item_min": {}}
        # (2) 感知哈希
        th, trows = _hashes(titems)
        for c0 in range(0, len(th), chunk):
            d = hamming_matrix(th[c0 : c0 + chunk], ev_all) if len(ev_all) else np.zeros((len(th[c0 : c0 + chunk]), 0), np.uint8)
            for j, en in enumerate(ev_names):
                lo, hi = bounds[j], bounds[j + 1]
                if hi == lo:
                    continue
                sub = d[:, lo:hi]
                near = sub[sub <= tmax]
                if near.size:
                    per_eval[en]["pair_counts"] += np.bincount(near, minlength=tmax + 1)[: tmax + 1]
                mins = sub.min(axis=1)
                args = sub.argmin(axis=1)
                for r in np.nonzero(mins <= tmax)[0]:
                    k = trows[c0 + int(r)]
                    ei = ev_rows[lo + int(args[r])]
                    per_eval[en]["item_min"][k] = (int(mins[r]), ei)
        for en in ev_names:
            pe = per_eval[en]
            eitems = evals[en]["items"]
            for k, (dist, ei) in pe["item_min"].items():
                if dist <= max_hamming:
                    hits.setdefault(k, []).append({"eval_set": en, "by": "phash", "matched": eitems[ei][0], "matched_path": eitems[ei][2], "hamming": dist})
            cum_pairs = np.cumsum(pe["pair_counts"])
            item_d = np.array([v[0] for v in pe["item_min"].values()], dtype=np.int64)
            ex_sorted = sorted(pe["item_min"].items(), key=lambda kv: (kv[1][0], titems[kv[0]][2]))[:N_EXAMPLES]
            excluded_here = {k for k, hs in hits.items() if any(h["eval_set"] == en for h in hs)}
            pairs.append(
                {
                    "train": tn,
                    "eval": en,
                    "n_train": len(titems),
                    "n_eval": len(eitems),
                    "n_same_key": len(pe["same_key"]),
                    "same_key_examples": pe["same_key"][:N_EXAMPLES],
                    "n_phash_pairs_le": {str(t): int(cum_pairs[t]) for t in thresholds},
                    "n_train_items_phash_le": {str(t): int((item_d <= t).sum()) for t in thresholds},
                    "phash_examples": [{"train_key": titems[k][0], "train_path": titems[k][2], "eval_key": eitems[ei][0], "eval_path": eitems[ei][2], "hamming": dist} for k, (dist, ei) in ex_sorted],
                    "n_train_excluded": len(excluded_here),
                }
            )
        exclusions[tn] = [{"path": titems[k][2], "key": titems[k][0], "reasons": hs} for k, hs in sorted(hits.items(), key=lambda kv: titems[kv[0]][2] or "")]
    return {"pairs": pairs, "exclusions": exclusions, "train_sets": list(train), "eval_sets": ev_names}


def calibrate(sets: Mapping[str, Mapping[str, Any]], *, thresholds: Sequence[int] = REPORT_THRESHOLDS, max_hamming: int = PHASH_MAX_HAMMING, n_random: int = N_RANDOM_PAIRS, seed: int = 0) -> dict[str, Any]:
    """阈值标定：已知重合（不同集合里同键、都有感知哈希）的距离分布；随机不同键图对的距离分布。"""
    by_key: dict[str, list[tuple[str, int]]] = {}
    all_h, all_k = [], []
    for name, s in sets.items():
        for it in s["items"]:
            if it[1]:
                v = int(it[1], 16)
                by_key.setdefault(it[0], []).append((name, v))
                all_h.append(v)
                all_k.append(it[0])
    groups: dict[str, list[int]] = {}
    for key, occ in by_key.items():
        first: dict[str, int] = {}
        for name, v in occ:
            first.setdefault(name, v)
        names = sorted(first)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                groups.setdefault(f"{names[a]} ~ {names[b]}", []).append(bin(first[names[a]] ^ first[names[b]]).count("1"))
    known = [d for ds in groups.values() for d in ds]

    def summary(ds: list[int]) -> dict[str, Any]:
        if not ds:
            return {"n": 0}
        a = np.array(ds)
        return {"n": len(ds), "max": int(a.max()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "frac_le": {str(t): round(float((a <= t).mean()), 6) for t in thresholds}, "hist": {str(int(k)): int(v) for k, v in zip(*np.unique(a, return_counts=True))}}

    rand: dict[str, Any] = {"n": 0}
    if len(all_h) >= 2:
        rng = np.random.default_rng(seed)
        h = np.array(all_h, dtype=np.uint64)
        i = rng.integers(0, len(h), n_random)
        j = rng.integers(0, len(h), n_random)
        keys = np.array(all_k, dtype=object)
        ok = (i != j) & (keys[i] != keys[j])
        d = _popcount(h[i[ok]] ^ h[j[ok]]).astype(np.int64)
        rand = {"n": int(ok.sum()), "seed": seed, "min": int(d.min()) if d.size else None, "p01": float(np.percentile(d, 1)) if d.size else None, "frac_le": {str(t): float((d <= t).mean()) if d.size else None for t in thresholds}}
    k_sum = summary(known)
    verdict = []
    if k_sum["n"]:
        miss = sum(1 for d in known if d > max_hamming)
        verdict.append(f"已知重合 {k_sum['n']} 对，距离 ≤ {max_hamming} 的占 {k_sum['frac_le'].get(str(max_hamming), 'n/a')}（{miss} 对超出阈值）")
    if rand["n"]:
        verdict.append(f"随机不同键图对 {rand['n']} 对，距离 ≤ {max_hamming} 的比例 {rand['frac_le'].get(str(max_hamming))}")
    return {"max_hamming": max_hamming, "known_duplicates": {**k_sum, "by_pair": {g: summary(ds) for g, ds in sorted(groups.items())}}, "distinct_random_pairs": rand, "verdict": "；".join(verdict) or "没有可用于标定的已知重合"}


def d11_facts(facts: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    rt = (facts.get("refcoco_train") or {}).get("stats", {})
    if "d11_1" in rt:
        out["d11_1"] = {**rt["d11_1"], "per_dataset": rt.get("per_dataset")}
    nd = (facts.get("nyu_depth_train") or {}).get("stats", {})
    if "d11_2" in nd:
        out["d11_2"] = nd["d11_2"]
    return out


def build_report(inputs: Sequence[tuple[str, dict[str, Any]]], *, max_hamming: int = PHASH_MAX_HAMMING) -> tuple[dict[str, Any], dict[str, Any]]:
    """inputs = [(来源说明, 导出格式)]。返回 (报告, 训练剔除清单)。"""
    sets, facts = merge_inputs([d for _, d in inputs])
    cmp = compare(sets, max_hamming=max_hamming)
    cal = calibrate(sets, max_hamming=max_hamming)
    created = utc_now()
    summaries = [input_summary(w, d) for w, d in inputs]
    report = {
        "schema": REPORT_SCHEMA,
        "created_utc": created,
        "git": git_info(REPO_ROOT),
        "max_hamming": max_hamming,
        "thresholds": list(REPORT_THRESHOLDS),
        "inputs": summaries,
        "train_sets": cmp["train_sets"],
        "eval_sets": cmp["eval_sets"],
        "sets": {n: {"role": s["role"], "source": s["source"], "n_items": len(s["items"]), "n_with_phash": sum(1 for it in s["items"] if it[1])} for n, s in sets.items()},
        "d11": d11_facts(facts),
        "pairs": cmp["pairs"],
        "calibration": cal,
        "exclusions_summary": {t: len(v) for t, v in cmp["exclusions"].items()},
    }
    excl = {"schema": EXCLUSIONS_SCHEMA, "created_utc": created, "max_hamming": max_hamming,
            "rule": "训练样本与任一评测集同键，或感知哈希汉明距离 ≤ max_hamming，即剔除；eval_sets 是参与比较的全部评测集，"
                    "sets 里每个参与比较的训练集都有键（没有被剔除的样本时是空列表），缺键就是没比较过",
            "train_sets": cmp["train_sets"], "eval_sets": cmp["eval_sets"], "inputs": summaries, "sets": cmp["exclusions"]}
    return report, excl


def render_md(report: Mapping[str, Any]) -> str:
    L = [f"# 去重报告（硬约束 7；D11-7）", "", f"- 生成：{report['created_utc']}，git {((report.get('git') or {}).get('commit') or '')[:10]}", f"- 感知哈希阈值：汉明距离 ≤ {report['max_hamming']}（src/data/constants.PHASH_MAX_HAMMING）", ""]
    L += ["## 输入", ""] + [f"- {i['what']}（host {i.get('host')}，{i.get('created_utc')}）" for i in report["inputs"]] + [""]
    L += ["## 集合", "", "| 集合 | 角色 | 图像数 | 有感知哈希 |", "|---|---|---:|---:|"]
    L += [f"| {n} | {s['role']} | {s['n_items']} | {s['n_with_phash']} |" for n, s in report["sets"].items()] + [""]
    d = report.get("d11", {})
    L += ["## D11-1、D11-2 过滤前后", ""]
    if "d11_1" in d:
        x = d["d11_1"]
        L.append(f"- D11-1（RefCOCO 训练图）：三者 train 图像并集 {x['n_train_union']}，评测划分图像并集 {x['n_eval_union']}，剔除 {x['n_removed']}，保留 {x['n_kept']}；交叉 {x.get('cross')}")
    if "d11_2" in d:
        x = d["d11_2"]
        L.append(f"- D11-2（NYU 深度训练集）：BTS 清单 {x['n_pairs']} 对，按场景去掉 {x['n_removed']}，保留 {x['n_kept']}；去掉的目录 {x.get('removed_dirs')}")
    if not d:
        L.append("- （输入里没有 refcoco_train / nyu_depth_train 的清单）")
    t = report["thresholds"]
    L += ["", "## 每对集合的重合", "", "| 训练集 | 评测集 | 同键 | " + " | ".join(f"≤{x} 的训练图" for x in t) + " | 剔除 |", "|---|---|---:|" + "---:|" * len(t) + "---:|"]
    for p in report["pairs"]:
        L.append(f"| {p['train']} | {p['eval']} | {p['n_same_key']} | " + " | ".join(str(p["n_train_items_phash_le"][str(x)]) for x in t) + f" | {p['n_train_excluded']} |")
    L += ["", "### 例子（每对最多 10 个）", ""]
    for p in report["pairs"]:
        if p["same_key_examples"] or p["phash_examples"]:
            L.append(f"- {p['train']} × {p['eval']}：同键 {p['same_key_examples']}；感知哈希 " + "，".join(f"{e['train_key']} ~ {e['eval_key']}（{e['hamming']}）" for e in p["phash_examples"]))
    c = report["calibration"]
    L += ["", "## 阈值标定", "", f"- {c['verdict']}"]
    for g, s in c["known_duplicates"].get("by_pair", {}).items():
        L.append(f"- 已知重合 {g}：n={s['n']}，最大 {s.get('max')}，中位 {s.get('p50')}，p95 {s.get('p95')}")
    r = c["distinct_random_pairs"]
    if r.get("n"):
        L.append(f"- 随机不同键图对：n={r['n']}，最小 {r['min']}，1% 分位 {r['p01']}，各档比例 {r['frac_le']}")
    L += ["", "## 训练剔除清单", ""] + [f"- {k}：{v} 条" for k, v in report["exclusions_summary"].items()]
    L += [f"- 参与比较的评测集（剔除清单的 eval_sets，{len(report.get('eval_sets') or [])} 个）：{report.get('eval_sets')}"]
    L += ["", "训练加载器（src/data/train_loaders.py）默认读 `<训练数据根>/dedupe/train_exclusions.json`，自动排除这些样本；"
          "训练集在清单里没有键就报错（没比较过）。", ""]
    return "\n".join(L)


def write_outputs(report: Mapping[str, Any], excl: Mapping[str, Any], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"report_json": out / "dedupe_report.json", "report_md": out / "dedupe_report.md", "exclusions": out / Path(EXCLUSIONS_REL).name}
    paths["report_json"].write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    paths["report_md"].write_text(render_md(report))
    paths["exclusions"].write_text(json.dumps(excl, ensure_ascii=False, indent=1) + "\n")
    return paths


def read_exclusions_doc(path: str | Path) -> dict[str, Any]:
    """读剔除清单并核对格式（模块说明）：schema 1（旧格式）或 2；schema 2 必须有 max_hamming、train_sets、eval_sets、inputs，
    且 train_sets 与 sets 的键一一对应。"""
    doc = json.loads(Path(path).read_text())
    schema = doc.get("schema")
    if schema not in EXCLUSIONS_SCHEMAS_READABLE:
        raise ValueError(f"{path}：剔除清单的 schema {schema} 不在 {EXCLUSIONS_SCHEMAS_READABLE} 里")
    if not isinstance(doc.get("sets"), dict):
        raise ValueError(f"{path}：剔除清单没有 sets")
    if schema >= 2:
        missing = [k for k in ("max_hamming", "train_sets", "eval_sets", "inputs") if k not in doc]
        if missing:
            raise ValueError(f"{path}：schema {schema} 的剔除清单缺少 {missing}")
        if not all(isinstance(x, str) for x in doc["eval_sets"]) or not all(isinstance(x, str) for x in doc["train_sets"]):
            raise ValueError(f"{path}：train_sets / eval_sets 必须是集合名的列表")
        if set(doc["train_sets"]) != set(doc["sets"]):
            raise ValueError(f"{path}：train_sets {sorted(doc['train_sets'])} 与 sets 的键 {sorted(doc['sets'])} 不一致")
    return doc


def load_exclusions(path: str | Path) -> dict[str, dict[str, dict[str, Any]]]:
    """训练剔除清单 → {训练集: {相对路径: 记录}}。两种格式都能读（read_exclusions_doc）。"""
    doc = read_exclusions_doc(path)
    return {t: {r["path"]: r for r in rows} for t, rows in doc["sets"].items()}


def exclusions_meta(path: str | Path) -> dict[str, Any]:
    """剔除清单的元数据（不含逐条记录）：schema、legacy（旧格式，没有 eval_sets）、max_hamming、train_sets（旧格式取 sets 的键）、
    eval_sets（旧格式为 None）、inputs（旧格式为 None）、created_utc、文件的 sha256。"""
    p = Path(path)
    doc = read_exclusions_doc(p)
    legacy = doc["schema"] < 2
    return {"schema": doc["schema"], "legacy": legacy, "max_hamming": doc.get("max_hamming"),
            "train_sets": sorted(doc["sets"]) if legacy else list(doc["train_sets"]),
            "eval_sets": None if legacy else list(doc["eval_sets"]), "inputs": None if legacy else doc["inputs"],
            "created_utc": doc.get("created_utc"), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}

