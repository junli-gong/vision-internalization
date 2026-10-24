#!/usr/bin/env python
"""管理员用：W2 在 devtune800 上调参（W2 预注册 §5 与"补记"）的三个子命令，由 RUN-0066 的 tune_w2.sh 依次调用。

  rank   --oracle <oracle_pick 的 predictions.jsonl> --base <w2 配置> --out ranking.json
         按 best_index（真值，并列取编号小的）给 2..K 号生成器按"作为最优候选的次数"从多到少排序，并列时编号小的在前。
  arm    --base <w2 配置> --out <臂的配置> --name <臂名> [--layout grid|separate] [--thinking on|off]
         [--ranking ranking.json --k K] [--propose box|box_points]
         由底配置写一个臂：只改给出的项；--k 时保留 1 号与排名前 K−1 的生成器（编号按原清单顺序）；--propose box_points 时，
         把保留的生成器里排名最低的非 1 号换成 grabcut_points（缺省参数），K 不变。臂的配置 frozen = false。
  select --w1f <w1f 的 predictions.jsonl> --arm 名=<w2a 的 predictions.jsonl> …（按"简单"的顺序给，越前越简单）
         [--tol 0.005] --out choice.json [--md choice.md]
         每个臂的 cIoU 与"减 W1-F"的配对差（精确的 Σinter/Σunion），取差最大的臂；与最大者相差不到 tol 的，取最简单的。
         另报：随机、最优候选（若给 --random/--oracle）、挑中最优、挑 1 号、"都不对"、解析失败的比例。
  freeze --arm-config <选中的臂的配置> --decision D<nnn> [--final final.json]
         冻结（W2 预注册 §5"冻结"）：把选中的臂写成新的 configs/w2.json（内容与臂相同，只改 name、status，frozen = true），
         并把它的 sha256 填进 configs/w2c_train.json 的 w2_config.sha256。给 --final 时核对臂的配置 sha256 等于 final.json 记的。
只读运行的输出，不跑模型（freeze 只写两个配置文件）。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _jsonl(p: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def _names(gens: list[dict]) -> list[str]:
    from src.w2.candidates import resolve_generator

    return [resolve_generator(g, i)["name"] for i, g in enumerate(gens)]


def cmd_rank(a) -> int:
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    gens = base["candidates"]["generators"]
    names = _names(gens)
    recs = _jsonl(a.oracle)
    if any(r.get("setting") != "oracle_pick" for r in recs):
        raise SystemExit("--oracle 必须是 oracle_pick 的记录")
    with_c = [r for r in recs if r.get("candidates")]
    for r in with_c:
        got = [c["name"] for c in r["candidates"]]
        if got != names:
            raise SystemExit(f"{r['id']}：记录里的候选名 {got} 与底配置的 {names} 不同")
    cnt = Counter(r["best_index"] for r in with_c)
    order = sorted(range(2, len(gens) + 1), key=lambda i: (-cnt.get(i, 0), i))
    doc = {"rule": "W2 预注册 §5 补记：按 best_index（真值，并列取编号小的）计数，2..K 号从多到少，并列编号小的在前；1 号永远保留",
           "oracle": str(a.oracle), "n_with_candidates": len(with_c),
           "counts": {f"{i}:{names[i - 1]}": cnt.get(i, 0) for i in range(1, len(gens) + 1)},
           "ranking": [{"index": i, "name": names[i - 1], "best_count": cnt.get(i, 0)} for i in order]}
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(doc["ranking"], ensure_ascii=False))
    return 0


def cmd_arm(a) -> int:
    base_text = Path(a.base).read_text(encoding="utf-8")
    d = json.loads(base_text)
    if d.get("frozen") is True:
        raise SystemExit("底配置已冻结，不再生成调参的臂")
    changes = []
    if a.layout:
        d["render"]["layout"] = a.layout
        changes.append(f"layout={a.layout}")
    if a.thinking:
        d["select"]["thinking"] = a.thinking == "on"
        changes.append(f"thinking={a.thinking}")
    gens = d["candidates"]["generators"]
    rk = None
    if a.k is not None or a.propose == "box_points":
        if not a.ranking:
            raise SystemExit("--k 或 --propose box_points 要给 --ranking")
        rk = json.loads(Path(a.ranking).read_text(encoding="utf-8"))
    if a.k is not None:
        names = _names(gens)
        for e in rk["ranking"]:  # --k 只用于完整的原清单：编号与名字逐个核对
            if e["index"] > len(gens) or names[e["index"] - 1] != e["name"]:
                raise SystemExit(f"ranking 与底配置的清单对不上（--k 要以完整的原清单为底）：{e}")
        if not 2 <= a.k <= len(gens):
            raise SystemExit(f"--k 必须在 2..{len(gens)}")
        keep = sorted([1] + [e["index"] for e in rk["ranking"]][: a.k - 1])
        gens = [gens[i - 1] for i in keep]
        changes.append(f"k={a.k} keep={keep}")
    if a.propose:
        d["propose"]["mode"] = a.propose
        changes.append(f"propose={a.propose}")
        if a.propose == "box_points":
            names_now = _names(gens)
            ranked_names = [e["name"] for e in rk["ranking"]]  # 2..K 号，从多到少
            kept = [n for n in ranked_names if n in names_now[1:]]
            if not kept:
                raise SystemExit("没有可替换的非 1 号生成器")
            drop_name = kept[-1]  # 保留的非 1 号里排名最低的
            pos = names_now.index(drop_name)
            gens = list(gens)
            gens[pos] = {"type": "grabcut_points"}
            changes.append(f"swap {drop_name}->grabcut_points")
    d["candidates"]["generators"] = gens
    base_sha = hashlib.sha256(base_text.encode("utf-8")).hexdigest()
    d["name"] = f"W2 调参臂 {a.name}（W2 预注册 §5；scripts/mgr/w2_tune.py arm）"
    d["status"] = f"调参用，只能跑 devtune800。底配置 {a.base}（sha256 {base_sha}）；改动：{'; '.join(changes) or '无'}。"
    d["frozen"] = False
    Path(a.out).write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{a.name}: {'; '.join(changes)} -> {a.out}")
    return 0


def _sums(recs: list[dict]) -> dict[str, tuple[int, int]]:
    return {r["id"]: (int(r["score"]["inter"]), int(r["score"]["union"])) for r in recs}


def _ciou(s: dict[str, tuple[int, int]], ids) -> Fraction:
    I = sum(s[i][0] for i in ids)
    U = sum(s[i][1] for i in ids)
    return Fraction(I, U) if U else Fraction(0)


def _rates(recs: list[dict]) -> dict:
    wc = [r for r in recs if r.get("candidates")]
    n = len(wc)
    f = lambda k: (sum(1 for r in wc if k(r)) / n) if n else None  # noqa: E731
    prop_fail = (sum(1 for r in recs if (r.get("proposal") or {}).get("failure")) / len(recs)) if recs else None
    return {"n_with_candidates": n, "propose_failure": prop_fail, "final_is_best": f(lambda r: r.get("final_is_best")),
            "final_is_1": f(lambda r: r.get("final_index") == 1), "none": f(lambda r: r.get("fallback") == "none"),
            "parse_failure": f(lambda r: r.get("fallback") == "parse_failure")}


def cmd_select(a) -> int:
    w1f = _jsonl(a.w1f)
    sw = _sums(w1f)
    ids = sorted(sw)
    arms = []
    for spec in a.arm:
        name, path = spec.split("=", 1)
        recs = _jsonl(path)
        s = _sums(recs)
        if sorted(s) != ids:
            raise SystemExit(f"臂 {name} 的 id 集合与 W1-F 不同")
        diff = _ciou(s, ids) - _ciou(sw, ids)
        arms.append({"name": name, "predictions": path, "ciou": float(_ciou(s, ids)), "diff_vs_w1f": float(diff),
                     "diff_exact": f"{diff.numerator}/{diff.denominator}", "_diff": diff, **_rates(recs)})
    # 仪器有效性（W2 预注册 §5 补记 2，D121）：挑选的解析失败率或出框的失败率超过上限的臂不参选（照样列出）
    for x in arms:
        bad = []
        if a.max_parse_failure is not None and (x["parse_failure"] or 0) > a.max_parse_failure:
            bad.append(f"parse_failure {x['parse_failure']:.3f} > {a.max_parse_failure}")
        if a.max_propose_failure is not None and (x["propose_failure"] or 0) > a.max_propose_failure:
            bad.append(f"propose_failure {x['propose_failure']:.3f} > {a.max_propose_failure}")
        x["eligible"] = not bad
        x["ineligible_reason"] = "; ".join(bad) or None
    pool = [x for x in arms if x["eligible"]]
    if not pool:
        raise SystemExit("没有一个臂通过仪器有效性过滤")
    best = max(pool, key=lambda x: x["_diff"])
    tol = Fraction(str(a.tol))
    chosen = next(x for x in pool if x["_diff"] >= best["_diff"] - tol)  # 按给出的顺序，越前越简单
    for x in arms:
        x.pop("_diff")
    refs = {}
    for lab, p in (("w1f", a.w1f), ("random_pick", a.random), ("oracle_pick", a.oracle)):
        if p:
            refs[lab] = float(_ciou(_sums(_jsonl(p)), ids))
    doc = {"rule": f"W2 预注册 §5：取减 W1-F 的配对 cIoU 差最大的臂；与最大者相差不到 {a.tol} 的取最简单的（给出的顺序，越前越简单）"
                   + (f"；只在仪器有效的臂里选（挑选解析失败 ≤ {a.max_parse_failure}、出框失败 ≤ {a.max_propose_failure}；§5 补记 2，D121）"
                      if (a.max_parse_failure is not None or a.max_propose_failure is not None) else ""),
           "n": len(ids), "references": refs, "arms": arms, "max_arm": best["name"], "chosen": chosen["name"]}
    Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.md:
        lines = [f"# W2 调参：{a.step or ''} 选臂", "", f"- 规则：{doc['rule']}", f"- n = {len(ids)}；参照 cIoU：" +
                 "，".join(f"{k} {v:.4f}" for k, v in refs.items()), "",
                 "| 臂（简单→复杂） | cIoU | 减 W1-F | 挑中最优 | 挑 1 号 | 都不对 | 解析失败 | 出框失败 | 参选 |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        fm = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
        for x in arms:
            lines.append(f"| {x['name']} | {x['ciou']:.4f} | {x['diff_vs_w1f']:+.4f} | {fm(x['final_is_best'])} | {fm(x['final_is_1'])} | "
                         f"{fm(x['none'])} | {fm(x['parse_failure'])} | {fm(x['propose_failure'])} | "
                         f"{'是' if x['eligible'] else '否（' + x['ineligible_reason'] + '）'} |")
        lines += ["", f"**差最大**：{best['name']}；**选中**：{chosen['name']}"]
        Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(chosen["name"])
    return 0


def cmd_freeze(a) -> int:
    w2 = ROOT / "configs" / "w2.json"
    cur = json.loads(w2.read_text(encoding="utf-8"))
    if cur.get("frozen") is True:
        raise SystemExit("configs/w2.json 已冻结")
    arm_bytes = Path(a.arm_config).read_bytes()
    arm_sha = hashlib.sha256(arm_bytes).hexdigest()
    if a.final:
        fin = json.loads(Path(a.final).read_text(encoding="utf-8"))
        if fin.get("sha256") != arm_sha:
            raise SystemExit(f"臂的配置 sha256 {arm_sha} 与 final.json 记的 {fin.get('sha256')} 不同")
    d = json.loads(arm_bytes)
    d["name"] = cur["name"]
    d["status"] = (f"已冻结（{a.decision}）：W2 预注册 §5 在 devtune800 上选中的臂 {Path(a.arm_config).stem}"
                   f"（{a.arm_config}，sha256 {arm_sha}），内容与该臂相同，只改 name、status、frozen。之后不再改动；devcheck2 与 test 只用这一份。")
    d["frozen"] = True
    text = json.dumps(d, ensure_ascii=False, indent=1) + "\n"
    w2.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    tc = ROOT / "configs" / "w2c_train.json"
    t = json.loads(tc.read_text(encoding="utf-8"))
    t["w2_config"]["sha256"] = sha
    tc.write_text(json.dumps(t, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"冻结：configs/w2.json sha256 {sha}；configs/w2c_train.json 的 w2_config.sha256 已填")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rank")
    r.add_argument("--oracle", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--out", required=True)
    m = sub.add_parser("arm")
    m.add_argument("--base", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--name", required=True)
    m.add_argument("--layout", choices=["grid", "separate"])
    m.add_argument("--thinking", choices=["on", "off"])
    m.add_argument("--ranking")
    m.add_argument("--k", type=int)
    m.add_argument("--propose", choices=["box", "box_points"])
    s = sub.add_parser("select")
    s.add_argument("--w1f", required=True)
    s.add_argument("--arm", action="append", required=True)
    s.add_argument("--random")
    s.add_argument("--oracle")
    s.add_argument("--tol", type=float, default=0.005)
    s.add_argument("--out", required=True)
    s.add_argument("--md")
    s.add_argument("--step")
    s.add_argument("--max-parse-failure", type=float, help="仪器有效性：挑选解析失败率的上限（§5 补记 2）")
    s.add_argument("--max-propose-failure", type=float, help="仪器有效性：出框失败率的上限（§5 补记 2）")
    f = sub.add_parser("freeze")
    f.add_argument("--arm-config", required=True)
    f.add_argument("--decision", required=True)
    f.add_argument("--final")
    a = p.parse_args(argv)
    return {"rank": cmd_rank, "arm": cmd_arm, "select": cmd_select, "freeze": cmd_freeze}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
