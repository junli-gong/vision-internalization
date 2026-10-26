"""W2-c 在 devcheck2 上的误差拆解（探索日志 X-0）：按提出框的 IoU 分档，各档占总并集与总损失（U − I）的份额、W2-c 与最优候选的 cIoU；
好框（IoU ≥ 0.9）上各生成器的 cIoU 与最优候选 IoU 的分布。只读 RUN-0069 的结果目录。devcheck2 自 G-W2 之后只作开发集（D126-3）。

用法：python -I scripts/mgr/w2c_error_decomp.py
"""
import json, sys
R = __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "../../experiments/W2/0069_w2c_hpcgpu06/0069_eval/qwen3.5-9b/")
def rows(p): return {json.loads(l)["id"]: json.loads(l) for l in open(R + p)}
w2c = rows("refseg/w2c/devcheck2/predictions.jsonl"); prop = rows("refexp_box/propose_w2c/devcheck2/predictions.jsonl")
w1f = rows("refseg/w1f/devcheck2/predictions.jsonl"); pb = rows("refexp_box/baseline_e1/devcheck2/predictions.jsonl")
def stats(ids, rr, key="final"):
    I = U = 0
    for i in ids:
        r = rr[i]
        if key == "final": I += r["score"]["inter"]; U += r["score"]["union"]
        else:
            c = [x for x in (r.get("candidates") or []) if x.get("inter") is not None]
            if c:
                b = max(c, key=lambda x: x["iou"]); I += b["inter"]; U += b["union"]
            else: I += r["score"]["inter"]; U += r["score"]["union"]
    return I, U
ids = sorted(w2c)
Itot, Utot = stats(ids, w2c); Iorc, Uorc = stats(ids, w2c, "oracle")
print(f"n={len(ids)} W2-c cIoU={Itot/Utot:.4f} oracle={Iorc/Uorc:.4f}  W1-F={stats(ids,w1f)[0]/stats(ids,w1f)[1]:.4f}")
for name, P in (("W2-c box", prop), ("base box", pb)):
    acc = sum((P[i]["score"] or {}).get("correct", False) for i in ids) / len(ids); print(f"{name} Acc@0.5={acc:.3f}")
edges = [0, .3, .5, .7, .9, 1.01]
print("box IoU bucket | n | share of union | share of loss(U-I) | W2-c cIoU | oracle cIoU")
for lo, hi in zip(edges, edges[1:]):
    b = [i for i in ids if lo <= ((prop[i]["score"] or {}).get("iou") or 0.0) < hi]
    if not b: continue
    I, U = stats(b, w2c); Io, Uo = stats(b, w2c, "oracle")
    print(f"[{lo:.1f},{hi:.1f}) | {len(b)} | {U/Utot:.3f} | {(U-I)/(Utot-Itot):.3f} | {I/U:.3f} | {Io/Uo:.3f}")
fail = [i for i in ids if prop[i].get("failure") is not None]; print("propose failures", len(fail))
good = [i for i in ids if ((prop[i]["score"] or {}).get("iou") or 0) >= 0.9]
names = {}
for i in good:
    for c in w2c[i]["candidates"]:
        if c.get("inter") is None: continue
        a = names.setdefault(c["name"], [0, 0, 0]); a[0] += c["inter"]; a[1] += c["union"]; a[2] += 1
print("good-box per-generator cIoU:", {k: round(v[0]/v[1], 3) for k, v in names.items()})
best = [max(c["iou"] for c in w2c[i]["candidates"] if c.get("iou") is not None) for i in good]
import statistics as st
print("good-box best-candidate IoU: median", round(st.median(best),3), " <0.5:", sum(b<0.5 for b in best), " <0.7:", sum(b<0.7 for b in best), " >=0.85:", sum(b>=0.85 for b in best), "of", len(best))
# box-as-mask IoU proxy is not available (no GT box here); size effect
big = sorted(good, key=lambda i: -w2c[i]["score"]["union"])[:len(good)//4]
I,U = stats(big, w2c); print(f"good-box largest quarter by union: n={len(big)} share_of_union={U/sum(w2c[i]['score']['union'] for i in good):.3f} cIoU={I/U:.3f}")
