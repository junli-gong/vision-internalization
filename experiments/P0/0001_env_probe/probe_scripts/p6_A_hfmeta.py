"""RUN-0001 §6 group A: HF metadata-only probe (no data download).
For each HF dataset repo: HfApi.dataset_info(files_metadata=True) + README (in memory)
+ datasets-server /splits /info /size /first-rows (features + first 2 rows, long strings truncated).
usage: python p6_A_hfmeta.py <out_dir> <repo_id> [<repo_id> ...]"""
import sys, os, json, time, urllib.request, urllib.parse, urllib.error
from huggingface_hub import HfApi

OUT = sys.argv[1]
api = HfApi()
DS = "https://datasets-server.huggingface.co"


def get_json(url, tries=4):
    tok = None
    try:
        from huggingface_hub import get_token
        tok = get_token()
    except Exception:
        pass
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"} if tok else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return {"status": r.status, "json": json.loads(r.read())}
        except urllib.error.HTTPError as e:
            body = e.read()[:2000].decode("utf-8", "replace")
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(2 ** i * 2); continue
            return {"status": e.code, "error": body}
        except Exception as e:
            if i < tries - 1:
                time.sleep(2 ** i * 2); continue
            return {"status": None, "error": repr(e)}


def get_text(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:
        return f"<<error {e!r}>>"


def trunc(x, n=300):
    if isinstance(x, str):
        return x if len(x) <= n else x[:n] + f"...<+{len(x)-n} chars>"
    if isinstance(x, list):
        out = [trunc(v, n) for v in x[:5]]
        if len(x) > 5: out.append(f"...<+{len(x)-5} items>")
        return out
    if isinstance(x, dict):
        return {k: trunc(v, n) for k, v in x.items()}
    return x


for rid in sys.argv[2:]:
    rec = {"requested_id": rid, "probe_time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        info = api.dataset_info(rid, files_metadata=True)
        rec["resolved_id"] = info.id
        rec["sha"] = info.sha
        rec["last_modified"] = str(info.last_modified)
        rec["created_at"] = str(getattr(info, "created_at", None))
        rec["gated"] = info.gated
        rec["private"] = info.private
        rec["downloads"] = info.downloads
        rec["likes"] = info.likes
        rec["tags"] = info.tags
        cd = info.card_data.to_dict() if info.card_data else {}
        rec["card_license"] = cd.get("license")
        rec["card_data_keys"] = sorted(cd.keys())
        rec["card_configs"] = cd.get("configs")
        rec["card_dataset_info"] = cd.get("dataset_info")
        files = [{"path": s.rfilename, "size": s.size, "lfs": bool(s.lfs)} for s in (info.siblings or [])]
        rec["n_files"] = len(files)
        rec["total_bytes_hf_api"] = sum((f["size"] or 0) for f in files)
        rec["files"] = files if len(files) <= 400 else files[:400] + [{"note": f"+{len(files)-400} more"}]
    except Exception as e:
        rec["dataset_info_error"] = repr(e)
        rid_res = rid
    rid_res = rec.get("resolved_id", rid)
    rec["readme"] = get_text(f"https://huggingface.co/datasets/{rid_res}/raw/main/README.md")[:20000]
    q = urllib.parse.quote(rid_res, safe="")
    rec["ds_splits"] = get_json(f"{DS}/splits?dataset={q}")
    rec["ds_size"] = get_json(f"{DS}/size?dataset={q}")
    rec["ds_info"] = get_json(f"{DS}/info?dataset={q}")
    fr = {}
    sp = rec["ds_splits"].get("json", {}).get("splits", []) if rec["ds_splits"].get("json") else []
    seen_cfg = set()
    for s in sp:
        cfg, split = s["config"], s["split"]
        if cfg in seen_cfg: continue  # one first-rows call per config
        seen_cfg.add(cfg)
        r = get_json(f"{DS}/first-rows?dataset={q}&config={urllib.parse.quote(cfg)}&split={urllib.parse.quote(split)}")
        if r.get("json"):
            j = r["json"]
            fr[f"{cfg}/{split}"] = {"features": j.get("features"), "n_rows_returned": len(j.get("rows", [])),
                                    "truncated": j.get("truncated"),
                                    "rows_first2": [trunc(x.get("row")) for x in j.get("rows", [])[:2]]}
        else:
            fr[f"{cfg}/{split}"] = r
        if len(seen_cfg) >= 6: break
    rec["ds_first_rows"] = fr
    fn = os.path.join(OUT, "hf_" + rid.replace("/", "__") + ".json")
    with open(fn, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"{rid:45s} -> {rec.get('resolved_id')} files={rec.get('n_files')} bytes={rec.get('total_bytes_hf_api')} "
          f"license={rec.get('card_license')} splits_status={rec['ds_splits'].get('status')} -> {fn} ({os.path.getsize(fn)} B)")
