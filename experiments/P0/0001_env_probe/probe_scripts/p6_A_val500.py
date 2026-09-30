"""RUN-0001 §6 group A: size estimates for a 500-image COCO val2017 subset from HEAD Content-Length (all 5000 images).
usage: python p6_A_val500.py <head_val2017_all5000.tsv> <instances_val2017.json> <out.json>"""
import csv, json, sys, random, statistics
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter="\t") if r["status"] == "200"]
size = {int(r["url"].rsplit("/", 1)[1].split(".")[0]): int(r["content_length"]) for r in rows}
v = json.load(open(sys.argv[2])); withann = sorted(set(a["image_id"] for a in v["annotations"]))
ids = sorted(size); mean = statistics.mean(size.values())
out = {"n_images_head_200": len(size), "total_bytes_all5000": sum(size.values()), "mean_bytes": round(mean, 1),
       "mean_x_500": round(mean * 500), "mean_bytes_images_with_annotations": round(statistics.mean(size[i] for i in withann), 1),
       "examples_of_concrete_500_subsets": {}}
for seed in (0, 1, 2):
    s = random.Random(seed).sample(ids, 500)
    out["examples_of_concrete_500_subsets"][f"random.Random({seed}).sample(sorted all 5000 ids, 500)"] = sum(size[i] for i in s)
out["examples_of_concrete_500_subsets"]["first 500 by sorted image id"] = sum(size[i] for i in ids[:500])
out["examples_of_concrete_500_subsets"]["random.Random(0).sample(sorted 4952 ids with annotations, 500)"] = sum(size[i] for i in random.Random(0).sample(withann, 500))
json.dump(out, open(sys.argv[3], "w"), indent=1); print(json.dumps(out, indent=1))
