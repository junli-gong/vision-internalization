"""RUN-0001 §6 group A: shrink head_train2014_union.tsv (>5 MB) to image_id/status/content_length/accept_ranges/content_type
(URL is COCO_train2014_{id:012d}.jpg under http://images.cocodataset.org/train2014/). usage: python p6_A_compact_head.py <in.tsv> <out.tsv>"""
import csv, sys
with open(sys.argv[2], "w") as f:
    f.write("image_id\tstatus\tcontent_length\taccept_ranges\tcontent_type\terror\n")
    for r in csv.DictReader(open(sys.argv[1]), delimiter="\t"):
        assert r["final_url"] == r["url"]
        iid = int(r["url"].rsplit("_", 1)[1].split(".")[0])
        f.write(f"{iid}\t{r['status']}\t{r['content_length']}\t{r['accept_ranges']}\t{r['content_type']}\t{r['error']}\n")
