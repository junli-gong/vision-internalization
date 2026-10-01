| 源 | 退出码 | 墙钟 s | 文件数 | 落地字节 | 估算字节 | 核对项（通过/总数） | 不可达 URL | 上游 |
|---|---|---|---|---|---|---|---|---|
| refcoco | 0 | 1089 | 6552 | 1,128,662,793 | 1,245,957,297 | 18/18 | 0 | https://web.archive.org/web/20220413011718if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip; https://web.archive.org/web/20220413011656if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip; https://web.archive.org/web/20220413012904if_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip |
| coco_val2017 | 0 | 50 | 501 | 84,838,146 | 106,838,125 | 1/1 | 0 | http://images.cocodataset.org/annotations/annotations_trainval2017.zip |
| odinw13 | 0 | 285 | 476 | 28,871,045 | 135,000,000 | 0/0 | 0 | https://huggingface.co/GLIPModel/GLIP/resolve/66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1/odinw_35/AerialMaritimeDrone.zip; https://huggingface.co/GLIPModel/GLIP/resolve/66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1/odinw_35/Aquarium.zip; https://huggingface.co/GLIPModel/GLIP/resolve/66ee3ae9a3b8cee0cf78f10ef5fc9a3725db02a1/odinw_35/CottontailRabbits.zip … |
| countbench | 0 | 4 | 492 | 23,699,768 | 25,353,242 | 3/3 | 0 | https://huggingface.co/datasets/nielsr/countbench/resolve/e32ce2541299d12755894e1d487f6dd75bb0176c/data/train-00000-of-00001-cf54c241ba947306.parquet; https://raw.githubusercontent.com/teaching-clip-to-count/teaching-clip-to-count.github.io/main/CountBench.json |
| nyuv2 | 0 | 326 | 1410 | 1,253,118,584 | 1,548,800,000 | 1/1 | 0 | https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat |
| blink_reldepth | 0 | 2 | 125 | 4,655,259 | 5,604,789 | 1/1 | 0 | https://huggingface.co/datasets/BLINK-Benchmark/BLINK/resolve/a3666eb249237ba3d5eca8db21176cc47967e040/Relative_Depth/val-00000-of-00001.parquet |
| cvbench | 0 | 30 | 2639 | 407,183,609 | 409,808,364 | 7/7 | 0 | https://huggingface.co/datasets/nyu-visionx/CV-Bench/resolve/bc284db50d036958861cb60cdd7b77612052ce0d/test_2d.jsonl; https://huggingface.co/datasets/nyu-visionx/CV-Bench/resolve/bc284db50d036958861cb60cdd7b77612052ce0d/test_3d.jsonl; https://huggingface.co/datasets/nyu-visionx/CV-Bench/resolve/bc284db50d036958861cb60cdd7b77612052ce0d/test_2d.parquet … |
| embspatial | 0 | 21 | 3641 | 190,906,612 | 200,000,000 | 2/2 | 0 | https://huggingface.co/datasets/Phineas476/EmbSpatial-Bench/resolve/e2733cacf0f20371518c687176e881d79644e61e/embspatial_bench.json |

未通过的核对项与不可达 URL：无

各源核对项明细（what → ok）：

**refcoco**（landed_at 2026-10-01T06:42:56+00:00，script commit 9c517fa5814e）
- refcoco：refs pickle 的 sha256 与 P0 记录一致 → True
- refcoco：评测划分计数（refs, sentences, images）= P0 → True
- HF jxu124/refcoco@9fba8200c5 data/validation-00000-of-00001-bfeafdc84ca37aa2.parquet → True
- HF jxu124/refcoco@9fba8200c5 data/test-00000-of-00001-82af0c1b600890ac.parquet → True
- HF jxu124/refcoco@9fba8200c5 data/testB-00000-of-00001-60990e4598892dc1.parquet → True
- refcoco：与镜像逐条比对 → True
- refcoco+：refs pickle 的 sha256 与 P0 记录一致 → True
- refcoco+：评测划分计数（refs, sentences, images）= P0 → True
- HF jxu124/refcocoplus@12e20b0b60 data/validation-00000-of-00001-8c57d66282bc60c9.parquet → True
- HF jxu124/refcocoplus@12e20b0b60 data/test-00000-of-00001-2b8e5d26906553b9.parquet → True
- HF jxu124/refcocoplus@12e20b0b60 data/testB-00000-of-00001-4f1178d399f1874a.parquet → True
- refcoco+：与镜像逐条比对 → True
- refcocog：refs pickle 的 sha256 与 P0 记录一致 → True
- refcocog：评测划分计数（refs, sentences, images）= P0 → True
- HF jxu124/refcocog@55319436ad data/validation-00000-of-00001-15168dfe7b5961e5.parquet → True
- HF jxu124/refcocog@55319436ad data/test-00000-of-00001-2316f36b19cd7f72.parquet → True
- refcocog：与镜像逐条比对 → True
- 图像宽高与 instances.json 一致 → True

**coco_val2017**（landed_at 2026-10-01T06:43:47+00:00，script commit 9c517fa5814e）
- instances_val2017.json 的 sha256 与字节数 = P0 记录 → True

**odinw13**（landed_at 2026-10-01T06:48:32+00:00，script commit 9c517fa5814e）

**countbench**（landed_at 2026-10-01T06:48:36+00:00，script commit 9c517fa5814e）
- HF nielsr/countbench@e32ce25412 data/train-00000-of-00001-cf54c241ba947306.parquet → True
- 官方 CountBench.json 的 sha256 = P0 记录 → True
- HF 打包版与官方 CountBench.json 逐条同序一致（image_url, text, number） → True

**nyuv2**（landed_at 2026-10-01T06:54:02+00:00，script commit 9c517fa5814e）
- splits.mat 的 sha256 = P0 记录（P0 已核对本地与官方相同） → True

**blink_reldepth**（landed_at 2026-10-01T06:54:04+00:00，script commit 9c517fa5814e）
- HF BLINK-Benchmark/BLINK@a3666eb249 Relative_Depth/val-00000-of-00001.parquet → True

**cvbench**（landed_at 2026-10-01T06:54:34+00:00，script commit 9c517fa5814e）
- HF nyu-visionx/CV-Bench@bc284db50d test_2d.jsonl → True
- HF nyu-visionx/CV-Bench@bc284db50d test_3d.jsonl → True
- HF nyu-visionx/CV-Bench@bc284db50d test_2d.parquet → True
- HF nyu-visionx/CV-Bench@bc284db50d test_3d.parquet → True
- 每一行都从 parquet 取到了图像 → True
- parquet 的非图像列与 jsonl 逐条一致 → True
- idx 全局唯一且每行都有 → True

**embspatial**（landed_at 2026-10-01T06:54:55+00:00，script commit 9c517fa5814e）
- HF Phineas476/EmbSpatial-Bench@e2733cacf0 embspatial_bench.json → True
- question_id 唯一 → True
