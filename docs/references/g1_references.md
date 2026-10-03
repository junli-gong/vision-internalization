# G1 reference lines: public numbers for our model families and specialist SOTA

Compiled 2026-10-03 for the G1 gate report. Read-only web research; nothing here was run by us.

**Conventions**
- Every number gives its source URL, table and row/column.
- **UNVERIFIED** means it was not confirmed on a primary page that was actually fetched.
- "secondary" means the number was read from another paper's comparison table.
- Kind labels:
  - **S**: specialist, a non-LLM CV model.
  - **L+E**: an LLM plus a learned CV decoder or expert, e.g. SAM or a mask decoder.
  - **VLM**: a plain VLM with text output only.
  - **VLM+tok**: a VLM whose special output tokens are decoded by a learned detokenizer, e.g. PaliGemma's VQ-VAE mask tokens.
- RefCOCO REC is Acc@0.5 unless stated otherwise.
- RES numbers are cIoU (= oIoU) unless marked mIoU/gIoU.

---

## A. Public numbers for our model families

### A1. Qwen3.5: our exact models 0.8B / 2B / 4B / 9B

**Sources**
- 0.8B/2B: https://huggingface.co/Qwen/Qwen3.5-2B/raw/main/README.md. The 0.8B card carries the same table.
- 4B/9B: https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/README.md. The 4B card carries the same table.
- Rows come from the "Vision Language" table, "Spatial Intelligence" section.

| Row | Qwen3.5-0.8B (T/NT) | Qwen3.5-2B (T/NT) | Qwen3.5-4B | Qwen3.5-9B |
|---|---|---|---|---|
| RefCOCO(avg) | 79.3 / 77.8 | 84.8 / 84.3 | 88.1 | 89.7 |
| CountBench | 77.0 / 68.6 | 91.4 / 86.8 | 96.3 | 97.2 |
| EmbSpatialBench | 68.6 / 54.6 | 77.9 / 66.4 | 81.3 | 83.0 |
| ODInW13 | 31.6 / 33.2 | 35.9 / 40.5 | not reported | not reported |
| ERQA | 34.5 / 23.8 | 43.8 / 33.0 | 54.0 | 55.5 |
| RefSpatialBench | 23.5 / 21.7 | 32.9 / 30.0 | 54.6 | 58.5 |
| BLINK, CV-Bench, COCO AP, RES | not reported | not reported | not reported | not reported |

**Settings**
- **0.8B/2B card:** the footnote says "Scores of Qwen3.5 models are reported as Thinking / Non-thinking".
  - Thinking mode uses temperature 0.6.
  - Non-thinking mode uses top_p 0.8, top_k 20, presence_penalty 1.5, temperature 0.7.
- **4B/9B card:** one value per row, with no mode stated. The card says "Qwen3.5 models operate in thinking mode by default", so these are probably thinking-mode numbers (UNVERIFIED).
- **Not stated anywhere:** the RefCOCO metric (presumably Acc@0.5), the prompt, and the resolution.
- **RefCOCO(avg):** an average over the eight RefCOCO/+/g splits. No per-split numbers are published.
- **ODinW-13 protocol (Qwen3-VL report, same family):** mAP with every confidence set to 1.0 and all categories in one prompt. This sits close to our "no confidence" setting, but it is still AP, not our F1.

**Larger Qwen3.5 sizes**, same rows: RefCOCO(avg) / CountBench / ODInW13 / EmbSpatialBench.
- 397B-A17B: 92.3 / 97.2 / 47.0 / 84.5 (https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/README.md).
- 27B: 90.9 / 97.8 / 41.1 / 84.5 (https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/README.md).
- 35B-A3B: 89.2 / 97.8 / 42.6 / 83.1 (same 27B card).
- 122B-A10B: 91.3 / 97.0 / 44.5 / 83.9 (same 27B card).

**Report:** no Qwen3.5 technical report was found on arXiv.

### A2. Qwen predecessors (labelled: not our models)

**Qwen3-VL Technical Report** (arXiv 2511.21631v2, https://arxiv.org/pdf/2511.21631v2), Table 4, "small-sized Qwen3-VL models".

| Row | 2B Think | 2B Inst | 4B Think | 4B Inst | 8B Think | 8B Inst |
|---|---|---|---|---|---|---|
| RefCOCO-avg | 84.8 | 85.6 | 88.2 | 89.0 | 88.2 | 89.1 |
| CountBench | 84.1 | 88.4 | 89.4 | 84.9 | 91.5 | 80.5 |
| ODinW-13 (mAP) | 36.0 | 43.4 | 39.4 | 48.2 | 39.8 | 44.7 |
| EmbSpatialBench | 75.9 | 69.2 | 80.7 | 79.6 | 81.1 | 78.5 |
| BLINK | 57.2 | 53.8 | 63.4 | 65.8 | 64.7 | 69.1 |

- The Qwen3.5 cards compare against the Thinking columns.
- GPT-5-nano (high/minimal) in the same table: CountBench 80.0/62.9, EmbSpatialBench 74.2/50.7, BLINK 58.3/42.2.
- Larger sizes:
  - 32B Inst (Table 3): RefCOCO-avg 91.9, CountBench 94.9.
  - 235B-A22B Think/Inst (Table 2): RefCOCO-avg 92.1/91.9, CountBench 93.7/93.0, EmbSpatialBench 84.3/83.1, BLINK 67.1/70.7.
- Prompts (appendix):
  - RefCOCO asks for a bbox in JSON.
  - CountBench is multiple-choice. **This differs from our exact-answer protocol.**

**Qwen2.5-VL Technical Report** (arXiv 2502.13923, https://arxiv.org/pdf/2502.13923), Table 6, "grounding".

| Row | 3B | 7B | 72B |
|---|---|---|---|
| RefCOCO val / testA / testB | 89.1 / 91.7 / 84.0 | 90.0 / 92.5 / 85.4 | 92.7 / 94.6 / 89.7 |
| RefCOCO+ val / testA / testB | 82.4 / 88.0 / 74.1 | 84.2 / 89.1 / 76.9 | 88.9 / 92.2 / 83.7 |
| RefCOCOg val / test | 85.2 / 85.7 | 87.2 / 87.2 | 89.9 / 90.3 |
| ODinW-13 (mAP) | 37.5 | 37.3 | 43.1 |
| CountBench (Table 7) | n/r | n/r | 93.6 |
| BLINK val (Table 3) | 47.6 | 56.4 | 64.4 |

- Absolute-pixel coordinates.
- CountBench uses a detect-then-count style prompt, per the report text.
- Same Table 6 reference row: Grounding-DINO RefCOCO val 90.6, ODinW 55.0.

### A3. Gemma 4: our exact models E2B-it / E4B-it

**None of our task benchmarks are reported.** That covers RefCOCO, CountBench(QA), BLINK, CV-Bench, EmbSpatial, COCO and ODinW.

**Sources**
- https://huggingface.co/google/gemma-4-E4B-it/raw/main/README.md and the -E2B-it card.
- https://ai.google.dev/gemma/docs/core/model_card_4.
- Gemma 4 Technical Report, arXiv 2607.02770 (https://arxiv.org/html/2607.02770).
  - The sub-agent fetched it via a summarising tool. Its table numbers should be spot-checked before quoting.

**What is reported**
- The vision numbers are MMMU Pro, MATH-Vision, MedXPertQA MM, OmniDocBench 1.5 and InfoVQA.
  - E2B: MMMU Pro 44.2, MATH-Vision 52.4.
  - E4B: MMMU Pro 52.6, MATH-Vision 59.5.
- These are thinking-mode numbers at the maximum of 1120 vision tokens (report Table 6 caption). Report Table 12 repeats them at 280 tokens.
- The card claims "Object detection … and pointing" support but gives no numbers.

**Gemma 3** (predecessor; tech report arXiv 2503.19786, https://arxiv.org/html/2503.19786)
- **Table 11, pretrained (PT) checkpoints, no Pan&Scan:**
  - CountBenchQA: 4B 26.1, 12B 17.8, 27B 68.0.
  - BLINK: 38.0 / 35.9 / 39.6.
  - The 27B PT values (CountBenchQA 68.0, BLINK 39.6) are confirmed by the official card https://ai.google.dev/gemma/docs/core/model_card_3 (PT table).
- **Table 12, after per-benchmark fine-tuning:** Gemma 3 27B CountBenchQA 87.8, against PaliGemma 2 87.4.
- No IT-model numbers for our tasks.

**PaliGemma 1** (arXiv 2407.07726, https://arxiv.org/pdf/2407.07726). This is the origin of CountBenchQA.

- **CountBenchQA, Table 1** (3B):
  - 81.9 ±1.6 at 224px, 83.1 ±2.1 at 448px.
  - Appendix B.30 describes the protocol as "Zero-shot evaluation of the model trained for TallyQA. Metric: Exact match (full test split)."
- **Appendix D** describes how CountBenchQA was built from CountBench's 540 LAION-400M images (60 per count, 2..10):
  - Questions were "manually annotated" with the template "How many {object class} are there in the image?", with 3 exceptions.
  - Only 490 of the 540 images are still downloadable (HF `nielsr/countbench`). The paper notes "we have two splits, although we computed our metrics using the full split".
  - The dataset is released in big_vision `datasets/countbenchqa/`.
- **RES, Table 1** (3B), mIoU per Appendix B.22:
  - RefCOCO testA 75.7 / 77.9 / 78.7 at 224/448/896.
  - RefCOCOg test 68.2 / 71.0 / 72.7.
  - Masks are emitted as 16 `<seg>` tokens decoded by a learned VQ-VAE (lines "add 128 VQVAE tokenized single-object mask tokens"), so the kind is **VLM+tok**.

**PaliGemma 2** (arXiv 2412.03555, https://arxiv.org/html/2412.03555), Table 13, all fine-tuned transfer results.
- **CountBenchQA:**
  - 224px: 3B 81.0, 10B 84.0, 28B 86.4.
  - 448px: 3B 82.0, 10B 85.3, 28B 87.4.
- **RES mIoU, RefCOCO val:**
  - 224px: 73.4 / 75.9 / 75.0 (3B/10B/28B).
  - 448px: 76.3 / 78.2 / 77.3.
- The HTML rendering adds a stray leading "1" to every Table 13 value; it was stripped. The values are consistent with Gemma 3's Table 12 quote of 87.4.

### A4. InternVL3.5: our exact models 2B / 4B (we run the -HF Transformers-format repos)

**Source:** arXiv 2508.18265 (https://arxiv.org/html/2508.18265), **Table 7 (visual grounding)**.
- The same table image was checked directly: https://huggingface.co/OpenGVLab/InternVL3_5-241B-A28B/resolve/main/images/performance_grounding.jpg (linked from the 2B model card).
- BLINK comes from **Table 5** ("multi-image and real-world understanding"), column "BLINK (val)".

| Model | RefCOCO val / testA / testB | RefCOCO+ val / testA / testB | RefCOCOg val / test | Overall | BLINK val |
|---|---|---|---|---|---|
| InternVL3.5-1B | 85.4 / 89.7 / 80.2 | 77.7 / 85.5 / 69.5 | 81.9 / 81.6 | 81.4 | 44.0 |
| **InternVL3.5-2B** | **88.7 / 91.6 / 84.8** | 82.7 / 88.4 / 76.6 | 85.6 / 85.5 | 85.5 | **51.3** |
| **InternVL3.5-4B** | **92.5 / 94.3 / 88.2** | 87.6 / 92.3 / 81.6 | 89.6 / 89.3 | 89.4 | **58.1** |
| InternVL3.5-8B | 92.4 / 94.7 / 88.7 | 87.9 / 92.4 / 82.4 | 89.6 / 89.4 | 89.7 | 59.5 |
| InternVL3.5-241B-A28B | 94.1 / 96.3 / 91.5 | 91.6 / 94.6 / 86.9 | 92.0 / 92.1 | 92.4 | – |
| InternVL3-2B (predecessor) | 89.8 / 92.6 / 86.4 | 84.0 / 89.2 / 76.5 | 87.6 / 87.2 | 86.7 | – |

**Settings**
- Numbers are "obtained without applying TTS" (paper; model card).
- Thinking mode needs a special system prompt, so these are presumably non-thinking (UNVERIFIED for Table 7).
- The metric is not named in the image (presumably Acc@0.5).

**Not reported** for 2B/4B: CountBench, CV-Bench, EmbSpatial, RES, COCO, ODinW.
- VSI-Bench appears only for large sizes. The sub-agent parse of its value was suspicious, so it is omitted.

---

## B. Reference lines per task

### B0. At a glance

| Task | Best specialist (S) | Best L+E | Best VLM / text-only | Closest to our models |
|---|---|---|---|---|
| REC, RefCOCO val | OneRef-L 93.21 | – | InternVL3.5-241B 94.1 | InternVL3.5-2B 88.7, -4B 92.5; Qwen3.5 avg only |
| RES cIoU, RefCOCO val | UNINEXT-H 82.19 | PSALM 83.6 | UFO-8B 80.0; pure text: Text4Seg 74.7 | none (no Qwen3.5/Gemma 4/InternVL3.5 RES number) |
| CountBench(QA) | Count-aware CLIP 75.93 (classification) | – | Qwen3.5-27B/35B-A3B 97.8 | Qwen3.5-9B 97.2, -4B 96.3, -2B 91.4, -0.8B 77.0 |
| OVD COCO zero-shot AP | DINO-X Pro 56.0 | – | – | none |
| OVD ODinW13 zero-shot AP | GLIP-L 52.1 | – | Qwen3-VL-235B Inst 48.6 | Qwen3.5-2B 35.9/40.5, -0.8B 31.6/33.2 |
| BLINK Relative_Depth | DepthAnything 97.58 (dev) | GPT-4 Turbo+Tool 70.2 | GPT-4o 74.19 (val) | none for the subtask |
| NYUv2 point pairs | our construction, no public number | – | – | – |
| CV-Bench | – | – | Cambrian-1-34B 74.0 / 79.7 (2D/3D) | none first-party; Qwen2.5-VL-7B 75.0 / 83.1 (secondary) |
| EmbSpatial-Bench | – | – | Qwen3.5-397B / 27B 84.5 | Qwen3.5-9B 83.0, -4B 81.3 |

Human ceilings:
- BLINK Relative_Depth: 99.19 (test).
- EmbSpatial-Bench: 90.33.

### B1. REC Acc@0.5 (RefCOCO / RefCOCO+ / RefCOCOg)

| Model | Kind | Setting | RefCOCO val / tA / tB | RefCOCO+ val / tA / tB | RefCOCOg val / test | Source |
|---|---|---|---|---|---|---|
| MDETR ENB3 | S | pretrain COCO+VG+Flickr, FT per dataset | 87.51 / 90.40 / 82.67 | 81.13 / 85.52 / 72.96 | 83.35 / 83.31 | arxiv 2104.12763 Tab. 2 (ar5iv) |
| OFA-L | S | FT per dataset, 512px | 90.05 / 92.93 / 85.26 | 85.80 / 89.87 / 79.22 | 85.89 / 86.55 | arxiv 2202.03052 Tab. 4 |
| OFA-Huge | S | same | 92.04 / 94.03 / 88.44 | 87.86 / 91.70 / 80.71 | 88.07 / 88.78 | same Tab. 4 |
| Grounding DINO-L | S | pretrain O365+OI+GoldG+Cap4M+COCO+RefC, FT | 90.56 / 93.19 / 88.24 | 82.75 / 88.95 / 75.92 | 86.13 / 87.02 | arxiv 2303.05499 Tab. 5 |
| UNINEXT-H | S | one model, joint RefCOCO/+/g | 92.64 / 94.33 / 91.46 | 85.24 / 89.63 / 79.79 | 88.73 / 89.37 | arxiv 2303.06674 Tab. 3 |
| ONE-PEACE | S | (secondary) | 92.6 / 94.2 / 89.3 | 88.8 / 92.2 / 83.2 | 89.2 / 89.3 | InternVL3.5 Tab. 7 (image above) |
| **OneRef-L** | S | mixed RefC+ReferIt pretrain | **93.21** / 95.43 / 90.11 | 88.35 / 92.11 / 82.70 | 87.81 / 88.83 | arxiv 2410.08021 (table no. not captured) |
| OneRef-L | S | FT per dataset | 92.87 / 94.01 / 90.19 | 87.98 / 91.57 / 83.73 | 88.11 / 89.29 | same |
| PolyFormer-L | S | pretrain REC, joint FT | 90.38 / UNVERIFIED | 84.98 / UNVERIFIED | 85.83 / UNVERIFIED | arxiv 2302.07387 Tab. 2 |
| Shikra-7B | VLM | generalist, boxes as text | 87.01 / 90.61 / 80.24 | 81.60 / 87.36 / 72.12 | 82.27 / 82.19 | arxiv 2306.15195 Tab. 3 |
| Text4Seg-InternVL2-8B, no SAM | VLM | trained on RefC union | 88.3 / 91.4 / 85.8 | 83.5 / 88.2 / 77.9 | 82.4 / 82.5 | arxiv 2410.09855 Tab. 3 |
| Qwen2.5-VL-72B | VLM | generalist | 92.7 / 94.6 / 89.7 | 88.9 / 92.2 / 83.7 | 89.9 / 90.3 | arxiv 2502.13923 Tab. 6 |
| **InternVL3.5-241B-A28B** | VLM | generalist | **94.1** / 96.3 / 91.5 | 91.6 / 94.6 / 86.9 | 92.0 / 92.1 | arxiv 2508.18265 Tab. 7 |

- MDETR ENB5 does not appear in MDETR's REC table; only R101 and ENB3 do (UNVERIFIED that ENB5 REC exists elsewhere).
- Not checked: SimVG, HiVG, Florence-2, mPLUG-2, GLEE, Grounding DINO 1.5 / DINO-X on RefCOCO.

### B2. Referring segmentation, cIoU (RefCOCO / RefCOCO+ / RefCOCOg)

| Model | Kind | Setting | RefCOCO val / tA / tB | RefCOCO+ val / tA / tB | RefCOCOg val / test | Source |
|---|---|---|---|---|---|---|
| LAVT Swin-B | S | per dataset | 72.73 / 75.82 / 68.79 | 62.14 / 68.38 / 55.10 | 61.24 / 62.09 | arxiv 2112.02244 Tab. 1 (oIoU) |
| ReLA Swin-B | S | RES data | 73.82 / 76.48 / 70.18 | 66.04 / 71.02 / 57.65 | 65.00 / 65.97 | arxiv 2306.00968 Tab. 7 |
| **PolyFormer-L** (polygon vertex sequence) | S | REC pretrain, joint FT | 75.96 / 78.29 / 73.25 | 69.33 / 74.56 / 61.87 | 69.20 / 70.19 | arxiv 2302.07387 Tab. 1 |
| **UNINEXT-H** | S | joint RefC | **82.19** / 83.44 / 81.33 | 72.47 / 76.42 / 66.22 | 74.67 / 76.37 | arxiv 2303.06674 Tab. 4 |
| LISA-7B | L+E (SAM decoder) | generalist | 74.1 / 76.5 / 71.1 | 62.4 / 67.4 / 56.5 | 66.4 / 68.5 | arxiv 2308.00692 Tab. 2 |
| LISA-7B (ft) | L+E | + ReferSeg FT | 74.9 / 79.1 / 72.3 | 65.1 / 70.8 / 58.1 | 67.9 / 70.6 | same |
| GLaMM | L+E (SAM-based) | FT on RefC | 79.5 / 83.2 / 76.9 | 72.6 / 78.7 / 64.6 | 74.2 / 74.9 | arxiv 2311.03356 Tab. 4 (caption says mIoU? see C) |
| **PSALM** | L+E (Mask2Former) | joint multi-task | **83.6** / 84.7 / 81.6 | 72.9 / 75.5 / 70.1 | 73.8 / 74.4 | arxiv 2403.14598 Tab. 8 |
| Text4Seg + SAM-H (InternVL2-8B) | L+E (SAM refiner) | RefC union | 79.2 / 81.7 / 75.6 | 72.8 / 77.9 / 66.5 | 74.0 / 75.3 | arxiv 2410.09855 Tab. 1 |
| UFO-InternVL2.5-8B | VLM, mask read out by embedding dot-product, no learned decoder | non-finetuned row | 80.0 / 81.6 / 78.1 | 76.7 / 79.9 / 72.3 | 75.5 / 76.3 | arxiv 2503.01342 Tab. 3 |
| **Text4Seg 32×32, no refiner** | **VLM (pure text)** | RefC union | **74.7** / 77.4 / 71.6 | 68.5 / 73.6 / 62.9 | 70.7 / 71.6 | arxiv 2410.09855 Tab. 1 |
| Text4Seg 16×16, no refiner | VLM (pure text) | same | 70.3 / 71.9 / 68.7 | 65.0 / 68.9 / 60.8 | 66.7 / 67.6 | same |
| PaliGemma 2 10B @448 (**mIoU**) | VLM+tok (VQ-VAE) | per-task transfer | 78.2 / 79.7 / 76.2 | 74.4 / 77.7 / 71.1 | 74.4 / 74.8 | arxiv 2412.03555 Tab. 13 |

- Secondary only, kinds unverified (from UFO Tab. 3):
  - HiMTok-8B: 81.1 / 81.2 / 79.2.
  - VistaLLM-7B: 74.5 / 76.0 / 72.7.
- Not checked: EVF-SAM, UniLSeg, MagNet.

### B3. CountBench / CountBenchQA (accuracy, %)

| Reference | Kind | Variant / setting | Value | Source |
|---|---|---|---|---|
| **Qwen3.5-27B / Qwen3.5-35B-A3B** | VLM | "CountBench", mode unstated | **97.8** | HF card Qwen3.5-27B, Spatial Intelligence rows |
| Qwen3.5-9B / -397B-A17B | VLM | same | 97.2 / 97.2 | HF cards |
| Qwen3-VL-32B Inst | VLM | multiple-choice prompt | 94.9 | arxiv 2511.21631 Tab. 3 |
| Qwen2.5-VL-72B | VLM | detect-then-count style | 93.6 | arxiv 2502.13923 Tab. 7 |
| Molmo-72B | VLM | CountBenchQA | 91.2 | arxiv 2409.17146 Tab. 1 |
| Claude 3.5 Sonnet / GPT-4o-0513 / Gemini 1.5 Pro | VLM | CountBenchQA | 89.7 / 87.9 / 85.8 | same Tab. 1 |
| Gemma 3 27B, fine-tuned | VLM | CountBenchQA, transfer | 87.8 | arxiv 2503.19786 Tab. 12 |
| PaliGemma 2 28B @448 | VLM | CountBenchQA, zero-shot from TallyQA-transfer model | 87.4 ±1.0 | arxiv 2412.03555 Tab. 13 |
| PaliGemma 3B @448 | VLM | same | 83.1 ±2.1 | arxiv 2407.07726 Tab. 1, App. B.30 |
| Count-aware CLIP-B/32 | S | original CountBench, top-1 classification | 75.93 | arxiv 2302.12066 Tab. 1 |
| CountGD | S (GDINO-based) | CountBench, text prompt | MAE 0.86 only, no accuracy | arxiv 2407.04619 Tab. 2 |

- Gemma 3 PT checkpoints are low on CountBenchQA: 4B 26.1, 27B 68.0 (A3).
- No Gemma 4 number exists.
- Best specialist accuracy: none; CountGD reports MAE only.

### B4. Open-vocabulary detection, zero-shot AP (we report F1@0.5; not comparable)

| Reference | Kind | Pretraining | COCO val2017 AP | ODinW13 avg AP | Source |
|---|---|---|---|---|---|
| GLIP-T (full) | S | not re-checked | – | 46.5 | arxiv 2112.03857 Tab. 11 (ar5iv) |
| GLIP-L | S | FourODs+GoldG+Cap24M | 49.8 | **52.1** | same Tab. 2 / Tab. 11 |
| Grounding DINO-L | S | O365+OI+GoldG (true zero-shot) | 52.5 | ODinW (likely 35) 26.1 | arxiv 2303.05499 Tab. 2 / Tab. 4 |
| Grounding DINO 1.5 Pro | S | not re-checked | 54.3 (zero-shot status unclear) | – | arxiv 2405.10300 abstract |
| **DINO-X Pro** | S | not re-checked | **56.0** | not reported | arxiv 2411.14347 Tab. 1 + abstract |
| OWL-ViT L / YOLO-World v8-L / MM-G-DINO-L / T-Rex2-L (text) | S | various (secondary) | 42.2 / 45.1 / 53.0 / 52.2 | – | DINO-X Tab. 1 |
| Qwen3-VL-235B-A22B Inst | VLM | conf = 1.0, all classes in prompt | – | 48.6 | arxiv 2511.21631 Tab. 2 / Sec. 5 |
| Qwen2.5-VL-72B | VLM | – | – | 43.1 | arxiv 2502.13923 Tab. 6 |

- Qwen2.5-VL Tab. 6 also lists "Grounding DINO" ODinW 55.0, with variant and setting unstated.
- The pretraining mixes behind these "zero-shot" numbers were not re-checked for COCO-image overlap.

### B5. BLINK Relative_Depth (2-way multiple choice, chance 50; n = 124 per split)

| Reference | Kind | Split | Value | Source |
|---|---|---|---|---|
| Human | human | test / val | 99.19 | arxiv 2404.12390 Tab. 1, Tab. 4 |
| **DepthAnything** | S | dev (= val) | **97.58** | same Tab. 2 "Specialist" row |
| GPT-4o | VLM | val | 74.19 (best VLM in paper) | same Tab. 4 |
| GPT-4 Turbo | VLM | test | 66.94 (best on test) | same Tab. 1 |
| GPT-4 Turbo + depth tool | L+E | AURORA re-eval, split unstated | 70.2 | arxiv 2412.03548 Tab. 1 |
| LLaVA fine-tuned on depth labels / LLaVA-Aurora | VLM / VLM+tok | same | 68.5 / 64.5 | same Tab. 1 |

- BLINK Table 2 gives human 99.59 on dev.
- Only BLINK **overall** is published for our families:
  - Qwen3-VL-2B 57.2 / 53.8 and -4B 63.4 / 65.8 (T/I; 2511.21631 Tab. 4).
  - InternVL3.5-2B 51.3, -4B 58.1 (val; 2508.18265 Tab. 5).
  - Qwen2.5-VL-3B 47.6 (2502.13923 Tab. 3).
  - Gemma 3 4B PT 38.0 (2503.19786 Tab. 11).

### B6. NYUv2: context only

**Our 5-pairs-per-image, ≥15% relative-depth-gap protocol is our own construction. No public number exists for it.**

| Reference | Kind | Setting | Metric | Source |
|---|---|---|---|---|
| Depth Anything V2 ViT-L | S | zero-shot relative depth, NYU-D | AbsRel 0.045, δ1 0.979 | arxiv 2406.09414 Tab. 2 |
| Depth Anything V1 ViT-L | S | same | AbsRel 0.043, δ1 0.981 | same Tab. 2 |
| Depth Anything V2 ViT-L | S | metric, fine-tuned on NYU | δ1 0.984, AbsRel 0.056 | same, metric fine-tuning table (a) |
| Depth Anything V2 ViT-L / ViT-G | S | DA-2K sparse point-pair relative depth | Acc 97.1 / 97.4 | same Tab. 5 |

- DA-2K, a sparse pair-ordering benchmark, is the closest public analogue to our protocol.
- Earlier ordinal-pair precedent is UNVERIFIED (not fetched): Zoran et al. 2015 ordinal pairs on NYU, and Chen et al. 2016 DIW / WHDR.
- Also not fetched: Metric3D v2, UniDepth, Marigold.

### B7. CV-Bench (Cambrian-1; 2D = ADE20K + COCO, 3D = Omni3D)

| Reference | Kind | 2D / 3D (overall) | Source |
|---|---|---|---|
| Cambrian-1-34B | VLM | 74.0 / 79.7 | arxiv 2406.16860 Tab. 9 |
| Cambrian-1-8B / 13B | VLM | 72.3 / 72.0; 72.5 / 71.8 | same |
| Mini-Gemini-HD-34B / LLaVA-NeXT-34B | VLM | 71.5 / 79.2; 73.0 / 74.8 | same |
| GPT-4V | VLM | 64.3 / 73.8 | same |
| Qwen2.5-VL-7B (secondary) | VLM | 75.0 / 83.1 (79.0) | arxiv 2510.08531 (SpatialLadder) Tab. 11 |
| Qwen2.5-VL-3B (secondary) | VLM | 69.1 / 72.2 (70.6) | same |
| GPT-4o / InternVL2.5-4B (secondary) | VLM | 69.4 / 81.3 (75.4); 73.5 / 75.1 (74.4) | same |

- InternVL3-8B "82.0 avg" appeared only in a search snippet and is **UNVERIFIED**.
- No first-party CV-Bench number exists for Qwen3.5, Qwen3-VL, Gemma 4 or InternVL3.5.
- Not fetched: Cambrian-S, SAT.

### B8. EmbSpatial-Bench (test; 3,640 QA, 2,181 images)

| Reference | Kind | Setting | Value | Source |
|---|---|---|---|---|
| Human | human | – | 90.33 | arxiv 2406.05756 Tab. 2 |
| **Qwen3.5-397B-A17B / Qwen3.5-27B** | VLM | card, mode unstated | **84.5** | HF cards |
| Qwen3-VL-235B-A22B Think / Inst | VLM | prompt "<image> {question}" | 84.3 / 83.1 | arxiv 2511.21631 Tab. 2 |
| Qwen3.5-9B / -4B | VLM | card | 83.0 / 81.3 | HF card Qwen3.5-9B |
| MiniGPT-v2 + EmbSpatial-SFT | VLM | likelihood scoring | 78.10 | arxiv 2406.05756 Tab. 3 |
| GPT-5-nano (high) | VLM | – | 74.2 | arxiv 2511.21631 Tab. 4 |
| Qwen-VL-Max | VLM | zero-shot, generation (best in original paper) | 49.11 | arxiv 2406.05756 Tab. 2 |
| GPT-4V | VLM | same | 36.07 | same |

---

## C. Comparability caveats (one line per table)

- **A1 Qwen3.5**
  - Thinking vs non-thinking moves CountBench by up to 8 pts at 0.8B and EmbSpatial by 11–14 pts.
  - The 4B/9B mode is unstated.
  - RefCOCO is an 8-split average, not per split.
  - No prompt, resolution or metric definition is given.
- **A2 Qwen3-VL / Qwen2.5-VL**
  - Qwen3-VL CountBench is multiple-choice, not our exact-answer protocol.
  - Instruct vs Thinking differ by up to 11 pts.
  - Qwen2.5-VL uses absolute pixel coordinates.
- **A3 Gemma**
  - Gemma 4 publishes nothing on our tasks.
  - Gemma 3 Table 11 is PT checkpoints and Table 12 is per-task fine-tuning.
  - CountBenchQA was built and released by Google's PaliGemma team (2407.07726 App. D). Google-family models may therefore have been tuned against it, though no statement says CountBenchQA data was used in training.
  - PaliGemma's own number comes from the TallyQA-transferred model.
- **A4 InternVL3.5**
  - Per-split REC is directly comparable to our splits.
  - Thinking mode is off by default.
  - The metric is unnamed; Acc@0.5 is the convention.
  - RefCOCO train is in InternVL's SFT grounding data (UNVERIFIED; standard practice).
- **B1 REC**
  - Specialists are fine-tuned on RefCOCO train, which uses COCO train2014 images.
  - Grounding DINO-L pretraining includes COCO.
  - Best numbers come from different regimes: per-dataset FT (OFA, MDETR, OneRef) vs one joint model (UNINEXT).
  - VLM numbers are generalist.
  - We evaluate the full splits zero-shot; check that our protocol matches.
- **B2 RES**
  - cIoU (= oIoU) vs mIoU/gIoU are not interchangeable. PaliGemma, SeqTR and the OneRef-RES numbers are mIoU.
  - GLaMM's caption metric was unclear; it was cross-checked as cIoU.
  - Our 1,000-sentence subsample per split is not the full split.
  - Polygon-as-text caps boundary fidelity. PolyFormer (specialist polygon decoder) and Text4Seg-no-refiner (pure text) are the fair analogues.
  - LISA, GLaMM and PSALM are L+E, which our rules exclude from comparison.
- **B3 CountBench**
  - Three variants are in use:
    - original CountBench, image classification;
    - CountBenchQA, PaliGemma's human-written open questions;
    - Qwen3-VL's multiple-choice prompt.
  - 50 of the 540 images are no longer downloadable (490 remain). PaliGemma scored the "full split"; check our denominator against that.
  - The Qwen3.5 protocol is unstated.
- **B4 OVD**
  - Every public number is AP with scores. Our F1@0.5 without confidence is not comparable.
  - The closest public protocol is Qwen3-VL's ODinW-13 mAP with every score set to 1.0.
  - Our COCO numbers use a 500-image subset; the published ones use the full 5k val2017.
  - Grounding DINO's "ODinW" is likely ODinW35, not 13.
  - "Zero-shot" pretraining often includes VG/GoldG, which shares images with COCO.
- **B5 BLINK depth**
  - The original paper reports val and test separately; AURORA's re-evaluation gives a different GPT-4o score (53.2 vs 74.19), so the protocols differ.
  - The DepthAnything row is a depth map plus a rule, not a VLM.
  - n = 124, so a binomial 95% CI is about ±8 pts (p = 0.5–0.75).
- **B6 NYUv2**
  - Our pair protocol has no public counterpart.
  - δ1/AbsRel are dense-depth metrics, context only.
  - DA-2K pairwise accuracy is the closest analogue (different images, sparse human-annotated pairs).
- **B7 CV-Bench**
  - Cambrian reports 2D and 3D separately, while later papers often report a single overall.
  - Secondary numbers come from papers with their own prompts.
  - Nothing first-party exists for our 8 models.
- **B8 EmbSpatial**
  - Generation vs likelihood scoring differ by up to 45 pts (MiniGPT-v2 SFT: 78.10 vs 32.97).
  - The 2024 numbers (≤49) predate embodied/spatial SFT. Qwen3/3.5 likely trained on spatial data, so possible distribution overlap (UNVERIFIED).

**Coverage**
- **Not found or not checked:**
  - Gemma 4 numbers on any of our tasks (none published).
  - First-party CV-Bench numbers for any of our families.
  - BLINK Relative_Depth for any of our families.
  - MDETR ENB5 REC.
  - SimVG, HiVG, EVF-SAM and Florence-2 lines.
  - ODinW13 for Grounding DINO 1.5 / DINO-X / OWLv2.
  - Ordinal-depth precedents (Zoran, DIW).
  - Metric3D v2, UniDepth, Cambrian-S, SAT.
- **How sources were read:** most arXiv tables were read through a summarising fetch tool or ar5iv. Exceptions:
  - PaliGemma (2407.07726) was read from the PDF text.
  - InternVL3.5 Tab. 7 was read from the original image.
  - Qwen3-VL and Qwen2.5-VL were read from the PDF text by a sub-agent.
  - Table numbers from ar5iv may be off by one versus the latest arXiv version.
