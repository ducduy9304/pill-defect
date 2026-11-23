# Unsupervised / few-shot anomaly detection architectures for surface defects (state as of November 2026)

Scope of these notes: methods that learn from good images only, judged against one use case: white "betway" lettering plus a small lot-code line on clear carrier film, 5 MP colour camera at about 0.17 mm/px, rectified images, 5-20 good images per SKU, many SKUs, one desktop GPU, a check in well under a few seconds. Current baseline: DINOv2 ViT-B/14 patch features (blocks 5..10), nearest-neighbour cosine distance to good patches with a per-tile position constraint.

Reading notes:
- All numbers are percentages copied from the cited paper. "I-AUROC" = image-level AUROC, "P-AUROC" = pixel AUROC, "PRO"/"AUPRO" = per-region overlap, "P-F1" = pixel F1-max, "SegF1" = the MVTec AD 2 pixel F1 at a fixed threshold.
- Papers were read through an automatic page-to-text extraction. Where a table looked ambiguous or two sources disagree, this is flagged. Check the PDF before quoting a single number in a final document.
- Papers dated 2026 are newer than most survey material; they are preprints or fresh conference papers and their numbers are self-reported.

## 1. How do the method families compare on MVTec AD, VisA, MVTec AD 2 and Real-IAD (full-shot and 1/2/4/8-shot)?

### Takeaway
In the few-shot regime the ranking is stable across 2024-2026 papers: frozen DINO features with patch nearest-neighbour or subspace scoring (AnomalyDINO, SubspaceAD, DuoAD, RAD) beat CLIP-based prompt methods (WinCLIP, PromptAD, APRIL-GAN) and beat WideResNet PatchCore/SPADE/PaDiM by 10+ points of image AUROC and PRO. In full-shot, trained ViT reconstruction models (Dinomaly, INP-Former) and the training-free RAD are on par at the top, and all of them are far ahead of older CNN methods on pixel AP/F1, which is the metric that reflects small defects.

### Cited Findings

Few-shot, MVTec AD and VisA (good images only)

- AnomalyDINO (WACV 2025 oral, DINOv2 ViT-S/14, training-free patch kNN). MVTec AD I-AUROC at 1/2/4/8/16-shot: 96.5/96.7/97.6/98.0/98.3 at 448 px and 96.6/96.9/97.7/98.2/98.4 at 672 px. Same table: SPADE 81.0/82.9/84.8 (1/2/4), PatchCore 83.4/86.3/88.8, GraphCore 89.9/91.9/92.9/95.9 (1/2/4/8), WinCLIP+ 93.1/94.4/95.2, APRIL-GAN 92.0/92.4/92.8/93.1/93.2, ADP 95.4/96.2/97.0/97.0 (2/4/8/16) — [AnomalyDINO, arXiv 2405.14529](https://arxiv.org/html/2405.14529v3)
- AnomalyDINO MVTec AD segmentation (P-AUROC / P-F1 / PRO): 1-shot 96.3/57.9/91.7 at 448 and 96.8/60.2/92.7 at 672; 4-shot 96.7/59.2/92.4 and 97.2/61.8/93.4; 8-shot 97.0/59.6/92.7 and 97.4/62.3/93.8; 16-shot 97.1/60.0/92.9 and 97.5/62.7/94.0. Baselines at 1-shot: SPADE 91.2/42.4/83.9, PatchCore 92.0/50.4/79.7, WinCLIP+ 95.2/55.9/87.1, APRIL-GAN 95.1/54.2/90.6 — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- AnomalyDINO VisA (I-AUROC / P-AUROC / PRO) at 672 px: 1-shot 87.4/97.8/92.5, 2-shot 89.7/98.0/93.4, 4-shot 92.6/98.2/94.1, 8-shot 93.8/98.4/94.8, 16-shot 94.8/98.5/95.3. Baselines 1-shot: SPADE 79.5/95.6/84.1, PaDiM 62.8/89.9/64.3, PatchCore 79.9/95.4/80.5, WinCLIP+ 83.8/96.4/85.1, APRIL-GAN 91.2/96.0/90.0; 4-shot: PaDiM 72.8/93.2/72.6, PatchCore 85.3/96.8/84.9, WinCLIP+ 87.3/97.2/87.6, APRIL-GAN 92.6/96.2/90.2 — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- AnomalyDINO VisA 1-shot P-F1 is only 41.9 (448) / 45.1 (672): pixel F1 on VisA's small defects stays low even when P-AUROC is 97.5+ — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- SubspaceAD (CVPR 2026, PCA subspace on frozen DINOv2-G/14 features at 672 px, training-free). I-AUROC / P-AUROC / PRO. MVTec AD: 1-shot 97.1/97.5/94.5, 2-shot 97.5/97.8/94.9, 4-shot 98.0/97.9/95.1. VisA: 1-shot 93.2/98.2/95.5, 2-shot 93.8/98.3/95.7, 4-shot 94.7/98.4/96.0. Baselines in the same table, 1-shot MVTec: PromptAD 94.6/95.9/87.9, IIPAD 94.2/96.4/89.8, AnomalyDINO 96.6/96.8/92.7; 1-shot VisA: PromptAD 86.9/96.7/85.1, IIPAD 85.4/96.9/87.3, AnomalyDINO 87.4/97.8/92.5 — [SubspaceAD, arXiv 2602.23013](https://arxiv.org/html/2602.23013)
- SubspaceAD numbers differ between sources: the paper table gives 97.1 (MVTec 1-shot I-AUROC) while a README mirror lists 98.0 and the paper's own tau-ablation shows 98.0-98.1; treat the exact value as uncertain — [SubspaceAD paper](https://arxiv.org/html/2602.23013); [search summary of repository mirrors](https://www.alphaxiv.org/abs/2602.23013)
- DuoAD (arXiv July 2026, training-free, patch kNN on layers {8,10,12} of ViT-B plus [CLS]-attention reweighting and self-calibrated augmentation). I-AUROC / PRO. MVTec AD 1-shot: 97.3/94.3 (DINOv2-B/14 reg, 448 px) and 97.7/94.7 (DINOv3-B/16, 512 px); 4-shot 98.2/94.8 and 98.6/95.7. VisA 1-shot 92.7/93.2 and 93.2/93.2; 4-shot 94.9/94.5 and 95.5/94.8. The same table lists PromptAD 94.6/87.9, Kag-prompt 95.8/90.8, UniVAD 97.5/92.8, AnomalyDINO 96.5/91.7, FoundAD 96.1/92.8 for MVTec 1-shot — [DuoAD, arXiv 2607.23924](https://www.alphaxiv.org/abs/2607.23924.md)
- DuoAD pixel F1 (MVTec AD): 1-shot about 62.8, 4-shot about 65.2 with DINOv3; VisA 1-shot 50.5, 4-shot 53.6. The extraction of this table had ambiguous column order, verify in the PDF — [DuoAD](https://www.alphaxiv.org/abs/2607.23924.md)
- UniVAD (training-free; Grounded-SAM component segmentation + CLIP-L/14@336 and DINOv2-G/14 features). 1-shot I-AUROC / P-AUROC: MVTec AD 97.8/96.5, VisA 93.5/98.2, MVTec LOCO 71.0/75.1. Baselines in its table: PatchCore 84.0/89.9 (MVTec), 74.8/93.4 (VisA); AnomalyGPT 94.1/95.3, 87.4/96.2; WinCLIP 93.1/95.2, 83.8/96.2 — [UniVAD, arXiv 2412.03342](https://arxiv.org/html/2412.03342)
- UniVAD's own numbers (97.8 / 93.5) differ slightly from how DuoAD quotes them (97.5 / 92.9) — [UniVAD](https://arxiv.org/html/2412.03342); [DuoAD](https://www.alphaxiv.org/abs/2607.23924.md)
- INP-Former (CVPR 2025 per its repository listing; trained ViT reconstruction with prototypes from the test image; DINOv2-R ViT-B/14, 392 px). Few-shot, image AUROC/AP/F1 and pixel AUROC/AP/F1/AUPRO. 1-shot MVTec AD 96.6/98.2/96.4 and 97.0/64.2/64.0/92.6; VisA 91.4/92.2/88.6 and 96.3/42.5/47.3/89.5. 4-shot MVTec AD 97.6/98.6/97.0 and 97.0/65.9/65.6/92.9; VisA 96.4/96.0/93.0 and 97.7/49.3/54.3/93.1 — [INP-Former, arXiv 2503.02424](https://arxiv.org/html/2503.02424)
- FoundAD (ICLR 2026; small projector, 11.8 M parameters, trained on frozen DINOv3 features with CutPaste-style synthetic anomalies). 1-shot I-AUROC 96.1 on MVTec AD, 92.6 on VisA; 2-shot 96.8 / 93.5. Figures come from a third-party summary; its "PRO 98.0" for VisA is probably a pixel AUROC — [FoundAD summary, papernotes](https://en.papernotes.org/ICLR2026/anomaly_detection/foundation_visual_encoders_are_secretly_few-shot_anomaly_detectors/)
- RAD (arXiv Jan 2026, v3 June 2026; training-free multi-layer DINOv3 memory bank with global-then-patch retrieval): with one good image on MVTec AD it reaches 96.7 P-AUROC against 98.5 with full data; in 1/2/4-shot it reports +2.1/+2.7/+2.7 AUPRO over IIPAD and +16.8/+16.9/+14.9 AUPRO over Dinomaly trained on the same few images — [RAD, arXiv 2601.22763](https://arxiv.org/html/2601.22763v3)
- RegAD (ECCV 2022 oral; one category-agnostic registration network, no fine-tuning for a new category) reported 3-8 points of AUC over the few-shot methods of its time on MVTec AD and MPDD. Older result; it is no longer listed as a competitive baseline in the 2025-2026 tables above — [RegAD, arXiv 2207.07361](https://arxiv.org/abs/2207.07361)
- MetaUAS (NeurIPS 2024): pure-vision, one normal image as prompt, anomaly segmentation framed as change segmentation and meta-trained on synthetic image pairs; has a soft feature alignment module for geometric differences between prompt and query; no training on the target — [MetaUAS, arXiv 2505.09265](https://arxiv.org/abs/2505.09265)

Zero-shot vision-language (no good images of the product)

- Zero-shot I-AUROC: WinCLIP 91.8 (MVTec AD) / 78.1 (VisA); AnomalyCLIP 91.5 / 82.1; APRIL-GAN 86.1 / 78.0 — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- WinCLIP zero-shot localisation is weak: MVTec AD 91.8 I-AUROC / 85.1 P-AUROC / 64.6 PRO; VisA 78.1 / 79.6 / 56.8 — [SubspaceAD](https://arxiv.org/html/2602.23013)
- "Batched zero-shot" (unlabelled test batch, no references): MuSc 97.8 / 92.8-94.1, AnomalyDINO-S(672) 94.2 / 90.7, SubspaceAD 96.6 / 94.1 (MVTec / VisA I-AUROC) — [AnomalyDINO](https://arxiv.org/html/2405.14529v3); [SubspaceAD](https://arxiv.org/html/2602.23013)
- On MVTec AD 2 (VAND 4.0 challenge, CVPR 2026) zero-shot approaches trail the regular setting by about 15 SegF1 points; the organisers conclude that using normal data of the product is essential for precise localisation — [VAND 4.0 report, arXiv 2610.04392](https://chatpaper.com/paper/357114)

Full-shot, multi-class (one model for all classes), image AUROC/AP/F1 | pixel AUROC/AP/F1/AUPRO

- MVTec AD: RD4AD 94.6/96.5/95.2 | 96.1/48.6/53.8/91.1; SimpleNet 95.3/98.4/95.8 | 96.9/45.9/49.7/86.5; DeSTSeg 89.2/95.5/91.6 | 93.1/54.3/50.9/64.8; UniAD 96.5/98.8/96.2 | 96.8/43.4/49.5/90.7; ViTAD 98.3/99.4/97.3 | 97.7/55.3/58.7/91.4; MambaAD 98.6/99.6/97.8 | 97.7/56.3/59.2/93.1; Dinomaly 99.6/99.8/99.0 | 98.4/69.3/69.2/94.8 — [Dinomaly, arXiv 2405.14325](https://arxiv.org/html/2405.14325)
- VisA: RD4AD 92.4/92.4/89.6 | 98.1/38.0/42.6/91.8; SimpleNet 87.2/87.0/81.8 | 96.8/34.7/37.8/81.4; UniAD 88.8/90.8/85.8 | 98.3/33.7/39.0/85.5; ViTAD 90.5/91.7/86.3 | 98.2/36.6/41.1/85.1; MambaAD 94.3/94.5/89.4 | 98.5/39.4/44.0/91.0; Dinomaly 98.7/98.9/96.2 | 98.7/53.2/55.7/94.5 — [Dinomaly](https://arxiv.org/html/2405.14325)
- Real-IAD: RD4AD 82.4/79.0/73.9 | 97.3/25.0/32.7/89.6; SimpleNet 57.2/53.4/61.5 | 75.7/2.8/6.5/39.0; UniAD 83.0/80.9/74.3 | 97.3/21.1/29.2/86.7; MambaAD 86.3/84.6/77.0 | 98.5/33.0/38.7/90.5; Dinomaly 89.3/86.8/80.2 | 98.8/42.8/47.1/93.9 — [Dinomaly](https://arxiv.org/html/2405.14325)
- INP-Former multi-class: MVTec AD 99.7/99.9/99.2 | 98.5/71.0/69.7/94.9; VisA 98.9/99.0/96.6 | 98.9/51.2/54.7/94.4; Real-IAD 90.5/88.1/81.5 | 99.0/47.5/50.3/95.0 — [INP-Former](https://arxiv.org/html/2503.02424)
- RAD (training-free) multi-class: MVTec AD 99.6/99.8/99.0 | 98.5/75.6/71.3/94.9; VisA 97.5/97.3/95.2 | 99.1/55.3/57.5/94.6; Real-IAD 92.0/89.9/83.0 | 99.1/53.7/54.2/96.1. It is below Dinomaly on VisA image metrics and above it on all pixel metrics — [RAD](https://arxiv.org/html/2601.22763v3)

Full-shot, one model per class

- I-AUROC / AUPRO on MVTec AD, VisA, Real-IAD: PatchCore 99.1/93.5, 94.7/91.8, 89.4/91.5; RD4AD 98.5/93.9, 96.0/70.9, 87.1/93.8; SimpleNet 99.6/90.0, 97.1/90.7, 88.5/84.6; Dinomaly 99.7/95.0, 98.9/95.1, 92.0/95.1 — [Dinomaly](https://arxiv.org/html/2405.14325)
- INP-Former single-class (I-AUROC / P-AP / AUPRO): MVTec AD 99.7/70.2/95.4, VisA 98.5/49.2/93.8, Real-IAD 92.1/48.1/95.6; PatchCore in the same table 99.1/56.1/93.5 and 95.1/40.1/91.2 — [INP-Former](https://arxiv.org/html/2503.02424)
- AnomalyDINO-S full-shot (I-AUROC / P-AUROC / P-F1 / PRO): MVTec AD 99.5/98.2/64.3/95.0 at 672 px; VisA 97.6/98.8/53.8/96.1 — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- SubspaceAD full-shot (I-AUROC / P-AUROC / PRO): MVTec AD 99.2/98.2/95.6; VisA 98.2/99.1/96.9 — [SubspaceAD](https://arxiv.org/html/2602.23013)
- EfficientAD (WACV 2024 per its repository; arXiv 2023), image AUROC on MVTec AD / MVTec LOCO / VisA: EfficientAD-S 98.8/90.0/97.5, EfficientAD-M 99.1/90.7/98.1; PatchCore 98.7/80.3/94.3; PatchCore ensemble 99.3/79.4/97.7; SimpleNet 98.2/77.6/87.9; FastFlow 96.9/79.2/93.9; DSR 98.1/82.6/91.8; AST 98.9/83.4/94.9. Mean segmentation AU-PRO over the three collections: EfficientAD-M 93.3, EfficientAD-S 92.5, PatchCore 80.9, SimpleNet 74.4, DSR 78.6, FastFlow 86.5 — [EfficientAD, arXiv 2303.14535](https://arxiv.org/html/2303.14535)
- GLASS (synthetic anomalies by gradient ascent, 2024) claims 99.9 detection AUROC on MVTec AD and says it targets "weak defects" that resemble normal regions; it reports an industrial woven-fabric application — [GLASS, arXiv 2407.09359](https://arxiv.org/abs/2407.09359)

Real-IAD few-shot (harder, 30 objects, multi-view)

- 4-shot (image AUROC | pixel AUROC/AP/F1/AUPRO): PatchCore 66.0 | 92.9/9.8/16.1/68.6; WinCLIP 73.0 | 93.8/13.3/21.0/76.4; PromptAD 59.7 | 86.9/8.7/16.1/61.9; INP-Former 76.7 | 97.3/32.2/36.7/89.0. 1-shot INP-Former 67.5 | 94.9/20.0/25.8/81.8 — [INP-Former](https://arxiv.org/html/2503.02424)
- Real-IAD multi-view I-AUROC / PRO: AnomalyDINO 80.6/90.1 (1-shot), 85.3/91.8 (4-shot); DuoAD (DINOv2) 85.1/95.5 and 88.7/96.3; WinCLIP 74.7 and 73.6 I-AUROC; AdaptCLIP 81.8 and 82.6 — [DuoAD](https://www.alphaxiv.org/abs/2607.23924.md)

MVTec AD 2 (details in section 5)

- Original 2025 baselines at 256x256, mean AU-PRO@0.05 on TEST_priv / TEST_priv,mix: EfficientAD 30.8/19.2, PatchCore 28.8/26.0, RD++ 27.1/25.2, RD 26.4/25.0, MSFlow 24.3/11.9, SimpleNet 21.1/13.7, DSR 20.3/17.4 — [MVTec AD 2, arXiv 2503.21622](https://arxiv.org/html/2503.21622)
- 2026 best (SuperADD, training-free DINOv3 memory bank): SegF1 57.42 / 54.35 on TEST_priv / TEST_priv,mix — [SuperADD, arXiv 2605.14808](https://arxiv.org/html/2605.14808)

### Inferences
- For 5-20 good images per SKU the published evidence favours frozen DINO features plus a non-parametric normal model. The gap to trained reconstruction models closes only at full-shot (hundreds of images), and RAD shows a training-free bank can match or beat Dinomaly there on pixel metrics.
- Among few-shot methods the differences at the top are small (MVTec 1-shot I-AUROC 96.5 to 97.8, PRO 92 to 95). The current baseline already sits in this family, so the realistic gains are from details (multi-layer fusion, attention reweighting, augmentation of the bank, resolution), not from switching family.
- Pixel F1/AP is where methods separate (MVTec P-F1: PatchCore few-shot about 50, AnomalyDINO about 60, DuoAD/INP-Former about 63-66, full-shot Dinomaly 69, RAD 71). This is the metric closest to "find a thin scratch", so it should weigh more than image AUROC when choosing.
- Zero-shot CLIP methods are not suitable as the main detector here: their localisation is far weaker and the product (white print on clear film) is unlike what text prompts describe well.
- None of these benchmarks contains a printed label on transparent film; numbers transfer only roughly. A small in-house test set with real pen marks, dust and scratches is needed to rank candidates.

### Gaps
- No sourced numbers were retrieved for DRAEM, STFPM, PaDiM full-shot (canonical WideResNet setting), RD++ on MVTec/VisA, AdaCLIP, InCTRL, AnomalyGPT few-shot beyond the UniVAD table, MetaUAS benchmark scores, or AnomalyCLIP in few-shot mode. They are not invented here.
- 8-shot numbers exist only for AnomalyDINO, APRIL-GAN, GraphCore and ADP in the sources read; most newer papers stop at 4-shot.
- The INP-Former 1-shot and 2-shot baseline rows (SPADE, PatchCore) appear swapped compared with the AnomalyDINO/SubspaceAD tables in the extracted text; check the PDF.
- Dinomaly's class-separated table as extracted shows odd pixel AUROC entries (99.9 for MVTec, 90.1 for RD4AD on VisA), so only its I-AUROC and AUPRO are quoted above.
- GLASS pixel-level numbers and its VisA/MPDD results were not retrieved (abstract only).
- No ADer benchmark or Papers-with-Code style leaderboard was consulted directly.

## 2. Which methods need per-SKU training, and what are speed and memory?

### Takeaway
Memory-bank and subspace methods on frozen DINO features need no training: adding a SKU means one forward pass over its good images (about 0.5 s for one image, about 6 s for 16 images in AnomalyDINO). Reconstruction and student-teacher methods need thousands of iterations per SKU or a multi-class retrain. Inference for ViT-S/B at 448-672 px is 40-140 ms per image on one GPU.

### Cited Findings
- AnomalyDINO is training-free with no fine-tuning or meta-learning. On one NVIDIA A40 (MVTec AD, ViT-S, 448 px): inference 0.060 s/image at 1-shot, 0.063 s at 16-shot, 0.130 s full-shot; building the memory bank takes 0.52 s (1-shot), 6.02 s (16-shot), 93.7 s (full-shot) — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- AnomalyDINO inference by resolution (ViT-S): 0.043 s at 224, 0.060 s at 448, 0.086 s at 672, 0.141 s at 896. By model size at 448: S 0.060 s, B 0.084 s, L 0.141 s, G 0.306 s. Masking plus rotation augmentation adds about 23% — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- SubspaceAD: training-free; stores only a PCA mean and basis, under 1 MB per category; about 300 ms per 672x672 image on an H100 with DINOv2-G (226 ms backbone + 74 ms scoring). Its table lists backbone times S 36 ms, B 56 ms, L 112 ms, G 127 ms at 448 px. AnomalyDINO's scoring head measured in the same setup: about 80 ms — [SubspaceAD](https://arxiv.org/html/2602.23013)
- RAD: training-free; the uncompressed multi-layer bank is large (32.6 GB for MVTec AD full-shot, 40.7 GB VisA, 60.8 GB Real-IAD), a coreset cuts MVTec to 0.37 GB with small loss (P-AUROC 98.50 to 98.06, P-AP 75.6 to 73.9, AUPRO 94.90 to 94.75). Global [CLS] retrieval of reference images first cuts latency from 293.7 ms to 44.7 ms per image on MVTec AD — [RAD](https://arxiv.org/html/2601.22763v3)
- SuperAD and SuperADD are training-free; SuperAD runs on one 24 GB GPU with DINOv2 ViT-L/14 and a 16-image memory bank per class — [SuperAD, arXiv 2505.19750](https://arxiv.org/html/2505.19750v2); [SuperADD](https://arxiv.org/html/2605.14808)
- UniVAD is training-free but chains RAM, Grounding DINO, SAM, CLIP-L and DINOv2-G; its authors state that inference latency limits real-time use — [UniVAD](https://arxiv.org/html/2412.03342)
- FoundAD trains a projector (11.8 M parameters, 97.8 M total with DINOv3) on normal images with synthetic anomalies; inference 128.7 ms per image on an RTX 3090 using 1386 MiB VRAM. Training time is not stated in the summary read — [FoundAD summary](https://en.papernotes.org/ICLR2026/anomaly_detection/foundation_visual_encoders_are_secretly_few-shot_anomaly_detectors/)
- Dinomaly needs training: 10,000 iterations for multi-class MVTec AD / VisA, 50,000 for Real-IAD, 5,000 iterations per class in class-separated mode, on an RTX 3090. Inference throughput on RTX 3090 (batch 16, 392 px): ViT-S 153.6 img/s, ViT-B 58.1 img/s, ViT-L 24.2 img/s — [Dinomaly](https://arxiv.org/html/2405.14325)
- INP-Former trains for 200 epochs (RTX 4090); 139.8 M parameters / 98.0 G FLOPs for the base model, 35.1 M / 24.6 G for the small one — [INP-Former](https://arxiv.org/html/2503.02424)
- RoBiS (INP-Former with sliding windows, MVTec AD 2) trains one model per category for 200 epochs — [RoBiS, arXiv 2505.21152](https://arxiv.org/html/2505.21152v1)
- EfficientAD trains for 70,000 iterations per scenario at 256x256; its limitations section mentions twenty minutes of autoencoder training in their setup. Latency on an RTX A6000: EfficientAD-S 2.2 ms (614 img/s), EfficientAD-M 4.5 ms (269 img/s); PatchCore 32 ms, PatchCore ensemble 148 ms, SimpleNet 12 ms, FastFlow 17 ms, DSR 17 ms, AST 53 ms — [EfficientAD](https://arxiv.org/html/2303.14535)
- PatchCore with WideResNet at near-native MVTec AD 2 resolution takes about 2 s per image, and SimpleNet runs out of memory at the largest size — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- Tiled ensemble training time for PatchCore (ResNet18): 213 s for one 256 px model, 2273 s for one 512 px model, 1811 s for nine 256 px tile models — [Tiled ensemble, arXiv 2403.04932](https://arxiv.org/html/2403.04932)
- The VAND 4.0 organisers state that top methods rely on heavy architectures and that efficiency is largely neglected; they propose an efficiency-aware metric — [VAND 4.0 report](https://chatpaper.com/paper/357114)
- RegAD and MetaUAS train one category-agnostic model once and need no retraining for a new category — [RegAD](https://arxiv.org/abs/2207.07361); [MetaUAS](https://arxiv.org/abs/2505.09265)

### Inferences
- "Add a SKU without long training" rules out Dinomaly, INP-Former, RD4AD, EfficientAD, SimpleNet, GLASS and DRAEM/DSR as the primary method unless minutes-to-hours of training per SKU is acceptable. Training-free options: AnomalyDINO-style kNN, SubspaceAD, DuoAD, RAD, SuperAD/SuperADD; train-once options: RegAD, MetaUAS, FoundAD (if its projector is category-agnostic).
- Rough timing for this project (estimate, not measured): a 5 MP frame cut into 448-518 px tiles gives about 20-30 tiles; at 60-85 ms per tile for ViT-S/B unbatched that is roughly 1.5-2.5 s, and batching the tiles on a desktop GPU should bring it well under that. ViT-L/H backbones as used by the challenge winners would exceed the time budget unless only label regions are processed.
- With 5-20 images per SKU the memory bank is small (a few tens of thousands of patch vectors per layer), so RAD's storage problem does not apply; coreset subsampling is optional.
- A PCA subspace per tile position (SubspaceAD idea) would make per-SKU storage tiny and scoring independent of bank size, at the cost of a fixed linear model of "normal".

### Gaps
- No measured GPU memory figures for AnomalyDINO, Dinomaly or INP-Former inference were found in the sources read.
- Exact wall-clock training time per class for Dinomaly, INP-Former, SimpleNet, GLASS, DRAEM, DSR, RD4AD, STFPM was not found.
- No timing for tiled inference on 5 MP images with ViT-B was found; the estimate above must be measured locally.

## 3. Which methods are strongest for very small / thin defects and high-resolution inputs (resolution, tiling, patch size)?

### Takeaway
Every source agrees that effective resolution is the main lever for small defects: scores rise sharply with input size, and the top MVTec AD 2 entries all process the image in overlapping high-resolution windows with a DINO ViT. No controlled comparison of patch size 14 vs 8 vs 16 was found.

### Cited Findings
- MVTec AD 2 authors: for a 10-pixel defect in a 5 MP image, a 5% false-positive-rate limit still allows a wrongly segmented area more than 25,000 times larger than the defect; this is why they use AU-PRO@0.05 and a thresholded SegF1 — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- MVTec AD 2 mean AU-PRO@0.05 (TEST_priv) by input size: PatchCore 28.8 at 256x256, 41.9 at 512x512, 62.3 at half the original resolution; RD 26.4 / 37.3 / 53.7; RD++ 27.1 / 39.5 / 55.3; EfficientAD 30.8 / 40.0 / 50.8; MSFlow 24.3 / 39.2 / 58.2; SimpleNet 21.1 / 32.0 / 45.7 — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- The same paper notes runtime and memory grow by more than an order of magnitude at large inputs and lists efficient tiling and multi-scale strategies as future work — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- SuperADD (winner of the VAND 4.0 industrial track per anomalib's release notes): DINOv3 ViT-H+/16, images downscaled by 0.625, cut into 640x640 windows with at least 128 px overlap, redundant predictions at window borders discarded, features from blocks 7/15/23/31, patch kNN memory bank. SegF1 62.61 (public) / 57.42 (private) / 54.35 (private mixed lighting). The authors say window-wise processing helped most classes but not fabric or vial; no ablation numbers given — [SuperADD](https://arxiv.org/html/2605.14808); [anomalib releases](https://github.com/open-edge-platform/anomalib/releases)
- RoBiS (VAND 3.0, 2025): 1024x1024 sliding windows with 10% overlap, each resized to 518x518 for INP-Former with DINOv2-R ViT-B/14; SegF1 51.00 / 46.52 on TEST_priv / TEST_priv,mix versus 21.8 / 9.0 for the best original baseline (MSFlow). The paper attributes +3.7 F1-max to the window cropping and +1.9 to noise/exposure augmentation — [RoBiS](https://arxiv.org/html/2505.21152v1)
- SuperAD (VAND 3.0, 2025): DINOv2 ViT-L/14 on the whole image with the shorter side at 672 px, no tiling; SegF1 47.18 / 42.51 — [SuperAD](https://arxiv.org/html/2505.19750v2)
- AnomalyDINO, 448 vs 672 px: MVTec AD 1-shot P-F1 57.9 to 60.2 and PRO 91.7 to 92.7; VisA 1-shot I-AUROC 85.6 to 87.4, P-F1 41.9 to 45.1, PRO 90.7 to 92.5; full-shot VisA P-F1 50.5 to 53.8 — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- Dinomaly input-size ablation on MVTec AD (I-AUROC / P-AUROC / AUPRO): 280 px 99.6/98.2/93.6, 336 px 99.6/98.3/94.2, 392 px 99.6/98.4/94.8: image-level is flat, localisation improves with size — [Dinomaly](https://arxiv.org/html/2405.14325)
- SubspaceAD reports that gains saturate above 256 px with 448-672 px similar and a peak at 672 px (figure only) — [SubspaceAD](https://arxiv.org/html/2602.23013)
- RAD: moving from 224 to 448 px improves P-AUPRO for almost all encoders tested — [RAD](https://arxiv.org/html/2601.22763v3)
- UniVAD reports a slight drop at lower resolutions (224 and 336 px) — [UniVAD](https://arxiv.org/html/2412.03342)
- Tiled ensemble (Rolih, Ameln, Vaidya, Akcay; implemented in anomalib): 512x512 image split into nine overlapping 256x256 tiles, one model per tile location. Image AUROC / AUPRO, PatchCore (ResNet18): MVTec AD 97.7/92.8 (single 256), 98.0/94.5 (single 512), 97.8/95.3 (9-tile ensemble); VisA 92.0/87.7, 97.2/94.0, 95.4/93.7. PaDiM VisA 83.7/82.1, 81.9/86.9, 86.3/86.9. The gain is largest on VisA, whose anomalies are small, and overlapping tiles beat non-overlapping ones — [Tiled ensemble](https://arxiv.org/html/2403.04932)
- Thin-defect weakness remains even for the best method: on MVTec AD 2 "Can" (reflective, tiny defects) SuperADD gets SegF1 11.59 / 0.38, RoBiS 1.86 / 0.84, SuperAD 17.3 / 1.9 — [SuperADD](https://arxiv.org/html/2605.14808)
- Pixel AP/F1 gap between CNN-era and ViT-era models on VisA (small defects): SimpleNet P-AP 34.7, RD4AD 38.0, UniAD 33.7, MambaAD 39.4, Dinomaly 53.2, RAD 55.3 — [Dinomaly](https://arxiv.org/html/2405.14325); [RAD](https://arxiv.org/html/2601.22763v3)
- GLASS explicitly targets weak defects that are close to normal appearance by synthesising near-in-distribution anomalies — [GLASS](https://arxiv.org/abs/2407.09359)

### Inferences
- At 0.17 mm/px a 14 px patch covers about 2.4 mm. A scratch 2-3 px wide fills well under a quarter of a patch even at native resolution, so any downscaling before the ViT throws away the signal. Native-resolution (or upsampled) overlapping tiles are the right design; the challenge winners downscale by only 0.625 from much larger images.
- The current per-tile design matches what SuperADD and RoBiS do. Two cheap upgrades supported by the sources: overlap the tiles and drop or average predictions near tile borders (ViT border patches are less reliable), and aggregate the image score as the mean of the top 1% of patch distances (AnomalyDINO: 96.5 vs 95.0 I-AUROC against plain max).
- Upsampling a tile before the ViT (for example 1.5x to 2x for the lot-code line and letter edges) is the practical substitute for a smaller patch size, since no public DINOv2/DINOv3 ViT with patch 8 exists in the sources read.
- If thin scratches on clear film remain undetected at feature level, a complementary pixel-level check (for example a trained synthetic-anomaly discriminator like GLASS, or a classical difference image against an aligned golden reference) is the direction the "weak defect" literature points to. This is a hypothesis; no source tested it on transparent film.

### Gaps
- No controlled ablation of ViT patch size 14 vs 16 vs 8 for anomaly detection was found. DINOv2 (patch 14) and DINOv3 (patch 16) comparisons always change the weights too.
- No paper reports defect-size-binned recall (for example recall for defects under 5 px wide) for the few-shot methods.
- SuperADD gives no ablation with/without tiling and no runtime; RoBiS gives no direct sliding-window vs resize table that could be extracted.

## 4. Does the backbone matter (DINOv2 vs DINOv3 vs CLIP vs WideResNet-50), and which ViT layers work best?

### Takeaway
The backbone matters a lot between families (self-supervised DINO features are clearly better than CLIP and ImageNet CNN/ViT features for patch-level anomaly detection). DINOv2 vs DINOv3 is close and depends on using several layers. Mid-to-late layers beat the last layer in every ablation found, which supports the baseline's choice of blocks 5..10 of ViT-B.

### Cited Findings
- AnomalyDINO: DINOv2 gives at least +4 points AUROC over an ImageNet-pretrained ViT on MVTec AD and VisA (224 px, no pre/post-processing). The smallest model (ViT-S) is best on MVTec AD; larger ones are slightly better on VisA for 8 shots and more — [AnomalyDINO](https://arxiv.org/html/2405.14529v3)
- Dinomaly encoder ablation (MVTec AD I-AUROC, same decoder): DINOv2-R 99.60, DINOv2 99.55, iBOT 99.22, DINO 98.97, D-iGPT 98.75, MoCo v3 98.47, DeiT 98.19, MAE 96.27. Size: ViT-S 99.26 / AUPRO 94.36, ViT-B 99.60 / 94.79, ViT-L 99.77 / 95.09 — [Dinomaly](https://arxiv.org/html/2405.14325)
- RAD encoder ranking by P-AUPRO: DINOv3 highest, then DINOv2, iBOT, DINO, D-iGPT, MoCo v3, DeiT, with masked-image-modelling encoders (MAE, BEiT, BEiTv2) lowest; the ranking is the same at 224 and 448 px — [RAD](https://arxiv.org/html/2601.22763v3)
- FoundAD 1-shot MVTec AD (I-AUROC / P-AUROC / PRO) by frozen encoder: DINOv3 96.1/96.8/92.8, DINOv2 95.2/96.4/92.5, DINO 88.3/96.2/87.8, SigLIP 87.8/86.0/71.1, CLIP 79.0/90.9/70.9 — [FoundAD summary](https://en.papernotes.org/ICLR2026/anomaly_detection/foundation_visual_encoders_are_secretly_few-shot_anomaly_detectors/)
- DuoAD, plain patch kNN without its modules (1-shot I-AUROC / PRO): DINOv2-B/14 94.3/90.1 (MVTec), 83.0/80.5 (VisA); DINOv3-B/16 92.1/85.5 and 81.9/72.9; MetaCLIP 2 B/16 87.3/80.4 and 78.2/71.4. With multi-layer fusion only: DINOv2 95.6/93.3 and 90.0/90.5; DINOv3 95.0/93.8 and 90.5/90.4. With all modules: DINOv2 97.3/94.3, 92.7/93.2; DINOv3 97.7/94.7, 93.2/93.2. The authors conclude multi-level aggregation is essential for DINOv3 — [DuoAD](https://www.alphaxiv.org/abs/2607.23924.md)
- DuoAD single-layer ablation, ViT-B, 1-shot I-AUROC / PRO. DINOv2 on MVTec: layer 7 94.8/92.8, 8 96.4/93.5, 9 96.7/93.7, 10 97.1/93.8, 11 96.8/93.7, 12 96.0/92.9. DINOv2 on VisA: layer 7 88.6/91.3, 10 92.9/93.0, 12 88.2/88.8. DINOv3 on VisA: layer 10 93.6/93.2, 12 85.4/85.2. Fusing {8,10,12} gives 97.3/94.3 (DINOv2 MVTec) — [DuoAD](https://www.alphaxiv.org/abs/2607.23924.md)
- SubspaceAD layer ablation with DINOv2-G (4-shot, I-AUROC / PRO): mean of middle 7 layers (22-28 of 40) 98.4/95.0 MVTec and 94.1/95.1 VisA; mean of final 7 layers 98.2/93.3 and 91.9/93.5; last layer only 97.6/92.5 and 89.3/91.2 — [SubspaceAD](https://arxiv.org/html/2602.23013)
- SubspaceAD with DINOv3-7B is generally below DINOv2-G: 1-shot MVTec 96.2/97.3/94.2 and VisA 92.8/98.0/94.7 (I-AUROC / P-AUROC / PRO) versus 97.1/97.5/94.5 and 93.2/98.2/95.5 — [SubspaceAD](https://arxiv.org/html/2602.23013)
- Layer choices used by leading methods: SuperAD blocks 6/12/18/24 of DINOv2 ViT-L; SuperADD blocks 7/15/23/31 of DINOv3 ViT-H+/16; RAD layers 4/7/10/12 of DINOv3 ViT-B/16 with equal weights; Dinomaly the middle 8 layers of ViT-B; FoundAD best single layer 10 of ViT-B — [SuperAD](https://arxiv.org/html/2505.19750v2); [SuperADD](https://arxiv.org/html/2605.14808); [RAD](https://arxiv.org/html/2601.22763v3); [Dinomaly](https://arxiv.org/html/2405.14325); [FoundAD search summary](https://www.emergentmind.com/topics/foundad)
- In the VAND 4.0 industrial track (MVTec AD 2, CVPR 2026) DINOv3 backbones clearly dominate — [VAND 4.0 report](https://chatpaper.com/paper/357114)
- WideResNet/ResNet PatchCore in few-shot is far behind DINO kNN in every table: MVTec 1-shot 83.4 I-AUROC / 79.7 PRO vs AnomalyDINO 96.6 / 92.7 — [SubspaceAD](https://arxiv.org/html/2602.23013)

### Inferences
- Staying on DINOv2 ViT-B/14 is defensible. Evidence for DINOv3 is mixed: better in FoundAD, RAD and the challenge, equal or worse in SubspaceAD and in DuoAD's plain-kNN setting. A swap to DINOv3 ViT-B/16 is cheap to test but should be combined with multi-layer fusion and judged on the project's own images.
- Layers: for a 12-block ViT, layer 10 is the best single layer and the final layer is the worst of the late ones; fusing a few layers (per-layer distance maps averaged) is consistently better than one layer. The baseline's blocks 5..10 cover the right range; whether to add 11-12 or drop 5-7 is an empirical question (DuoAD shows layer 7 already weaker than 8-11 for image AUROC, but shallow layers carry the fine texture that thin scratches need, which benchmarks do not isolate).
- Bigger backbones help on VisA-like small defects but cost latency linearly; ViT-S is about 30% faster than ViT-B and nearly as good on MVTec-like data.
- CLIP-family features are a poor choice for the patch-matching core.

### Gaps
- No ablation isolates shallow layers (1-6) for thin-scratch detection.
- No same-resolution, same-patch-grid DINOv2 vs DINOv3 comparison on MVTec AD 2 was found; the challenge report excerpt only states dominance.
- Whether AnomalyDINO uses only the final layer was not confirmed from the text read.
- FoundAD figures come from a third-party summary, not the paper.

## 5. What does MVTec AD 2 show about methods under hard conditions (transparent, reflective, lighting change)?

### Takeaway
All 2022-2023 methods collapse on MVTec AD 2 at standard resolution, and several lose half their score when lighting differs from training. By 2026 the best results come from training-free DINO memory banks with overlapping high-resolution windows and intensity augmentation, at about 57 SegF1; transparent objects are handled reasonably, reflective cans with tiny defects are still unsolved.

### Cited Findings
- Dataset: 8 scenarios, 8,004 images, object sizes from 1400x1900 (vial) to 4224x1056 (sheet metal), most at 2448x2048 (5 MP). Vial and fruit jelly are transparent; can and sheet metal are reflective; rice has transparent plastic contaminants. Training uses regular lighting only; each object was captured under at least four lighting conditions and TEST_priv,mix mixes seen and unseen lighting — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- Baseline mean AU-PRO@0.05 at 256x256 (TEST_priv / TEST_priv,mix): PatchCore 28.8/26.0, RD 26.4/25.0, RD++ 27.1/25.2, EfficientAD 30.8/19.2, MSFlow 24.3/11.9, SimpleNet 21.1/13.7, DSR 20.3/17.4 — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- Lighting robustness: RD drops only 1.4 points under mixed lighting, PatchCore 2.8, while EfficientAD drops 11.6 and MSFlow 12.4 — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- Baseline pixel F1 (mean, TEST_priv / mix): PatchCore 3.7/3.4, RD 18.1/16.7, RD++ 19.2/16.7, EfficientAD 15.4/8.0, MSFlow 21.8/9.0, SimpleNet 17.7/8.7, DSR 10.9/9.5. Image-level F1: PatchCore 8.0/9.9, RD 70.8/72.9, EfficientAD 76.8/77.8, SimpleNet 73.3/77.1, DSR 78.0/78.0 — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- Transparent objects, baseline AU-PRO@0.05 at 256: Vial PatchCore 62.2/59.1, RD 62.1/60.3, RD++ 63.0/57.0, EfficientAD 55.6/47.7, MSFlow 38.0/4.0, SimpleNet 47.8/23.3. Fruit Jelly PatchCore 46.7/46.7, RD++ 54.4/54.4, EfficientAD 50.5/47.6 — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- Reflective can, baseline AU-PRO@0.05: PatchCore 4.7/4.6, EfficientAD 9.6/1.3, DSR 13.9/3.5 — [MVTec AD 2](https://arxiv.org/html/2503.21622)
- 2025 challenge (VAND 3.0) entries, SegF1 on TEST_priv / TEST_priv,mix: SuperAD (training-free DINOv2 ViT-L kNN, 16-image coreset bank) 47.18/42.51 with AU-PRO@0.05 60.51/58.37; RoBiS (trained INP-Former, sliding windows, noise and exposure augmentation, SAM-refined binarisation) 51.00/46.52 with AU-PRO@0.05 67.25/59.65 — [SuperAD](https://arxiv.org/html/2505.19750v2); [RoBiS](https://arxiv.org/html/2505.21152v1)
- 2026 (VAND 4.0): SuperADD 57.42/54.35; ISVL 53.81/51.43; ASEG 47.5/44.49 (as listed in the SuperADD paper). The organisers report the best regular-setting method at about 57 SegF1 and call robustness to distribution shift an open problem — [SuperADD](https://arxiv.org/html/2605.14808); [VAND 4.0 report](https://chatpaper.com/paper/357114)
- Per-object SegF1 (TEST_priv / mix) on transparent classes: Vial SuperADD 64.37/63.52, RoBiS 48.73/48.83, SuperAD 42.8/40.8, EfficientAD 30.5/26.5, PatchCore 2.3/2.2. Fruit Jelly SuperADD 68.03/67.75, ISVL 68.1/67.76, RoBiS 53.63/52.62, SuperAD 41.3/40.9 — [SuperADD](https://arxiv.org/html/2605.14808)
- Reflective classes: Can SuperADD 11.59/0.38, ISVL 17.03/9.21, SuperAD 17.3/1.9, RoBiS 1.86/0.84; Sheet Metal RoBiS 70.98/70.92, ISVL 69.69/69.74, SuperADD 55.74/55.47 — [SuperADD](https://arxiv.org/html/2605.14808)
- What SuperADD changed to survive lighting shift: intensity augmentation of reference images (pixel values multiplied by a random factor in [0.8, 1.2]), overlapping window processing, kNN-density-based memory-bank subsampling for better coverage, iterative morphological closing; one hyperparameter set for all classes, with the threshold taken from held-out good images (95th percentile of anomaly-map values times 1.3-1.5) — [SuperADD](https://arxiv.org/html/2605.14808)
- SuperAD applied PCA foreground masking for Vial and Wallplugs and hole filling for Fabric and Walnuts, i.e. per-class handling — [SuperAD](https://arxiv.org/html/2505.19750v2)

### Inferences
- The mixed-lighting drop of the DINO memory-bank methods is small for transparent classes (Vial 64.4 to 63.5, Fruit Jelly 68.0 to 67.8) once intensity augmentation is in the bank. For this project, where exposure is fixed but lighting has been re-adjusted before, adding brightness-scaled copies of the good images to the bank is a cheap protection.
- PatchCore's near-zero pixel F1 at 256 px on this dataset is a resolution and thresholding failure, not a failure of memory banks as such: the same idea with DINO features at high resolution wins the benchmark.
- "Can" shows a limit that is relevant here: small defects on a specular, lighting-dependent surface are missed by every method. Clear film with glare has the same character, so optical control (polarisers, dark-field or back-light for the film area) will matter more than the model choice for scratches on the film.
- The winning recipes set the threshold from held-out good images only, which fits the "no defect labels" constraint.

### Gaps
- No Dinomaly or plain INP-Former result on MVTec AD 2 under the official protocol was found (RoBiS is a modified INP-Former).
- The VAND 4.0 report was read through a summary page; team names, per-team methods and the efficiency metric values were not available.
- No public result separates "defect on the transparent part" from "defect on the opaque part" of an object.

## 6. Is there evidence that position-aware or aligned matching helps for aligned objects?

### Takeaway
There is supporting but mostly indirect evidence. Several strong methods restrict matching spatially (RAD's local neighbourhood radius, PNI's position histograms, FAPM's per-location banks, per-tile banks in the tiled ensemble), and RAD tunes the radius per dataset, with a small radius for aligned data. A clean ablation with numbers for "global kNN vs position-constrained kNN" on aligned objects was not retrieved.

### Cited Findings
- RAD (training-free, top pixel metrics in 2026) matches each query patch only against reference patches within an l-infinity radius rho of the same grid position: rho = 1 for MVTec AD, 2 for VisA, 0 (same position only) for Real-IAD, 1 for 3D-ADAM. It first retrieves globally similar reference images by [CLS] similarity, then does the local patch match — [RAD](https://arxiv.org/html/2601.22763v3)
- PNI (ICCV 2023 per its repository; arXiv 2022) models normal features conditioned on neighbourhood features and on position (a histogram of representative features at each position) and reports 99.56 image AUROC and 98.98 pixel AUROC on MVTec AD, state of the art at the time — [PNI, arXiv 2211.12634](https://arxiv.org/abs/2211.12634)
- FAPM stores patch embeddings per location (patch-wise memory, layers split into grids), on the stated assumption that industrial images are roughly aligned — [FAPM, arXiv 2211.07381](https://arxiv.org/pdf/2211.07381)
- PaDiM (one Gaussian per position) is weak in few-shot: VisA 1/2/4-shot I-AUROC 62.8/67.4/72.8 and PRO 64.3/70.1/72.6, below position-free PatchCore (79.9/81.6/85.3; PRO 80.5/82.6/84.9); MVTec 4-shot 80.4 vs 88.8 — [AnomalyDINO](https://arxiv.org/html/2405.14529v3); [INP-Former](https://arxiv.org/html/2503.02424)
- Tiled ensemble restricts PatchCore's kNN search to embeddings from the same tile location. Localisation improved on MVTec AD (AUPRO 95.3 vs 94.5 for a single 512 px model) but image AUROC was slightly lower (97.8 vs 98.0), and on VisA both were slightly lower (95.4/93.7 vs 97.2/94.0); the authors attribute this to the kNN search being limited to within-tile embeddings — [Tiled ensemble](https://arxiv.org/html/2403.04932)
- RegAD aligns features by a learned registration (spatial transformer) before comparing with support images, and this category-agnostic registration gave 3-8 AUC points over 2022 few-shot baselines — [RegAD](https://arxiv.org/abs/2207.07361)
- MetaUAS includes a soft feature alignment module to handle geometric differences between the normal prompt image and the query — [MetaUAS](https://arxiv.org/abs/2505.09265)
- Rotation augmentation of the reference set (the opposite of a position constraint) helps unaligned objects and hurts aligned ones: AnomalyDINO Screw 1-shot rises from 65.6 to 89.2 with rotations, while PCB1/2/3, Macaroni1 and Transistor get worse; SubspaceAD Transistor drops from 96.6 to 89.1 I-AUROC with rotations; DuoAD's self-calibration rejects rotation/flip for Capsule, Pill, Transistor, Toothbrush, Zipper — [AnomalyDINO](https://arxiv.org/html/2405.14529v3); [SubspaceAD](https://arxiv.org/html/2602.23013); [DuoAD](https://www.alphaxiv.org/abs/2607.23924.md)
- Patch-based nearest-neighbour methods miss "wrong content at the right-looking place" anomalies when matching is global: AnomalyDINO scores 50.2 AUROC (chance) on the MVTec "cable swap" defect; SubspaceAD lists missing/misplaced components (transistor, 1-shot PRO 68.9) as a limitation — [AnomalyDINO](https://arxiv.org/html/2405.14529v3); [SubspaceAD](https://arxiv.org/html/2602.23013)

### Inferences
- For a rectified label at fixed position and scale, position-constrained matching is the right prior and is in line with the strongest 2026 training-free method (RAD uses radius 0-2 patches). It also covers the failure that global kNN has: a pen stroke that looks like a letter edge, or a missing piece of a letter, matches some other good patch elsewhere in the image but not the patch at that position.
- PaDiM's poor few-shot numbers say that a strict one-Gaussian-per-position model is too brittle with few samples and residual misalignment. A small search window (plus or minus 1-2 patches) around the position, as in RAD, is the safer form than exact same-cell matching. The radius should cover the residual registration error of the rectification in patch units.
- Do not add rotation or flip augmentation to the bank for this product; brightness and small sub-patch shift augmentations are the ones consistent with an aligned setup. Sub-patch shifts (for example 3-7 px) are a hypothesis from first principles to reduce patch-grid aliasing at letter edges; no source tested them.
- The tiled-ensemble result suggests that a per-tile bank costs a little image-level accuracy on objects that are not aligned (VisA), which does not apply here.

### Gaps
- RAD's Appendix H ablation of the spatial radius was not retrieved, so the size of the gain from the constraint is unknown.
- No paper compares global vs position-constrained kNN on DINOv2 features in the few-shot setting with numbers.
- PNI's improvement over PatchCore is not quantified in the abstract; RegAD per-shot numbers were not retrieved.
- No evidence was found on how much registration error position-constrained matching tolerates.

## 7. What is available in anomalib v2.x and how mature is it?

### Takeaway
anomalib (latest release seen: v2.7.0, 6 October) now ships most of the relevant models, including AnomalyDINO, Dinomaly, INP-Former, GLASS, FoundAD, RAD, SuperADD, EfficientAD, PatchCore and WinCLIP, with DINOv3 backbones through timm. The ViT-based models were added within the last year and have had breaking changes in two recent releases, so they are usable as references but not yet stable APIs.

### Cited Findings
- Image models listed in the current docs (33): AnomalyVFM, AnomalyDINO, CFA, CFM, C-Flow, CS-Flow, DFKDE, DFM, Dinomaly, DRAEM, DSR, EfficientAD, INP-Former, FastFlow, FoundAD, FRE, GANomaly, GLASS, GeneralAD, L2BT, MH-PatchCore, PaDiM, PatchCore, PatchFlow, RAD, Reverse Distillation, STFPM, SuperSimpleNet, U-Flow, UniNet, VLM-AD, SuperADD, WinCLIP — [anomalib image models](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/index.html)
- Not in that list: SimpleNet (only SuperSimpleNet), RD++, UniAD, MambaAD, ViTAD, AnomalyCLIP, APRIL-GAN, AnomalyGPT, PromptAD, AdaCLIP, InCTRL, UniVAD, MetaUAS, SubspaceAD, DuoAD, SPADE, RegAD — [anomalib image models](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/index.html)
- Release history (year not shown on the page; dates as displayed): v2.4.0 (24 Apr) added L2BT, GeneralAD, PatchFlow and Dinomaly's context-aware recentering from Dinomaly2; v2.4.2 half-precision training for Dinomaly; v2.5.0 (29 May) added INP-Former, GLASS, AnomalyVFM, CFM; v2.5.1 (17 Jul) moved AnomalyDINO, Dinomaly, INP-Former, L2BT and PatchFlow to timm feature extraction, which adds DINOv3 backbones such as `vit_large_patch16_dinov3.lvd1689m`; v2.6.0 (25 Jul) added SuperADD, described as the VAND 4.0 winning method, with `patch_size` and `patch_overlap` options; v2.7.0 (06 Oct) added MH-PatchCore, RAD, FoundAD — [anomalib releases](https://github.com/open-edge-platform/anomalib/releases)
- AnomalyDINO was introduced in v2.3.0 — [anomalib search summary, PyPI 2.3.0](https://pypi.org/project/anomalib/2.3.0/)
- Breaking changes: v2.5.1 changed AnomalyDINO's default encoder name and the encoder parameters of Dinomaly, INP-Former, L2BT and PatchFlow to timm names; v2.7.0 no longer loads pre-timm ViT checkpoints for Dinomaly, INP-Former, AnomalyDINO, PatchFlow and L2BT (retrain or re-export needed) and removed the legacy ONNX exporter in favour of the dynamo exporter — [anomalib releases](https://github.com/open-edge-platform/anomalib/releases)
- Recent bug fixes that indicate young code: SuperADD's async CPU copy corrupted stored embeddings (fixed in v2.7.0); random tiling ignored the configured tile width (fixed in v2.6.1) — [anomalib releases](https://github.com/open-edge-platform/anomalib/releases)
- anomalib's AnomalyDINO defaults: encoder `vit_small_patch14_dinov2`, `num_neighbours=1`, `masking=False`, `coreset_subsampling=False` (`sampling_ratio=0.1`), float32 or float16, default pre-processor resizes to 252x252, single-GPU only; the example trains on a full MVTec category and enables masking for capsule, hazelnut, pill, screw and toothbrush. No benchmark numbers are given on the page, and the docs describe the image score two ways (99th percentile vs mean of top 1%) — [anomalib AnomalyDINO docs](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/anomaly_dino.html)
- WinCLIP needs the optional `[clip]` extra — [anomalib search summary](https://pypi.org/project/anomalib/2.3.0/)
- The tiled ensemble (one model per tile location, overlapping tiles, seam smoothing) is implemented in anomalib and needs no change to the underlying model — [Tiled ensemble](https://arxiv.org/html/2403.04932)
- OpenVINO and dynamo-ONNX export gained `dynamic_shapes` support in v2.5.1 — [anomalib releases](https://github.com/open-edge-platform/anomalib/releases)

### Inferences
- Maturity tiers (judgement from release history, not from testing): long-standing and stable: PatchCore, PaDiM, STFPM, Reverse Distillation, EfficientAD, FastFlow, DRAEM, DSR, WinCLIP. Recent and still moving: Dinomaly, AnomalyDINO, INP-Former, GLASS, UniNet, SuperSimpleNet. Very new (2026, one or two releases old): SuperADD, RAD, FoundAD, MH-PatchCore, AnomalyVFM.
- anomalib's AnomalyDINO default (ViT-S at 252x252, global kNN, no position constraint) is far from what this project needs; the project's own implementation (ViT-B, multi-layer, tiles, position constraint) is already more specialised. anomalib is most useful here as a quick way to run comparison baselines (SuperADD, RAD, Dinomaly, INP-Former, EfficientAD, GLASS) on the same images, and SuperADD's overlapping-window code is the closest reference implementation.
- Pin the anomalib version if it is used: two of the last four feature releases broke checkpoints or parameter names of the ViT models.

### Gaps
- anomalib's own benchmark tables for these models were not found on the pages read, so no anomalib-reproduced numbers are given.
- The release page shows dates without years; v2.7.0 on 6 October is presumed to be 2026.
- Whether anomalib's RAD exposes the spatial radius and whether its AnomalyDINO supports tiling or position constraints was not checked in the code.
- Issue-tracker health (open bugs per model) was not reviewed.
