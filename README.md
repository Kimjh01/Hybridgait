# HybridGait: A Robust Multimodal Gait Recognition Pipeline Against Clothing Variations

> **Summary in One Line**
> **HybridGait** provides an end-to-end integrated pipeline — from **dataset preprocessing → DLCR-based (clothing reconstruction) augmentation → single-GPU training with OpenGait**.
> The paper introduces **ΔJoint (inter-frame joint difference)**, **Upper/Lower-body separation weights (UL)**, and **Cross-Attention / FlashAttention fusion**, achieving robust gait recognition under clothing and background variations on **CCVID, CASIA-B, GREW**, and custom in-the-wild datasets.

---

## Table of Contents

* [Background & Contributions](#background--contributions)
* [Repository Structure](#repository-structure)
* [Quick Start](#quick-start)
* [Dataset Preprocessing](#dataset-preprocessing)
* [DLCR-Based Clothing Augmentation](#dlcr-based-clothing-augmentation)
* [Model & Training (Single GPU)](#model--training-single-gpu)
* [References](#references)
* [Attribution & Original Source Notice](#attribution--original-source-notice)

---

## Background & Contributions

* **Problem**
  Silhouette-based gait recognition suffers from **clothing, carried objects, illumination, and viewpoint variations**.
  In contrast, skeleton-based approaches are robust to appearance changes but highly sensitive to **pose estimation quality and frame consistency**.
  Therefore, robust gait recognition requires **silhouette–skeleton fusion** and **dataset expansion reflecting real-world variability**.

* **Solution**

  1. Introduce **DLCR** for synthetic data augmentation to diversify clothing conditions and simulate real-world distributions.
  2. Employ **Cross-Attention (FlashAttention)** for fine-grained interaction between silhouette and skeleton modalities.
  3. Apply **ΔJoint + Upper/Lower (UL) weighting** to model temporal motion and emphasize lower-body gait dynamics.
  4. Refactor **OpenGait** for **single-GPU FP16 training** with reproducibility and efficiency.

---

## Repository Structure

```
HybridGait/
├── datasets_preprocess/
│   ├── casia_b/
│   ├── ccvid/
│   ├── ccvid_aug/
│   └── requirements.txt
│
├── dlcr/
│   ├── llava/
│   ├── llama/
│   ├── schp/
│   └── sd_inpaint/
│
├── opengait/
│   ├── configs/
│   ├── opengait/
│   └── logs/ or results/
│
├── opengait_ccvid/
└── README.md
```

> **Tip:** Use lowercase `snake_case` for all folders and files.
> Script naming convention: `verb_object.py` (e.g., `split_frames.py`, `make_probe_gallery.py`)

---

## Quick Start

```bash
# 0) Environment setup
conda create -n hybridgait python=3.10 -y
conda activate hybridgait
pip install -r datasets_preprocess/requirements.txt

# 1) Dataset preprocessing (example: CASIA-B)
python datasets_preprocess/casia_b/split_frames.py
python datasets_preprocess/casia_b/make_pose_gei.py
python datasets_preprocess/casia_b/rearrange_to_grew.py

# 2) (Optional) Augmented CCVID preprocessing
python datasets_preprocess/ccvid_aug/crop_augmented_k5.py
python datasets_preprocess/ccvid_aug/rearrange_to_grew.py
python datasets_preprocess/ccvid_aug/make_probe_gallery.py

# 3) DLCR-based clothing augmentation
python dlcr/llava/extract_clothes_descriptions.py --src <img_root> --ckpt <llava_ckpt>
python dlcr/llama/summarize_clothes_descriptions.py --src <desc.jsonl>
python dlcr/schp/parse_clothes.py --src <img_root> --part upper
python dlcr/sd_inpaint/inpaint_clothes.py --src <img_root> --prompts <prompts.jsonl> --n 10

# 4) Training (OpenGait single-GPU)
python opengait/opengait/main.py \
  --cfgs opengait/configs/skeletongait/skeletongait++_grew.yaml \
  --phase train --log_to_file
```

---

## Dataset Preprocessing

* Standardize all datasets (e.g., CCVID, CASIA-B) into a **GREW-compatible format**.
* Includes **detection → alignment → GEI generation → pose extraction (COCO-17)** → sequence synchronization and quality filtering.

---

## DLCR-Based Clothing Augmentation

* **Pipeline Overview:**
  (1) Extract clothing descriptions via LLaVA →
  (2) Summarize/refine via LLaMA →
  (3) Parse body/clothing masks using SCHP →
  (4) Perform Stable Diffusion inpainting.
  The same subject, pose, and background are maintained while **only clothing appearance varies**.
  About **25% of training images** are expanded to **10 new clothing variants** each.

* **Goal:**
  Generate synthetic silhouettes and skeletons to improve robustness under clothing variations.

---

## Model & Training (Single GPU)

* **Cross-Attention (FlashAttention)**
  Converts silhouette/skeleton features into Q/K/V and models fine-grained modality interaction using FlashAttention.

* **ΔJoint + Upper/Lower (UL) weighting**

  * ΔJoint: Inter-frame joint difference using a central difference kernel `[-0.5, 0, +0.5]` to encode temporal dynamics.
  * UL separation: Split upper/lower body weights (initial upper: 0.3, lower: 0.7, optimized via softmax).

* **Training Setup**
  Single RTX 3070 (8GB), FP16 precision, Triplet + CE loss, 30-frame sequences (interval 4), up to 720 frames for testing.

---

### Appendix: Command Summary

```bash
# CCVID preprocessing
python datasets_preprocess/ccvid/rename_ids.py
python datasets_preprocess/ccvid/rearrange_to_grew.py
python datasets_preprocess/ccvid/make_probe_gallery.py

# CCVID augmentation (optional)
python datasets_preprocess/ccvid_aug/crop_augmented_k5.py
python datasets_preprocess/ccvid_aug/rearrange_to_grew.py
python datasets_preprocess/ccvid_aug/make_probe_gallery.py

# Training (example: GREW)
python opengait/opengait/main.py \
  --cfgs opengait/configs/skeletongait/skeletongait++_grew.yaml \
  --phase train --log_to_file
```

---

## References

* **OpenGait**
  Fan et al., *OpenGait: A Comprehensive Benchmark Study for Gait Recognition Towards Better Practicality*, IEEE TPAMI, 2025.
  [GitHub: https://github.com/ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait)

* **DLCR (Diffusion + LLM Clothes Reconstruction)**
  Siddiqui et al., *DLCR: A Generative Data Expansion Framework via Diffusion for Clothes-Changing Person Re-Identification*, IEEE/CVF WACV, 2025.
  [GitHub: https://github.com/CroitoruAlin/dlcr](https://github.com/CroitoruAlin/dlcr)

* **Datasets**

  * **CASIA-B** — Yu et al., *A Benchmark for Gait Recognition*, IEEE TPAMI, 2006.
    [Official dataset link](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
  * **CCVID** — Hou et al., *Clothes-Changing Gait Recognition via Shape-Texture Disentanglement*, CVPR 2022.
    [Paper link](https://openaccess.thecvf.com/content/CVPR2022/html/Hou_Clothes-Changing_Gait_Recognition_via_Shape-Texture_Disentanglement_CVPR_2022_paper.html)
  * **GREW** — Zhang et al., *GREW: A Large-scale Benchmark for Gait Recognition in the Wild*, CVPR 2021.
    [GitHub: https://github.com/Gait3D/GREW-Benchmark](https://github.com/Gait3D/GREW-Benchmark)

> Usage Notice
> Please check and comply with the **license and usage terms** of all referenced datasets and open-source frameworks.
> This repository is intended for **research purposes only**, and proper attribution is required for any derivative or commercial use.

---

## Attribution & Original Source Notice

This repository is **modified and extended** based on the following open-source projects:

* **OpenGait**
  Original repository: [ShiqiYu/OpenGait](https://github.com/ShiqiYu/OpenGait)
  Fan et al., *OpenGait: Revisiting Gait Recognition Towards Better Practicality*, CVPR 2023 (Extended version: TPAMI 2025).
  Customized here for single-GPU training and structural simplification.

* **DLCR**
  Original repository: [CroitoruAlin/dlcr](https://github.com/CroitoruAlin/dlcr)
  DLCR: Data Expansion via Diffusion + LLM for Clothes-Changing Person Re-ID.
  Selected modules have been restructured and integrated into the gait recognition pipeline.

---

> Note: Please respect the copyright and usage terms of the original repositories.
> The OpenGait repository is designated for **academic use only**,
> and the DLCR repository does **not explicitly specify a license**—redistribution or commercial use should be verified individually.


