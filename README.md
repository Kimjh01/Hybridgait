# HybridGait: Integrated Pipeline for Gait Data Preprocessing, Augmentation, and Training

**HybridGait** provides an end-to-end pipeline for gait recognition —
from **dataset preprocessing** to **data augmentation (DLCR)** and **model training** based on a customized OpenGait framework.

This project performs **DLCR (Diffusion + LLM Clothes Reconstruction)**–based augmentation
and improves **OpenGait** for **single-GPU training efficiency** and better dataset compatibility.

---

## 1. Overview

HybridGait aims to:

* Standardize multiple gait datasets (CCVID, CASIA-B, GREW, etc.) into the **GREW format**
* Enhance dataset diversity through **DLCR-based appearance and pose augmentation**
* Improve the **OpenGait framework** for single-GPU operation and simplified training

The repository includes all three main stages:
**Dataset Preprocessing → DLCR Augmentation → OpenGait Training.**

---

## 2. Repository Structure

Below is an overview of the HybridGait repository layout.

```
HybridGait/
├── Datasets_Preprocess/
│   ├── CASIA-B_origin/
│   │   ├── frame_split.py           # Split video into frame images
│   │   ├── rearranged_casia_b.py    # Rearrange into GREW format
│   │   └── make_pose_gei.py
│   ├── CCVID_origin/
│   │   ├── rename.py                # Fix ID conflicts
│   │   ├── rearrangement_main.py    # GREW-style rearrangement
│   │   └── probe.py                 # Create probe/gallery sets
│   ├── CCVID_augmentation/
│   │   ├── crop_augmented_data_2.py
│   │   ├── crop_augmented_data_5.py
│   │   ├── crop_augmented_data_10.py
│   │   ├── rename.py
│   │   ├── rearrangement_main.py
│   │   └── probe.py
│   └── requirements.txt
│
├── dlcr/
│   ├── LLaVA/
│   │   └── extract_clothes_descriptions.py
│   ├── llama/
│   │   └── summarize_clothes_descriptions.py
│   ├── Self-Correction-Human-Parsing/
│   │   └── simple_extractor.py
│   ├── stable-diffusion/
│   │   ├── generate_data.py
│   │   └── utils/
│   └── DG/
│       └── train.py
│
├── Opengait/
│   └── OpenGait-master/
│       ├── configs/
│       │   └── skeletongait/
│       │       └── skeletongait++_GREW.yaml
│       ├── opengait/
│       │   ├── main.py
│       │   ├── modeling/
│       │   ├── utils/
│       │   └── data/
│       └── logs/
│
├── opengait_CCVID/
│   ├── configs/
│   │   └── ccvid.yaml
│   ├── opengait/
│   │   ├── main.py
│   │   ├── modeling/
│   │   ├── utils/
│   │   └── data/
│   └── results/
│
└── README.md
```

---

## 3. Dataset Preprocessing (`Datasets_Preprocess`)

This stage converts raw datasets (e.g., CASIA-B, CCVID) into a **GREW-compatible structure**
for unified training input across datasets.

### CASIA-B Example

```bash
# (1) Split videos into frames
python Datasets_Preprocess/CASIA-B_origin/frame_split.py

# (2) Generate pose and GEI images
python Datasets_Preprocess/CASIA-B_origin/make_pose_gei.py

# (3) Rearrange into GREW directory format
python Datasets_Preprocess/CASIA-B_origin/rearranged_casia_b.py
```

### CCVID Example

```bash
# (1) Fix ID conflicts
python Datasets_Preprocess/CCVID_origin/rename.py

# (2) Convert to GREW format
python Datasets_Preprocess/CCVID_origin/rearrangement_main.py

# (3) Build probe/gallery sets
python Datasets_Preprocess/CCVID_origin/probe.py
```

### CCVID Augmentation (Optional)

```bash
# (1) Crop by number of people per frame
python Datasets_Preprocess/CCVID_augmentation/crop_augmented_data_5.py

# (2) Rename and rearrange
python Datasets_Preprocess/CCVID_augmentation/rename.py
python Datasets_Preprocess/CCVID_augmentation/rearrangement_main.py

# (3) Generate probe/gallery configuration
python Datasets_Preprocess/CCVID_augmentation/probe.py
```

---

## 4. DLCR Augmentation Pipeline (`dlcr`)

**DLCR (Diffusion + LLM Clothes Reconstruction)** performs appearance- and pose-level augmentation.
It extracts clothing descriptions, generates masks, and synthesizes new variants using diffusion models.

| Module               | Function                           | Example Command                                                        |
| -------------------- | ---------------------------------- | ---------------------------------------------------------------------- |
| **LLaVA**            | Extract clothing descriptions      | `python extract_clothes_descriptions.py -s <path> --model_path <ckpt>` |
| **LLaMA**            | Summarize & parse descriptions     | `python summarize_clothes_descriptions.py`                             |
| **SCHP**             | Generate person masks              | `python simple_extractor.py --dataset lip ...`                         |
| **Stable Diffusion** | Generate synthetic variants        | `python generate_data.py --use_discriminator True ...`                 |
| **DG**               | Train discriminator for refinement | `python train.py --datadir <path>`                                     |

All generated data follow GREW-like folder structures,
allowing immediate use for training in OpenGait.

---

## 5. Training Pipeline (`Opengait/OpenGait-master`, `opengait_CCVID`)

After preprocessing and augmentation,
the GREW-style dataset can be used to train **SkeletonGait++** models
via the customized single-GPU version of OpenGait.

### OpenGait (GREW-based Training)

```bash
python Opengait/OpenGait-master/opengait/main.py \
  --cfgs Opengait/OpenGait-master/configs/skeletongait/skeletongait++_GREW.yaml \
  --phase train \
  --log_to_file
```

### CCVID-Specific Training (Optional)

```bash
python opengait_CCVID/opengait/main.py \
  --cfgs opengait_CCVID/configs/ccvid.yaml \
  --phase train
```

> The OpenGait version in HybridGait has been refactored for **single-GPU training**,
> with simplified data loading and improved efficiency.
> Further architecture details and benchmark results will be added later.

---

## 6. Full Execution Summary

```bash
# 1️. Dataset Preprocessing
python Datasets_Preprocess/CASIA-B_origin/frame_split.py
python Datasets_Preprocess/CASIA-B_origin/rearranged_casia_b.py

# 2️. DLCR Augmentation
python dlcr/LLaVA/extract_clothes_descriptions.py -s <path> --model_path <ckpt>
python dlcr/llama/summarize_clothes_descriptions.py
python dlcr/Self-Correction-Human-Parsing/simple_extractor.py --dataset lip ...
python dlcr/stable-diffusion/generate_data.py --use_discriminator True ...
python dlcr/DG/train.py --datadir <path>

# 3️. OpenGait Training
python Opengait/OpenGait-master/opengait/main.py \
  --cfgs Opengait/OpenGait-master/configs/skeletongait/skeletongait++_GREW.yaml \
  --phase train
```

---

> **HybridGait** provides a unified end-to-end framework that covers
> **Dataset Structuring → GREW Conversion → DLCR Augmentation → OpenGait Training.**
> Future updates will include refined model architectures and performance evaluations.

