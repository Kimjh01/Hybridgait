# GREW Dataset Preprocessing & Training Pipeline

This document outlines the full preprocessing and training pipeline for the GREW dataset using the SkeletonGait++ model in OpenGait.

---

## 🔧 Step 1: Rearranging the Raw GREW Dataset

Rearrange the dataset structure to match OpenGait requirements:

```bash
python datasets/GREW/rearrange_GREW.py \
  --input_path D:\Grew_reduce \
  --output_path D:\Grew_reduce_rearranged

python datasets/GREW/rearrange_GREW_pose.py \
  --input_path D:\Grew_reduce \
  --output_path D:\Grew_reduce_pose_rearranged
```

---

## 📆 Step 2: Preprocess to PKL Format

Convert silhouettes and pose data to `.pkl` format:

```bash
# For silhouettes
python datasets/pretreatment.py \
  --input_path D:\Grew_reduce_rearranged \
  --output_path D:\Grew_reduce_pkl \
  --dataset GREW

# For pose data
python datasets/pretreatment.py \
  --input_path D:\Grew_reduce_pose_rearranged \
  --output_path D:\Grew_reduce_pose_pkl \
  --pose \
  --dataset GREW
```

---

## 🔥 Step 3: Generate Pose Heatmaps

> **Note:** Requires modification to `pretreatment_heatmap.py` for GREW pose format support.

```bash
python datasets/pretreatment_heatmap.py \
  --pose_data_path D:\Grew_reduce_pose_pkl \
  --save_root D:\Grew_reduce_posemap \
  --dataset_name GREW
```

---

## 🪨 Step 4: Link Silhouettes and Heatmaps

> **Note:** Requires modification to `ln_sil_heatmap.py` to support GREW structure.

```bash
python datasets/ln_sil_heatmap.py \
  --heatmap_data_path D:\Grew_reduce_posemap\GREW_sigma_8.0_\pkl \
  --silhouette_data_path D:\Grew_reduce_pkl \
  --output_path D:\Grew_reduce_dataset
```

---

## 🏃‍♂️ Step 5: Train SkeletonGait++ Model

Train the model with customized GREW configuration:

```bash
python opengait/main.py \
  --cfgs configs/skeletongait/skeletongait++_GREW.yaml \
  --phase train \
  --log_to_file
```

### ✨ Required File Modifications

* `utils/msg_manager.py`: Adjust logging for GREW
* `utils/common.py`: Update utility methods
* `data/sampler.py`: Modify sampling for GREW sessions
* `modeling/losses/base.py`: Apply any custom loss functions

---

## 📅 Summary

This pipeline covers:

1. GREW dataset restructuring
2. Silhouette and pose data conversion
3. Pose heatmap generation
4. Combined dataset creation
5. Training with SkeletonGait++

Ensure all script paths and required `.py` modifications are correctly handled before execution.
