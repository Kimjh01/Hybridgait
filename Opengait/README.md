# GREW Dataset Preprocessing & Training Pipeline (OpenGait · SkeletonGait++ -> HybridGait)

This document describes all **required modifications** and the **end-to-end pipeline** for preprocessing and training the GREW dataset using the SkeletonGait++ model in OpenGait.
You can adapt this process for other datasets by changing the dataset paths and parameters.

---

## 0) Environment Setup

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>

# Install PyTorch with CUDA 12.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

---

## 1) Code and Configuration Modifications (Apply Before Running)

### 1-1. File: `OpenGait-master/configs/skeletongait/skeletongait++_GREW.yaml`

Edit the following items to match your system and dataset:

```yaml
data_cfg:
  dataset_root: "D:/Grew_reduce_dataset"  # Change to your dataset root path
  dataset_partition: "OpenGait-master/datasets/GREW/GREW.json"  # Update to absolute/relative path

sampler:
  batchsize: [16, 2]  # Change from [32, 4]

trainer_cfg:
  save_iter: 1000     # Change from 30000
  # (Optional) Modify log iteration interval
  # log_iter: 100
```

**Note:**
`dataset_root` should point to the folder containing your final preprocessed GREW dataset
(i.e., where the linked silhouette and pose data are stored).

---

### 1-2. File: `OpenGait-master/opengait/utils/common.py`

At **line 144**, update the config path to an explicit location:

```diff
- cfg_file = "./configs/default.yaml"
+ cfg_file = "OpenGait-master/configs/default.yaml"
```

(You may use an absolute path depending on your directory structure.)

---

## 2) Execution Order

### Step 1. Rearrange the Raw GREW Dataset

Reorganize the dataset to fit OpenGait’s expected directory structure:

```bash
python OpenGait-master/datasets/GREW/rearrange_GREW.py \
  --input_path D:\Grew_reduce \
  --output_path D:\Grew_reduce_rearranged

python OpenGait-master/datasets/GREW/rearrange_GREW_pose.py \
  --input_path D:\Grew_reduce \
  --output_path D:\Grew_reduce_pose_rearranged
```

---

### Step 2. Convert to PKL Format

Convert silhouettes and pose data into `.pkl` format for model compatibility.

```bash
# Silhouette preprocessing
python OpenGait-master/datasets/pretreatment.py \
  --input_path D:\Grew_reduce_rearranged \
  --output_path D:\Grew_reduce_pkl \
  --dataset GREW

# Pose preprocessing
python OpenGait-master/datasets/pretreatment.py \
  --input_path D:\Grew_reduce_pose_rearranged \
  --output_path D:\Grew_reduce_pose_pkl \
  --pose \
  --dataset GREW
```

---

### Step 3. Generate Pose Heatmaps

(Modify `pretreatment_heatmap.py` to support GREW’s pose data format if needed.)

```bash
python OpenGait-master/datasets/pretreatment_heatmap.py \
  --pose_data_path D:\Grew_reduce_pose_pkl \
  --save_root D:\Grew_reduce_posemap \
  --dataset_name GREW
```

---

### Step 4. Link Silhouettes and Heatmaps

(Modify `ln_sil_heatmap.py` to support the GREW directory hierarchy.)

```bash
python OpenGait-master/datasets/ln_sil_heatmap.py \
  --heatmap_data_path D:\Grew_reduce_posemap\GREW_sigma_8.0_\pkl \
  --silhouette_data_path D:\Grew_reduce_pkl \
  --output_path D:\Grew_reduce_dataset
```

---

### Step 5. Train the SkeletonGait++ Model

After all preprocessing steps are completed and configuration files are updated, start training:

```bash
python OpenGait-master/opengait/main.py \
  --cfgs OpenGait-master/configs/skeletongait/skeletongait++_GREW.yaml \
  --phase train \
  --log_to_file
```

---

### Additional File Adjustments (if needed)

* `utils/msg_manager.py`: Modify logging messages or paths if dataset names differ.
* `utils/common.py`: Ensure YAML paths and file reading functions are compatible.
* `data/sampler.py`: Adjust GREW-specific sampling behavior if session handling differs.
* `modeling/losses/base.py`: Add or modify custom loss definitions if your model version requires it.

---

## 3) Summary

This pipeline performs:

1. GREW dataset restructuring
2. Silhouette and pose data conversion to PKL
3. Pose heatmap generation
4. Linking silhouette and pose data
5. Model training using customized `skeletongait++_GREW.yaml`

Ensure all paths, YAML configurations, and code modifications are applied **before running the training command** to avoid runtime errors.
