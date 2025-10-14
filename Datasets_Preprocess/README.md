# Gait Recognition Dataset Preprocessing Script (GREW Format Conversion)

This repository provides preprocessing scripts for various gait recognition datasets (e.g., **CCVID**, **CASIA-B**) and converts them into the **GREW dataset structure**.  
The main goal is to **standardize dataset organization** for training gait recognition models such as **SkeletonGait++**.

---

## 1. Create Virtual Environment & Install Dependencies

```bash
# Create virtual environment
conda create -n gait_gpu python=3.9

# Activate environment
conda activate gait_gpu

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install required libraries
conda install -c conda-forge numpy tqdm matplotlib seaborn pillow
conda install -c conda-forge ultralytics
````

Alternatively, install via `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 2. Directory Structure

```bash
Datasets_Preprocess/
├── CASIA-B_origin/
│   ├── frame_split.py           
│   ├── rearranged_casia_b.py     
│
├── CCVID_origin/
│   ├── rename.py
│   ├── rearrangement_main.py
│   ├── probe.py
│
├── CCVID_augmentation/
│   ├── crop_augmented_data_2.py
│   ├── crop_augmented_data_5.py
│   ├── crop_augmented_data_10.py
│   ├── rename.py
│   ├── rearrangement_main.py
│   ├── probe.py
│
├── requirements.txt
```

---

## 3. CASIA-B Preprocessing Steps

```bash
# 1. Split video into frames
python frame_split.py

# 2. Generate pose and GEI images
python make_pose_gei.py

# 3. Reorganize into GREW format
python rearranged_casia_b.py
```

---

## 4. CCVID_origin Preprocessing Steps

```bash
python rename.py                 # Resolve ID conflicts
python rearrangement_main.py     # Convert to GREW structure
python probe.py                  # Build probe set
```

---

## 5. CCVID_augmentation Preprocessing Steps

```bash
# 1. Crop each person (choose depending on number of people per frame)
python crop_augmented_data_2.py     # 2 people per frame
python crop_augmented_data_5.py     # 5 people per frame
python crop_augmented_data_10.py    # 10 people per frame

# 2. Resolve ID conflicts
python rename.py

# 3. Convert to GREW structure
python rearrangement_main.py

# 4. Build probe set
python probe.py
```

---

## 6. CCVID Dataset Download Links

* **Original CCVID Dataset:**
  [Google Drive Link](https://drive.google.com/file/d/1vkZxm5v-aBXa_JEi23MMeW4DgisGtS4W/view)

* **Augmented CCVID Dataset:**
  [Huggingface Link](https://huggingface.co/datasets/ihaveamoose/DLCR/tree/main)

---

## 7. Important Notes

* The scripts are designed for preparing datasets for **gait recognition models** such as *SkeletonGait++*.
* GEI and silhouette images are resized to **64×64**.
* Pose data can be configured for both **2D and 3D estimation**.
* The final dataset is automatically split into:

  ```
  train/
  test/gallery/
  test/probe/
  ```


