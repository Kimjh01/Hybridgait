# import torch
# print(torch.cuda.is_available())         # True
# print(torch.cuda.get_device_name(0))     # 'NVIDIA GeForce RTX 3070'



# import os
# import json

# json_path = "C:/Users/user/Desktop/capstone/Opengait-main/OpenGait-master/datasets/CCVID/CCVID.json"
# dataset_root = "C:/Users/user/Desktop/capstone/CCVID2_dataset"

# with open(json_path, "r") as f:
#     data = json.load(f)

# broken_samples = []
# total_samples = 0

# for pid in data["TRAIN_SET"]:
#     pid_path = os.path.join(dataset_root, pid)
#     if not os.path.isdir(pid_path):
#         print(f"❌ {pid} 디렉토리 없음")
#         continue

#     for root, _, files in os.walk(pid_path):
#         for file in files:
#             if file.endswith("0_heatmap.pkl"):
#                 total_samples += 1
#                 full_path = os.path.join(root, file)
#                 if os.path.islink(full_path):
#                     target = os.readlink(full_path)
#                     if not os.path.exists(target):
#                         broken_samples.append((pid, full_path, target))
#                         print(f"⚠️ 깨진 링크: {full_path} → {target}")

# # 요약 출력
# print("\n📊 요약")
# print(f"전체 heatmap 샘플 수: {total_samples}")
# print(f"깨진 heatmap 링크 수: {len(broken_samples)}")

# if total_samples > 0:
#     ratio = len(broken_samples) / total_samples * 100
#     print(f"깨진 비율: {ratio:.2f}%")
# else:
#     print("⚠️ heatmap 샘플이 없습니다.")

# # (선택) 깨진 샘플 목록 ID만 보기
# broken_ids = sorted(set(pid for pid, _, _ in broken_samples))
# print(f"\n⚠️ 깨진 ID 수: {len(broken_ids)}")
# print("깨진 ID 목록:", broken_ids)



# import os
# import pickle
# from collections import defaultdict

# base_path = "C:/Users/user/Desktop/capstone/CCVID2_dataset"
# len_counts = defaultdict(int)

# for root, dirs, files in os.walk(base_path):
#     for file in files:
#         if file.endswith(".pkl"):
#             file_path = os.path.join(root, file)
#             try:
#                 with open(file_path, "rb") as f:
#                     data = pickle.load(f)
#                 l = len(data)
#                 len_counts[l] += 1
#             except Exception as e:
#                 print(f"Error loading {file_path}: {e}")

# # 정렬해서 출력
# print("\n📊 len() 값별 파일 개수:")
# for length in sorted(len_counts.keys()):
#     print(f"len = {length}: {len_counts[length]}개")


import os

root_dir = "C:/Users/user/Desktop/capstone/CCVID2_dataset"
missing_pose = []
missing_sil = []
complete = []
total = 0

for pid in os.listdir(root_dir):
    pid_path = os.path.join(root_dir, pid)
    if not os.path.isdir(pid_path):
        continue
    for seq_type in os.listdir(pid_path):
        type_path = os.path.join(pid_path, seq_type)
        for view in os.listdir(type_path):
            view_path = os.path.join(type_path, view)
            if not os.path.isdir(view_path):
                continue
            total += 1
            files = os.listdir(view_path)
            has_pose = any("pose" in f and f.endswith(".pkl") for f in files)
            has_sil = any("sil" in f and f.endswith(".pkl") for f in files)

            if has_pose and has_sil:
                complete.append(view_path)
            elif not has_pose:
                missing_pose.append(view_path)
                print(f"⚠️ Missing pose file: {view_path}")
            elif not has_sil:
                missing_sil.append(view_path)
                print(f"⚠️ Missing silhouette file: {view_path}")

print("\n🔎 검사 결과:")
print(f"총 시퀀스 폴더 수: {total}")
print(f" - 정상 (pose + sil): {len(complete)}")
print(f" - pose 없음: {len(missing_pose)}")
print(f" - sil 없음: {len(missing_sil)}")

