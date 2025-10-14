import os
import json

# 기준 루트 경로 설정
root_dir = r"C:\Users\user\Desktop\capstone\CCVID2_dataset"  # 사용자에 맞게 수정

# 결과 저장용 딕셔너리 초기화
data_split = {
    "TRAIN_SET": [],
    "TEST_SET": []
}

# 폴더 이름 순회
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        if 'train' in folder_name:
            data_split["TRAIN_SET"].append(folder_name)
        else:
            data_split["TEST_SET"].append(folder_name)

# JSON으로 저장
output_path = os.path.join(root_dir, "ccvid2_split.json")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data_split, f, indent=4, ensure_ascii=False)

print(f"JSON 파일 저장 완료: {output_path}")
