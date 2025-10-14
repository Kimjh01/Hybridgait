import os
import shutil

# probe 경로
probe_root = r"C:\Users\PC-3\Desktop\dictory\Grew_ccvid\test\probe"

# 모든 ID 폴더 순회
for id_name in os.listdir(probe_root):
    id_path = os.path.join(probe_root, id_name)
    
    if not os.path.isdir(id_path):
        continue  # 파일 무시

    # 폴더 내부 순회
    for item in os.listdir(id_path):
        item_path = os.path.join(id_path, item)

        # 시퀀스 폴더이면 폴더 이름 바꿔서 이동
        if os.path.isdir(item_path):
            new_folder_name = f"{id_name}_{item}"  # ⬅ 여기에서 _로 이름 연결
            new_folder_path = os.path.join(probe_root, new_folder_name)
            shutil.move(item_path, new_folder_path)

        # PNG 파일이면 probe 루트로 이동
        elif item.endswith(".png"):
            new_file_path = os.path.join(probe_root, item)
            shutil.move(item_path, new_file_path)

    # 원래 ID 폴더 비었으면 삭제
    try:
        os.rmdir(id_path)
    except OSError as e:
        print(f"[오류] {id_path} 삭제 실패: {e}")

print("✅ 구조 변경 및 이미지 이동 완료")
