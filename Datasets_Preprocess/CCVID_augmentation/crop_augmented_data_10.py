import os
from PIL import Image

# 경로 설정
input_root = r"C:\Users\PC-3\Desktop\CCVID_processed"
output_root = r"C:\Users\PC-3\Desktop\CCVID_reduce\CCVID"

# .txt 파일 삭제 함수
def delete_txt_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                os.remove(os.path.join(root, file))
                print(f"[TXT Deleted] {os.path.join(root, file)}")

# 이미지 자르고 저장 함수
def process_images(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    width, height = img.size

                    person_width = width // 10  # 10명 기준
                    for i in range(10):  # 🔥 10명 전체로 수정
                        left = i * person_width
                        right = (i + 1) * person_width
                        cropped = img.crop((left, 0, right, height))

                        # 저장 경로 구성
                        relative_path = os.path.relpath(root, input_dir)
                        save_dir = os.path.join(output_dir, relative_path)
                        os.makedirs(save_dir, exist_ok=True)

                        base_name = os.path.splitext(file)[0]
                        new_filename = f"{base_name}_{i+1:02d}.png"
                        save_path = os.path.join(save_dir, new_filename)
                        cropped.save(save_path)
                        print(f"[Saved] {save_path}")

                except Exception as e:
                    print(f"[Error] {img_path} - {e}")

# 실행
delete_txt_files(input_root)
process_images(input_root, output_root)
