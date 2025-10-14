import os
import re

def rename_folders_only(base_path):
    """
    CCVID_reduce/session3 내 폴더명만 변경하는 함수
    예: 001_01 → 227_01 (기존 숫자 + 226)
    """
    session3_path = os.path.join(base_path, "session3")

    print("=== 폴더명 변경 시작 ===")

    if os.path.exists(session3_path):
        folders = [f for f in os.listdir(session3_path)
                   if os.path.isdir(os.path.join(session3_path, f)) and re.match(r'\d{3}_\d{2}', f)]
        folders.sort()

        print(f"총 {len(folders)}개 폴더 발견")

        for folder in reversed(folders):
            old_num = int(folder[:3])
            new_num = old_num + 226
            new_folder = f"{new_num:03d}" + folder[3:]

            old_path = os.path.join(session3_path, folder)
            new_path = os.path.join(session3_path, new_folder)

            try:
                os.rename(old_path, new_path)
                print(f"[변경 완료] {folder} → {new_folder}")
            except Exception as e:
                print(f"[오류] {folder} 변경 실패: {e}")
    else:
        print(f"[경로 오류] session3 폴더 없음: {session3_path}")

def main():
    # base_path는 CCVID_reduce의 절대 경로여야 함
    base_path = "C:/Users/PC-3/Desktop/CCVID_reduce"

    abs_base = os.path.abspath(base_path)
    session3_path = os.path.join(abs_base, "session3")

    print(f"작업 기준 경로: {abs_base}")
    print(f"session3 경로 존재 여부: {os.path.exists(session3_path)}")

    print("\n⚠️  경고: 이 작업은 폴더명을 실제로 변경합니다. 백업을 권장합니다.")
    response = input("계속 진행할까요? (y/N): ")

    if response.lower() in ['y', 'yes']:
        rename_folders_only(abs_base)
        print("\n✅ 폴더 이름 변경 작업 완료!")
    else:
        print("❌ 작업 취소됨.")

if __name__ == "__main__":
    main()
