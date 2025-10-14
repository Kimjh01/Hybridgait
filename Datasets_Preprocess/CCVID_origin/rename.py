import os
import re

def rename_folders_and_update_files(base_path):
    """
    폴더명을 변경하고 텍스트 파일 내용을 업데이트하는 함수
    """
    # 현재 dictory 폴더에서 실행 중이므로 직접 CCVID 참조
    ccvid_path = "CCVID"
    session3_path = os.path.join(ccvid_path, "session3")
    
    # 1. session3 폴더 내 폴더명 변경
    print("=== 폴더명 변경 시작 ===")
    
    if os.path.exists(session3_path):
        # 폴더 목록 가져오기 (숫자 순서로 정렬)
        folders = [f for f in os.listdir(session3_path) 
                  if os.path.isdir(os.path.join(session3_path, f)) and re.match(r'\d{3}_\d{2}', f)]
        folders.sort()
        
        print(f"발견된 폴더: {folders}")
        
        # 폴더명 변경 (역순으로 처리하여 충돌 방지)
        for folder in reversed(folders):
            if re.match(r'\d{3}_\d{2}', folder):
                old_num = int(folder[:3])
                new_num = old_num + 226
                new_folder = f"{new_num:03d}" + folder[3:]
                
                old_path = os.path.join(session3_path, folder)
                new_path = os.path.join(session3_path, new_folder)
                
                try:
                    os.rename(old_path, new_path)
                    print(f"폴더명 변경: {folder} → {new_folder}")
                except Exception as e:
                    print(f"폴더명 변경 실패 - {folder}: {e}")
    else:
        print(f"session3 폴더를 찾을 수 없습니다: {session3_path}")
    
    # 2. 텍스트 파일 내용 업데이트
    print("\n=== 파일 내용 업데이트 시작 ===")
    
    text_files = ["gallery.txt", "query.txt", "train.txt"]
    
    for filename in text_files:
        file_path = os.path.join(ccvid_path, filename)
        
        if os.path.exists(file_path):
            try:
                # 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # session3/XXX_XX XXX 패턴 찾아서 변경
                def replace_pattern(match):
                    folder_part = match.group(1)  # XXX_XX
                    number_part = match.group(2)  # XXX
                    rest_part = match.group(3)    # 나머지 u1_l1...

                    old_num = int(number_part)
                    new_num = old_num + 226
                    new_folder = f"{new_num:03d}" + folder_part[3:]

                    return f"session3/{new_folder} {new_num:03d}\t{rest_part}"
                
                # 정규표현식으로 패턴 변경
                pattern = r'session3/(\d{3}_\d{2})\s+(\d{3})\s+(u\d+_.*)$'
                content = re.sub(pattern, replace_pattern, content, flags=re.MULTILINE)
                
                # 변경사항이 있는 경우에만 파일 쓰기
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"파일 업데이트 완료: {filename}")
                else:
                    print(f"변경사항 없음: {filename}")
                    
            except Exception as e:
                print(f"파일 처리 실패 - {filename}: {e}")
        else:
            print(f"파일 없음: {filename}")

def main():
    # 기본 경로 설정 (현재 폴더가 dictory이므로 현재 디렉토리 사용)
    base_path = "."
    
    print(f"작업 경로: {os.path.abspath(base_path)}")
    
    # 경로 확인
    ccvid_path = "CCVID"
    session3_path = os.path.join(ccvid_path, "session3")
    
    print(f"CCVID 경로: {os.path.abspath(ccvid_path)} - 존재: {os.path.exists(ccvid_path)}")
    print(f"session3 경로: {os.path.abspath(session3_path)} - 존재: {os.path.exists(session3_path)}")
    
    # 텍스트 파일 확인
    text_files = ["gallery.txt", "query.txt", "train.txt"]
    for filename in text_files:
        file_path = os.path.join(ccvid_path, filename)
        print(f"{filename}: {os.path.abspath(file_path)} - 존재: {os.path.exists(file_path)}")
    
    print("\n작업을 시작합니다...\n")
    
    # 백업 권장 메시지
    print("⚠️  중요: 작업 전 데이터 백업을 권장합니다!")
    response = input("계속 진행하시겠습니까? (y/N): ")
    
    if response.lower() in ['y', 'yes']:
        rename_folders_and_update_files(base_path)
        print("\n작업 완료!")
    else:
        print("작업이 취소되었습니다.")

if __name__ == "__main__":
    main()