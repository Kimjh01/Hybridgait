import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
import random
import shutil
import re

# 경로 설정
src_root = os.path.normpath('./CCVID')
dst_root = os.path.normpath('./CCVID_augmentation')

# YOLO 모델 로드
seg_model = YOLO('yolov8m-seg.pt')
pose_model = YOLO('yolov8m-pose.pt')

# COCO 17 keypoints 정보
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))

def scan_directory_structure():
    """디렉토리 구조를 스캔하여 데이터 수집 - 각 시퀀스를 독립적인 ID로 처리"""
    print("\n=== 디렉토리 구조 스캔 (독립적 시퀀스 처리) ===")
    
    all_data = []
    
    if not os.path.exists(src_root):
        print(f"Error: {src_root} 디렉토리가 존재하지 않습니다!")
        return []
    
    # session 디렉토리들 탐색
    for session_name in os.listdir(src_root):
        session_path = os.path.join(src_root, session_name)
        
        if not os.path.isdir(session_path):
            continue
            
        print(f"스캔 중: {session_name}")
        
        # 세션 내의 개별 폴더들 탐색
        for folder_name in os.listdir(session_path):
            folder_path = os.path.join(session_path, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            # 이미지 파일이 있는지 확인
            img_files = [f for f in os.listdir(folder_path) if is_image_file(f)]
            if not img_files:
                continue
            
            print(f"  폴더: {folder_name} -> 이미지 {len(img_files)}개")
            
            # 🔥 폴더명 전체를 고유 ID로 사용 (00001_01, 00001_02를 각각 다른 ID로)
            unique_id = folder_name  # 예: 00001_01, 00001_02, 00002_01 등
            path_info = f"{session_name}/{folder_name}"
            all_data.append((path_info, unique_id))
            print(f"    -> 등록: {path_info} -> 독립 ID: {unique_id}")
    
    print(f"\n전체 발견된 독립 시퀀스: {len(all_data)}개")
    return all_data

def parse_all_data():
    """모든 데이터를 스캔하고 6:2:2로 분할 - 각 시퀀스를 독립적으로 처리"""
    all_data = scan_directory_structure()
    
    if not all_data:
        print("데이터를 찾을 수 없습니다!")
        return {}
    
    # 🔥 각 시퀀스를 독립적인 ID로 처리 (그룹화 없음)
    print(f"전체 독립 시퀀스 수: {len(all_data)}")
    
    # 각 시퀀스별 데이터 수 출력
    print("\n독립 시퀀스별 데이터 분포:")
    for path, unique_id in sorted(all_data):
        full_path = os.path.join(src_root, path)
        if os.path.exists(full_path):
            img_count = len([f for f in os.listdir(full_path) if is_image_file(f)])
            print(f"  ID {unique_id}: {img_count}개 이미지 ({path})")
    
    # 🔥 전체 시퀀스 리스트를 섞어서 무작위로 분할
    random.shuffle(all_data)
    
    # 6:2:2 비율로 분할
    total_sequences = len(all_data)
    train_size = int(total_sequences * 0.6)
    gallery_size = int(total_sequences * 0.2)
    
    train_data = all_data[:train_size]
    gallery_data = all_data[train_size:train_size + gallery_size]
    probe_data = all_data[train_size + gallery_size:]
    
    print(f"\n독립 시퀀스 분할:")
    print(f"  Train 시퀀스 ({len(train_data)}개):")
    for path, uid in train_data[:5]:  # 처음 5개만 출력
        print(f"    - {uid} ({path})")
    if len(train_data) > 5:
        print(f"    ... 외 {len(train_data)-5}개")
    
    print(f"  Gallery 시퀀스 ({len(gallery_data)}개):")
    for path, uid in gallery_data[:5]:  # 처음 5개만 출력
        print(f"    - {uid} ({path})")
    if len(gallery_data) > 5:
        print(f"    ... 외 {len(gallery_data)-5}개")
    
    print(f"  Probe 시퀀스 ({len(probe_data)}개):")
    for path, uid in probe_data[:5]:  # 처음 5개만 출력
        print(f"    - {uid} ({path})")
    if len(probe_data) > 5:
        print(f"    ... 외 {len(probe_data)-5}개")
    
    # 각 split에 데이터 할당
    data_dict = {
        'train': train_data,
        'gallery': gallery_data,
        'probe': probe_data
    }
    
    return data_dict

def extract_silhouette_and_resize(img_path, target_size=64):
    """이미지에서 사람 실루엣 추출하고 64x64로 리사이즈"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # YOLOv8 segmentation 실행
        results = seg_model(img, verbose=False)
        
        # 결과 확인
        if not results or not results[0].masks or len(results[0].masks) == 0:
            return None
        
        # 사람 클래스(0)인 마스크만 선택
        person_masks = []
        for i, cls in enumerate(results[0].boxes.cls):
            if int(cls) == 0:  # 0은 person 클래스
                person_masks.append(results[0].masks.data[i].cpu().numpy())
        
        if len(person_masks) == 0:
            return None
        
        # 여러 사람이 검출된 경우, 가장 큰 마스크 선택
        if len(person_masks) > 1:
            areas = [mask.sum() for mask in person_masks]
            largest_idx = np.argmax(areas)
            final_mask = person_masks[largest_idx]
        else:
            final_mask = person_masks[0]
        
        # 마스크를 원본 이미지 크기로 리사이즈
        h, w = img.shape[:2]
        final_mask = cv2.resize(final_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 이진화 (0 또는 255)
        silhouette = (final_mask > 0.5).astype(np.uint8) * 255
        
        # 실루엣에서 사람 영역 찾기
        coords = np.where(silhouette > 0)
        if len(coords[0]) == 0:
            return None
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # 바운딩 박스로 크롭
        cropped_silhouette = silhouette[y_min:y_max+1, x_min:x_max+1]
        cropped_h, cropped_w = cropped_silhouette.shape[:2]
        
        # 비율 계산
        scale = min(target_size / cropped_w, target_size / cropped_h)
        
        # 새로운 크기 계산
        new_w = int(cropped_w * scale)
        new_h = int(cropped_h * scale)
        
        # 리사이즈
        resized = cv2.resize(cropped_silhouette, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 패딩 계산
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # 패딩 적용 (검은색 배경)
        padded = cv2.copyMakeBorder(
            resized,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0
        )
        
        return padded
        
    except Exception as e:
        print(f"실루엣 추출 오류: {e}")
        return None

def extract_pose_yolo(img_path):
    """🔥 강화된 YOLOv8 포즈 추출 - 더 안전한 예외 처리"""
    try:
        # 이미지 로드 확인
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지 로드 실패: {img_path}")
            return None
        
        # 이미지 크기 확인
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            print(f"유효하지 않은 이미지 크기: {w}x{h}")
            return None
        
        # YOLOv8 pose 실행
        results = pose_model(img, verbose=False)
        
        # 🔥 결과 존재 여부 다단계 확인
        if not results:
            print(f"YOLO 결과 없음: {img_path}")
            return None
        
        if len(results) == 0:
            print(f"YOLO 결과 리스트 비어있음: {img_path}")
            return None
        
        result = results[0]
        if result is None:
            print(f"첫 번째 결과가 None: {img_path}")
            return None
        
        # keypoints 속성 확인
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            print(f"keypoints 속성 없음: {img_path}")
            return None
        
        # keypoints 데이터 확인
        if len(result.keypoints) == 0:
            print(f"keypoints 데이터 비어있음: {img_path}")
            return None
        
        # xy 좌표 확인
        if not hasattr(result.keypoints, 'xy') or result.keypoints.xy is None:
            print(f"xy 좌표 없음: {img_path}")
            return None
        
        if len(result.keypoints.xy) == 0:
            print(f"xy 좌표 데이터 비어있음: {img_path}")
            return None
        
        # confidence 확인
        if not hasattr(result.keypoints, 'conf') or result.keypoints.conf is None:
            print(f"confidence 데이터 없음: {img_path}")
            return None
        
        if len(result.keypoints.conf) == 0:
            print(f"confidence 데이터 비어있음: {img_path}")
            return None
        
        # 🔥 첫 번째 사람의 keypoints 안전하게 가져오기
        try:
            keypoints = result.keypoints.xy[0].cpu().numpy()  # (17, 2) 형태
            confidences = result.keypoints.conf[0].cpu().numpy()  # (17,) 형태
        except Exception as e:
            print(f"keypoints 데이터 추출 실패: {e}")
            return None
        
        # 데이터 형태 확인
        if keypoints is None or confidences is None:
            print(f"keypoints 또는 confidences가 None")
            return None
        
        if keypoints.shape[0] != 17 or confidences.shape[0] != 17:
            print(f"keypoints 형태 오류: keypoints={keypoints.shape}, confidences={confidences.shape}")
            return None
        
        # 🔥 포즈 데이터 생성
        pose_data = []
        for i in range(17):  # COCO는 17개 keypoints
            try:
                x, y = keypoints[i]
                conf = confidences[i]
                
                # NaN 값 확인
                if np.isnan(x) or np.isnan(y) or np.isnan(conf):
                    x, y, conf = 0.0, 0.0, 0.0
                
                # 정규화된 좌표로 변환
                x_norm = float(x / w) if w > 0 else 0.0
                y_norm = float(y / h) if h > 0 else 0.0
                conf_val = float(conf)
                
                # 범위 확인 (0~1)
                x_norm = max(0.0, min(1.0, x_norm))
                y_norm = max(0.0, min(1.0, y_norm))
                conf_val = max(0.0, min(1.0, conf_val))
                
                pose_data.append([x_norm, y_norm, 0.0, conf_val])  # z는 0으로 설정
                
            except Exception as e:
                print(f"keypoint {i} 처리 오류: {e}")
                pose_data.append([0.0, 0.0, 0.0, 0.0])  # 기본값
        
        return pose_data
        
    except Exception as e:
        print(f"포즈 추출 전체 오류 ({img_path}): {e}")
        return None

def save_pose_to_txt(pose_data, txt_path):
    """COCO 17 keypoints를 txt 파일로 저장"""
    try:
        if pose_data is None or len(pose_data) != 17:
            print(f"유효하지 않은 pose_data: {pose_data}")
            return False
        
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, 'w') as f:
            # 헤더 추가
            f.write("# COCO 17 keypoints format\n")
            f.write("# x_normalized y_normalized z confidence\n")
            for i, landmark in enumerate(pose_data):
                if len(landmark) != 4:
                    print(f"keypoint {i} 형태 오류: {landmark}")
                    continue
                f.write(f"{landmark[0]:.6f} {landmark[1]:.6f} {landmark[2]:.6f} {landmark[3]:.6f}\n")
        
        return True
        
    except Exception as e:
        print(f"포즈 저장 오류: {e}")
        return False

def generate_gei(silhouette_folder, output_path):
    """실루엣 이미지들로부터 GEI 생성"""
    try:
        if not os.path.exists(silhouette_folder):
            print(f"실루엣 폴더가 존재하지 않습니다: {silhouette_folder}")
            return False
        
        # 실루엣 파일 찾기
        all_files = os.listdir(silhouette_folder)
        silhouette_files = [f for f in all_files 
                           if f.endswith('.png') and not f.endswith('_gei.png') and not f.endswith('_origin.jpg')]
        
        if not silhouette_files:
            print(f"실루엣 파일이 없습니다: {silhouette_folder}")
            return False
        
        # 모든 실루엣 이미지 로드
        silhouettes = []
        for file in sorted(silhouette_files):
            img_path = os.path.join(silhouette_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                silhouettes.append(img.astype(np.float32))
        
        if not silhouettes:
            print(f"유효한 실루엣 이미지가 없습니다: {silhouette_folder}")
            return False
        
        # GEI 계산 (평균)
        gei = np.mean(silhouettes, axis=0).astype(np.uint8)
        
        # GEI 저장 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # GEI 저장
        success = cv2.imwrite(output_path, gei)
        return success
        
    except Exception as e:
        print(f"GEI 생성 오류: {e}")
        return False

def process_sequence(src_folder, dst_folder, unique_id, split_type, save_original=False):
    """단일 시퀀스 처리 및 GEI 생성 - 독립적 ID 처리"""
    try:
        os.makedirs(dst_folder, exist_ok=True)
        
        processed = 0
        failed = 0
        pose_success = 0
        pose_failed = 0
        
        # 이미지 파일 처리
        img_files = [f for f in os.listdir(src_folder) if is_image_file(f)]
        
        for file in sorted(img_files):
            src_file = os.path.join(src_folder, file)
            base_name = os.path.splitext(file)[0]
            file_ext = os.path.splitext(file)[1]
            
            try:
                # 원본 이미지 복사 (옵션)
                if save_original:
                    origin_path = os.path.join(dst_folder, f"{base_name}_origin{file_ext}")
                    shutil.copy2(src_file, origin_path)
                
                # 실루엣 추출 및 리사이즈
                silhouette = extract_silhouette_and_resize(src_file)
                if silhouette is None:
                    failed += 1
                    continue
                
                # 실루엣 저장 (png 형식)
                silhouette_path = os.path.join(dst_folder, f"{base_name}.png")
                cv2.imwrite(silhouette_path, silhouette)
                processed += 1
                
                # 🔥 Pose 추출 (강화된 오류 처리)
                pose_data = extract_pose_yolo(src_file)
                if pose_data is not None:
                    # Pose 데이터 저장
                    pose_path = os.path.join(dst_folder, f"{base_name}_2d_pose.txt")
                    if save_pose_to_txt(pose_data, pose_path):
                        pose_success += 1
                    else:
                        pose_failed += 1
                else:
                    pose_failed += 1
                
            except Exception as e:
                print(f"파일 처리 오류 ({file}): {str(e)}")
                failed += 1
        
        # 시퀀스 처리 완료 후 즉시 GEI 생성
        gei_generated = False
        if processed > 0:
            # 🔥 GEI 저장 경로 - 독립적 ID 사용
            if split_type == 'train':
                gei_path = os.path.join(dst_root, 'train', unique_id, f"{unique_id}_gei.png")
            elif split_type == 'gallery':
                gei_path = os.path.join(dst_root, 'test', 'gallery', unique_id, f"{unique_id}_gei.png")
            else:  # probe
                gei_path = os.path.join(dst_root, 'test', 'probe', unique_id, f"{unique_id}_gei.png")
            
            gei_generated = generate_gei(dst_folder, gei_path)
        
        # 🔥 포즈 처리 결과 출력
        if pose_success + pose_failed > 0:
            pose_rate = pose_success / (pose_success + pose_failed) * 100
            print(f"  포즈 처리: {pose_success}성공/{pose_failed}실패 ({pose_rate:.1f}%)")
        
        return processed, failed, gei_generated
        
    except Exception as e:
        print(f"시퀀스 처리 오류: {e}")
        return 0, 0, False

def process_data(data_dict, save_original=False):
    """데이터 처리 - 실루엣, pose, GEI 생성 (독립적 시퀀스 처리)"""
    total_processed = 0
    total_failed = 0
    gei_generated = 0
    
    for split_type, data_list in data_dict.items():
        print(f"\nProcessing {split_type}...")
        
        with tqdm(total=len(data_list), desc=f"{split_type}") as pbar:
            for path_info, unique_id in data_list:
                # path_info: session1/00001_01 형태
                # unique_id: 00001_01 (독립적 ID)
                
                # 원본 폴더 경로
                src_folder = os.path.join(src_root, path_info)
                if not os.path.exists(src_folder):
                    print(f"\nWarning: {src_folder} not found")
                    pbar.update(1)
                    continue
                
                # 🔥 대상 폴더 경로 설정 - 독립적 ID 사용
                if split_type == 'train':
                    dst_folder = os.path.join(dst_root, 'train', unique_id, '01')  # 기본 시퀀스 번호
                elif split_type == 'gallery':
                    dst_folder = os.path.join(dst_root, 'test', 'gallery', unique_id, '01')
                else:  # probe
                    dst_folder = os.path.join(dst_root, 'test', 'probe', unique_id, '01')
                
                # 시퀀스 처리 및 GEI 생성
                processed, failed, gei_success = process_sequence(
                    src_folder, dst_folder, unique_id, split_type, save_original
                )
                
                total_processed += processed
                total_failed += failed
                if gei_success:
                    gei_generated += 1
                
                pbar.update(1)
    
    return total_processed, total_failed, gei_generated

def print_statistics(data_dict):
    """통계 출력 - 독립적 시퀀스 처리"""
    print("\n=== 독립적 시퀀스 분할 통계 ===")
    
    for split_type, data_list in data_dict.items():
        session_counts = defaultdict(int)
        person_counts = defaultdict(int)  # 원래 사람 ID별 카운트
        
        for path_info, unique_id in data_list:
            session = path_info.split('/')[0]
            session_counts[session] += 1
            
            # 원래 사람 ID 추출 (00001_01 → 00001)
            person_match = re.match(r'(\d+)', unique_id)
            if person_match:
                original_person_id = person_match.group(1)
                person_counts[original_person_id] += 1
        
        print(f"\n{split_type}:")
        print(f"  - 독립 시퀀스 수: {len(data_list)}")
        print(f"  - 원래 사람 수: {len(person_counts)}")
        print(f"  - Session 분포: {dict(session_counts)}")
        print(f"  - 사람별 시퀀스 수: {dict(list(person_counts.items())[:5])}{'...' if len(person_counts) > 5 else ''}")
    
    # 전체 비율 확인
    total = sum(len(data_list) for data_list in data_dict.values())
    print(f"\n전체 독립 시퀀스 수: {total}")
    print(f"Train : Gallery : Probe = {len(data_dict['train'])} : {len(data_dict['gallery'])} : {len(data_dict['probe'])}")
    if total > 0:
        print(f"비율: {len(data_dict['train'])/total*100:.1f}% : {len(data_dict['gallery'])/total*100:.1f}% : {len(data_dict['probe'])/total*100:.1f}%")

def main(save_original=False):
    """메인 함수"""
    # 출력 디렉토리가 이미 존재하는지 확인
    if os.path.exists(dst_root):
        response = input(f"\n{dst_root}가 이미 존재합니다. 덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("작업을 취소했습니다.")
            return
        else:
            print(f"{dst_root} 삭제 중...")
            shutil.rmtree(dst_root)
    
    #모든 시퀀스를 독립적으로 스캔하고 6:2:2로 분할
    data_dict = parse_all_data()
    
    if not data_dict or not any(data_dict.values()):
        print("처리할 데이터가 없습니다!")
        return
    
    # 통계 출력
    print_statistics(data_dict)
    
    # 원본 저장 옵션 안내
    save_option_text = "원본 이미지도 함께 저장됩니다." if save_original else "원본 이미지는 저장되지 않습니다."
    print(f"\n설정: {save_option_text}")
    print("⚠️  각 시퀀스(00001_01, 00001_02 등)를 독립적인 ID로 처리합니다.")
    print("🔧 포즈 추출 오류 처리가 강화되었습니다.")
    
    # 사용자 확인
    response = input("\n계속 진행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("작업을 취소했습니다.")
        return
    
    # 데이터 처리
    processed, failed, gei_count = process_data(data_dict, save_original)
    
    print(f'\n=== 처리 완료 ===')
    print(f'성공적으로 처리된 이미지: {processed}')
    print(f'처리 실패한 이미지: {failed}')
    print(f'생성된 GEI 이미지: {gei_count}')
    if processed + failed > 0:
        print(f'성공률: {processed/(processed+failed)*100:.1f}%')

if __name__ == '__main__':
    # 원본 이미지 저장 여부 설정
    SAVE_ORIGINAL = False
    
    main(save_original=SAVE_ORIGINAL)
