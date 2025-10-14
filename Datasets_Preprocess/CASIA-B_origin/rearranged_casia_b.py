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
src_root = os.path.normpath('./CASIA-B-frame-reorganized')
dst_root = os.path.normpath('./CASIA-B-augmentation')

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

def parse_casia_folder_name(folder_name):
    """CASIA-B 폴더명 파싱: 001-bg-01 -> (person_id=001, condition=bg, sequence=01)"""
    parts = folder_name.split('-')
    if len(parts) == 3:
        person_id = parts[0]  # 001
        condition = parts[1]  # bg, cl, nm
        sequence = parts[2]   # 01, 03, 05
        return person_id, condition, sequence
    return None, None, None

def scan_casia_structure():
    """CASIA-B 구조 스캔 - train/test 폴더 모두 처리"""
    print("\n=== CASIA-B 구조 스캔 ===")
    
    all_data = []
    
    if not os.path.exists(src_root):
        print(f"Error: {src_root} 디렉토리가 존재하지 않습니다!")
        return []
    
    # train과 test 폴더 모두 스캔
    for split_folder in ['train', 'test']:
        split_path = os.path.join(src_root, split_folder)
        
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} 폴더가 존재하지 않습니다.")
            continue
            
        print(f"스캔 중: {split_folder}")
        
        # 각 person-condition-sequence 폴더 스캔
        for folder_name in os.listdir(split_path):
            folder_path = os.path.join(split_path, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            # 폴더명 파싱
            person_id, condition, sequence = parse_casia_folder_name(folder_name)
            if person_id is None:
                print(f"  알 수 없는 폴더 형식: {folder_name}")
                continue
            
            # 각 angle 폴더 스캔 (000, 018, 036, ...)
            angle_folders = [d for d in os.listdir(folder_path) 
                           if os.path.isdir(os.path.join(folder_path, d))]
            
            if not angle_folders:
                print(f"  각도 폴더 없음: {folder_name}")
                continue
            
            for angle in sorted(angle_folders):
                angle_path = os.path.join(folder_path, angle)
                
                # 이미지 파일 확인
                img_files = [f for f in os.listdir(angle_path) if is_image_file(f)]
                if not img_files:
                    continue
                
                print(f"  {folder_name}/{angle} -> 이미지 {len(img_files)}개")
                
                # 고유 ID 생성: person_condition_sequence_angle
                unique_id = f"{person_id}_{condition}_{sequence}_{angle}"
                path_info = f"{split_folder}/{folder_name}/{angle}"
                
                all_data.append({
                    'path_info': path_info,
                    'unique_id': unique_id,
                    'person_id': person_id,
                    'condition': condition,
                    'sequence': sequence,
                    'angle': angle,
                    'original_split': split_folder,
                    'img_count': len(img_files)
                })
    
    print(f"\n전체 발견된 시퀀스: {len(all_data)}개")
    return all_data

def reorganize_casia_data():
    """CASIA-B 데이터를 새로운 구조로 재구성"""
    all_data = scan_casia_structure()
    
    if not all_data:
        print("데이터를 찾을 수 없습니다!")
        return {}
    
    # 사람별로 그룹화
    person_groups = defaultdict(list)
    for data in all_data:
        person_groups[data['person_id']].append(data)
    
    print(f"\n전체 사람 수: {len(person_groups)}")
    
    # 각 사람별 데이터 분포 출력 (처음 10명만)
    print("\n사람별 데이터 분포 (처음 10명):")
    for i, (person_id, sequences) in enumerate(sorted(person_groups.items())[:10]):
        conditions = defaultdict(int)
        angles = defaultdict(int)
        for seq in sequences:
            conditions[seq['condition']] += 1
            angles[seq['angle']] += 1
        print(f"  Person {person_id}: {len(sequences)}개 시퀀스 - 조건{dict(conditions)} 각도{dict(angles)}")
    
    if len(person_groups) > 10:
        print(f"  ... 외 {len(person_groups)-10}명")
    
    # 사람 ID 리스트를 섞어서 6:2:2로 분할
    person_ids = list(person_groups.keys())
    random.shuffle(person_ids)
    
    total_persons = len(person_ids)
    train_size = int(total_persons * 0.6)
    gallery_size = int(total_persons * 0.2)
    
    train_persons = person_ids[:train_size]
    gallery_persons = person_ids[train_size:train_size + gallery_size]
    probe_persons = person_ids[train_size + gallery_size:]
    
    # 각 split에 데이터 할당
    data_dict = {
        'train': [],
        'gallery': [],
        'probe': []
    }
    
    for person_id in train_persons:
        data_dict['train'].extend(person_groups[person_id])
    
    for person_id in gallery_persons:
        data_dict['gallery'].extend(person_groups[person_id])
    
    for person_id in probe_persons:
        data_dict['probe'].extend(person_groups[person_id])
    
    print(f"\n데이터 분할 결과:")
    print(f"  Train: {len(train_persons)}명 ({len(data_dict['train'])}개 시퀀스)")
    print(f"  Gallery: {len(gallery_persons)}명 ({len(data_dict['gallery'])}개 시퀀스)")
    print(f"  Probe: {len(probe_persons)}명 ({len(data_dict['probe'])}개 시퀀스)")
    
    # 각 split의 처음 5개 시퀀스 출력
    for split_name, split_data in data_dict.items():
        print(f"\n{split_name} 시퀀스 (처음 5개):")
        for data in split_data[:5]:
            print(f"  - {data['unique_id']} ({data['path_info']}) - {data['img_count']}장")
        if len(split_data) > 5:
            print(f"  ... 외 {len(split_data)-5}개")
    
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
    """YOLOv8 포즈 추출"""
    try:
        # 이미지 로드 확인
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # 이미지 크기 확인
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return None
        
        # YOLOv8 pose 실행
        results = pose_model(img, verbose=False)
        
        # 결과 존재 여부 확인
        if not results or len(results) == 0:
            return None
        
        result = results[0]
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return None
        
        if len(result.keypoints) == 0:
            return None
        
        if not hasattr(result.keypoints, 'xy') or result.keypoints.xy is None:
            return None
        
        if len(result.keypoints.xy) == 0:
            return None
        
        if not hasattr(result.keypoints, 'conf') or result.keypoints.conf is None:
            return None
        
        # 첫 번째 사람의 keypoints 가져오기
        try:
            keypoints = result.keypoints.xy[0].cpu().numpy()  # (17, 2)
            confidences = result.keypoints.conf[0].cpu().numpy()  # (17,)
        except Exception:
            return None
        
        if keypoints.shape[0] != 17 or confidences.shape[0] != 17:
            return None
        
        # 포즈 데이터 생성
        pose_data = []
        for i in range(17):
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
                
                pose_data.append([x_norm, y_norm, 0.0, conf_val])
                
            except Exception:
                pose_data.append([0.0, 0.0, 0.0, 0.0])
        
        return pose_data
        
    except Exception:
        return None

def save_pose_to_txt(pose_data, txt_path):
    """COCO 17 keypoints를 txt 파일로 저장"""
    try:
        if pose_data is None or len(pose_data) != 17:
            return False
        
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, 'w') as f:
            f.write("# COCO 17 keypoints format\n")
            f.write("# x_normalized y_normalized z confidence\n")
            for landmark in pose_data:
                if len(landmark) != 4:
                    continue
                f.write(f"{landmark[0]:.6f} {landmark[1]:.6f} {landmark[2]:.6f} {landmark[3]:.6f}\n")
        
        return True
        
    except Exception:
        return False

def generate_gei(silhouette_folder, output_path):
    """실루엣 이미지들로부터 GEI 생성"""
    try:
        if not os.path.exists(silhouette_folder):
            return False
        
        # 실루엣 파일 찾기
        all_files = os.listdir(silhouette_folder)
        silhouette_files = [f for f in all_files 
                           if f.endswith('.png') and not f.endswith('_gei.png') and not f.endswith('_origin.jpg')]
        
        if not silhouette_files:
            return False
        
        # 모든 실루엣 이미지 로드
        silhouettes = []
        for file in sorted(silhouette_files):
            img_path = os.path.join(silhouette_folder, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                silhouettes.append(img.astype(np.float32))
        
        if not silhouettes:
            return False
        
        # GEI 계산 (평균)
        gei = np.mean(silhouettes, axis=0).astype(np.uint8)
        
        # GEI 저장 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # GEI 저장
        success = cv2.imwrite(output_path, gei)
        return success
        
    except Exception:
        return False

def process_sequence(data_info, split_type, save_original=False):
    """단일 시퀀스 처리"""
    try:
        src_folder = os.path.join(src_root, data_info['path_info'])
        
        # 목적지 폴더 설정
        if split_type == 'train':
            person_folder = os.path.join(dst_root, 'train', data_info['person_id'])
            dst_folder = os.path.join(person_folder, 
                                    f"{data_info['condition']}-{data_info['sequence']}-{data_info['angle']}")
        elif split_type == 'gallery':
            person_folder = os.path.join(dst_root, 'test', 'gallery', data_info['person_id'])
            dst_folder = os.path.join(person_folder, 
                                    f"{data_info['condition']}-{data_info['sequence']}-{data_info['angle']}")
        else:  # probe
            person_folder = os.path.join(dst_root, 'test', 'probe', data_info['person_id'])
            dst_folder = os.path.join(person_folder, 
                                    f"{data_info['condition']}-{data_info['sequence']}-{data_info['angle']}")
        
        os.makedirs(dst_folder, exist_ok=True)
        os.makedirs(person_folder, exist_ok=True)  # person 폴더도 생성
        
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
                
                # 실루엣 저장
                silhouette_path = os.path.join(dst_folder, f"{base_name}.png")
                cv2.imwrite(silhouette_path, silhouette)
                processed += 1
                
                # Pose 추출
                pose_data = extract_pose_yolo(src_file)
                if pose_data is not None:
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
        
        # 🔥 GEI 생성 - 상위 폴더(person 폴더)에 저장
        gei_generated = False
        if processed > 0:
            gei_filename = f"{data_info['condition']}-{data_info['sequence']}-{data_info['angle']}_gei.png"
            gei_path = os.path.join(person_folder, gei_filename)  # dst_folder가 아닌 person_folder에 저장
            gei_generated = generate_gei(dst_folder, gei_path)
        
        return processed, failed, pose_success, pose_failed, gei_generated
        
    except Exception as e:
        print(f"시퀀스 처리 오류: {e}")
        return 0, 0, 0, 0, False


def process_data(data_dict, save_original=False):
    """데이터 처리"""
    total_processed = 0
    total_failed = 0
    total_pose_success = 0
    total_pose_failed = 0
    gei_generated = 0
    
    for split_type, data_list in data_dict.items():
        print(f"\nProcessing {split_type}...")
        
        with tqdm(total=len(data_list), desc=f"{split_type}") as pbar:
            for data_info in data_list:
                processed, failed, pose_success, pose_failed, gei_success = process_sequence(
                    data_info, split_type, save_original
                )
                
                total_processed += processed
                total_failed += failed
                total_pose_success += pose_success
                total_pose_failed += pose_failed
                if gei_success:
                    gei_generated += 1
                
                pbar.update(1)
    
    return total_processed, total_failed, total_pose_success, total_pose_failed, gei_generated

def print_statistics(data_dict):
    """통계 출력"""
    print("\n=== CASIA-B 데이터 분할 통계 ===")
    
    for split_type, data_list in data_dict.items():
        person_counts = defaultdict(int)
        condition_counts = defaultdict(int)
        angle_counts = defaultdict(int)
        
        for data in data_list:
            person_counts[data['person_id']] += 1
            condition_counts[data['condition']] += 1
            angle_counts[data['angle']] += 1
        
        print(f"\n{split_type}:")
        print(f"  - 시퀀스 수: {len(data_list)}")
        print(f"  - 사람 수: {len(person_counts)}")
        print(f"  - 조건 분포: {dict(condition_counts)}")
        print(f"  - 각도 분포: {dict(angle_counts)}")
    
    # 전체 통계
    total_sequences = sum(len(data_list) for data_list in data_dict.values())
    total_persons = len(set(data['person_id'] for data_list in data_dict.values() for data in data_list))
    
    print(f"\n전체 통계:")
    print(f"  - 전체 시퀀스: {total_sequences}")
    print(f"  - 전체 사람: {total_persons}")
    print(f"  - 분할 비율: {len(data_dict['train'])} : {len(data_dict['gallery'])} : {len(data_dict['probe'])}")

def main(save_original=False):
    """메인 함수"""
    # 출력 디렉토리 확인
    if os.path.exists(dst_root):
        response = input(f"\n{dst_root}가 이미 존재합니다. 덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("작업을 취소했습니다.")
            return
        else:
            print(f"{dst_root} 삭제 중...")
            shutil.rmtree(dst_root)
    
    # CASIA-B 데이터 구조 분석 및 재구성
    data_dict = reorganize_casia_data()
    
    if not data_dict or not any(data_dict.values()):
        print("처리할 데이터가 없습니다!")
        return
    
    # 통계 출력
    print_statistics(data_dict)
    
    # 설정 안내
    save_option_text = "원본 이미지도 함께 저장됩니다." if save_original else "원본 이미지는 저장되지 않습니다."
    print(f"\n설정: {save_option_text}")
    print("🔧 CASIA-B 데이터셋 구조에 맞춰 처리합니다.")
    print("📁 출력 구조: person_id/condition-sequence-angle/")
    
    # 사용자 확인
    response = input("\n계속 진행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("작업을 취소했습니다.")
        return
    
    # 데이터 처리
    processed, failed, pose_success, pose_failed, gei_count = process_data(data_dict, save_original)
    
    print(f'\n=== 처리 완료 ===')
    print(f'성공적으로 처리된 이미지: {processed}')
    print(f'처리 실패한 이미지: {failed}')
    print(f'포즈 추출 성공: {pose_success}')
    print(f'포즈 추출 실패: {pose_failed}')
    print(f'생성된 GEI 이미지: {gei_count}')
    if processed + failed > 0:
        print(f'실루엣 추출 성공률: {processed/(processed+failed)*100:.1f}%')
    if pose_success + pose_failed > 0:
        print(f'포즈 추출 성공률: {pose_success/(pose_success+pose_failed)*100:.1f}%')

if __name__ == '__main__':
    # 원본 이미지 저장 여부 설정
    SAVE_ORIGINAL = False
    
    main(save_original=SAVE_ORIGINAL)
