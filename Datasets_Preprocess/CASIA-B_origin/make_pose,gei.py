import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
import shutil

# 경로 설정
src_root = os.path.normpath('./CASIA-B-reorganized')
dst_root = os.path.normpath('./CASIA-B-processed')

# YOLO 모델 로드
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

def scan_casia_structure():
    """CASIA-B-reorganized 구조를 스캔하여 데이터 수집"""
    print("\n=== CASIA-B 구조 스캔 ===")
    
    data_dict = {
        'train': [],
        'test_gallery': [],
        'test_probe': []
    }
    
    if not os.path.exists(src_root):
        print(f"Error: {src_root} 디렉토리가 존재하지 않습니다!")
        return data_dict
    
    # train 폴더 스캔
    train_path = os.path.join(src_root, 'train')
    if os.path.exists(train_path):
        print("스캔 중: train/")
        for person_condition in os.listdir(train_path):
            person_condition_path = os.path.join(train_path, person_condition)
            if os.path.isdir(person_condition_path):
                # person_condition: "001-bg-01" 형태
                parts = person_condition.split('-')
                if len(parts) >= 3:
                    person_id = parts[0]  # "001"
                    condition = f"{parts[1]}-{parts[2]}"  # "bg-01"
                    
                    # 각도 폴더들 스캔
                    for angle_folder in os.listdir(person_condition_path):
                        angle_path = os.path.join(person_condition_path, angle_folder)
                        if os.path.isdir(angle_path):
                            img_files = [f for f in os.listdir(angle_path) if is_image_file(f)]
                            if img_files:
                                data_dict['train'].append({
                                    'person_id': person_id,
                                    'condition': condition,
                                    'angle': angle_folder,
                                    'path': angle_path,
                                    'images': len(img_files)
                                })
    
    # test/gallery 폴더 스캔
    gallery_path = os.path.join(src_root, 'test', 'gallery')
    if os.path.exists(gallery_path):
        print("스캔 중: test/gallery/")
        for person_condition in os.listdir(gallery_path):
            person_condition_path = os.path.join(gallery_path, person_condition)
            if os.path.isdir(person_condition_path):
                parts = person_condition.split('-')
                if len(parts) >= 3:
                    person_id = parts[0]
                    condition = f"{parts[1]}-{parts[2]}"
                    
                    for angle_folder in os.listdir(person_condition_path):
                        angle_path = os.path.join(person_condition_path, angle_folder)
                        if os.path.isdir(angle_path):
                            img_files = [f for f in os.listdir(angle_path) if is_image_file(f)]
                            if img_files:
                                data_dict['test_gallery'].append({
                                    'person_id': person_id,
                                    'condition': condition,
                                    'angle': angle_folder,
                                    'path': angle_path,
                                    'images': len(img_files)
                                })
    
    # test/probe 폴더 스캔
    probe_path = os.path.join(src_root, 'test', 'probe')
    if os.path.exists(probe_path):
        print("스캔 중: test/probe/")
        for person_condition in os.listdir(probe_path):
            person_condition_path = os.path.join(probe_path, person_condition)
            if os.path.isdir(person_condition_path):
                parts = person_condition.split('-')
                if len(parts) >= 3:
                    person_id = parts[0]
                    condition = f"{parts[1]}-{parts[2]}"
                    
                    for angle_folder in os.listdir(person_condition_path):
                        angle_path = os.path.join(person_condition_path, angle_folder)
                        if os.path.isdir(angle_path):
                            img_files = [f for f in os.listdir(angle_path) if is_image_file(f)]
                            if img_files:
                                data_dict['test_probe'].append({
                                    'person_id': person_id,
                                    'condition': condition,
                                    'angle': angle_folder,
                                    'path': angle_path,
                                    'images': len(img_files)
                                })
    
    print(f"발견된 데이터:")
    print(f"  Train: {len(data_dict['train'])}개 시퀀스")
    print(f"  Test Gallery: {len(data_dict['test_gallery'])}개 시퀀스")
    print(f"  Test Probe: {len(data_dict['test_probe'])}개 시퀀스")
    
    return data_dict

def resize_silhouette_with_ratio(silhouette_img, target_size=64):
    """실루엣 이미지를 비율에 맞춰 64x64로 리사이즈"""
    if silhouette_img is None:
        return None
    
    # 이미지가 컬러인지 그레이스케일인지 확인하고 그레이스케일로 변환
    if len(silhouette_img.shape) == 3:
        silhouette_img = cv2.cvtColor(silhouette_img, cv2.COLOR_BGR2GRAY)
    
    # 실루엣에서 사람 영역 찾기
    coords = np.where(silhouette_img > 128)  # 임계값 128
    if len(coords[0]) == 0:
        return None
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    # 바운딩 박스로 크롭
    cropped_silhouette = silhouette_img[y_min:y_max+1, x_min:x_max+1]
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

def extract_pose_from_silhouette(silhouette_path):
    """실루엣 이미지에서 YOLOv8을 사용하여 COCO 17 keypoints 추출"""
    try:
        # 이미지 로드
        img = cv2.imread(silhouette_path)
        if img is None:
            return None
        
        # 이미지가 그레이스케일인지 확인하고 3채널로 변환
        if len(img.shape) == 2:
            # 그레이스케일인 경우 3채널로 변환
            img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            # 단일 채널인 경우 3채널로 변환
            img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # 이미 3채널인 경우 그대로 사용
            img_3ch = img
        else:
            return None
        
        # YOLOv8 pose 실행
        results = pose_model(img_3ch, verbose=False)
        
        # 결과 검증
        if not results or len(results) == 0:
            return None
            
        result = results[0]
        
        # keypoints 속성 확인
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return None
            
        # keypoints 데이터 확인
        if len(result.keypoints) == 0 or result.keypoints.xy is None:
            return None
            
        # 첫 번째 사람의 keypoints 가져오기
        if len(result.keypoints.xy) == 0:
            return None
            
        keypoints = result.keypoints.xy[0].cpu().numpy()  # (17, 2) 형태
        confidences = result.keypoints.conf[0].cpu().numpy()  # (17,) 형태
        
        # keypoints 데이터 검증
        if keypoints.shape[0] != 17 or confidences.shape[0] != 17:
            return None
        
        # 이미지 크기로 정규화
        h, w = img_3ch.shape[:2]
        
        pose_data = []
        for i in range(17):  # COCO는 17개 keypoints
            x, y = keypoints[i]
            conf = confidences[i]
            # 정규화된 좌표로 변환
            x_norm = float(x / w) if w > 0 else 0.0
            y_norm = float(y / h) if h > 0 else 0.0
            conf_val = float(conf) if not np.isnan(conf) else 0.0
            pose_data.append([x_norm, y_norm, 0.0, conf_val])  # z는 0으로 설정
        
        return pose_data
        
    except Exception as e:
        # 디버깅을 위해 상세한 오류 정보 출력 (선택적)
        # print(f"Error in pose extraction for {silhouette_path}: {str(e)}")
        return None

def save_pose_to_txt(pose_data, txt_path):
    """COCO 17 keypoints를 txt 파일로 저장"""
    try:
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, 'w') as f:
            # 헤더 추가
            f.write("# COCO 17 keypoints format\n")
            f.write("# x_normalized y_normalized z confidence\n")
            for i, landmark in enumerate(pose_data):
                f.write(f"{landmark[0]:.6f} {landmark[1]:.6f} {landmark[2]:.6f} {landmark[3]:.6f}\n")
        return True
    except Exception as e:
        return False

def generate_gei(silhouette_folder, output_path):
    """실루엣 이미지들로부터 GEI 생성"""
    try:
        silhouette_files = [f for f in os.listdir(silhouette_folder) 
                           if f.endswith('.png') and not f.endswith('_gei.png')]
        
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
        cv2.imwrite(output_path, gei)
        return True
        
    except Exception as e:
        return False

def process_sequence(seq_data, split_type):
    """단일 시퀀스 처리"""
    person_id = seq_data['person_id']
    condition = seq_data['condition']
    angle = seq_data['angle']
    src_path = seq_data['path']
    
    processed = 0
    failed = 0
    pose_success = 0
    
    # 대상 폴더 경로 설정
    if split_type == 'train':
        dst_folder = os.path.join(dst_root, 'train', f"{person_id}-{condition}", angle)
    elif split_type == 'test_gallery':
        dst_folder = os.path.join(dst_root, 'test', 'gallery', f"{person_id}-{condition}", angle)
    else:  # test_probe
        dst_folder = os.path.join(dst_root, 'test', 'probe', f"{person_id}-{condition}", angle)
    
    os.makedirs(dst_folder, exist_ok=True)
    
    # 이미지 파일들 처리
    img_files = [f for f in os.listdir(src_path) if is_image_file(f)]
    
    for img_file in sorted(img_files):
        src_img_path = os.path.join(src_path, img_file)
        base_name = os.path.splitext(img_file)[0]
        
        try:
            # 원본 실루엣 이미지 로드 (안전하게)
            silhouette = cv2.imread(src_img_path)
            if silhouette is None:
                # 다른 방법으로 시도
                silhouette = cv2.imread(src_img_path, cv2.IMREAD_UNCHANGED)
                if silhouette is None:
                    failed += 1
                    continue
            
            # 64x64로 리사이즈 (비율 유지)
            resized_silhouette = resize_silhouette_with_ratio(silhouette, target_size=64)
            if resized_silhouette is None:
                failed += 1
                continue
            
            # 리사이즈된 실루엣 저장
            silhouette_path = os.path.join(dst_folder, f"{base_name}.png")
            cv2.imwrite(silhouette_path, resized_silhouette)
            
            # 원본 실루엣에서 포즈 추출 (더 정확한 추출을 위해)
            pose_data = extract_pose_from_silhouette(src_img_path)
            if pose_data is not None:
                # 포즈 데이터 저장
                pose_path = os.path.join(dst_folder, f"{base_name}_2d_pose.txt")
                if save_pose_to_txt(pose_data, pose_path):
                    pose_success += 1
            
            processed += 1
            
        except Exception as e:
            failed += 1
    
    # GEI 생성
    gei_generated = False
    if processed > 0:
        # GEI 저장 경로
        if split_type == 'train':
            gei_path = os.path.join(dst_root, 'train', f"{person_id}-{condition}", f"{person_id}-{condition}-{angle}_gei.png")
        elif split_type == 'test_gallery':
            gei_path = os.path.join(dst_root, 'test', 'gallery', f"{person_id}-{condition}", f"{person_id}-{condition}-{angle}_gei.png")
        else:  # test_probe
            gei_path = os.path.join(dst_root, 'test', 'probe', f"{person_id}-{condition}", f"{person_id}-{condition}-{angle}_gei.png")
        
        gei_generated = generate_gei(dst_folder, gei_path)
    
    return processed, failed, gei_generated, pose_success

def process_all_data(data_dict):
    """모든 데이터 처리"""
    total_processed = 0
    total_failed = 0
    total_gei = 0
    total_pose = 0
    
    for split_type, seq_list in data_dict.items():
        if not seq_list:
            continue
            
        print(f"\nProcessing {split_type}...")
        
        with tqdm(total=len(seq_list), desc=f"{split_type}") as pbar:
            for seq_data in seq_list:
                processed, failed, gei_success, pose_success = process_sequence(seq_data, split_type)
                
                total_processed += processed
                total_failed += failed
                total_pose += pose_success
                if gei_success:
                    total_gei += 1
                
                pbar.set_postfix({
                    'img': f'{processed}/{processed+failed}',
                    'pose': pose_success,
                    'gei': 'OK' if gei_success else 'FAIL'
                })
                pbar.update(1)
    
    return total_processed, total_failed, total_gei, total_pose

def print_statistics(data_dict):
    """통계 출력"""
    print("\n=== 데이터 통계 ===")
    
    for split_type, seq_list in data_dict.items():
        if not seq_list:
            continue
            
        unique_persons = set()
        unique_conditions = set()
        unique_angles = set()
        total_images = 0
        
        for seq_data in seq_list:
            unique_persons.add(seq_data['person_id'])
            unique_conditions.add(seq_data['condition'])
            unique_angles.add(seq_data['angle'])
            total_images += seq_data['images']
        
        print(f"\n{split_type}:")
        print(f"  - 시퀀스 수: {len(seq_list)}")
        print(f"  - 고유 인물 수: {len(unique_persons)}")
        print(f"  - 고유 조건 수: {len(unique_conditions)}")
        print(f"  - 고유 각도 수: {len(unique_angles)}")
        print(f"  - 총 이미지 수: {total_images}")

def main():
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
    
    # CASIA-B 구조 스캔
    data_dict = scan_casia_structure()
    
    if not any(data_dict.values()):
        print("처리할 데이터가 없습니다!")
        return
    
    # 통계 출력
    print_statistics(data_dict)
    
    # 사용자 확인
    response = input("\n계속 진행하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("작업을 취소했습니다.")
        return
    
    # 데이터 처리
    processed, failed, gei_count, pose_count = process_all_data(data_dict)
    
    print(f'\n=== 처리 완료 ===')
    print(f'성공적으로 처리된 이미지: {processed}')
    print(f'처리 실패한 이미지: {failed}')
    print(f'성공적으로 추출된 포즈: {pose_count}')
    print(f'생성된 GEI 이미지: {gei_count}')
    if processed + failed > 0:
        print(f'이미지 처리 성공률: {processed/(processed+failed)*100:.1f}%')
    if processed > 0:
        print(f'포즈 추출 성공률: {pose_count/processed*100:.1f}%')

if __name__ == '__main__':
    main()
