import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from tqdm import tqdm
import random
import shutil

# 경로 설정
src_root = os.path.normpath('./CCVID')
dst_root = os.path.normpath('./Grew_ccvid')

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

def check_txt_files():
    """txt 파일 구조 확인"""
    print("\n=== TXT 파일 확인 ===")
    
    txt_dir = os.path.join(src_root)
    
    for filename in ['train.txt', 'gallery.txt', 'query.txt']:
        txt_path = os.path.join(txt_dir, filename)
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                print(f"\n{filename}:")
                print(f"  총 라인 수: {len(lines)}")
                if lines:
                    print(f"  첫 번째 라인: {lines[0].strip()}")
                    print(f"  마지막 라인: {lines[-1].strip()}")
                    
                    # 세션 분포 확인
                    session_counts = defaultdict(int)
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            session = parts[0].split('/')[0]
                            session_counts[session] += 1
                    print(f"  세션 분포: {dict(session_counts)}")
        else:
            print(f"\n{txt_path} 파일이 없습니다!")

def parse_all_data():
    """모든 데이터를 먼저 읽고 6:2:2로 분할"""
    all_data = []
    
    txt_dir = os.path.join(src_root)
    
    # 모든 txt 파일에서 데이터 수집
    for filename in ['train.txt', 'gallery.txt', 'query.txt']:
        txt_path = os.path.join(txt_dir, filename)
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        path = parts[0]  # session1/001_01
                        id_info = parts[1]  # 001
                        
                        # 실제 폴더 존재 확인
                        full_path = os.path.join(src_root, path)
                        if os.path.exists(full_path):
                            all_data.append((path, id_info))
                        else:
                            print(f"Warning: {full_path} not found")
    
    print(f"\n전체 데이터 수: {len(all_data)}")
    
    # ID별로 그룹화
    id_groups = defaultdict(list)
    for path, id_info in all_data:
        id_groups[id_info].append(path)
    
    print(f"전체 고유 ID 수: {len(id_groups)}")
    
    # ID 리스트를 섞어서 무작위로 분할
    all_ids = list(id_groups.keys())
    random.shuffle(all_ids)
    
    # 6:2:2 비율로 분할
    total_ids = len(all_ids)
    train_size = int(total_ids * 0.6)
    gallery_size = int(total_ids * 0.2)
    
    train_ids = set(all_ids[:train_size])
    gallery_ids = set(all_ids[train_size:train_size + gallery_size])
    probe_ids = set(all_ids[train_size + gallery_size:])
    
    print(f"\nID 분할:")
    print(f"  Train IDs: {len(train_ids)}")
    print(f"  Gallery IDs: {len(gallery_ids)}")
    print(f"  Probe IDs: {len(probe_ids)}")
    
    # 각 split에 데이터 할당
    data_dict = {
        'train': [],
        'gallery': [],
        'probe': []
    }
    
    for path, id_info in all_data:
        if id_info in train_ids:
            data_dict['train'].append((path, id_info))
        elif id_info in gallery_ids:
            data_dict['gallery'].append((path, id_info))
        elif id_info in probe_ids:
            data_dict['probe'].append((path, id_info))
    
    return data_dict

def extract_silhouette_and_resize(img_path, target_size=64):
    """이미지에서 사람 실루엣 추출하고 64x64로 리사이즈"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # YOLOv8 segmentation 실행
    results = seg_model(img)
    
    if len(results[0].masks) == 0:
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

def extract_pose_yolo(img_path):
    """YOLOv8을 사용하여 이미지에서 COCO 17 keypoints 추출"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # YOLOv8 pose 실행
    results = pose_model(img)
    
    if len(results[0].keypoints) == 0:
        return None
    
    # 첫 번째 사람의 keypoints 가져오기
    keypoints = results[0].keypoints.xy[0].cpu().numpy()  # (17, 2) 형태
    confidences = results[0].keypoints.conf[0].cpu().numpy()  # (17,) 형태
    
    # 이미지 크기로 정규화
    h, w = img.shape[:2]
    
    pose_data = []
    for i in range(17):  # COCO는 17개 keypoints
        x, y = keypoints[i]
        conf = confidences[i]
        # 정규화된 좌표로 변환
        x_norm = x / w
        y_norm = y / h
        pose_data.append([x_norm, y_norm, 0.0, conf])  # z는 0으로 설정
    
    return pose_data

def save_pose_to_txt(pose_data, txt_path):
    """COCO 17 keypoints를 txt 파일로 저장"""
    with open(txt_path, 'w') as f:
        # 헤더 추가
        f.write("# COCO 17 keypoints format\n")
        f.write("# x_normalized y_normalized z confidence\n")
        for i, landmark in enumerate(pose_data):
            f.write(f"{landmark[0]:.6f} {landmark[1]:.6f} {landmark[2]:.6f} {landmark[3]:.6f}\n")

def generate_gei(silhouette_folder, output_path):
    """실루엣 이미지들로부터 GEI 생성"""
    silhouette_files = [f for f in os.listdir(silhouette_folder) 
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
    
    # GEI 저장
    cv2.imwrite(output_path, gei)
    return True

def process_sequence(src_folder, dst_folder, id_info, seq, split_type, save_original=False):
    """단일 시퀀스 처리 및 GEI 생성"""
    processed = 0
    failed = 0
    
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
            
            # Pose 추출 (YOLOv8 사용)
            pose_data = extract_pose_yolo(src_file)
            if pose_data is not None:
                # Pose 데이터 저장
                pose_path = os.path.join(dst_folder, f"{base_name}_2d_pose.txt")
                save_pose_to_txt(pose_data, pose_path)
            
            processed += 1
            
        except Exception as e:
            print(f"\nError processing {src_file}: {str(e)}")
            failed += 1
    
    # 시퀀스 처리 완료 후 즉시 GEI 생성
    gei_generated = False
    if processed > 0:
        # GEI 저장 경로
        if split_type == 'train':
            gei_path = os.path.join(dst_root, 'train', id_info, f"{id_info}_{seq}_gei.png")
        elif split_type == 'gallery':
            gei_path = os.path.join(dst_root, 'test', 'gallery', id_info, f"{id_info}_{seq}_gei.png")
        else:  # probe
            gei_path = os.path.join(dst_root, 'test', 'probe', id_info, f"{id_info}_{seq}_gei.png")
        
        gei_generated = generate_gei(dst_folder, gei_path)
    
    return processed, failed, gei_generated

def process_data(data_dict, save_original=False):
    """데이터 처리 - 실루엣, pose, GEI 생성"""
    total_processed = 0
    total_failed = 0
    gei_generated = 0
    
    for split_type, data_list in data_dict.items():
        print(f"\nProcessing {split_type}...")
        
        with tqdm(total=len(data_list), desc=f"{split_type}") as pbar:
            for path_info, id_info in data_list:
                # path_info: session1/001_01 형태
                parts = path_info.split('/')
                if len(parts) != 2:
                    pbar.update(1)
                    continue
                    
                session = parts[0]  # session1 or session2
                folder_name = parts[1]  # 001_01
                
                # 폴더명에서 시퀀스 추출
                folder_parts = folder_name.split('_')
                if len(folder_parts) < 2:
                    pbar.update(1)
                    continue
                    
                seq = folder_parts[1]  # 01
                
                # 원본 폴더 경로
                src_folder = os.path.join(src_root, session, folder_name)
                if not os.path.exists(src_folder):
                    print(f"\nWarning: {src_folder} not found")
                    pbar.update(1)
                    continue
                
                # 대상 폴더 경로 설정
                if split_type == 'train':
                    dst_folder = os.path.join(dst_root, 'train', id_info, seq)
                elif split_type == 'gallery':
                    dst_folder = os.path.join(dst_root, 'test', 'gallery', id_info, seq)
                else:  # probe
                    dst_folder = os.path.join(dst_root, 'test', 'probe', id_info, seq)
                
                os.makedirs(dst_folder, exist_ok=True)
                
                # 시퀀스 처리 및 GEI 생성
                processed, failed, gei_success = process_sequence(
                    src_folder, dst_folder, id_info, seq, split_type, save_original
                )
                
                total_processed += processed
                total_failed += failed
                if gei_success:
                    gei_generated += 1
                
                pbar.update(1)
    
    return total_processed, total_failed, gei_generated

def print_statistics(data_dict):
    """통계 출력"""
    print("\n=== 데이터 분할 통계 ===")
    
    # 각 split별 unique ID 수집
    all_ids = {'train': set(), 'gallery': set(), 'probe': set()}
    
    for split_type, data_list in data_dict.items():
        unique_ids = set()
        session_counts = defaultdict(int)
        
        for path_info, id_info in data_list:
            unique_ids.add(id_info)
            all_ids[split_type].add(id_info)
            session = path_info.split('/')[0]
            session_counts[session] += 1
        
        print(f"\n{split_type}:")
        print(f"  - Total samples: {len(data_list)}")
        print(f"  - Unique IDs: {len(unique_ids)}")
        print(f"  - Session distribution: {dict(session_counts)}")
    
    # ID 중복 검사
    print("\n=== ID 중복 검사 ===")
    train_gallery_overlap = all_ids['train'] & all_ids['gallery']
    train_probe_overlap = all_ids['train'] & all_ids['probe']
    gallery_probe_overlap = all_ids['gallery'] & all_ids['probe']
    
    print(f"Train-Gallery 중복 ID: {len(train_gallery_overlap)} 개")
    print(f"Train-Probe 중복 ID: {len(train_probe_overlap)} 개")
    print(f"Gallery-Probe 중복 ID: {len(gallery_probe_overlap)} 개")
    
    # 전체 비율 확인
    total = sum(len(data_list) for data_list in data_dict.values())
    print(f"\n전체 샘플 수: {total}")
    print(f"Train : Gallery : Probe = {len(data_dict['train'])} : {len(data_dict['gallery'])} : {len(data_dict['probe'])}")
    print(f"비율: {len(data_dict['train'])/total*100:.1f}% : {len(data_dict['gallery'])/total*100:.1f}% : {len(data_dict['probe'])/total*100:.1f}%")

def main(save_original=False):
    """메인 함수
    
    Args:
        save_original (bool): True면 원본 이미지도 함께 저장, False면 저장하지 않음
    """
    # 출력 디렉토리가 이미 존재하는지 확인
    if os.path.exists(dst_root):
        response = input(f"\n{dst_root}가 이미 존재합니다. 덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("작업을 취소했습니다.")
            return
        else:
            print(f"{dst_root} 삭제 중...")
            shutil.rmtree(dst_root)
    
    # txt 파일 구조 확인
    check_txt_files()
    
    # 모든 데이터를 읽고 6:2:2로 분할
    data_dict = parse_all_data()
    
    # 통계 출력
    print_statistics(data_dict)
    
    # 원본 저장 옵션 안내
    save_option_text = "원본 이미지도 함께 저장됩니다." if save_original else "원본 이미지는 저장되지 않습니다."
    print(f"\n설정: {save_option_text}")
    
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
    # True: 원본 이미지를 _origin 접미사로 함께 저장
    # False: 원본 이미지는 저장하지 않음 (64x64 실루엣과 포즈 데이터만 저장)
    SAVE_ORIGINAL = False
    
    main(save_original=SAVE_ORIGINAL)