import os
import cv2
import shutil
from pathlib import Path
import numpy as np

def extract_frames_from_video(video_path, output_dir, num_frames=60):
    """
    비디오에서 균등하게 분포된 프레임을 추출합니다.
    
    Args:
        video_path: 비디오 파일 경로
        output_dir: 프레임을 저장할 디렉토리
        num_frames: 추출할 프레임 수 (기본값: 60)
    
    Returns:
        추출된 프레임 수
    """
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0
    
    # 비디오 정보 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0:
        print(f"Error: No frames in video {video_path}")
        cap.release()
        return 0
    
    # 균등한 간격으로 프레임 인덱스 계산
    if total_frames <= num_frames:
        # 비디오 프레임이 원하는 수보다 적으면 모든 프레임 사용
        frame_indices = list(range(total_frames))
    else:
        # 균등한 간격으로 프레임 선택
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_count = 0
    
    for i, frame_idx in enumerate(frame_indices):
        # 특정 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # 파일명: 00001.png, 00002.png, ...
            filename = f"{i+1:05d}.png"
            output_path = os.path.join(output_dir, filename)
            
            # 프레임 저장
            cv2.imwrite(output_path, frame)
            extracted_count += 1
        else:
            print(f"Warning: Could not read frame {frame_idx} from {video_path}")
    
    cap.release()
    return extracted_count

def reorganize_casia_b_video_dataset(source_dir, target_dir, frames_per_video=60):
    """
    CASIA-B 비디오 데이터셋을 재구성합니다.
    
    원본 구조: CASIA-B-video/001-bg-01-000.avi
    목표 구조: CASIA-B-reorganized/train/001-bg-01/000/*.png
              CASIA-B-reorganized/test/gallery/001-bg-02/000/*.png  
              CASIA-B-reorganized/test/probe/001-cl-02/000/*.png
    """
    
    # 목표 디렉토리 생성
    train_dir = os.path.join(target_dir, 'train')
    test_gallery_dir = os.path.join(target_dir, 'test', 'gallery')
    test_probe_dir = os.path.join(target_dir, 'test', 'probe')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_gallery_dir, exist_ok=True)
    os.makedirs(test_probe_dir, exist_ok=True)
    
    # 분할 규칙 정의
    train_conditions = ['bg-01', 'cl-01', 'nm-01', 'nm-03', 'nm-05']
    test_gallery_conditions = ['bg-02', 'nm-02', 'nm-04']
    test_probe_conditions = ['cl-02', 'nm-06']
    
    # 비디오 확장자
    video_extensions = ['.avi', '.mp4', '.mov', '.mkv']
    
    total_processed = 0
    total_frames_extracted = 0
    
    # 소스 디렉토리의 비디오 파일들 순회
    for video_file in sorted(os.listdir(source_dir)):
        # 비디오 파일인지 확인
        if not any(video_file.lower().endswith(ext) for ext in video_extensions):
            continue
            
        # 파일명에서 정보 추출: 001-bg-01-000.avi
        filename_parts = video_file.replace('.avi', '').replace('.mp4', '').replace('.mov', '').replace('.mkv', '').split('-')
        
        if len(filename_parts) < 3:
            print(f"Invalid filename format: {video_file}, skipping...")
            continue
            
        person_id = filename_parts[0]
        condition_type = filename_parts[1]
        condition_number = filename_parts[2]
        angle = filename_parts[3] if len(filename_parts) > 3 else '000'
        
        condition_full = f"{condition_type}-{condition_number}"
        
        # 목적지 결정
        if condition_full in train_conditions:
            dest_base = train_dir
        elif condition_full in test_gallery_conditions:
            dest_base = test_gallery_dir
        elif condition_full in test_probe_conditions:
            dest_base = test_probe_dir
        else:
            print(f"Unknown condition: {condition_full}, skipping {video_file}...")
            continue
        
        # 새로운 폴더명: person_id-condition_type-condition_number
        new_condition_name = f"{person_id}-{condition_type}-{condition_number}"
        
        # 목적지 디렉토리 구조 생성
        dest_condition_dir = os.path.join(dest_base, new_condition_name)
        dest_angle_dir = os.path.join(dest_condition_dir, angle)
        os.makedirs(dest_angle_dir, exist_ok=True)
        
        # 비디오 파일 경로
        video_path = os.path.join(source_dir, video_file)
        
        # 프레임 추출
        frames_extracted = extract_frames_from_video(
            video_path, dest_angle_dir, frames_per_video
        )
        
        if frames_extracted > 0:
            total_processed += 1
            total_frames_extracted += frames_extracted
            print(f"Extracted {frames_extracted} frames from {video_file} -> {new_condition_name}/{angle}/")
        else:
            print(f"Failed to extract frames from {video_file}")

    print(f"\n=== Processing Complete ===")
    print(f"Total videos processed: {total_processed}")
    print(f"Total frames extracted: {total_frames_extracted}")

def print_dataset_statistics(target_dir):
    """데이터셋 통계 출력"""
    
    def count_samples(directory):
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len([f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return count
    
    def count_folders(directory):
        folder_count = 0
        for root, dirs, files in os.walk(directory):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                folder_count += 1
        return folder_count
    
    train_dir = os.path.join(target_dir, 'train')
    test_gallery_dir = os.path.join(target_dir, 'test', 'gallery')
    test_probe_dir = os.path.join(target_dir, 'test', 'probe')
    
    train_samples = count_samples(train_dir)
    test_gallery_samples = count_samples(test_gallery_dir)
    test_probe_samples = count_samples(test_probe_dir)
    
    train_folders = count_folders(train_dir)
    test_gallery_folders = count_folders(test_gallery_dir)
    test_probe_folders = count_folders(test_probe_dir)
    
    print("\n=== Dataset Statistics ===")
    print(f"Train: {train_samples} frames in {train_folders} sequences")
    print(f"Test Gallery: {test_gallery_samples} frames in {test_gallery_folders} sequences")
    print(f"Test Probe: {test_probe_samples} frames in {test_probe_folders} sequences")
    print(f"Total: {train_samples + test_gallery_samples + test_probe_samples} frames")
    
    # 폴더 구조 샘플 출력
    print("\n=== Sample Directory Structure ===")
    
    # Train 구조
    print("train/")
    if os.path.exists(train_dir):
        folders = sorted(os.listdir(train_dir))[:3]
        for folder in folders:
            folder_path = os.path.join(train_dir, folder)
            if os.path.isdir(folder_path):
                print(f"  ├── {folder}/")
                angle_folders = sorted(os.listdir(folder_path))[:3]
                for i, angle in enumerate(angle_folders):
                    angle_path = os.path.join(folder_path, angle)
                    if os.path.isdir(angle_path):
                        image_count = len([f for f in os.listdir(angle_path) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        prefix = "  │   ├──" if i < len(angle_folders)-1 else "  │   └──"
                        print(f"{prefix} {angle}/ ({image_count} images)")
        if len(folders) > 3:
            print("  └── ...")
    
    # Test 구조
    print("\ntest/")
    print("  ├── gallery/")
    if os.path.exists(test_gallery_dir):
        folders = sorted(os.listdir(test_gallery_dir))[:2]
        for folder in folders:
            folder_path = os.path.join(test_gallery_dir, folder)
            if os.path.isdir(folder_path):
                print(f"  │   ├── {folder}/")
                angle_folders = sorted(os.listdir(folder_path))[:2]
                for angle in angle_folders:
                    angle_path = os.path.join(folder_path, angle)
                    if os.path.isdir(angle_path):
                        image_count = len([f for f in os.listdir(angle_path) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        print(f"  │   │   └── {angle}/ ({image_count} images)")
    
    print("  └── probe/")
    if os.path.exists(test_probe_dir):
        folders = sorted(os.listdir(test_probe_dir))[:2]
        for folder in folders:
            folder_path = os.path.join(test_probe_dir, folder)
            if os.path.isdir(folder_path):
                print(f"      ├── {folder}/")
                angle_folders = sorted(os.listdir(folder_path))[:2]
                for angle in angle_folders:
                    angle_path = os.path.join(folder_path, angle)
                    if os.path.isdir(angle_path):
                        image_count = len([f for f in os.listdir(angle_path) 
                                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                        print(f"      │   └── {angle}/ ({image_count} images)")

# 사용 예시
if __name__ == "__main__":
    source_directory = r"C:\Users\user\Desktop\Data_pre\CASIA-B-video"  # 원본 비디오 데이터셋 경로
    target_directory = r"C:\Users\user\Desktop\Data_pre\CASIA-B-frame-reorganized"  # 재구성된 데이터셋 저장 경로
    
    print("Starting CASIA-B video dataset reorganization...")
    print(f"Source: {source_directory}")
    print(f"Target: {target_directory}")
    print(f"Frames per video: 60")
    print()
    
    reorganize_casia_b_video_dataset(source_directory, target_directory, frames_per_video=60)
    
    print("\nReorganization completed!")
    print_dataset_statistics(target_directory)