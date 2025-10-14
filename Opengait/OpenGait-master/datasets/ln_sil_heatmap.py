import os
import sys
import argparse
from tqdm import tqdm
from glob import glob
from pathlib import Path          # ★ 추가
import shutil                     # ★ 추가

def get_args():
    parser = argparse.ArgumentParser(
        description='Symlink silhouette data와 heatmap 데이터를 동일 폴더에 모아 SkeletonGait++ 학습용으로 정리.'
    )
    parser.add_argument('--heatmap_data_path', type=str, required=True,
                        help="heatmap 루트 경로(절대경로).")
    parser.add_argument('--silhouette_data_path', type=str, required=True,
                        help="silhouette 루트 경로(절대경로).")
    parser.add_argument('--dataset_pkl_ext_name', type=str, default='.pkl',
                        help="silhouette PKL 확장자(기본 .pkl).")
    parser.add_argument('--output_path', type=str, required=True,
                        help="결과 데이터가 저장될 루트 경로")
    return parser.parse_args()

# ★ 공통 링크/복사 유틸 함수 -------------------------------
def _link_or_copy(src, dst):
    """
    src → dst 로 심볼릭 링크 생성.
    권한 문제로 실패(OSError)하면 파일을 복사로 대체.
    dst 가 이미 존재하면 덮어쓴다.
    """
    dst_path = Path(dst)
    if dst_path.exists():
        dst_path.unlink()
    try:
        os.symlink(src, dst_path)
    except OSError:                      # Windows 권한·지원 안 될 때
        shutil.copy2(src, dst_path)
# ----------------------------------------------------------

def main():
    opt = get_args()
    heatmap_root = opt.heatmap_data_path
    sil_root     = opt.silhouette_data_path

    if not os.path.exists(heatmap_root):
        sys.exit(f"[ERROR] heatmap 경로가 존재하지 않습니다: {heatmap_root}")
    if not os.path.exists(sil_root):
        sys.exit(f"[ERROR] silhouette 경로가 존재하지 않습니다: {sil_root}")

    heatmap_files   = sorted(glob(os.path.join(heatmap_root, "*/*/*/*.pkl")))
    silhouette_files = sorted(glob(os.path.join(
        sil_root, f"*/*/*/*{opt.dataset_pkl_ext_name}"
    )))

    # heatmap 수가 더 많거나 같은 경우
    if len(heatmap_files) >= len(silhouette_files):
        for h_file in tqdm(heatmap_files, desc="link heatmap→sil"):
            parts = Path(h_file).parts        # (..., ID, seq, view, file.pkl)
            id_, seq, view = parts[-4:-1]

            sil_folder = Path(sil_root, id_, seq, view)
            if not sil_folder.exists():
                print(f"[WARN] silhouette 폴더 없음: {sil_folder}")
                continue
            sil_file = sorted(sil_folder.glob(f"*{opt.dataset_pkl_ext_name}"))[0]

            out_dir = Path(opt.output_path, id_, seq, view)
            out_dir.mkdir(parents=True, exist_ok=True)

            _link_or_copy(sil_file, out_dir / "1_sil.pkl")
            _link_or_copy(h_file,   out_dir / "0_heatmap.pkl")

    # silhouette 수가 더 많은 경우
    else:
        for s_file in tqdm(silhouette_files, desc="link sil→heatmap"):
            parts = Path(s_file).parts
            id_, seq, view = parts[-4:-1]

            h_folder = Path(heatmap_root, id_, seq, view)
            if not h_folder.exists():
                print(f"[WARN] heatmap 폴더 없음: {h_folder}")
                continue
            h_file = sorted(h_folder.glob("*.pkl"))[0]

            out_dir = Path(opt.output_path, id_, seq, view)
            out_dir.mkdir(parents=True, exist_ok=True)

            _link_or_copy(s_file, out_dir / "1_sil.pkl")
            _link_or_copy(h_file, out_dir / "0_heatmap.pkl")

    print(f"✔ 완료! 결과 데이터: {opt.output_path}")

if __name__ == "__main__":
    main()

