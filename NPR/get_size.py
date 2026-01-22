import os
import glob
import cv2
import pandas as pd
from PIL import Image
from collections import Counter
from tqdm import tqdm

def get_image_size(filepath):
    try:
        with Image.open(filepath) as img:
            return img.size # (width, height)
    except:
        return None

def get_video_size(filepath):
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened(): return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return (w, h)
    except:
        return None

def main():
    target_folder = 'test' # 조사할 폴더
    
    # 확장자 필터
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif')
    vid_exts = ('.mp4', '.avi', '.mkv', '.mov')
    
    print(f"📂 '{target_folder}' 폴더 스캔 중...")
    all_files = glob.glob(os.path.join(target_folder, '**', '*'), recursive=True)
    
    target_files = [f for f in all_files if f.lower().endswith(img_exts + vid_exts)]
    print(f"🔍 총 {len(target_files)}개 파일 분석 시작")

    # 빈도수 계산을 위한 Counter
    resolution_counts = Counter()

    for filepath in tqdm(target_files):
        size = None
        f_lower = filepath.lower()
        
        if f_lower.endswith(img_exts):
            size = get_image_size(filepath)
        elif f_lower.endswith(vid_exts):
            size = get_video_size(filepath)
            
        if size:
            # (Width, Height) 튜플을 카운트
            resolution_counts[size] += 1

    # 결과 정리
    # 데이터프레임으로 변환 (보기 좋게)
    df = pd.DataFrame.from_dict(resolution_counts, orient='index', columns=['count'])
    df.index.name = 'resolution (w, h)'
    df.reset_index(inplace=True)
    
    # 개수 많은 순서대로 정렬
    df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

    print("\n" + "="*40)
    print("📊 해상도별 빈도수 (상위 20개)")
    print("="*40)
    print(df.head(20)) # 상위 20개 출력
    print("="*40)

    # CSV 저장
    df.to_csv('resolution_counts.csv', index=False)
    print(f"✅ 전체 통계가 'resolution_counts.csv'에 저장되었습니다.")

if __name__ == '__main__':
    main()