import cv2
import os
import glob
import kagglehub
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

# ================= Configuration =================
TARGET_SIZE = 384       # 최종 리사이즈 크기
FRAME_INTERVAL_SEC = 1  # 1초에 1프레임 추출
CROP_SCALE = 1.3        # 얼굴 박스 확대 비율 (1.3배 -> 귀, 머리카락 포함)
MODEL_PATH = 'yolov12n-face.pt' # YOLO 모델 경로

# 저장할 루트 폴더
OUTPUT_ROOT = "./processed_dataset_384_padded"
# =================================================

def setup_environment():
    print("Checking Dataset...")
    try:
        # 이미 다운받았으면 캐시 경로를 반환합니다.
        path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")
        print(f"Dataset Path: {path}")
        return path
    except Exception as e:
        print(f"Dataset download check failed: {e}")
        return "." 

def smart_resize(img, target_size=384):
    """
    이미지가 target보다 크면 축소(LANCZOS4), 작으면 확대(CUBIC)
    """
    h, w = img.shape[:2]
    if h > target_size or w > target_size:
        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

def get_padded_box(box, img_w, img_h, scale=1.3):
    """
    YOLO 박스 좌표를 중심으로 scale배 만큼 확장하되, 이미지 범위를 벗어나지 않게 보정
    """
    x1, y1, x2, y2 = map(int, box)
    
    # 중심점과 너비/높이 계산
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w // 2
    cy = y1 + h // 2
    
    # 확장된 너비/높이 (정사각형에 가깝게 만들기 위해 max 사용하기도 하지만, 여기선 비율 유지)
    # 얼굴은 보통 세로가 기므로 max(w, h)를 기준으로 정사각형 crop을 하기도 함.
    # 여기서는 원본 비율 유지하면서 확대
    new_w = w * scale
    new_h = h * scale
    
    # 새로운 좌표 계산
    new_x1 = int(cx - new_w / 2)
    new_y1 = int(cy - new_h / 2)
    new_x2 = int(cx + new_w / 2)
    new_y2 = int(cy + new_h / 2)
    
    # 이미지 경계 처리 (Clamping)
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(img_w, new_x2)
    new_y2 = min(img_h, new_y2)
    
    return new_x1, new_y1, new_x2, new_y2

def process_video(video_path, output_base_dir, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    # Fake 여부 판단 (파일 경로에 fake, manipulated 등이 있는지)
    path_lower = video_path.lower()
    is_fake_video = 'dfd_manipulated_sequences' in path_lower
    
    # 저장 경로: output/real 혹은 output/fake
    label_dir = "fake" if is_fake_video else "real"
    save_dir = os.path.join(output_base_dir, label_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30
    
    frame_interval = int(fps * FRAME_INTERVAL_SEC)
    if frame_interval == 0: frame_interval = 1
    
    video_name = Path(video_path).stem
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # YOLO Inference
            results = model.predict(frame, conf=0.5, verbose=False)
            boxes = results[0].boxes
            num_faces = len(boxes)
            
            # === [수정된 로직] ===
            should_process = False
            
            if num_faces == 0:
                should_process = False
                
            elif is_fake_video:
                # Fake 영상: 오직 1명일 때만 처리 (누가 가짜인지 모르므로 다수는 스킵)
                if num_faces == 1:
                    should_process = True
                else:
                    should_process = False # 2명 이상이면 Skip
                    
            else: 
                # Real 영상: 1명이든 100명이든 전부 진짜 얼굴이므로 모두 처리
                should_process = True
            
            # === 크롭 및 저장 ===
            if should_process:
                h_img, w_img = frame.shape[:2]
                
                for i, box in enumerate(boxes):
                    # 원본 좌표
                    coords = box.xyxy[0].tolist()
                    
                    # 1.3배 확장된 좌표 계산
                    x1, y1, x2, y2 = get_padded_box(coords, w_img, h_img, scale=CROP_SCALE)
                    
                    # 너무 작으면 무시 (확장 후에도 50px 미만이면 버림)
                    if (x2 - x1) < 50 or (y2 - y1) < 50:
                        continue
                    
                    # Crop
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # Resize (384) - Smart Resize
                    face_resized = smart_resize(face_crop, TARGET_SIZE)
                    
                    # Save
                    save_filename = f"{video_name}_fr{frame_count}_face{i}.jpg"
                    save_path = os.path.join(save_dir, save_filename)
                    cv2.imwrite(save_path, face_resized)
                    
        frame_count += 1

    cap.release()

def main():
    dataset_root = setup_environment()
    
    print(f"Loading YOLO model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 비디오 파일 검색
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    
    print("Searching video files...")
    for ext in video_extensions:
        # 데이터셋 구조에 맞춰 검색 (recursive)
        video_files.extend(glob.glob(os.path.join(dataset_root, "**", ext), recursive=True))
    
    print(f"Found {len(video_files)} videos. Starting processing...")
    
    for video_path in tqdm(video_files):
        try:
            process_video(video_path, OUTPUT_ROOT, model)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print(f"\nProcessing Complete! Images saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()