import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

def save_shifted_crop_512(img, box, save_dir, base_name, suffix):
    """
    얼굴을 중심으로 512x512를 자르되,
    이미지 경계를 벗어나면 0으로 채우지 않고
    크롭 박스 위치를 이동시켜서(Shift) 이미지 내부 데이터로만 채움.
    """
    CROP_SIZE = 512
    HALF_SIZE = CROP_SIZE // 2 # 256
    
    # 이미지 전체 크기
    h_img, w_img = img.shape[:2]

    # --- [예외 처리] 원본이 512보다 작은 경우 ---
    # 물리적으로 이동(Shift)이 불가능하므로, 이 경우에만 검은 배경 사용
    if w_img < CROP_SIZE or h_img < CROP_SIZE:
        # print(f"  [Warning] 이미지가 너무 작음 ({w_img}x{h_img}). 패딩 처리함.")
        canvas = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        
        # 중앙 배치 계산
        paste_x = (CROP_SIZE - w_img) // 2 if w_img < CROP_SIZE else 0
        paste_y = (CROP_SIZE - h_img) // 2 if h_img < CROP_SIZE else 0
        
        # 잘라낼 영역 (이미지 전체)
        crop_w = min(w_img, CROP_SIZE)
        crop_h = min(h_img, CROP_SIZE)
        
        # 원본에서 가져올 좌표 (이미지 안쪽으로만)
        src_x = 0 if w_img < CROP_SIZE else (w_img - CROP_SIZE) // 2
        src_y = 0 if h_img < CROP_SIZE else (h_img - CROP_SIZE) // 2
        
        canvas[paste_y:paste_y+crop_h, paste_x:paste_x+crop_w] = img[src_y:src_y+crop_h, src_x:src_x+crop_w]
        
        save_name = f"{base_name}_{suffix}.jpg"
        cv2.imwrite(os.path.join(save_dir, save_name), canvas)
        return True

    # --- [메인 로직] 위치 이동 (Shift) ---
    
    # 1. 바운딩 박스 중심점 구하기
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # 2. 크롭 시작 좌표 계산 (중심 - 256)
    start_x = cx - HALF_SIZE
    start_y = cy - HALF_SIZE

    # 3. 좌표 보정 (Shift)
    # 3-1. 왼쪽/위쪽 경계 벗어남 -> 0으로 고정 (오른쪽/아래로 밈)
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
        
    # 3-2. 오른쪽/아래쪽 경계 벗어남 -> (이미지길이 - 512)로 고정 (왼쪽/위로 당김)
    if start_x + CROP_SIZE > w_img:
        start_x = w_img - CROP_SIZE
    if start_y + CROP_SIZE > h_img:
        start_y = h_img - CROP_SIZE

    # 4. 최종 크롭 및 저장
    # 위 로직 덕분에 start_x, start_y는 무조건 안전한 범위 내에 있음 (이미지가 512보다 큰 이상)
    cropped_img = img[start_y:start_y+CROP_SIZE, start_x:start_x+CROP_SIZE]
    
    if cropped_img.shape[0] == CROP_SIZE and cropped_img.shape[1] == CROP_SIZE:
        save_name = f"{base_name}_{suffix}.jpg"
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, cropped_img)
        return True
    
    return False

def process_media(folder_path='test', model_path='yolov12n-face.pt'):
    # 1. 모델 확인
    if not os.path.exists(model_path):
        print(f"오류: {model_path} 파일이 없습니다.")
        return
    model = YOLO(model_path)
    
    # 2. 저장 폴더
    save_dir = 'cropped_shifted_512'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 3. 파일 리스트
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.mp4', '*.avi', '*.mkv', '*.mov']
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(folder_path, ext)))
        all_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    all_files = sorted(list(set(all_files)))
    print(f"총 {len(all_files)}개 파일 처리 중...")

    for file_path in all_files:
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        
        # [A] 동영상
        if ext in ['.mp4', '.avi', '.mkv', '.mov']:
            cap = cv2.VideoCapture(file_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 0:
                mid_frame = total_frames // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                if ret:
                    results = model.predict(frame, conf=0.5, verbose=False)
                    for result in results:
                        for i, box in enumerate(result.boxes):
                            save_shifted_crop_512(frame, box, save_dir, name_only, f"vid_{mid_frame}_face{i}")
            cap.release()

        # [B] 이미지
        else:
            img = cv2.imread(file_path)
            if img is not None:
                results = model.predict(img, conf=0.5, verbose=False)
                for result in results:
                    for i, box in enumerate(result.boxes):
                        save_shifted_crop_512(img, box, save_dir, name_only, f"face{i}")
        
        print(f"처리중: {filename}", end='\r')

if __name__ == "__main__":
    INPUT_FOLDER = 'train/fake/change' 
    MODEL_FILE = 'yolov12n-face.pt'
    
    process_media(INPUT_FOLDER, MODEL_FILE)
    print("\n완료되었습니다. 'cropped_shifted_512' 폴더를 확인하세요.")