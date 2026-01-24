import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
import pandas as pd
from tqdm import tqdm
import cv2  # OpenCV 추가

# 모델 정의 파일이 같은 경로에 있어야 합니다.
from efficientnet import efficientnetb2_custom

# --- 설정 (Configuration) ---
class Config:
    CROP_SIZE = 512
    MODEL_PATH = 'best_model_f1.pth' 
    INPUT_FOLDER = 'sorted_groups/group_009_'
    OUTPUT_CSV = 'inference_results.csv'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device):
    """모델 초기화 및 가중치 로드"""
    print(f"[Info] Loading model from {model_path}...")
    model = efficientnetb2_custom()
    
    checkpoint = torch.load(model_path, map_location=device)
    
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace("module.", "") 
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def get_transform():
    """학습과 동일한 전처리 파이프라인 반환"""
    return transforms.Compose([
        transforms.CenterCrop(Config.CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path, transform):
    """
    이미지 또는 MP4 파일의 첫 프레임을 로드하여 전처리
    """
    try:
        img = None
        
        # 1. MP4 파일인 경우 첫 프레임 추출
        if image_path.lower().endswith('.mp4'):
            cap = cv2.VideoCapture(image_path)
            
            # 1. 전체 프레임 수 확인
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 2. 중간 프레임 인덱스 계산 (0보다 커야 함)
            if frame_count > 0:
                mid_frame_index = frame_count // 2
                # 3. 해당 위치로 이동
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
            
            # 4. 프레임 읽기 (이동한 위치의 프레임을 읽음)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"[Error] Could not read frame from video: {image_path}")
                return None
            
            # OpenCV(BGR) -> PIL(RGB) 변환
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            
        # 2. 일반 이미지 파일인 경우
        else:
            img = Image.open(image_path).convert('RGB')
        
        # 3. 공통 전처리 로직 (리사이즈 및 Transform)
        if img is not None:
            # 학습 코드의 리사이즈 로직 유지 (홀수 해상도 처리)
            w, h = img.size
            if w % 2 == 1: w += 1
            if h % 2 == 1: h += 1
            img = img.resize((w, h))
            
            # Transform 적용
            input_tensor = transform(img)
            return input_tensor.unsqueeze(0) # Batch 차원 추가 (1, C, H, W)
            
    except Exception as e:
        print(f"[Error] Failed to process {image_path}: {e}")
        return None

def main():
    # 1. 모델 로드
    device = Config.DEVICE
    if not os.path.exists(Config.MODEL_PATH):
        print(f"[Error] Model file not found: {Config.MODEL_PATH}")
        return

    model = load_model(Config.MODEL_PATH, device)
    transform = get_transform()

    # 2. 파일 리스트 확보
    # mp4가 이미 포함되어 있었으므로 그대로 둡니다.
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif', '.mp4')
    image_files = []
    
    raw_files = glob.glob(os.path.join(Config.INPUT_FOLDER, '**', '*'), recursive=True)
    image_files = sorted([f for f in raw_files if f.lower().endswith(valid_extensions)])

    if not image_files:
        print(f"[Warning] No files found in {Config.INPUT_FOLDER}")
        return

    print(f"[Info] Found {len(image_files)} files. Starting inference...")

    results = []

    # 3. 추론 루프
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Inferencing"):
            input_tensor = preprocess_image(img_path, transform)
            
            if input_tensor is None:
                continue
            
            input_tensor = input_tensor.to(device)
            
            # Forward
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            
            pred_label = 1 if prob > 0.5 else 0
            label_str = "Fake" if pred_label == 1 else "Real"
            
            results.append({
                'filename': os.path.basename(img_path),
                'path': img_path,
                'probability': prob,
                'prediction': pred_label,
                'label': label_str
            })

    # 4. 결과 출력 및 저장
    df = pd.DataFrame(results)
    df.to_csv(Config.OUTPUT_CSV, index=False)
    
    print("-" * 50)
    print(f"[Result] Inference complete. Saved to {Config.OUTPUT_CSV}")
    print("-" * 50)
    
    if not df.empty:
        print(df[['filename', 'label', 'probability']].head(10))

if __name__ == "__main__":
    if not os.path.exists(Config.INPUT_FOLDER):
        os.makedirs(Config.INPUT_FOLDER, exist_ok=True)
        print(f"Created input folder: {Config.INPUT_FOLDER}. Please put images/videos inside.")
    else:
        main()