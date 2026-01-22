import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import glob
import pandas as pd
from tqdm import tqdm

# 모델 정의 파일이 같은 경로에 있어야 합니다.
from efficientnet import efficientnetb2_custom

# --- 설정 (Configuration) ---
class Config:
    CROP_SIZE = 512
    # 학습된 모델 가중치 경로
    MODEL_PATH = 'best_model_f1.pth' 
    # 추론할 이미지가 있는 폴더 경로
    INPUT_FOLDER = 'cropped_shifted_512'
    # 결과 저장 파일명
    OUTPUT_CSV = 'inference_results.csv'
    # 사용할 장치
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, device):
    """모델 초기화 및 가중치 로드"""
    print(f"[Info] Loading model from {model_path}...")
    model = efficientnetb2_custom()
    
    # 가중치 로드 (CPU 매핑 포함하여 안전하게 로드)
    checkpoint = torch.load(model_path, map_location=device)
    
    # state_dict 키 불일치 방지 (혹시 모를 DataParallel의 'module.' 접두사 제거)
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
    """단일 이미지 로드 및 전처리"""
    try:
        img = Image.open(image_path).convert('RGB')
        
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

    # 2. 이미지 파일 리스트 확보
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif')
    image_files = []
    
    # 하위 폴더까지 검색할지 여부에 따라 recursive 조정
    raw_files = glob.glob(os.path.join(Config.INPUT_FOLDER, '**', '*'), recursive=True)
    image_files = sorted([f for f in raw_files if f.lower().endswith(valid_extensions)])

    if not image_files:
        print(f"[Warning] No images found in {Config.INPUT_FOLDER}")
        return

    print(f"[Info] Found {len(image_files)} images. Starting inference...")

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
            prob = torch.sigmoid(output).item() # 0~1 사이 확률값
            
            # Label 결정 (학습 코드 기준: Class 1 = Fake, Class 0 = Real)
            # 0.5 초과면 Fake(1), 이하면 Real(0)
            pred_label = 1 if prob > 0.5 else 0
            label_str = "Fake" if pred_label == 1 else "Real"
            
            # 결과 저장
            results.append({
                'filename': os.path.basename(img_path),
                'path': img_path,
                'probability': prob,  # 1(Fake)에 가까운 정도
                'prediction': pred_label,
                'label': label_str
            })

    # 4. 결과 출력 및 저장
    df = pd.DataFrame(results)
    
    # CSV 저장
    df.to_csv(Config.OUTPUT_CSV, index=False)
    
    print("-" * 50)
    print(f"[Result] Inference complete. Saved to {Config.OUTPUT_CSV}")
    print("-" * 50)
    
    # 터미널에 상위 10개 결과 출력 (확인용)
    if not df.empty:
        print(df[['filename', 'label', 'probability']].head(10))

if __name__ == "__main__":
    # 폴더가 없으면 에러가 나므로 미리 체크하거나 생성
    if not os.path.exists(Config.INPUT_FOLDER):
        os.makedirs(Config.INPUT_FOLDER, exist_ok=True)
        print(f"Created input folder: {Config.INPUT_FOLDER}. Please put images inside.")
    else:
        main()