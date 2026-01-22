import os
import shutil
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# 설정
SOURCE_DIR = "./test"
OUTPUT_DIR = "./sorted_groups"
MODEL_NAME = 'facebook/dinov2-base'
# MODEL_NAME = 'openai/clip-vit-base-patch32' # CLIP (VLM 방식, 의미적 유사성에 강함)

THRESHOLD = 0.1  # DBSCAN 거리 임계값 (0.0~0.2 사이 권장, 작을수록 엄격하게 구분)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_file_list(folder_path):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.webp', 'jfif')
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]

def load_image_or_video_frame(file_path):
    """이미지면 열고, 동영상(mp4)이면 첫 프레임을 추출"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mov']:
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        # BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image)
    else:
        try:
            return Image.open(file_path).convert("RGB")
        except:
            return None

def main():
    # 1. 모델 로드
    print(f"Loading Model: {MODEL_NAME}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()

    files = get_file_list(SOURCE_DIR)
    if not files:
        print("파일이 없습니다.")
        return

    embeddings = []
    valid_files = []

    # 2. 임베딩 추출
    print("Extracting features...")
    for f_path in tqdm(files):
        img = load_image_or_video_frame(f_path)
        if img is None:
            continue
        
        # 전처리 및 추론
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            # DINOv2나 ViT 계열은 pooler_output 혹은 last_hidden_state의 [CLS] 토큰 사용
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                emb = outputs.pooler_output
            else:
                # last_hidden_state의 첫번째 토큰(CLS) 사용
                emb = outputs.last_hidden_state[:, 0]
            
        # 정규화 (Cosine Similarity 사용을 위해)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(emb.cpu().numpy().squeeze())
        valid_files.append(f_path)

    if not embeddings:
        print("처리할 수 있는 이미지가 없습니다.")
        return

    embeddings = np.array(embeddings)

    # 3. 클러스터링 (Cosine Distance = 1 - Cosine Similarity)
    # metric='cosine'을 쓰면 eps는 '거리' 기준임 (0에 가까울수록 똑같은 이미지)
    print(f"Clustering with DBSCAN (eps={THRESHOLD})...")
    clustering = DBSCAN(eps=THRESHOLD, min_samples=2, metric='cosine').fit(embeddings)
    labels = clustering.labels_

    # 4. 폴더 이동
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 노이즈(-1)로 분류된 것들은 별도 처리 (유사한 짝이 없는 경우)
    unique_labels = set(labels)
    print(f"총 {len(unique_labels) - (1 if -1 in labels else 0)} 개의 그룹이 발견되었습니다.")

    for file_path, label in zip(valid_files, labels):
        file_name = os.path.basename(file_path)
        
        if label == -1:
            # 짝이 없는 파일 (Outlier)
            target_folder = os.path.join(OUTPUT_DIR, "outliers")
        else:
            # 그룹 폴더
            target_folder = os.path.join(OUTPUT_DIR, f"group_{label:03d}")
        
        os.makedirs(target_folder, exist_ok=True)
        shutil.copy2(file_path, os.path.join(target_folder, file_name)) # 원본 보존을 위해 copy 사용 (move로 변경 가능)

    print("완료되었습니다. ./sorted_groups 폴더를 확인하세요.")

if __name__ == "__main__":
    main()