from PIL import Image
import numpy as np
import cv2  # OpenCV 추가

def preprocess_image(image_path):
    img = None
    
    # 1. MP4 파일인 경우 첫 프레임 추출
    if image_path.lower().endswith('.mp4'):
        cap = cv2.VideoCapture(image_path)
        ret, frame = cap.read() # 첫 프레임 읽기
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

    print(img.size)
    return np.array(img)

    

img = preprocess_image('sorted_groups/group_000/TEST_392.mp4')
img2 = preprocess_image('sorted_groups/group_000/TEST_448.png')

cv2.imwrite('out1.png', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
print(np.sum(img == img2) / (img.shape[0] * img.shape[1] * 3))