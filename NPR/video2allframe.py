import cv2
import os

# 1. 설정: 동영상 경로
video_path = 'sorted_groups/group_000/TEST_392.mp4'

# 2. 저장할 폴더 생성 (파일명을 폴더명으로 사용)
# 예: TEST_086_frames 폴더가 생성됨
video_name = os.path.splitext(os.path.basename(video_path))[0]
save_dir = f"{video_name}_frames"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"[Info] Created directory: {save_dir}")
else:
    print(f"[Info] Directory already exists: {save_dir}")

# 3. 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"[Error] Could not open video: {video_path}")
    exit()

# 전체 프레임 수 확인 (진행상황 표시용)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[Info] Total frames to process: {total_frames}")

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # 더 이상 읽을 프레임이 없으면 종료
    
    # 4. 파일명 생성 (00001.jpg, 00002.jpg 형식으로 정렬되게 저장)
    # 5자리 숫자로 패딩 (필요에 따라 조절 가능)
    save_path = os.path.join(save_dir, f"{frame_count:05d}.jpg")
    
    # 이미지 저장
    cv2.imwrite(save_path, frame)
    
    frame_count += 1
    
    # 100프레임마다 로그 출력
    if frame_count % 100 == 0:
        print(f"Saved {frame_count}/{total_frames} frames...")

cap.release()
print("-" * 50)
print(f"[Done] All {frame_count} frames saved in '{save_dir}'")