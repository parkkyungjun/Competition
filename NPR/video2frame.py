import cv2
import os
import glob
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# ================= 설정 =================
SOURCE_ROOT = 'train'         # 원본 데이터 폴더
OUTPUT_ROOT = 'train_frames'  # 변환된 이미지가 저장될 폴더
NUM_WORKERS = 8               # CPU 코어 수 (본인 PC에 맞춰 조절, 보통 4~8)
# =======================================

def extract_frames(video_path):
    """
    동영상 하나를 처리하는 함수:
    1초 단위로 프레임을 추출하여 PNG로 저장합니다.
    """
    try:
        # 경로 처리: train/class1/video.mp4 -> train_frames/class1/video_mp4/
        path_obj = Path(video_path)
        
        # SOURCE_ROOT에 상대적인 경로 계산 (예: class1/video.mp4)
        relative_path = path_obj.relative_to(SOURCE_ROOT)
        
        # 파일명을 폴더명으로 변경 (확장자 충돌 방지 및 정리 목적)
        # 예: video.mp4 -> video_mp4
        new_folder_name = path_obj.stem + "_" + path_obj.suffix[1:]
        
        # 최종 저장 경로: train_frames/class1/video_mp4/
        save_dir = Path(OUTPUT_ROOT) / relative_path.parent / new_folder_name
        
        # 이미 변환된 폴더가 있다면 스킵 (이어하기 기능)
        if save_dir.exists() and any(save_dir.iterdir()):
            return f"Skipped: {video_path}"

        os.makedirs(save_dir, exist_ok=True)

        # 비디오 로드
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return f"Error opening: {video_path}"

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        count = 0
        # 0초부터 동영상 끝까지 1초 간격으로 반복
        for sec in range(int(duration) + 1):
            target_frame_idx = int(sec * fps)
            
            # 해당 프레임으로 이동
            # (프레임 수가 범위를 벗어나지 않도록 클램핑)
            if target_frame_idx >= frame_count:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # 파일명: 00000.png, 00001.png ... (초 단위)
                save_path = save_dir / f"{sec:05d}.png"
                cv2.imwrite(str(save_path), frame)
                count += 1
            else:
                break

        cap.release()
        return None # 성공

    except Exception as e:
        return f"Error processing {video_path}: {str(e)}"

def main():
    # 1. 비디오 파일 찾기
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.webm')
    print(f"📂 '{SOURCE_ROOT}' 폴더에서 동영상을 검색합니다...")
    
    all_files = glob.glob(os.path.join(SOURCE_ROOT, '**', '*'), recursive=True)
    video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    
    print(f"🔍 총 {len(video_files)}개의 동영상을 찾았습니다.")
    print(f"🚀 {NUM_WORKERS}개의 프로세스로 변환을 시작합니다. (대상 폴더: {OUTPUT_ROOT})")

    # 2. 멀티프로세싱으로 병렬 처리
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # tqdm을 사용하여 진행률 표시
        results = list(tqdm(executor.map(extract_frames, video_files), total=len(video_files)))

    # 3. 에러 로그 출력
    errors = [r for r in results if r is not None and "Error" in r]
    if errors:
        print(f"\n⚠️ {len(errors)}개의 파일 처리 중 에러 발생:")
        for err in errors[:5]: # 상위 5개만 출력
            print(err)
        print("...")
    
    print("\n✅ 모든 작업이 완료되었습니다!")

if __name__ == '__main__':
    # 윈도우/리눅스 멀티프로세싱 충돌 방지
    main()