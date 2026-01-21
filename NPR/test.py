import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def delete_small_images(root_dir):
    root_path = Path(os.path.expanduser(root_dir))
    
    # 이미지 확장자 정의
    valid_extensions = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'}
    
    deleted_count = 0
    error_count = 0
    
    # 모든 이미지 파일 리스트 확보
    all_files = [f for f in root_path.rglob('*') if f.suffix.lower() in valid_extensions]
    print(f"총 {len(all_files)}개의 이미지를 검사합니다...")

    for img_path in tqdm(all_files):
        try:
            # 이미지 헤더만 읽어서 사이즈 확인
            with Image.open(img_path) as img:
                w, h = img.size
            
            # 한 쪽이라도 512 미만인 경우 삭제
            if w < 512 or h < 512:
                img_path.unlink() # 파일 삭제
                deleted_count += 1
                
        except Exception as e:
            # 파일이 손상되었거나 접근 권한이 없는 경우
            error_count += 1
            continue

    print("\n" + "="*30)
    print(f"작업 완료 (경로: {root_dir})")
    print(f"- 삭제된 이미지: {deleted_count}개")
    if error_count > 0:
        print(f"- 처리 중 오류 발생: {error_count}개")
    print("="*30)

# 실행 (경로 확인 필수!)
target_path = '~/.cache/kagglehub/datasets/sautkin/imagenet1k1/versions/2'
delete_small_images(target_path)