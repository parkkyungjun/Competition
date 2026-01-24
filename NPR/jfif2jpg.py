import os
import glob

# ==========================================
# [설정] 원본 안전을 위한 모드 설정
# True: 실제로 변경하지 않고, 무엇이 바뀔지만 출력합니다. (안전 모드)
# False: 실제로 파일 이름을 변경합니다.
DRY_RUN = False 
# ==========================================

def rename_jfif_to_jpg_recursive(root_folder):
    # 대소문자 구분 없이 찾기 위해 패턴 설정 (리눅스 환경 고려하여 확장자 매칭)
    # glob은 대소문자를 구분하므로 .jfif와 .JFIF 등을 모두 찾으려면 약간의 처리가 필요할 수 있으나,
    # 보통 소문자로 되어있다고 가정하고 진행합니다. 필요시 패턴 추가 가능.
    pattern = os.path.join(root_folder, '**', '*.jfif')
    
    # recursive=True 옵션으로 하위 폴더까지 모두 탐색
    files = glob.glob(pattern, recursive=True)
    
    print(f"검색 경로: {root_folder}")
    print(f"발견된 .jfif 파일 수: {len(files)}개")
    print("-" * 60)

    if len(files) == 0:
        print("변경할 파일이 없습니다.")
        return

    success_count = 0
    skip_count = 0

    for old_path in files:
        # 파일 경로에서 확장자 부분을 분리하여 .jpg로 교체
        # rsplit을 사용하여 마지막 점(.) 기준 분리
        base_name, _ = os.path.splitext(old_path)
        new_path = base_name + ".jpg"

        # [안전 장치 1] 이미 변경하려는 이름의 파일이 존재하는지 확인
        if os.path.exists(new_path):
            print(f"[Skip] 이미 존재함: {os.path.basename(new_path)}")
            skip_count += 1
            continue

        try:
            if DRY_RUN:
                # [모의 실행] 출력만 함
                print(f"[Preview] 변경 예정: {old_path} -> {new_path}")
            else:
                # [실제 실행] 이름 변경 (내용 수정 없음, 단순 rename)
                os.rename(old_path, new_path)
                print(f"[Done] 변경 완료: {old_path} -> {new_path}")
            
            success_count += 1

        except Exception as e:
            print(f"[Error] 변경 실패 ({old_path}): {e}")
            skip_count += 1

    print("-" * 60)
    if DRY_RUN:
        print(f"모의 실행 완료. ({success_count}개 변경 예정)")
        print(">> 실제로 변경하려면 코드 상단의 'DRY_RUN = False'로 변경 후 다시 실행하세요.")
    else:
        print(f"작업 완료. 총 {success_count}개 파일의 확장자가 변경되었습니다.")
        if skip_count > 0:
            print(f"(중복되거나 에러로 건너뛴 파일: {skip_count}개)")

if __name__ == "__main__":
    # 작업할 최상위 폴더 경로
    TARGET_FOLDER = 'sorted_groups'
    
    if os.path.exists(TARGET_FOLDER):
        rename_jfif_to_jpg_recursive(TARGET_FOLDER)
    else:
        print(f"경로를 찾을 수 없습니다: {TARGET_FOLDER}")