import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from scipy.spatial.distance import euclidean
# pip install fastdtw 필요 (없으면 아래 dtw 부분 주석 처리)
try:
    from fastdtw import fastdtw
    use_dtw = True
except ImportError:
    print("fastdtw 라이브러리가 없어서 DTW는 패스합니다. (pip install fastdtw 추천)")
    use_dtw = False

from tqdm import tqdm

# 데이터 로드 (아까랑 동일)
df = pd.read_csv('train.csv')
df_grouped = df.groupby(['item_id', 'year', 'month'])['value'].sum().reset_index()
df_grouped['date'] = pd.to_datetime(df_grouped[['year', 'month']].assign(day=1))
pivot_df = df_grouped.pivot(index='date', columns='item_id', values='value').fillna(0)
items = pivot_df.columns

results_advanced = []

print("🚀 [고급 분석] 공적분(Cointegration) & DTW 분석 시작...")

for target in tqdm(items):
    for candidate in items:
        if target == candidate: continue
        
        y = pivot_df[target]
        x = pivot_df[candidate]
        
        # -------------------------------------------
        # 1. 공적분 검정 (Cointegration)
        # H0: 공적분 관계가 없다 (관계없음)
        # P-value < 0.05 이면: "장기적으로 묶여있다(관계있음)"
        # -------------------------------------------
        try:
            # coint 함수는 (t-stat, p-value, crit-values)를 반환
            score, p_value, _ = coint(y, x)
            
            is_coint = False
            if p_value < 0.05:
                is_coint = True
        except:
            p_value = 1.0
            is_coint = False

        # -------------------------------------------
        # 2. DTW (Dynamic Time Warping) 거리 계산
        # 거리가 짧을수록(0에 가까울수록) 패턴이 비슷함
        # -------------------------------------------
        dtw_dist = 99999
        if use_dtw:
            def normalize(series):
                return (series - series.mean()) / (series.std() + 1e-6)
            
            # [중요] .values를 써서 Numpy 배열로 변환해야 함
            x_norm = normalize(x).values
            y_norm = normalize(y).values
            
            # [수정] dist=euclidean 대신 lambda a, b: abs(a-b) 사용
            # 이유: 1차원 값(스칼라)끼리 비교 시 scipy euclidean은 차원 에러를 유발함
            distance, path = fastdtw(x_norm, y_norm, dist=lambda a, b: abs(a - b))
            dtw_dist = distance

        # -------------------------------------------
        # [조건 필터링]
        # 전략: 공적분이 존재하거나(P<0.05) OR DTW 거리가 매우 가깝거나
        # -------------------------------------------
        
        # 여기서는 예시로 "공적분 P-value 0.05 미만"인 것만 저장
        if p_value < 0.05 and dtw_dist < 50:
            results_advanced.append({
                'leading_item_id': candidate,
                'following_item_id': target,
                'p_value_coint': round(p_value, 4),
                'dtw_distance': round(dtw_dist, 2),
                'value': 9999999999 # F1 테스트용 더미 값
            })

df_adv = pd.DataFrame(results_advanced)

# 2. 분포 확인 (최솟값, 평균값, 25% 지점 등 확인)
print(df_adv['dtw_distance'].describe())

# 3. 하위 10% (거리가 가까운 순) 기준값 구하기
# threshold = df_adv['dtw_distance'].quantile(0.30) 
# print(f"상위 10% 커트라인 점수: {threshold}")

# # 4. 그 기준보다 작은(가까운) 애들만 남기기
# df_adv = df_adv[df_adv['dtw_distance'] <= threshold]

submission_clean = df_adv[['leading_item_id', 'following_item_id', 'value']]

# 파일 다시 저장
# df_adv.to_csv('df_adv.csv', index=False)
submission_clean.to_csv('submission_TEST_D_Cointegration.csv', index=False)

print(f"\n👉 공적분(Cointegration) 발견 쌍: {len(df_adv)}개")
print("DTW 값은 저장만 해둠. 나중에 필터링할 때 'dtw_distance'가 낮은 순으로 자르면 좋음.")