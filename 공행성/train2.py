import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# 설정
DATA_PATH = 'train.csv'
MAX_LAG = 6
DUMMY_VALUE = 9999999999 # 예측값 무시 (F1 테스트용)

# 1. 데이터 로드 (아까랑 동일)
df = pd.read_csv(DATA_PATH)
df_grouped = df.groupby(['item_id', 'year', 'month'])['value'].sum().reset_index()
df_grouped['date'] = pd.to_datetime(df_grouped[['year', 'month']].assign(day=1))
pivot_df = df_grouped.pivot(index='date', columns='item_id', values='value').fillna(0)
items = pivot_df.columns

# -------------------------------------------------------
# 실험 A: Granger Strict (P-value < 0.01)
# -------------------------------------------------------
print("\n🧪 [실험 A] Granger Strict (P < 0.01) 진행 중...")
results_granger_strict = []

for target in tqdm(items):
    for candidate in items:
        if target == candidate: continue
        
        data = pd.concat([pivot_df[target], pivot_df[candidate]], axis=1)
        if data.std().min() == 0: continue
        
        try:
            gc_res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
            is_causal = False
            for lag in range(1, MAX_LAG + 1):
                # P-value가 0.01 미만인 경우만 통과
                if gc_res[lag][0]['ssr_ftest'][1] < 0.01:
                    is_causal = True
                    break
            if is_causal:
                results_granger_strict.append({'leading_item_id': candidate, 'following_item_id': target, 'value': DUMMY_VALUE})
        except: continue

df_A = pd.DataFrame(results_granger_strict)
df_A.to_csv('submission_TEST_A_Granger_Strict.csv', index=False)
print(f"👉 실험 A 결과: {len(df_A)}개 쌍 발견")


# -------------------------------------------------------
# 실험 B: Mutual Information (Top 500)
# -------------------------------------------------------
print("\n🧪 [실험 B] Mutual Information (MI) 진행 중...")
results_mi = []

# MI는 계산이 좀 걸려서, 각 타겟별로 가장 MI 높은 Top 5만 뽑는 전략
for target in tqdm(items):
    mi_scores = []
    y = pivot_df[target]
    
    for candidate in items:
        if target == candidate: continue
        
        x = pivot_df[candidate]
        
        # Lag 1~6 중 가장 MI 높은 것 찾기
        best_mi = 0
        for lag in range(1, MAX_LAG + 1):
            x_shifted = x.shift(lag)
            valid = ~np.isnan(x_shifted) & ~np.isnan(y)
            if valid.sum() < 10: continue
            
            # MI 계산 (값이 클수록 관련성 높음)
            # reshape 필요
            mi = mutual_info_regression(x_shifted[valid].values.reshape(-1, 1), y[valid].values)[0]
            if mi > best_mi:
                best_mi = mi
        
        if best_mi > 0:
            mi_scores.append((candidate, best_mi))
    
    # 타겟별 상위 5개만 후보로 등록 (너무 많이 잡히는 거 방지)
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    for cand, score in mi_scores[:5]:
        # MI 점수가 너무 낮으면(0.1 미만) 버림
        if score > 0.15:
            results_mi.append({'leading_item_id': cand, 'following_item_id': target, 'value': DUMMY_VALUE})

df_B = pd.DataFrame(results_mi)
df_B.to_csv('submission_TEST_B_MutualInfo.csv', index=False)
print(f"👉 실험 B 결과: {len(df_B)}개 쌍 발견")


# -------------------------------------------------------
# 실험 C: 합집합 (Granger 0.05 + Pearson 0.7)
# -------------------------------------------------------
print("\n🧪 [실험 C] Union Strategy (Granger + Pearson) 진행 중...")

# 아까 만든 Granger(0.05) 파일이 있다고 가정하거나 리스트 재사용
# (여기서는 로직상 합치는 방법만 보여줌. 위에서 구한 strict 말고 loose한 0.05가 필요함)
# 편의상 아까 결과가 있다고 치고, 실험 A(Strict) + 피어슨 결과를 합쳐봄 (예시)

# 피어슨 구하기 (간단하게 다시 계산)
pearson_pairs = []
for target in items:
    for candidate in items:
        if target == candidate: continue
        # ... (피어슨 로직 생략, 결과만 78개 있다고 가정) ...
# 실전에서는 아까 만든 csv 불러와서 합치면 됨

# 여기선 '그레인저 2144개 파일'과 '피어슨 78개 파일'을 읽어와서 합치는 코드 제공
try:
    df_g = pd.read_csv('submission_GRANGER.csv') # 아까 0.23점 나온 파일
    df_p = pd.read_csv('submission_PEARSON.csv') # 아까 78개 파일
    
    # concat 후 drop_duplicates로 합집합 만들기
    df_C = pd.concat([df_g, df_p]).drop_duplicates(subset=['leading_item_id', 'following_item_id'])
    df_C['value'] = DUMMY_VALUE # 값은 더미로 통일
    
    df_C.to_csv('submission_TEST_C_Union.csv', index=False)
    print(f"👉 실험 C 결과: {len(df_C)}개 쌍 (합집합)")
except:
    print("⚠️ 이전에 만든 CSV 파일이 없어서 실험 C는 패스함.")

print("\n완료! 3개 파일 제출해서 점수 비교해보셈.")