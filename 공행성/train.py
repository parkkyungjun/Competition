import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm
import warnings

# 경고 메시지 무시 (Granger 테스트 시 시끄러울 수 있음)
warnings.filterwarnings("ignore")

# ==========================================
# [설정] 이 값들을 조절해서 실험해보세요
# ==========================================
DATA_PATH = 'train.csv'
MAX_LAG = 12           # 최대 6개월 전 데이터까지 확인
DUMMY_VALUE = 9999999999 # 예측을 일부러 틀리기 위한 큰 값

# 실험별 기준값 (Threshold)
PEARSON_THR = 0.4     # 상관계수 0.7 이상이면 짝꿍
SPEARMAN_THR = 0.7    # 상관계수 0.7 이상이면 짝꿍
GRANGER_P_VAL = 0.01  # P-value 0.05 미만이면 짝꿍 (유의수준 5%)
# ==========================================

def load_and_preprocess(path):
    """데이터 불러오기 및 피벗 테이블 생성"""
    print("📂 데이터 로딩 중...")
    df = pd.read_csv(path)
    
    # 월별 합계 계산
    df_grouped = df.groupby(['item_id', 'year', 'month'])['value'].sum().reset_index()
    
    # 날짜 인덱스 생성
    df_grouped['date'] = pd.to_datetime(df_grouped[['year', 'month']].assign(day=1))
    
    # 피벗 (행: 날짜, 열: 아이템, 값: 무역량)
    pivot_df = df_grouped.pivot(index='date', columns='item_id', values='value').fillna(0)
    print(f"✅ 데이터 준비 완료. (총 {len(pivot_df.columns)}개 품목)")
    # analyze_zero_counts(pivot_df)
    return pivot_df

def analyze_zero_counts(pivot_df):
    # 1. 각 아이템(컬럼)별 0의 개수 계산
    zero_counts = (pivot_df == 0).sum()
    
    # 2. 전체 데이터 길이 대비 0의 비율 (Sparsity 확인용)
    zero_ratios = (zero_counts / len(pivot_df)) * 100
    
    # 3. 데이터프레임으로 합쳐서 보기 좋게 정렬 (0이 많은 순서)
    analysis_df = pd.DataFrame({
        'zero_count': zero_counts,
        'zero_ratio (%)': zero_ratios
    }).sort_values(by='zero_count', ascending=False)
    
    print("📊 아이템별 0 값(결측치 포함) 통계:")
    print(analysis_df)
    analysis_df.to_csv('spase.csv')
    return analysis_df

# def load_and_preprocess(path, apply_log=True, apply_diff=True):
#     """
#     데이터 불러오기 및 전처리 (로그 변환 + 차분)
#     apply_log: 값의 스케일을 줄여 아웃라이어 영향 감소
#     apply_diff: 추세를 제거하여 정상성(Stationarity) 확보 (Granger 필수)
#     """
#     print("📂 데이터 로딩 중...")
#     df = pd.read_csv(path)
    
#     # 월별 합계 계산
#     df_grouped = df.groupby(['item_id', 'year', 'month'])['value'].sum().reset_index()
    
#     # 날짜 인덱스 생성
#     df_grouped['date'] = pd.to_datetime(df_grouped[['year', 'month']].assign(day=1))
    
#     # 피벗 (행: 날짜, 열: 아이템, 값: 무역량)
#     pivot_df = df_grouped.pivot(index='date', columns='item_id', values='value').fillna(0)
    
#     print(f"📊 원본 데이터: {pivot_df.shape}")

#     # -------------------------------------------------------
#     # [전처리 1] 로그 변환 (Outlier 완화)
#     # -------------------------------------------------------
#     if apply_log:
#         print("🔧 [전처리] 로그 변환 적용 (np.log1p)")
#         # log1p는 log(x+1)로, 0인 값도 에러 없이 처리해줌
#         pivot_df = np.log1p(pivot_df)

#     # -------------------------------------------------------
#     # [전처리 2] 차분 (Trend 제거 -> Stationarity 확보)
#     # -------------------------------------------------------
#     if apply_diff:
#         print("🔧 [전처리] 1차 차분 적용 (Differencing)")
#         pivot_df = pivot_df.diff().dropna() # 첫 행은 NaN 되므로 제거

#     # -------------------------------------------------------
#     # [옵션] 극단적 아웃라이어 캡핑 (Winsorizing)
#     # 로그 변환으로도 부족할 때 사용 (예: 상위 1% 값으로 제한)
#     # -------------------------------------------------------
#     # upper_limit = pivot_df.quantile(0.99)
#     # pivot_df = pivot_df.clip(upper=upper_limit, axis=1)

#     print(f"✅ 데이터 준비 완료. (최종 {len(pivot_df)}개월, {len(pivot_df.columns)}개 품목)")
#     return pivot_df

def run_correlation_method(pivot_df, method_name='pearson', threshold=0.7):
    """
    피어슨 또는 스피어만 상관계수로 짝꿍 찾기
    method_name: 'pearson' 또는 'spearman'
    """
    items = pivot_df.columns
    results = []
    
    print(f"\n🚀 [{method_name.upper()}] 분석 시작...")
    
    for target in tqdm(items, desc=f"{method_name}"): # B (후행)

        y = pivot_df[target].values
        if np.count_nonzero(y) < 12:
            continue

        for candidate in items: # A (선행)
            if target == candidate: continue

            x = pivot_df[candidate].values
            if np.count_nonzero(x) < 12:
                continue

            y = pivot_df[target]
            x = pivot_df[candidate]
            
            best_corr = 0
            
            # Lag 1 ~ MAX_LAG 탐색
            for lag in range(1, MAX_LAG + 1):
                # A를 lag만큼 밀어서 B와 비교
                x_shifted = x.shift(lag)
                
                # 결측치 제거 후 상관계수 계산
                valid_idx = ~np.isnan(x_shifted) & ~np.isnan(y)
                if valid_idx.sum() < 10: continue
                
                # 상관계수 계산 (method에 따라 다름)
                if method_name == 'pearson':
                    corr = np.corrcoef(x_shifted[valid_idx], y[valid_idx])[0, 1]
                elif method_name == 'spearman':
                    # numpy에는 스피어만이 없어서 pandas로 계산
                    temp_df = pd.DataFrame({'A': x_shifted[valid_idx], 'B': y[valid_idx]})
                    corr = temp_df.corr(method='spearman').iloc[0, 1]
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
            
            # 기준 넘으면 저장
            if abs(best_corr) >= threshold:
                results.append({
                    'leading_item_id': candidate,
                    'following_item_id': target,
                    'value': DUMMY_VALUE
                })
                
    return pd.DataFrame(results)

def run_granger_method(pivot_df, p_val_thr=0.05):
    """그레인저 인과관계 테스트로 짝꿍 찾기"""
    items = pivot_df.columns
    results = []
    
    print(f"\n🚀 [GRANGER] 분석 시작... (시간이 좀 걸립니다)")
    
    for target in tqdm(items, desc="Granger"): # B (후행)
        for candidate in items: # A (선행)
            if target == candidate: continue
            
            # 데이터 준비 (2차원 배열: [Target, Source])
            # statsmodels는 [현재값, 과거값] 순서가 중요함. 보통 [y, x] 순서로 넣음
            data = pd.concat([pivot_df[target], pivot_df[candidate]], axis=1)
            
            # 데이터가 모두 0이거나 변화가 없으면 에러나므로 스킵
            if data.std().min() == 0: continue
            
            try:
                # Granger 테스트 실행 (maxlag까지 한 번에 검사)
                gc_res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)
                
                is_causal = False
                
                # 모든 Lag에 대해 P-value 확인
                for lag in range(1, MAX_LAG + 1):
                    # F-test의 P-value 가져오기
                    p_value = gc_res[lag][0]['ssr_ftest'][1]
                    
                    if p_value < p_val_thr:
                        is_causal = True
                        break # 하나라도 인과성 있으면 통과
                
                if is_causal:
                    results.append({
                        'leading_item_id': candidate,
                        'following_item_id': target,
                        'value': DUMMY_VALUE
                    })
                    
            except Exception:
                continue

    return pd.DataFrame(results)
from collections import defaultdict
a = defaultdict(int)
def run_granger_method(pivot_df, p_val_thr=0.05, min_nonzero_ratio=0.5):
    """
    min_nonzero_ratio: 0이 아닌 데이터가 전체 기간 중 최소 이 비율 이상이어야 테스트 진행 (0.5 = 50%)
    """
    items = pivot_df.columns
    results = []
    
    print(f"\n🚀 [GRANGER] 분석 시작... (희소 데이터 필터링 적용)")
    
    # 전체 기간 길이 미리 계산
    total_len = len(pivot_df)

    for target in tqdm(items, desc="Granger"): 
        # y = pivot_df[target].values
        # if np.count_nonzero(y) < 12:
        #     continue
        s1 = pivot_df[target]
        if (s1 != 0).mean() < min_nonzero_ratio:
            # print(target)
            continue
        for candidate in items: 
            if target == candidate: continue

            # x = pivot_df[candidate].values
            # if np.count_nonzero(x) < 12:
            #     continue
            
            # 두 컬럼 데이터 추출
            s1 = pivot_df[target]
            s2 = pivot_df[candidate]

            # ---------------------------------------------------------
            # [추가된 로직] 데이터 희소성(Sparsity) 체크
            # ---------------------------------------------------------
            # 1. 각 아이템이 0이 아닌 구간이 너무 적으면 스킵 (노이즈 방지)
            if (s2 != 0).mean() < min_nonzero_ratio:
                # print(target, candidate)
                continue

            # 2. 두 아이템이 "동시에" 0이 아닌 구간이 너무 적어도 스킵
            # (교집합 구간이 없으면 인과성 판단 불가)
            common_nonzero = ((s1 != 0) & (s2 != 0)).sum()
            if common_nonzero < (total_len * 0.3): # 예: 겹치는 구간이 30% 미만이면 스킵
                print(target, candidate)
                continue
            # ---------------------------------------------------------

            data = pd.concat([s1, s2], axis=1)
            
            # 표준편차가 0이면(값 변화가 아예 없으면) 에러나므로 스킵
            if data.std().min() == 0: continue
            
            gc_res = grangercausalitytests(data, maxlag=MAX_LAG, verbose=False)

            min_p_value = 1.0
            best_lag = 0

            # 모든 Lag에 대해 검사해서 가장 강력한 신호(가장 낮은 P-value)를 찾음
            # for lag in range(1, MAX_LAG + 1):
            #     # ssr_ftest의 p-value 추출
            #     p_val = gc_res[lag][0]['ssr_ftest'][1]
            #     if p_val < min_p_value:
            #         min_p_value = p_val
            #         best_lag = lag

            # # 기준 통과 시 결과 저장 (P-value도 같이 저장!)
            # if min_p_value < p_val_thr:
            #     results.append({
            #         'leading_item_id': candidate,
            #         'following_item_id': target,
            #         'p_value': min_p_value,  # <-- 이걸 저장해야 나중에 비교 가능
            #         'lag': best_lag,
            #         'value': 0 # 예측값 (나중에 채움)
            #     })
            best_p_value = 1.0
            for lag in range(1, MAX_LAG + 1):
                    # F-test의 p-value
                    p_val = gc_res[lag][0]['ssr_ftest'][1]
                    
                    # 팁: 단순히 하나라도 통과하면 OK가 아니라,
                    # 가장 강력한 신호(최소 p-value)를 찾습니다.
                    if p_val < best_p_value:
                        best_p_value = p_val
                        best_lag = lag
                
                # 기준 통과 시
            if best_p_value < p_val_thr:
                results.append({
                    'leading_item_id': candidate,
                    'following_item_id': target,
                    'lag': best_lag,       # 몇 달 전 반응인지 저장
                    'p_value': best_p_value,
                    'value': 0 # 나중에 예측
                })

    # res_df = pd.DataFrame(results)

    # # A->B 와 B->A 가 둘 다 존재할 경우, P-value가 더 작은(더 확실한) 쪽만 남기기
    # # (선택 사항: 양방향성을 인정하고 싶으면 이 단계 생략 가능)
    # final_results = []
    # for idx, row in res_df.iterrows():
    #     # 반대 방향 쌍이 있는지 확인
    #     reverse_pair = res_df[
    #         (res_df['leading_item_id'] == row['following_item_id']) & 
    #         (res_df['following_item_id'] == row['leading_item_id'])
    #     ]
        
    #     if not reverse_pair.empty:
    #         # 반대 방향의 P-value와 비교
    #         reverse_p = reverse_pair.iloc[0]['p_value']
    #         if row['p_value'] < reverse_p:
    #             final_results.append(row) # 내가 더 쎄니까 내가 살아남음
    #     else:
    #         final_results.append(row) # 반대 방향 없으면 무조건 생존

    return pd.DataFrame(results)#[['leading_item_id', 'following_item_id', 'value']]
# ==========================================
# [메인 실행 코드]
# ==========================================

# 1. 데이터 로드
df_pivot = load_and_preprocess(DATA_PATH)

# 2. 피어슨 (Pearson) 실행 및 저장
# df_pearson = run_correlation_method(df_pivot, method_name='pearson', threshold=PEARSON_THR)
# df_pearson.to_csv('submission_PEARSON.csv', index=False)
# print(f"👉 피어슨 결과 저장 완료: {len(df_pearson)}개 쌍 발견")

# # 3. 스피어만 (Spearman) 실행 및 저장
# df_spearman = run_correlation_method(df_pivot, method_name='spearman', threshold=SPEARMAN_THR)
# df_spearman.to_csv('submission_SPEARMAN.csv', index=False)
# print(f"👉 스피어만 결과 저장 완료: {len(df_spearman)}개 쌍 발견")

# 4. 그레인저 (Granger) 실행 및 저장
df_granger = run_granger_method(df_pivot, p_val_thr=GRANGER_P_VAL)
df_granger.to_csv('submission_GRANGER.csv', index=False)
print(f"👉 그레인저 결과 저장 완료: {len(df_granger)}개 쌍 발견")
print(a)
print("\n🎉 모든 파일 생성 완료! 제출해서 F1 Score를 확인해보세요.")