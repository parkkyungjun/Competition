import pandas as pd
from collections import defaultdict

df = pd.read_csv('/home/pkj/rag/data/train.csv')

ps = ['1-1 2024 주요 재정통계 1권', '2024 나라살림 예산개요', '2024년도 성과계획서(총괄편)', '월간 나라재정 2023년 12월호', '재정통계해설']
last = defaultdict(int)
for data in df.iterrows():
    SAMPLE_ID, Source, Source_path, Question, Answer = data[1]
    if Source in ps:
        continue
    k = Answer.split(' ')[-1]
    last[k] = last[k] + 1  
print(last)