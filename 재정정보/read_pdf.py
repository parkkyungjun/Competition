import numpy as np
import pandas as pd
from collections import Counter

# df = pd.read_csv('train_infer.csv')
# pred = df['Answer']

# df = pd.read_csv('train.csv')
# gt = df['Answer']

def calculate_f1_score(true_sentence, predicted_sentence, sum_mode=True):
    true_counter = Counter(true_sentence)
    predicted_counter = Counter(predicted_sentence)

    #문자가 등장한 개수도 고려
    if sum_mode:
        true_positive = sum((true_counter & predicted_counter).values())
        predicted_positive = sum(predicted_counter.values())
        actual_positive = sum(true_counter.values())

    #문자 자체가 있는 것에 focus를 맞춤
    else:
        true_positive = len((true_counter & predicted_counter).values())
        predicted_positive = len(predicted_counter.values())
        actual_positive = len(true_counter.values())

    #f1 score 계산
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0
    recall = true_positive / actual_positive if actual_positive > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def calculate_average_f1_score(true_sentences, predicted_sentences):
    
    total_precision = 0
    total_recall = 0
    total_f1_score = 0
    
    for true_sentence, predicted_sentence in zip(true_sentences, predicted_sentences):
        precision, recall, f1_score = calculate_f1_score(true_sentence, predicted_sentence)
        total_precision += precision
        total_recall += recall
        total_f1_score += f1_score
    
    avg_precision = total_precision / len(true_sentences)
    avg_recall = total_recall / len(true_sentences)
    avg_f1_score = total_f1_score / len(true_sentences)
    
    return {
        'average_precision': avg_precision,
        'average_recall': avg_recall,
        'average_f1_score': avg_f1_score
    }

# result = calculate_average_f1_score(['2024년 중앙정부 재정체계는 예산(일반·특별회계)과 기금으로 구분되며, 2024년 기준으로 일반회계 1개, 특별회계 21개, 기금 68개로 구성되어 있습니다.'], ['중앙정부의 재정체계는 예산과 기금으로 구분됩니다. 2024년 기준으로 일반회계, 특별회계, 기금이 있습니다'])
# print(result)