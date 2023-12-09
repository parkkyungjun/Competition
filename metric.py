def average_precision(label, pred):
    label = label.dropna()[['series_id','event','step']].reset_index(drop=True)
    pred = pred.sort_values(['score'], ascending=False)[['series_id','event','step', 'score']].reset_index(drop=True)
    
    label = label.merge(pred.groupby(['series_id', 'event']).agg(pred_step=('step', list)),
               on=['series_id', 'event'], how='left')
    label['best'] = label[['step', 'pred_step']].apply(lambda x: np.argmin(np.abs(np.array(x[0])-np.array(x[1]))), axis=1)

    label['best_step'] = label[['pred_step', 'best']].apply(lambda x:x[0][x[1]], axis=1)
    label['best_gap'] = np.abs(label['step'] - label['best_step'])
    
    label, pred = reduce_mem(label), reduce_mem(pred)
    
    pred = pred.merge(label[label['best_gap'] < 360][['series_id','event','best_step','best_gap']], left_on=['series_id','event','step'], right_on=['series_id','event','best_step'], how='left')

    score_table={'event':[],'tol':[],'score':[],'pos_recall':[]}
    gaps=[12, 36, 60, 90, 120, 150, 180, 240, 300, 360]

    for event in ['onset', 'wakeup']:
        label_idx, pred_idx = label['event'] == event , pred['event'] == event
        for gap in gaps:
            wrong_cnt = (label.loc[label_idx, 'best_gap'] > gap).sum()
            match = (pred.loc[pred_idx, 'best_gap'] < gap).values.astype('int64')
            score = (pred.loc[pred_idx, 'score']).values

            recall = (sum(label_idx) - wrong_cnt) / sum(label_idx)

            ap = average_precision_score(match, score) * recall
            
            score_table['event'].append(event)
            score_table['tol'].append(gap)
            score_table['score'].append(ap)
            score_table['pos_recall'].append(recall)
    
    score_table=pd.DataFrame(score_table)
    
    display(score_table.round(3))
    display(score_table.groupby(['event']).mean().round(3))
    return score_table['score'].mean()



def macro_f1_score(y_true, y_pred, n_classes):
    return np.mean([f1_score((y_true == c), (y_pred == c)) for c in range(n_classes)])

def auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


