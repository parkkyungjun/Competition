import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from sklearn.model_selection import StratifiedKFold
#from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import cupy as cp
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset
#from transformers import DataCollatorWithPadding
import gc

DATA_PATH = "/opt/ml/Kaggle_HuBMAP-HPA-Hacking-the-Human-Body/Learning_Equality-Curriculum_Recommendations/"
topics = pd.read_csv(DATA_PATH + "topics.csv")
content = pd.read_csv(DATA_PATH + "content.csv")
correlations = pd.read_csv(DATA_PATH + "correlations.csv")

str_kf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 8746)

for train_index, test_index in str_kf.split(correlations, [0]*len(correlations)):
    X_train, X_test = correlations.iloc[train_index], correlations.iloc[test_index]
    _ = X_train
    correlations = X_test
    del X_train, X_test
    break
    
topics.rename(columns=lambda x: "topic_" + x, inplace=True)
content.rename(columns=lambda x: "content_" + x, inplace=True)

corr = correlations.merge(topics, how="left", on="topic_id")

l = []
for i in range(len(content)):
    t = str(content['content_title'].values[i])
    if str(content['content_description'].values[i]) != 'nan':
        t += '[SEP]' + str(content['content_description'].values[i])
    if str(str(content['content_text'].values[i])) != 'nan':
        t += '[SEP]'  + str(content['content_text'].values[i])
    l.append(t)
corr['topic'] = [str(i) + '[SEP]' + str(j) if str(j) != 'nan' else i for i,j in zip(corr['topic_title'].values, corr['topic_description'].values)]
content['content'] = l
del l, t, topics
corr['value'] = 1
corr = corr[['topic_id', 'content_ids', 'topic','value']]
content = content[['content_id', 'content']]

print('dataset on')
class uns_dataset2(Dataset):
    def __init__(self, df, col):
        self.texts = df[col].values
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        inputs = self.texts[item]
        return inputs

class uns_model2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = model
        self.pool = MeanPooling()
    def feature(self, inputs):
        outputs = self.model.encode(**inputs, show_progress_bar=False)
        last_hidden_state = outputs.last_hidden_state
        feature = self.pool(last_hidden_state, inputs['attention_mask'])
        return feature
    def forward(self, inputs):
        feature = self.feature(inputs)
        return feature

def get_embeddings2(loader, model, device):
    model.eval()
    preds = []
    for step, inputs in enumerate(tqdm(loader)):
        with torch.no_grad():
            y_preds = model.encode(inputs, show_progress_bar=False)
        preds.append(y_preds)
    preds = np.concatenate(preds)
    return preds
    
def get_pos_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = [set(i.split()) for i in y_pred]
    int_true = np.array([len(x[0] & x[1]) / len(x[0]) for x in zip(y_true, y_pred)])
    return round(np.mean(int_true), 5)

device = 'cuda'

topics_dataset = uns_dataset2(corr, 'topic')
content_dataset = uns_dataset2(content, 'content')

topics_loader = DataLoader(
    topics_dataset, 
    batch_size = 32, 
    shuffle = False, 
    #collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
    num_workers = 2, 
    pin_memory = True, 
    drop_last = False
)
content_loader = DataLoader(
    content_dataset, 
    batch_size = 32, 
    shuffle = False, 
    #collate_fn = DataCollatorWithPadding(tokenizer = cfg.tokenizer, padding = 'longest'),
    num_workers = 2, 
    pin_memory = True, 
    drop_last = False
    )

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
model.max_seq_length = 128
model.load_state_dict(torch.load('20epochs.pt'))

print('get embedding...')
topics_preds = get_embeddings2(topics_loader, model, device)
content_preds = get_embeddings2(content_loader, model, device)
print('get embedding done')
# topics_preds_gpu = cp.array(topics_preds)
# content_preds_gpu = cp.array(content_preds)

topics_preds_gpu = topics_preds
content_preds_gpu = content_preds

del topics_dataset, content_dataset, topics_loader, content_loader, topics_preds, content_preds
gc.collect()
torch.cuda.empty_cache()
print(' ')
print('Training KNN model...')
# content_preds_gpu = cp.float32(content_preds_gpu.get())
# topics_preds_gpu = cp.float32(topics_preds_gpu.get())

neighbors_model = NearestNeighbors(n_neighbors = 50, metric = 'cosine')
neighbors_model.fit(content_preds_gpu)
indices = neighbors_model.kneighbors(topics_preds_gpu, return_distance = False)
print('indices done')
p = []
t = []
for k in range(len(indices)):
    pred = indices[k]
    prediction = [content.loc[ind, 'content_id'] for ind in pred]
    p += prediction
    t += [corr.loc[k, 'topic_id']] * len(prediction)
    
neg = pd.DataFrame({'topic_id' : t, 'content_id':p, 'value': [0]*len(t)})

corr["content_id"] = corr["content_ids"].str.split(" ")
corr = corr.explode("content_id").drop(columns=["content_ids"])

corr = pd.concat([corr,neg])
del neg, p, t, prediction
corr = corr.merge(content, how="left", on="content_id")
corr = corr[['topic_id', 'content_id', 'topic', 'content', 'value']]
del content
corr = corr.sort_values(by='topic_id' ,ascending=True).reset_index(drop=True)

correlations2 = pd.read_csv(DATA_PATH + "correlations.csv")
topics = pd.read_csv(DATA_PATH + "topics.csv")

topics.rename(columns=lambda x: "topic_" + x, inplace=True)

corr2 = correlations2.merge(topics, how="left", on="topic_id")
corr2['topic'] = [str(i) + '[SEP]' + str(j) if str(j) != 'nan' else i for i,j in zip(corr2['topic_title'].values, corr2['topic_description'].values)]
corr2 = corr2[['topic_id', 'topic']]
del correlations2, topics
corr = corr[['topic_id', 'content_id', 'content', 'value']]
corr = corr.merge(corr2, how="left", on="topic_id")
del corr2
print('corr2 done')
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from torch.utils.data import DataLoader

#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cuda')
model.max_seq_length = 256

class s2_datset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        text = self.df.loc[i, 'topic'] + '[SEP]' + self.df.loc[i, 'content']
        label = self.df.loc[i, 'value']
        return text, torch.tensor(label).to(device)

print('make train dataset...')
train_dataset = s2_datset(corr)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=384)
# 0 : max_pos_top-50: 0.45608
# 1 : max_pos_top-50: 0.43154
# 20 : 0.51276
gc.collect()
torch.cuda.empty_cache()

class stage2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.bc = nn.Linear(384, 1)
        # Set requires_grad attribute of self.model parameters to False
        # for param in self.model.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        # Pass input through self.model but only compute output of self.bc
        encoded = self.model.encode(x)
        return self.bc(torch.tensor(encoded).to(device))

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


def f2_score(true_ids, pred_ids):
    true_positives = len(set(true_ids)&set(pred_ids))
    false_positives = len(set(pred_ids)-set(true_ids))
    false_negatives = len(set(true_ids)-set(pred_ids))

    beta = 2
    f2_score = ((1+beta**2)*true_positives)/((1+beta**2)*true_positives + beta**2*false_negatives + false_positives)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

# Instantiate stage2 model
model = stage2(model)
#model.load_state_dict(torch.load('3epochs_stage2.pt'))
model.load_state_dict(torch.load('3epochs_stage2_Frozen.pt'))
model.to(device)
model.eval()
threshold = 0.001
correlations = correlations.sort_values(by='topic_id' ,ascending=False)
for k in tqdm(range(999)):
    find = np.numpy()
    
    for text, _ in train_dataloader:
        output = model(text).unsqueeze().detach().cpu().numpy()
        find = np.concatenate(find, output)
    corr['out'] = find
    temp = corr[corr['out'] >= threshold]
    temp = temp.groupby('topic_id').agg({'content_id': ' '.join}).reset_index()
    temp = temp.sort_values(by='topic_id' ,ascending=False)
    if k == 0:
        if all([t==c for t, c in zip(temp['topic_id'], correlations['topic_id'])]) == False:
            print(len(temp), len(correlations))
            break
        else:
            print('good')
    avg = 0
    for i in range(len(temp)):
        avg += f2_score(list(correlations.loc[i, 'content_ids']), list(temp.loc[i, 'content_id']))
    val = avg/len(temp)
    if val > befo:
        print(threshold, val)
        befo = val
    threshold += 0.001 
