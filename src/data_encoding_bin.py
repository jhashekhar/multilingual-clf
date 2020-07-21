import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import BertTokenizer
from transformers import XLMRobertaTokenizer


TRAIN1_PATH = '../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv'
TRAIN2_PATH = '../input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv'
VALID_PATH = '../input/jigsaw-multilingual-toxic-comment-classification/validation.csv'
TEST_PATH = '../input/jigsaw-multilingual-toxic-comment-classification/test.csv'


train1_df = pd.read_csv(TRAIN1_PATH, usecols=['comment_text', 'toxic']).fillna('none')
train2_df = pd.read_csv(TRAIN2_PATH, usecols=['comment_text', 'toxic']).fillna('none')
train2_df.toxic = np.round(train2_df.toxic).astype(int)
train2 = train2_df.query("toxic==1")
trainfull_df = pd.concat([train1_df, train2_df], axis=0)
train_df = trainfull_df.sample(frac=1).head(400000)


class JigsawDataset(object):
    def __init__(self, input_ids=None, attention_mask=None, targets=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets
        
    def __len__(self):
        return self.input_ids.shape[0]
    
    def __getitem__(self, item):
        input_ids = self.input_ids[item]
        attention_mask = self.attention_mask[item]
        targets = self.targets[item]

        return {
            'input_ids': torch.from_numpy(input_ids),
            'attention_mask': torch.from_numpy(attention_mask),
            'targets': torch.from_numpy(targets)}


# Multiprocessing makes the encoding process 4 times faster
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 192
num_cores = 4
N = 1000000

# get dataframes
train1_df = pd.read_csv(TRAIN1_PATH, usecols=['comment_text', 'toxic']).fillna('none')
train2_df = pd.read_csv(TRAIN2_PATH, usecols=['comment_text', 'toxic']).fillna('none')
trainfull_df = pd.concat([train1_df, train2_df], axis=0).reset_index(drop=True)
train_df = trainfull_df.sample(frac=1, random_state=42).head(N)


# remove wierd spaces and convert to lower case
def preprocessing(text):
    text = str(text).strip().lower()
    return " ".join(text.split())


# encode string for each subprocess
def token_encoding(t, tokenizer=bert_tokenizer, max_len=192):
    texts = t[0]
    targets = t[1]
    
    input_ids = []
    token_type_ids = []
    attention_mask = []
    
    for i in tqdm(range(0, len(texts))):
        text = preprocessing(texts[i])
        inputs = tokenizer.encode_plus(text,
                                       pad_to_max_length=True, 
                                       max_length=max_len)
        input_ids.append(inputs['input_ids'])
        token_type_ids.append(inputs['token_type_ids'])
        attention_mask.append(inputs['attention_mask'])
        
    return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask), np.array(targets)


def encoding_process():
    
    text1 = train_df.comment_text.values[:int(N/4)]
    target1 = train_df.toxic.values[:int(N/4)]
    
    text2 = train_df.comment_text.values[int(N/4):int(N/2)]
    target2 = train_df.toxic.values[int(N/4):int(N/2)]
    
    text3 = train_df.comment_text.values[int(N/2):int(0.75 * N)]
    target3 = train_df.toxic.values[int(N/2):int(0.75 * N)]
    
    text4 = train_df.comment_text.values[int(0.75 * N):]
    target4 = train_df.toxic.values[int(0.75 * N):]
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 192
    
    pool = Pool(num_cores)
    
    chunk1 = (text1, target1)
    chunk2 = (text2, target2)
    chunk3 = (text3, target3)
    chunk4 = (text4, target4)
    
    chunks = [chunk1, chunk2, chunk3, chunk4]
    
    result = pool.map(token_encoding, chunks)
    
    input_ids = np.concatenate([r[0] for r in result], axis=0)
    token_type_ids = np.concatenate([r[1] for r in result], axis=0)
    attention_mask = np.concatenate([r[2] for r in result], axis=0)
    targets = np.concatenate([r[3] for r in result], axis=0)
    
    assert input_ids.shape[0] == attention_mask.shape[0] == targets.shape[0]
    
    return input_ids, token_type_ids, attention_mask, targets


def save_data(compressed=False):
    if compressed is True:
        np.savez_compressed('train-df-compressed-input-ids.npz', input_ids)
        np.savez_compressed('train-df-compressed-attention-mask.npz', attention_mask)
        np.savez_compressed('train-df-compressedtoken-type-ids.npz', token_type_ids)
        np.savez_compressed('train-df-compressed-targets.npz', targets)

    else:
        np.save('train-df-input-ids.npy', input_ids)
        np.save('train-df-attention-mask.npy', attention_mask)
        np.save('train-df-token-type-ids', token_type_ids)
        np.save('train-df-targets', targets)


input_ids, token_type_ids, attention_mask, targets = encoding_process()

save_data(compressed=True)

TOKEN_ID_PATH = '/kaggle/input/train-df-compressed-token-type-ids.npz'
ATTENTION_PATH = '/kaggle/input/train-df-compressed-attention-mask.npz'
INPUT_ID_PATH = '/kaggle/input/train-df-compressed-input-ids.npz'
TARGETS_PATH = '/kaggle/input/train-df-compressed-targets.npz'


# loading compressed numpy data
load_input_ids = np.load(INPUT_ID_PATH, mmap_mode='r')
load_token_type_ids = np.load(TOKEN_ID_PATH, mmap_mode='r')
load_attention_mask = np.load(ATTENTION_PATH, mmap_mode='r')
load_targets = np.load(TARGETS_PATH, mmap_mode='r')

input_ids = load_input_ids.f.arr_0
token_type_ids = load_token_type_ids.f.arr_0
attention_mask = load_attention_mask.f.arr_0
targets = load_targets.f.arr_0

# split train_rows = 392000, valid_rows = 8000
train_split = 392000

# training data
train_input_ids = input_ids[:train_split]
train_token_type_ids = token_type_ids[:train_split]
train_attention_mask = attention_mask[:train_split]
train_targets = targets[:train_split]

# validation data
valid_input_ids = input_ids[train_split:]
valid_token_type_ids = input_ids[train_split:]
valid_attention_mask = attention_mask[train_split:]
valid_targets = targets[train_split:]


# sanity check for the sizes
assert valid_input_ids.shape[0] == valid_attention_mask.shape[0] == valid_targets.shape[0]
assert train_input_ids.shape[0] == train_attention_mask.shape[0] == train_targets.shape[0]

# uncomment the lines below to check the sizes of data rows
# print(valid_input_ids.shape[0], valid_attention_mask.shape[0], valid_targets.shape[0])
# print(train_input_ids.shape[0], train_attention_mask.shape[0], train_targets.shape[0])