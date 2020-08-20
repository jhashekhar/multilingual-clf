'''
    Complete code of the training and inference
'''

# computational modules
import numpy as np
import pandas as pd

from tqdm import tqdm

# system modules
import time
import os
import sys

# multiprocessing modules
import multiprocessing as mp
from multiprocessing import Pool

# torch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# transformer modules
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from transformers import BertTokenizer, BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

# tpu-specific modules
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from sklearn import metrics

# local modules
from dataset import JigsawDataset

#############################################################################################
# dataset.py

TRAIN1_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv'
TRAIN2_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv'
train1_df = pd.read_csv(TRAIN1_PATH, usecols=['comment_text', 'toxic']).fillna('none')
train2_df = pd.read_csv(TRAIN2_PATH, usecols=['comment_text', 'toxic']).fillna('none')

# Multiprocessing makes the transformation 4 times faster

TRAIN1_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv'
TRAIN2_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv'
VALID_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv'
TEST_PATH = '/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv'

train1_df = pd.read_csv(TRAIN1_PATH, usecols=['comment_text', 'toxic']).fillna('none')
train2_df = pd.read_csv(TRAIN2_PATH, usecols=['comment_text', 'toxic']).fillna('none')
trainfull_df = pd.concat([train1_df, train2_df], axis=0).reset_index(drop=True)

train_df = trainfull_df.sample(frac=1, random_state=42)
valid_df = pd.read_csv(VALID_PATH, usecols=['comment_text', 'toxic'])
test_df = pd.read_csv(TEST_PATH, usecols=['content'])


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#xlm_roberta_tokenizer = XLMRobertaTokenizer.from_pretrained('')

MAX_LENGTH = 192
#num_cores = 4
train_N = 800000
test_N = test_df.shape[0]
valid_N = valid_df.shape[0]


# get dataframes
train_df = train_df.head(train_N)
test_df = test_df
valid_df = valid_df


# remove wierd spaces and convert to lower case
def preprocessing(text):
    text = str(text).strip().lower()
    return " ".join(text.split())


# encode string for each subprocess
def token_encoding(t, target, tokenizer, max_length):
    # there is no target for the test case
    if target is True:
        texts = t[0]
        targets = t[1]
    else:
        texts = t
    
    input_ids = []
    token_type_ids = []
    attention_mask = []
    
    for i in tqdm(range(0, len(texts))):
        text = preprocessing(texts[i])
        inputs = tokenizer.encode_plus(text,
                                       None,
                                       pad_to_max_length=True, 
                                       max_length=max_length)
        
        input_ids.append(inputs['input_ids'])
        token_type_ids.append(inputs['token_type_ids'])
        attention_mask.append(inputs['attention_mask'])
    
    if target is True:
        return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask), np.array(targets)
    
    else:
        return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)


def encoding_process(df, N, num_cores, tokenizer, max_length):
    
    text1 = df.comment_text.values[:int(N/4)]
    target1 = df.toxic.values[:int(N/4)]
    
    text2 = df.comment_text.values[int(N/4):int(N/2)]
    target2 = df.toxic.values[int(N/4):int(N/2)]
    
    text3 = df.comment_text.values[int(N/2):int(0.75 * N)]
    target3 = df.toxic.values[int(N/2):int(0.75 * N)]
    
    text4 = df.comment_text.values[int(0.75 * N):]
    target4 = df.toxic.values[int(0.75 * N):]
    
    process_pool = Pool(num_cores)
    
    chunk1 = ((text1, target1), True, tokenizer, max_length)
    chunk2 = ((text2, target2), True, tokenizer, max_length)
    chunk3 = ((text3, target3), True, tokenizer, max_length)
    chunk4 = ((text4, target4), True, tokenizer, max_length)
    
    chunks = [chunk1, chunk2, chunk3, chunk4]
    
    output = process_pool.starmap(token_encoding, chunks)
    
    input_ids = np.concatenate([out[0] for out in output], axis=0)
    token_type_ids = np.concatenate([out[1] for out in output], axis=0)
    attention_mask = np.concatenate([out[2] for out in output], axis=0)
    targets = np.concatenate([out[3] for out in output], axis=0)
    
    assert input_ids.shape[0] == token_type_ids.shape[0] \
            == attention_mask.shape[0] == targets.shape[0] == N
    
    return input_ids, token_type_ids, attention_mask, targets
    

def test_encoding_process(df, N, num_cores, tokenizer, max_length):
    
    text1 = df.content.values[:int(N/4)]
    
    text2 = df.content.values[int(N/4):int(N/2)]
    
    text3 = df.content.values[int(N/2):int(0.75 * N)]
    
    text4 = df.content.values[int(0.75 * N):]
    
    process_pool = Pool(num_cores)
    
    chunk1 = (text1, False, tokenizer, max_length)
    chunk2 = (text2, False, tokenizer, max_length)
    chunk3 = (text3, False, tokenizer, max_length)
    chunk4 = (text4, False, tokenizer, max_length)
    
    chunks = [chunk1, chunk2, chunk3, chunk4]
    
    output = process_pool.starmap(token_encoding, chunks)
    
    input_ids = np.concatenate([out[0] for out in output], axis=0)
    token_type_ids = np.concatenate([out[1] for out in output], axis=0)
    attention_mask = np.concatenate([out[2] for out in output], axis=0)
    
    assert input_ids.shape[0] == token_type_ids.shape[0] == attention_mask.shape[0] == N
    
    return input_ids, token_type_ids, attention_mask


def save_data(compressed=False):
    if compressed is True:
        np.savez_compressed('train-df-compressed-input-ids.npz', train_input_ids)
        np.savez_compressed('train-df-compressed-attention-mask.npz', train_attention_mask)
        np.savez_compressed('train-df-compressed-token-type-ids.npz', train_token_type_ids)
        np.savez_compressed('train-df-compressed-targets.npz', train_targets)
        
        # valid
        np.savez_compressed('valid-df-compressed-input-ids.npz', valid_input_ids)
        np.savez_compressed('valid-df-compressed-attention-mask.npz', valid_attention_mask)
        np.savez_compressed('valid-df-compressed-token-type-ids.npz', valid_token_type_ids)
        np.savez_compressed('valid-df-compressed-targets.npz', valid_targets)
        
        # test
        np.savez_compressed('test-df-compressed-input-ids.npz', test_input_ids)
        np.savez_compressed('test-df-compressed-attention-mask.npz', test_attention_mask)
        np.savez_compressed('test-df-compressed-token-type-ids.npz', test_token_type_ids)

    else:
        np.save('train-df-input-ids.npy', input_ids)
        np.save('train-df-attention-mask.npy', attention_mask)
        np.save('train-df-token-type-ids', token_type_ids)
        np.save('train-df-targets', targets)
        
        
train_input_ids, train_token_type_ids, train_attention_mask, train_targets \
                        = encoding_process(
                                    df=train_df,
                                    N=train_N,
                                    num_cores=4,
                                    tokenizer=bert_tokenizer,
                                    max_length=MAX_LENGTH)

print("Training data encoding complete.")


valid_input_ids, valid_token_type_ids, valid_attention_mask, valid_targets \
                        = encoding_process(
                                    df=valid_df,
                                    N=valid_N,
                                    num_cores=4,
                                    tokenizer=bert_tokenizer,
                                    max_length=MAX_LENGTH)

print("Validation data encoding complete.")


test_input_ids, test_token_type_ids, test_attention_mask \
                        = test_encoding_process(
                                    df=test_df,
                                    N=test_N,
                                    num_cores=4,
                                    tokenizer=bert_tokenizer,
                                    max_length=MAX_LENGTH)

print("Test data encoding complete.")

save_data(compressed=True)


# training data paths
TRAIN_INPUT_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/train-df-compressed-input-ids.npz'
TRAIN_TOKEN_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/train-df-compressed-token-type-ids.npz'
TRAIN_ATTENTION_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/train-df-compressed-attention-mask.npz'
TRAIN_TARGETS_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/train-df-compressed-targets.npz'

# validation data paths
VALID_INPUT_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/valid-df-compressed-input-ids.npz'
VALID_TOKEN_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/valid-df-compressed-token-type-ids.npz'
VALID_ATTENTION_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/valid-df-compressed-attention-mask.npz'
VALID_TARGETS_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/valid-df-compressed-targets.npz'

# test data paths
TEST_INPUT_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/test-df-compressed-input-ids.npz'
TEST_TOKEN_ID_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/test-df-compressed-token-type-ids.npz'
TEST_ATTENTION_PATH = '/kaggle/input/jigsaw-compressed-bert-tokens/test-df-compressed-attention-mask.npz'

# loading compressed numpy training data
load_train_input_ids = np.load(TRAIN_INPUT_ID_PATH, mmap_mode='r')
load_train_token_type_ids = np.load(TRAIN_TOKEN_ID_PATH, mmap_mode='r')
load_train_attention_mask = np.load(TRAIN_ATTENTION_PATH, mmap_mode='r')
load_train_targets = np.load(TRAIN_TARGETS_PATH, mmap_mode='r')

# training data
train_input_ids = load_train_input_ids.f.arr_0
train_token_type_ids = load_train_token_type_ids.f.arr_0
train_attention_mask = load_train_attention_mask.f.arr_0
train_targets = load_train_targets.f.arr_0

# loading compressed numpy validation data
load_valid_input_ids = np.load(VALID_INPUT_ID_PATH, mmap_mode='r')
load_valid_token_type_ids = np.load(VALID_TOKEN_ID_PATH, mmap_mode='r')
load_valid_attention_mask = np.load(VALID_ATTENTION_PATH, mmap_mode='r')
load_valid_targets = np.load(VALID_TARGETS_PATH, mmap_mode='r')

# validation data
valid_input_ids = load_valid_input_ids.f.arr_0
valid_token_type_ids = load_valid_token_type_ids.f.arr_0
valid_attention_mask = load_valid_attention_mask.f.arr_0
valid_targets = load_valid_targets.f.arr_0

# loading compressed numpy test data
load_test_input_ids = np.load(TEST_INPUT_ID_PATH, mmap_mode='r')
load_test_token_type_ids = np.load(TEST_TOKEN_ID_PATH, mmap_mode='r')
load_test_attention_mask = np.load(TEST_ATTENTION_PATH, mmap_mode='r')

# test data
test_input_ids = load_test_input_ids.f.arr_0
test_token_type_ids = load_test_token_type_ids.f.arr_0
test_attention_mask = load_test_attention_mask.f.arr_0


# sanity check for the sizes
assert train_input_ids.shape[0] == train_token_type_ids.shape[0] \
        == train_attention_mask.shape[0] == train_targets.shape[0]

assert valid_input_ids.shape[0] == valid_token_type_ids.shape[0] \
        == valid_attention_mask.shape[0] == valid_targets.shape[0]

assert test_input_ids.shape[0] == test_token_type_ids.shape[0] == test_attention_mask.shape[0]


# uncomment the lines below to check the sizes of the data rows
print(train_input_ids.shape[0], train_token_type_ids.shape[0], train_attention_mask.shape[0], train_targets.shape[0])
print(valid_input_ids.shape[0], valid_token_type_ids.shape[0], valid_attention_mask.shape[0], valid_targets.shape[0])
print(test_input_ids.shape[0], test_token_type_ids.shape[0], test_attention_mask.shape[0])

#############################################################################################
# model.py

class BertMultilingualModel(nn.Module):
    def __init__(self, model, dropout):
        super(BertMultilingualModel, self).__init__()
        self.bert_model = model
        self.fc = nn.Linear(768, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, out = self.bert_model(
                            input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
        out = self.dropout(out)
        out = self.fc(out)
        return out



class XLMRoberta(nn.Module):
    def __init__(self, model, dropout):
        super(XLMRoberta, self).__init__()
        self.roberta = model
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(True)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, out2 = self.roberta(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        
        out2 = self.dropout(out2)
        out = self.relu(self.fc1(out2))
        out = self.fc2(out)
        return out


#############################################################################################
# train.py


def loss_fn(output, target):
    return nn.BCEWithLogitsLoss()(output, target)


def train_fn(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    train_loss = []
    
    for i, data in enumerate(dataloader):
        input_ids = data['input_ids']
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']
        targets = data['targets']
        
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = targets.to(device)
        
        output = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fn(output, targets.unsqueeze(1))
        train_loss.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            xm.master_print(f"iteration: {i}, train loss: {loss:.4f}")
        
        if scheduler is not None:
            scheduler.step()
            
    return train_loss


def valid_fn(dataloader, model, device):
    valid_loss = []
    outputs = []
    targets = []
    
    with torch.no_grad():
      
        for i, data in enumerate(dataloader):
            input_ids = data['input_ids']
            token_type_ids = data['token_type_ids']
            attention_mask = data['attention_mask']
            targets = data['targets']

            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = targets.to(device)

            output = model(input_ids, token_type_ids, attention_mask)
            
            output_np = output.cpu().detach().numpy().tolist()
            target_np = targets.cpu().detach().numpy().tolist()
            
            outputs.extend(output_np)
            targets.extend(target_np)
            
    return outputs, targets


def run(
    epochs, 
    batch_size, 
    num_workers, 
    learning_rate, 
    warmup_steps,
    pretrained_model,
    dropout):
    
    
    # datasets, samplers and dataloaders
    trainset = JigsawDataset(
        input_ids=train_input_ids,
        token_type_ids=train_token_type_ids,
        attention_mask=train_attention_mask,
        targets=train_targets)
    
    validset = JigsawDataset(
        input_ids=valid_input_ids,
        token_type_ids=valid_token_type_ids,
        attention_mask=valid_attention_mask,
        targets=valid_targets)
    
    # samplers
    trainsampler = DistributedSampler(
        dataset=trainset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    
    validsampler = DistributedSampler(
        dataset=validset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
    
    # dataloaders
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        sampler=trainsampler,
        num_workers=num_workers,
        drop_last=True,)
    
    validloader = DataLoader(
        dataset=validset,
        batch_size=batch_size,
        sampler=validsampler,
        drop_last=True) 
    
    xm.master_print(f"Loading datasets....Complete!")
    
    # model
    device = xm.xla_device()
    model = BertBaseUncased(pretrained_model, dropout)
    model = model.to(device)
    xm.master_print(f"Loading model....Complete!")
    
    # training_parameters, optimizers and schedulers
    not_decay = ['LayerNorm.weight', 'LayerNorm.bias', 'bias']
    
    parameters = list(model.named_parameters())
    
    train_parameters = [
        {'params': [p for n, p in parameters if not any(nd in n for nd in not_decay)], 
         'weight_decay': 0.001},
        
        {'params': [p for n, p in parameters if any(nd in n for nd in not_decay)], 
         'weight_decay': 0.001 }]
    
    
    num_training_steps = int(len(trainset) / xm.xrt_world_size())
    xm.master_print(f"Iterations per epoch: {num_training_steps}")
    
    optimizer = AdamW(train_parameters, lr=learning_rate)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps)
    
    
    # training and evaluation
    for epoch in range(epochs):
        # train
        para_train_loader = pl.ParallelLoader(trainloader, [device])
        
        start_time = time.time()
        
        train_loss = train_fn(
            model, 
            para_train_loader.per_device_loader(device), 
            optimizer, 
            device, 
            scheduler=scheduler)
        
        end_time = time.time()
        time_per_epoch = end_time - start_time
        xm.master_print(f"Time taken: {time_per_epoch}")
        
        xm.master_print(f"epoch: {epoch+1}/{epochs}, train loss: {np.mean(train_loss):.4f}")
        
        # eval
        para_valid_loader = pl.ParallelLoader(validloader, [device])
        outputs, targets = valid_fn(
            para_valid_loader.per_device_loader(device),
            model,
            device)
        
        auc = metrics.roc_auc_score(np.array(targets) > 0.5, outputs)
        xm.master_print(f"auc_score: {auc:.4f}")


#############################################################################################
# main.py

# hyper parameters
MODEL_PATH = 'bert-base-multilingual-uncased'
BATCH_SIZE = 128
NUM_WORKERS = 8
DROPOUT = 0.3
LR = 0.4 * 1e-5
EPOCHS = 5
WARMUP_STEPS = 0

MODEL = BertModel.from_pretrained(MODEL_PATH)

def _mp_fn(rank, flags):
    
    a = run(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
        pretrained_model=MODEL,
        dropout=DROPOUT)
    

FLAGS = {}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')

def predictions():
    testset = JigsawDataset(
        input_ids=test_input_ids,
        token_type_ids=test_token_type_ids,
        attention_mask=test_attention_mask)
    
    testsampler = DistributedSampler(
        dataset=testset,
        num_replicas=x.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False)
    
    testloader = DataLoader(
        dataset=testset,
        batch_size=batch_size,
        sampler=testsampler,
        num_workers=num_workers,
        droplast=True)
    
    
    model = model.to(device)
    model.eval()
    
    outputs = []
    
    for i, data in enumerate(dataloader):
        input_ids = data['input_ids']
        token_type_ids = data['token_type_ids']
        attention_mask = data['attention_mask']
        
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        output = model(
            input_ids=input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask)
        
        output_np = output.cpu().detach().numpy().tolist()
        outputs.extend(output_np)
        
    return outputs