import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# transformer modules
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

# tpu-specific modules
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# sklearn modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# local modules
from dataset import JigsawDataset
from models import *


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def reduce_fn(vals):
    return sum(vals) / len(vals)


def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    
    train_loss = []
    
    for bi, data in enumerate(data_loader):

        ids = data['ids']
        mask = data['mask']
        targets = data['target']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        
        outputs = model(input_ids=ids, attention_mask=mask)
        loss = loss_fn(outputs, targets)
        
        train_loss.append(loss.item())
        
        if bi % 500 == 0:
            loss_reduced = xm.mesh_reduce('loss_reduce', loss, reduce_fn)
            xm.master_print(f'bi={bi}, loss={loss_reduced:.4f}')
        
        loss.backward()
        
        xm.optimizer_step(optimizer)

        if scheduler is not None:
            scheduler.step()
            
    return train_loss
    

def eval_loop_fn(data_loader, model, device):
    model.eval()
    
    fin_targets = []
    fin_outputs = []
    
    for bi, data in enumerate(data_loader):
        ids = data['ids']
        mask = data['mask']
        targets = data['target']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(input_ids=ids, attention_mask=mask)

        targets_np = targets.cpu().detach().numpy().tolist()
        outputs_np = outputs.cpu().detach().numpy().tolist()
        
        fin_targets.extend(targets_np)
        fin_outputs.extend(outputs_np)    
        
        del targets_np, outputs_np
        
        gc.collect()
        
    return fin_outputs, fin_targets


def _run():
    MAX_LEN = 192

    train_sampler = DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0)
    
    valid_sampler = DistributedSampler(
          valid_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False)

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        sampler=valid_sampler,
        drop_last=False,
        num_workers=0)

    device = xm.xla_device()
    
    model = mx.to(device)
    
    # print only once
    if fold == 0:
        xm.master_print('done loading model')

    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    lr = 0.5e-5 * xm.xrt_world_size()
    
    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    
    if args.scheduler == 'cosine':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps= int(0.01 * (num_train_steps)),
            num_training_steps=num_train_steps)
    
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps= int(0.01 * (num_train_steps)),
            num_training_steps=num_train_steps)
    
    xm.master_print(f'num_train_steps = {num_train_steps}, world_size={xm.xrt_world_size()}')

    gc.collect()


    for epoch in range(EPOCHS):
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        
        # print only once
        if epoch == 0: 
            xm.master_print('parallel loader created... training now')

        # train mode/function
        train_loss = train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)
        
        del para_loader
        
        # eval mode/function
        para_loader = pl.ParallelLoader(valid_data_loader, [device])
        o, t = eval_loop_fn(para_loader.per_device_loader(device), model, device)
        
        del para_loader
        gc.collect()
        
        auc = roc_auc_score(np.array(t) >= 0.5, o)
        auc_reduced = xm.mesh_reduce('auc_reduce',auc,reduce_fn)
        xm.master_print(f'Epoch: {epoch+1}/{EPOCHS} | train loss: {np.mean(train_loss):.4f} | val auc: {auc_reduced:.4f}')

    gc.collect()

    xser.save(model.state_dict(), f"f{fold+1}_roberta_large.pth")

