import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader import pl
import torch_xla.distributed.xla_multiprocessing as xmp

from sklearn import model_selection
from sklearn import metrics

# local modules
import config
from model import BertMultilingualModel
from dataset import JigsawDataset
from train import loss_fn, train_fn, valid_fn


def run():
    # dataframes
    train_df = pd.read_csv(config.TRAIN)
    # dataset
    trainset = JigsawDataset()
