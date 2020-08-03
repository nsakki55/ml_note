import sys

import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob

sys.path.append("../input/timm-efficientdet-pytorch/")
sys.path.append("../input/omegaconf/")

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
import warnings

warnings.filterwarnings("ignore")

import utils
import data_process
import trainer

SEED = 42

utils.seed_everything(SEED)


VERSION = "effdet-mixup-cutmix-cv" # 実験番号
logger = utils.create_logger(VERSION)


marking = data_process.get_marking(train_csv_path = '../input/global-wheat-detection/train.csv')
df_folds = data_process.get_folds(marking)

for fold_num in range(5):
    logger.info(f"{fold_num} fold start")
    # train config
    train_global_config = trainer.TrainGlobalConfig(n_epochs=60, batch_size=3, folder = f'effdet5-cutout-cutmix-mixup-{fold_num}CV-improve')
    
    # dataset
    dataset_dict = data_process.get_dataset(fold_num, df_folds, marking)
    train_dataset = dataset_dict['train']
    validation_dataset = dataset_dict['valid']
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_global_config.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=train_global_config.num_workers,
        collate_fn=trainer.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=train_global_config.batch_size,
        num_workers=train_global_config.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=trainer.collate_fn,
    )
    
    # net
    device = torch.device('cuda:0')
    net = trainer.get_net()
    net.to(device)
    
    fitter = trainer.Fitter(model=net, device=device, config=train_global_config)
    fitter.fit(train_loader, val_loader)
    
    logger.info(f"{fold_num} fold finished")
