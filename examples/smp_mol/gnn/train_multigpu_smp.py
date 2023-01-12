import argparse
import logging
import os
import time
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
# import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import GNN_SMP
from data import GNNTransformSMP
from atom3d.datasets import LMDBDataset, MolH5Dataset
from module import SMPLitModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from sklearn.metrics import mean_absolute_error
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_squared_error
import torch_geometric.transforms as T
from pytorch_lightning.callbacks import ModelSummary


def train(args):

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join('/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/', now)
    #log_dir= '/p/project/hai_drug_qm/atom3d/examples/lba_md/cnn3d/'   
    if not os.path.exists(log_dir):
        try : 
            os.makedirs(log_dir)
        except : 
            print('problem with log directory')
    
    h5_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/qm.h5"
    norm_h5_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/norm_qm.hdf5"


    train_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/train_norm.txt"
    val_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/val_norm.txt"
    test_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/test_norm.txt"

    train_dataset = MolH5Dataset(h5_file, train_idx, target=args.target_name, target_norm_file=norm_h5_file, transform=GNNTransformSMP(args.target_name), post_transform=T.RandomTranslate(0.25))
    val_dataset = MolH5Dataset(h5_file, val_idx, target=args.target_name, target_norm_file=norm_h5_file, transform=GNNTransformSMP(args.target_name))
    test_dataset = MolH5Dataset(h5_file, test_idx, target=args.target_name, target_norm_file=norm_h5_file, transform=GNNTransformSMP(args.target_name))
        
    train_loader = DataLoader(train_dataset, 128, shuffle=True, num_workers=5)
    val_loader = DataLoader(val_dataset, 128, shuffle=False, num_workers=5)
    test_loader = DataLoader(test_dataset, 128, shuffle=False, num_workers=5)

    print("data loaded")
    for data in train_loader:     
        print(data)   
        num_features = data.num_features
        break
        
    print("number gpus : ", torch.cuda.device_count())
    model = GNN_SMP(num_features, dim=args.hidden_dim)
    module = SMPLitModule(model)
    print("model",next(module.parameters()).is_cuda) 
    bar = TQDMProgressBar(refresh_rate = 10)
    logger = TensorBoardLogger(f"{log_dir}/tensor_board")

    print(f"Log dir for trainer{log_dir}")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", dirpath=f'{log_dir}/model_ckpts/', filename=f'gnn_{args.target_name}', verbose=True, every_n_epochs=1 )
    #trainer = Trainer(devices="auto", accelerator="auto", strategy="ddp", callbacks=[ModelSummary(max_depth=3), checkpoint_callback, bar], max_epochs=3, logger=logger,  gradient_clip_val=0.5) 
    trainer = Trainer(devices=1, accelerator="gpu", callbacks=[ModelSummary(max_depth=3), checkpoint_callback, bar], max_epochs=200, logger=logger,  gradient_clip_val=0.5) 

    # trainer = Trainer(devices="auto", accelerator="auto", strategy="horovod", callbacks=[checkpoint_callback, bar], max_epochs=11, logger=logger,  gradient_clip_val=1.5, profiler="simple", resume_from_checkpoint="/p/project/hai_drug_qm/atom3d/examples/lba_md/cnn3d/model_ckpts/23-10-22/epochs.ckpt") 
    # print("world size : ", trainer.world_size)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target_name', type=str, default="Electronegativity")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default='/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/logs')
    parser.add_argument('--tensor_board', type=str, default="/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/tensor_board/")
    parser.add_argument('--model_ckpt', type=str, default="/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/model_ckpts/")
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--flip', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--train_name', type=str)
    args = parser.parse_args()
    log_dir = args.log_dir
    
    train(args)
