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

def train(args, device, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    print(args)
    pl.seed_everything(12345, workers=True)
    
    h5_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/qm.h5"

    train_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/train_norm.txt"
    val_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/val_norm.txt"
    test_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/test_norm.txt"

    target_norm_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/norm_qm.hdf5"

    post_transform = T.Compose([
            # T.RandomJitter(0.5),
            T.RandomFlip(args.axis,args.flip),
            T.RandomScale((args.scale, args.scale))
        ])
    print(post_transform)
    train_dataset = MolH5Dataset(h5_file, train_idx, target=args.target_name, transform=GNNTransformSMP(args.target_name), post_transform=post_transform, sett="train", target_norm_file=target_norm_file)
    val_dataset = MolH5Dataset(h5_file, val_idx, target=args.target_name, transform=GNNTransformSMP(args.target_name), sett="val", target_norm_file=target_norm_file)
    test_dataset = MolH5Dataset(h5_file, test_idx, target=args.target_name, transform=GNNTransformSMP(args.target_name), sett="test", target_norm_file=target_norm_file)
        
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=48)
    val_dataloader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=16)
    test_dataloader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=16)

    print("data loaded")
    for data in train_dataloader:     
        print(data)   
        num_features = data.num_features
        break
        
    print("number gpus : ", torch.cuda.device_count())
    model = GNN_SMP(num_features, dim=args.hidden_dim)
    module = SMPLitModule(model)
    print("model",next(module.parameters()).is_cuda) 
    bar = TQDMProgressBar(refresh_rate = 10)
    logger = TensorBoardLogger(args.tensor_board, name="00")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", dirpath=args.model_ckpt, filename=args.train_name, verbose=True )
    # trainer = Trainer(devices=2, accelerator="gpu", strategy="ddp")
    trainer = Trainer(devices=1, accelerator="auto", callbacks=[checkpoint_callback, bar], max_epochs=args.num_epochs, profiler="simple",logger=logger, gradient_clip_val=1.5)
    
    print("world size : ", trainer.world_size)
    trainer.fit(module, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    train_file = os.path.join(log_dir, f'smp-rep{rep}.best.train.pt')
    val_file = os.path.join(log_dir, f'smp-rep{rep}.best.val.pt')
    test_file = os.path.join(log_dir, f'smp-rep{rep}.best.test.pt')

    trainer2 = Trainer(devices=1, accelerator="auto", resume_from_checkpoint=args.model_ckpt+"/"+args.train_name)
    
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=16)
    pred_train = trainer2.predict(module, dataloaders=train_dataloader)
    for i in range(len(pred_train)):
        print(f'\tTrain RMSE {math.sqrt(mean_squared_error(pred_train[i]["preds"], pred_train[i]["targets"]))}')
        torch.save({'targets':pred_train[i], 'predictions':pred_train[i]}, train_file)
    print("val prediction")

    pred_val = trainer2.predict(module, dataloaders=val_dataloader)

    print(f'\tVal RMSE {math.sqrt(mean_absolute_error(pred_val[0]["preds"], pred_val[0]["targets"]))}')
    torch.save({'targets':pred_val[0], 'predictions':pred_val[0]}, val_file)
    print("test prediction")

    pred_test = trainer2.predict(module, dataloaders=test_dataloader)
    for i in range(len(pred_test)):
        print(f'\tTest RMSE {math.sqrt(mean_squared_error(pred_test[i]["preds"], pred_test[i]["targets"]))}')
        torch.save({'targets':pred_test[i], 'predictions':pred_test[i]}, test_file)
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target_name', type=str, default="Ionization_Potential")
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default='/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/logs')
    parser.add_argument('--tensor_board', type=str, default="/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/tensor_board/")
    parser.add_argument('--model_ckpt', type=str, default="/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/model_ckpts/")
    parser.add_argument('--resume_path', type=str, default="")
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--flip', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--train_name', type=str, default="0")
    args = parser.parse_args()
    log_dir = args.log_dir
    target_name = args.target_name
  

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if log_dir is None:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.path.join('logs', now)
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_dir = os.path.join(log_dir, target_name, now)
        args.tensor_board = os.path.join(args.tensor_board, target_name, now)
        args.model_ckpt = os.path.join(args.model_ckpt, target_name, now)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    train(args, device, log_dir)
