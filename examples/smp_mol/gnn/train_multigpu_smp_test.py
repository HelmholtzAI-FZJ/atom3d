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
import h5py
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


def train(args, ckpt_path):

    
    h5_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/qm.h5"
    norm_h5_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/norm_qm.hdf5"


    target_norm = h5py.File(norm_h5_file, 'r') 
        
    target = args.target_name

    mean_data = target_norm['train'][target]['mean'][()]
    std_data = target_norm['train'][target]['std'][()]

    train_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/train_norm.txt"
    val_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/val_norm.txt"
    test_idx = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/mol_split/test_norm.txt"

    train_dataset = MolH5Dataset(h5_file, train_idx, target=args.target_name, target_norm_file=norm_h5_file, transform=GNNTransformSMP(args.target_name))
    val_dataset = MolH5Dataset(h5_file, val_idx, target=args.target_name, target_norm_file=norm_h5_file, transform=GNNTransformSMP(args.target_name))
    test_dataset = MolH5Dataset(h5_file, test_idx, target=args.target_name, target_norm_file=norm_h5_file, transform=GNNTransformSMP(args.target_name))
        
    train_loader = DataLoader(train_dataset, 64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, 64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, 64, shuffle=False, num_workers=0)

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
    print("start predicting ... ")
    # model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
    #trainer = Trainer(devices=1, accelerator="auto", resume_from_checkpoint="/p/project/hai_drug_qm/atom3d/examples/lba_md/cnn3d/model_ckpts/23-10-22/epochs.ckpt") 
    trainer = Trainer(devices=1, accelerator="auto") 
    predictions = trainer.predict(module, dataloaders=test_loader, ckpt_path=ckpt_path)
    print("predictions len", len(predictions))
    # pred_train.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))
        # model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
    
    print("predictions[0] ", predictions[0])

    print("predictions ", predictions)


    with open(test_idx) as file:
        pdb_ids = [line.rstrip() for line in file]

    print(f"Pdb ids {pdb_ids}")
    counter = 0
    pdb_id = pdb_ids[0]
    test_results = []
    for prediction in predictions:
        print(f"Counter {counter}")
        preds = prediction['preds']
        targets = prediction['targets']
        for pred, target in zip(preds, targets):
            
            pdb_id = pdb_ids[counter]
            temp_data = [pdb_id,(pred.item()*std_data)+mean_data, (target.item()*std_data)+mean_data]
            counter += 1
            test_results.append(temp_data)

    # Create the pandas DataFrame
    df = pd.DataFrame(test_results, columns=['pdb_id', 'prediction', 'target'])
    
    print(f"Test Predictions Shape {df.shape}")
    
    corr=df[['prediction', 'target']].corr(method='pearson')
    print(f"Pearson Correlation of Predictions and targets {corr}")

    corr=df[['prediction', 'target']].corr(method='spearman')
    print(f"Spearman Correlation of Predictions and targets {corr}")

    mae = mean_absolute_error(df['prediction'], df['target'])
    print(f"MAE of Predictions and targets {mae}")
    df.to_csv(f"/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/predictions/{args.target_name}.txt",index=False)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--target_name', type=str, default="Ionization_Potential")
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
    ckpt_path = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/gnn/2022-12-08-11-55-28/model_ckpts/gnn_Ionization_Potential.ckpt"
    train(args, ckpt_path)
