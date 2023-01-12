import argparse
import datetime
import json
import os
import time
import tqdm
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from atom3d.datasets import LMDBDataset, ProtMolH5Dataset
from scipy.stats import spearmanr, pearsonr

from pytorch_lightning.callbacks import ModelSummary

from model import GNN_LBA
from data import GNNTransformLBA

from module import LBA_MDLitModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error
# Construct model




def run_test(args, model_name, ckpt_path):

    print("Entered in New Test")
    useCA = True
    usePocket = False
    transform=GNNTransformLBA(useCA=useCA)

    #qmh5_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/qm.hdf5"
    #mdh5_file = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/MD_dataset_mapped.hdf5"
    if args.useCA:
        mdh5_file = '/p/project/hai_drug_qm/atom3d/examples/lba_md/data/MD_dataset_mapped_ca_Pres_Lat.hdf5'
        train_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/ca_train.txt"
        val_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/ca_val.txt"
        test_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/ca_test.txt"
    elif args.usePocket:
        mdh5_file = '/p/project/hai_drug_qm/atom3d/examples/lba_md/data/MD_dataset_mapped_protein_pocket.hdf5'
        train_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/pocket_train.txt"
        val_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/pocket_val.txt"
        test_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/pocket_test.txt"
    else:
        mdh5_file = '/p/project/hai_drug_qm/atom3d/examples/lba_md/data/MD_dataset_mapped.hdf5'
        train_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/train.txt"
        val_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/val.txt"
        test_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/test.txt"
    


    print(f"Train index {train_idx}")
    train_dataset = ProtMolH5Dataset(mdh5_file, train_idx, useCA=useCA, transform=transform)
    val_dataset = ProtMolH5Dataset(mdh5_file, val_idx, useCA=useCA, transform=transform)
    test_dataset = ProtMolH5Dataset(mdh5_file, test_idx, useCA=useCA, transform=transform)
    
    train_loader = DataLoader(train_dataset, 16, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, 16, shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, 16, shuffle=False, num_workers=16)

    print(f"length of train dataset  = {len(train_dataset)}")
    print(f"length of val dataset  = {len(val_dataset)}")
    print(f"length of test dataset  = {len(test_dataset)}")
    for data in train_loader:
        num_features = data.num_features
        print(data)
        break

    print("number of gpus ",torch.cuda.device_count())

    model = GNN_LBA(num_features, hidden_dim=args.hidden_dim)
    module = LBA_MDLitModule(model)
    bar = TQDMProgressBar(refresh_rate = 50)
    
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
    frame = 0
    pdb_id = pdb_ids[0]
    test_results = []
    for prediction in predictions:
        print(f"Counter {counter}")
        preds = prediction['preds']
        targets = prediction['targets']
        for pred, target in zip(preds, targets):
            
            if counter > 0 and counter % 100 ==0:
                frame = 0
                pdb_id = pdb_ids[counter//100]
            temp_data = [pdb_id, frame, pred.item(), target.item()]
            frame += 1
            counter += 1
            test_results.append(temp_data)

    # Create the pandas DataFrame
    df = pd.DataFrame(test_results, columns=['pdb_id', 'frame', 'prediction', 'target'])
    
    print(f"Test Predictions Shape {df.shape}")
    
    corr=df[['prediction', 'target']].corr(method='pearson')
    print(f"Pearson Correlation of Predictions and targets {corr}")

    corr=df[['prediction', 'target']].corr(method='spearman')
    print(f"Spearman Correlation of Predictions and targets {corr}")

    mae = mean_absolute_error(df['prediction'], df['target'])
    print(f"MAE of Predictions and targets {mae}")
    df.to_csv(f"/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/output/{model_name}.txt",index=False)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', action='store_true')
    parser.add_argument('--usePocket', type=bool, default=False)
    parser.add_argument('--useCA', type=bool, default=True)
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--flip', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=0.5)
    parser.add_argument('--ckpt_path', type=str)
    args = parser.parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #log_dir = args.log_dir

    # PATH_TO_DATA = "/p/project/hai_drug_qm/Dataset/paris/DB/qm.hdf5"
    # train_dataset = ProtMolH5Dataset(PATH_TO_DATA, transform=None)
    # dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # for batch in dataloader:
    #     in_dim = batch.num_features
    #     break
    model_name = 'gnn_CA_True_Pocket_False'
    args.ckpt_path = f'/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/model_ckpts/2022-12-06-16-39-25/model_ckpts/gnn_CA_True_Pocket_False.ckpt'
    
    run_test(args, model_name, args.ckpt_path)
