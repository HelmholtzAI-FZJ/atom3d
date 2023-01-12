import argparse
import logging
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import GNN_LBA
from data import GNNTransformLBA
from atom3d.datasets import  ProtMolH5Dataset
from scipy.stats import spearmanr
from tqdm import tqdm

from module import LBA_MDLitModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error,mean_squared_error

from pytorch_lightning.callbacks import ModelSummary
import torch_geometric.transforms as T

def train_loop(model, loader, optimizer, device):
    model.train()

    loss_all = 0
    total = 0
    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        loss = F.mse_loss(output.float(), data.y.float())
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        optimizer.step()
    return np.sqrt(loss_all / total)


@torch.no_grad()
def test(model, loader, device):
    model.eval()

    loss_all = 0
    total = 0

    y_true = []
    y_pred = []

    for data in tqdm(loader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr.view(-1), data.batch)
        loss = F.mse_loss(output, data.y)
        loss_all += loss.item() * data.num_graphs
        total += data.num_graphs
        y_true.extend(data.y.tolist())
        y_pred.extend(output.tolist())

    r_p = np.corrcoef(y_true, y_pred)[0,1]
    r_s = spearmanr(y_true, y_pred)[0]

    return np.sqrt(loss_all / total), r_p, r_s, y_true, y_pred

# def plot_corr(y_true, y_pred, plot_dir):
#     plt.clf()
#     sns.scatterplot(y_true, y_pred)
#     plt.xlabel('Actual -log(K)')
#     plt.ylabel('Predicted -log(K)')
#     plt.savefig(plot_dir)

def save_weights(model, weight_dir):
    torch.save(model.state_dict(), weight_dir)

def train(args, log_dir, rep=None, test_mode=False):
    # logger = logging.getLogger('lba')
    # logger.basicConfig(filename=os.path.join(log_dir, f'train_{split}_cv{fold}.log'),level=logging.INFO)

    # if args.precomputed:
    #     train_dataset = PTGDataset(os.path.join(args.data_dir, 'train'))
    #     val_dataset = PTGDataset(os.path.join(args.data_dir, 'val'))
    #     test_dataset = PTGDataset(os.path.join(args.data_dir, 'test'))
    # else:
    #     transform=GNNTransformLBA()
    #     train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'), transform=transform)
    #     val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'), transform=transform)
    #     test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'), transform=transform)
    # path="/p/project/hai_drug_qm/atom3d/"
    # os.chdir(path)
    

    # useCA = True
    # usePocket = False
        
    transform=GNNTransformLBA(args.useCA)

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
    

    post_transform = T.Compose([
        # T.RandomJitter(0.5),
        T.RandomFlip(args.axis,args.flip),
        T.RandomScale((args.scale, args.scale))
    ])
   

    train_dataset = ProtMolH5Dataset(mdh5_file, train_idx, useCA=args.useCA, transform=transform, post_transform=post_transform)
    val_dataset = ProtMolH5Dataset(mdh5_file, val_idx, useCA=args.useCA, transform=transform)
    test_dataset = ProtMolH5Dataset(mdh5_file, test_idx, useCA=args.useCA, transform=transform)
    
    train_loader = DataLoader(train_dataset, 8, shuffle=True, num_workers=48)
    val_loader = DataLoader(val_dataset, 8, shuffle=False, num_workers=16)
    test_loader = DataLoader(test_dataset, 8, shuffle=False, num_workers=16)

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
    logger = TensorBoardLogger(f"{log_dir}/tensor_board")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", dirpath=f'{log_dir}/ckpts/', filename=f'gnn_{args.useCA}_{args.usePocket}', verbose=True, every_n_epochs=1 )
    #callbacks=[ModelSummary(max_depth=3), checkpoint_callback, bar]
    trainer = Trainer(devices="auto", accelerator="auto", max_epochs=5, logger=logger,  gradient_clip_val=0.5, callbacks=[checkpoint_callback, bar]) 
    
    # trainer = Trainer(devices="auto", accelerator="auto", strategy="horovod", callbacks=[checkpoint_callback, bar], max_epochs=11, logger=logger,  gradient_clip_val=1.5, profiler="simple", resume_from_checkpoint="/p/project/hai_drug_qm/atom3d/examples/lba_md/cnn3d/model_ckpts/23-10-22/epochs.ckpt") 
    # print("world size : ", trainer.world_size)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    print("start predicting ... ")
    # model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
    #trainer = Trainer(devices=1, accelerator="auto", resume_from_checkpoint="/p/project/hai_drug_qm/atom3d/examples/lba_md/cnn3d/model_ckpts/23-10-22/epochs.ckpt") 
    trainer = Trainer(devices="auto", accelerator="auto") 

    predictions = trainer.predict(module, dataloaders=test_loader, ckpt_path=f"{log_dir}/gnn_{args.useCA}_{args.usePocket}.ckpt")
    # print("predictions len", len(predictions))
    # pred_train.to_pickle(os.path.join(args.output_dir, 'test_results.pkl'))
        # model.load_state_dict(torch.load(os.path.join(args.output_dir, f'best_weights.pt')))
    
    # print("predictions[0] ", predictions[0])

    # print("predictions ", predictions)


    with open(test_idx) as file:
        pdb_ids = [line.rstrip() for line in file]

    # print(f"Pdb ids {pdb_ids}")
    counter = 0
    frame = 0
    pdb_id = pdb_ids[0]
    test_results = []
    for prediction in predictions:
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

    mae = mean_squared_error(df['prediction'], df['target'], squared=False)
    print(f"MAE of Predictions and targets {mae}")
    df.to_csv(f"{log_dir}/gnn_out.csv",index=False)


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
    args = parser.parse_args()

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #log_dir = args.log_dir

    # PATH_TO_DATA = "/p/project/hai_drug_qm/Dataset/paris/DB/qm.hdf5"
    # train_dataset = ProtMolH5Dataset(PATH_TO_DATA, transform=None)
    # dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # for batch in dataloader:
    #     in_dim = batch.num_features
    #     break

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join('/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/model_ckpts/', now)
        
    if not os.path.exists(log_dir):
        try : 
            os.makedirs(log_dir)
        except : 
            print('problem with log directory')
    
    print(args)
    train(args, log_dir)