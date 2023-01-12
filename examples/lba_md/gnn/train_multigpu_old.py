import argparse
import logging
import os
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
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

def train(args, device, log_dir, rep=None, test_mode=False):
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

    transform=GNNTransformLBA()

    qmh5_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/qm.hdf5"
    mdh5_file = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/MD_dataset_mapped.hdf5"
    residue_file = "/p/project/hai_supreme/Prep/MD_dataset_mapped_ca.hdf5"

    train_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/train.txt"
    val_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/val.txt"
    test_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/test.txt"


    train_dataset = ProtMolH5Dataset(qmh5_file, mdh5_file, residue_file, val_idx, transform=transform)
    val_dataset = ProtMolH5Dataset(qmh5_file, mdh5_file, residue_file, val_idx, transform=transform)
    test_dataset = ProtMolH5Dataset(qmh5_file, mdh5_file, residue_file, test_idx, transform=transform)

    train_loader = DataLoader(train_dataset, 10, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, 10, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, 10, shuffle=False, num_workers=8)

    print(f"data loading finished with length {len(train_dataset)}")
    for data in train_loader:
        num_features = data.num_features
        print(data)
        break
        
    print("number of gpus ",torch.cuda.device_count())
    model = GNN_LBA(num_features, hidden_dim=args.hidden_dim).to(device)
    module = LBA_MDLitModule(model)
    bar = TQDMProgressBar(refresh_rate = 10)
    logger = TensorBoardLogger("/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/tensor_board", name="23-10-22")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", dirpath='/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/model_ckpts/23-10-22/', filename='last', verbose=True, every_n_epochs=1)
    #trainer = Trainer(devices=2, accelerator="gpu", strategy="ddp")
    print("Start training ...")
    trainer = Trainer(devices="auto", accelerator="auto", strategy="horovod", callbacks=[checkpoint_callback, bar], max_epochs=4, num_sanity_val_steps=5, logger=logger, num_nodes=1, gradient_clip_val=1.5, profiler="simple", resume_from_checkpoint='/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/model_ckpts/23-10-22/last.ckpt') 
    # print("world size : ", trainer.world_size)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    train_file = os.path.join(log_dir, f'lba-rep{rep}.best.train.pt')
    val_file = os.path.join(log_dir, f'lba-rep{rep}.best.val.pt')
    test_file = os.path.join(log_dir, f'lba-rep{rep}.best.test.pt')
    
    # cpt = torch.load(os.path.join(log_dir, f'best_weights_rep{rep}.pt'))
    # model.load_state_dict(cpt['model_state_dict'])
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False, num_workers=4 )
    # _, _, _, y_true_train, y_pred_train = trainer.test(module, dataloaders=train_loader)
    # torch.save({'targets':y_true_train, 'predictions':y_pred_train}, train_file)
    
    # _, _, _, y_true_val, y_pred_val = trainer.predict(module, dataloaders=val_loader)
    # torch.save({'targets':y_true_val, 'predictions':y_pred_val}, val_file)
    y_pred_test = trainer.test(module, dataloaders=test_dloader)
    print(y_pred_test)
    # rmse, pearson, spearman, y_true_test, y_pred_test = trainer.predict(module, dataloaders=test_dloader)
    # print(f'\tTest RMSE {rmse}, Test Pearson {pearson}, Test Spearman {spearman}')
    # torch.save({'targets':y_true_test, 'predictions':y_pred_test}, test_file)



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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = args.log_dir

    # PATH_TO_DATA = "/p/project/hai_drug_qm/Dataset/paris/DB/qm.hdf5"
    # train_dataset = ProtMolH5Dataset(PATH_TO_DATA, transform=None)
    # dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # for batch in dataloader:
    #     in_dim = batch.num_features
    #     break

    if args.mode == 'train':
        if log_dir is None:
            now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_dir = os.path.join('logs', now)
        else:
            log_dir = os.path.join('logs', log_dir)
        if not os.path.exists(log_dir):
            try : 
                os.makedirs(log_dir)
            except : 
                print()
        train(args, device, log_dir)
        
    elif args.mode == 'test':
        for rep, seed in enumerate(np.random.randint(0, 1000, size=3)):
            print('seed:', seed)
            log_dir = os.path.join('logs', f'lba_test_withH_{args.seqid}')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train(args, device, log_dir, rep, test_mode=True)
