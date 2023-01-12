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

from atom3d.datasets import LMDBDataset, ProtMolH5Dataset, ProtH5Dataset
from scipy.stats import spearmanr, pearsonr

from pytorch_lightning.callbacks import ModelSummary

from model import GNN_LBA, GNN_LBA_SoftHard
from data import GNNTransformLBA, GNNTransformLBA_SoftHard

from module import LBA_MDLitModule
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_error
import torch_geometric.transforms as T
# Construct model




def train(args):
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join('/p/project/hai_drug_qm/atom3d/examples/lba_md/gnn/model_ckpts/', now)
    #log_dir= '/p/project/hai_drug_qm/atom3d/examples/lba_md/cnn3d/'  
    print(log_dir) 
    if not os.path.exists(log_dir):
        try : 
            os.makedirs(log_dir)
        except : 
            print('problem with log directory')


    np.random.seed(5555)
    torch.manual_seed(5555)

   # train_dataset = LMDBDataset(os.path.join(args.data_dir, 'train'),
   #                             transform=CNN3D_TransformLBA(random_seed=args.random_seed))
   # val_dataset = LMDBDataset(os.path.join(args.data_dir, 'val'),
   #                           transform=CNN3D_TransformLBA(random_seed=args.random_seed))
   # test_dataset = LMDBDataset(os.path.join(args.data_dir, 'test'),
   #                            transform=CNN3D_TransformLBA(random_seed=args.random_seed))

    #train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    if args.useSoftHard:
        transform=GNNTransformLBA_SoftHard(useCA=args.useCA)
    else:
        transform=GNNTransformLBA(useCA=args.useCA)

    #qmh5_file = "/p/project/hai_drug_qm/atom3d/examples/smp_mol/data/qm/aeneas/h5/qm.hdf5"
    #mdh5_file = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/MD_dataset_mapped.hdf5"
    if args.useSoftHard:
        #mdh5_file = '/p/project/hai_drug_qm/atom3d/examples/lba_md/data/h5/MD_dataset_soft_hard_pocket_noH.hdf5'
        mdh5_file = '/p/project/hai_drug_qm/atom3d/examples/lba_md/data/h5/MD_dataset_soft_hard_noH.hdf5'
        train_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/train_soft_hard.txt"
        val_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/val_soft_hard.txt"
        test_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/test_soft_hard.txt"
        
        #train_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/small.txt"
        #val_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/small.txt"
        #test_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/small.txt"

    elif args.useCA:
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
    
    print(mdh5_file)
    print(train_idx)
    #train_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/pocket_train.txt"
    #val_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/pocket_val.txt"
    #test_idx = "/p/project/hai_drug_qm/atom3d/examples/lba_md/data/new_split/pocket_test.txt"

    post_transform = T.Compose([
        T.RandomTranslate(0.1)
        # T.RandomFlip(args.axis,args.flip),
        # T.RandomScale((args.scale, args.scale))
    ])
    print("with transform")
    print(args.useCA)
    if args.useSoftHard:
        
        train_dataset = ProtH5Dataset(mdh5_file, train_idx, useCA=args.useCA, transform=transform, post_transform=None)
        val_dataset = ProtH5Dataset(mdh5_file, val_idx, useCA=args.useCA, transform=transform)
        test_dataset = ProtH5Dataset(mdh5_file, test_idx, useCA=args.useCA, transform=transform)
    else:
        train_dataset = ProtMolH5Dataset(mdh5_file, train_idx, useCA=args.useCA, transform=transform, post_transform=None)
        val_dataset = ProtMolH5Dataset(mdh5_file, val_idx, useCA=args.useCA, transform=transform)
        test_dataset = ProtMolH5Dataset(mdh5_file, test_idx, useCA=args.useCA, transform=transform)
    
    train_loader = DataLoader(train_dataset, 16, shuffle=True, num_workers=48)
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

    if args.useSoftHard:
        model = GNN_LBA_SoftHard(num_features, hidden_dim=args.hidden_dim)
        module = LBA_MDLitModule(model)
    else:
        model = GNN_LBA(num_features, hidden_dim=args.hidden_dim)
        module = LBA_MDLitModule(model)
    
    
    bar = TQDMProgressBar(refresh_rate = 50)
    logger = TensorBoardLogger(f"{log_dir}/tensor_board")

    print(f"Log dir for trainer{log_dir}")
    checkpoint_callback = ModelCheckpoint(monitor="val/loss", dirpath=f'{log_dir}/model_ckpts/', filename=f'gnn_CA_{args.useCA}_Pocket_{args.usePocket}_SoftHard_{args.useSoftHard}', verbose=True, every_n_epochs=1 )
    trainer = Trainer(devices="auto", accelerator="auto", strategy="ddp", callbacks=[ModelSummary(max_depth=3), checkpoint_callback, bar], max_epochs=args.num_epochs, logger=logger,  gradient_clip_val=0.5) 
    
    # trainer = Trainer(devices="auto", accelerator="auto", strategy="horovod", callbacks=[checkpoint_callback, bar], max_epochs=11, logger=logger,  gradient_clip_val=1.5, profiler="simple", resume_from_checkpoint="/p/project/hai_drug_qm/atom3d/examples/lba_md/cnn3d/model_ckpts/23-10-22/epochs.ckpt") 
    # print("world size : ", trainer.world_size)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', action='store_true')
    parser.add_argument('--usePocket', type=bool, default=False)
    parser.add_argument('--useCA', type=bool, default=False)
    parser.add_argument('--useSoftHard', type=bool, default=True)
    parser.add_argument('--axis', type=int, default=0)
    parser.add_argument('--flip', type=float, default=0.5)
    parser.add_argument('--scale', type=float, default=0.5)
    args = parser.parse_args()
    print(args)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #log_dir = args.log_dir

    # PATH_TO_DATA = "/p/project/hai_drug_qm/Dataset/paris/DB/qm.hdf5"
    # train_dataset = ProtMolH5Dataset(PATH_TO_DATA, transform=None)
    # dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # for batch in dataloader:
    #     in_dim = batch.num_features
    #     break
    
    train(args)
