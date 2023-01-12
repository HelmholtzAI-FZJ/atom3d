import numpy as np
import os
import sys
from tqdm import tqdm
import torch
from atom3d.util.transforms import prot_graph_transform, mol_graph_transform
from atom3d.datasets import LMDBDataset
from torch_geometric.data import Data, Dataset, DataLoader
import atom3d.util.graph as gr

    
class GNNTransformLBA(object):
    def __init__(self, useCA=False):
        self.useCA = useCA
    
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        try:

            item = prot_graph_transform(item, atom_keys=['atoms_protein'], label_key='scores', useCA=self.useCA)
            #node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(item['atoms_protein'], allowable_feats=item['allowable_atoms']) 
            #item["atoms_protein"] = Data(node_feats, edge_index, edge_feats, y=item["scores"], pos=pos)
            # transform ligand into PTG graph
            item = mol_graph_transform(item, atom_key='atoms_ligand', label_key='scores', use_bonds=False, onehot_edges=False)
            node_feats, edges, edge_feats, node_pos = gr.combine_graphs(item['atoms_protein'], item['atoms_ligand'], edges_between=True)
            combined_graph = Data(node_feats, edges, edge_feats, y=item['scores'], pos=node_pos)
            return combined_graph
        except:
            print(f"Problem with PDB Id is {item['id']}")
        
class GNNTransformLBA_SoftHard(object):
    def __init__(self, useCA=False):
        self.useCA = useCA
    def __call__(self, item):
        # transform protein and/or pocket to PTG graphs
        try:

            item = prot_graph_transform(item, atom_keys=['atoms_protein'], label_key='scores', useCA=self.useCA)
            #node_feats, edge_index, edge_feats, pos = gr.prot_df_to_graph(item['atoms_protein'], allowable_feats=item['allowable_atoms']) 
            #item["atoms_protein"] = Data(node_feats, edge_index, edge_feats, y=item["scores"], pos=pos)
            # transform ligand into PTG graph
            #item = mol_graph_transform(item, atom_key='atoms_ligand', label_key='scores', use_bonds=False, onehot_edges=False)
            #print(item['scores'].shape)
            return item['atoms_protein']
        except:
            print(f"Problem with PDB Id is {item['id']}")   

        
if __name__=="__main__":
    seqid = sys.argv[1]
    save_dir = '/scratch/users/aderry/atom3d/lba_ptg_' + str(seqid)
    data_dir = f'/scratch/users/raphtown/atom3d_mirror/lmdb/LBA/splits/split-by-sequence-identity-{seqid}/data'
    # data_dir = '/scratch/users/aderry/atom3d/lba_30_withH/split/data'
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)
    train_dataset = LMDBDataset(os.path.join(data_dir, 'train'), transform=GNNTransformLBA())
    val_dataset = LMDBDataset(os.path.join(data_dir, 'val'), transform=GNNTransformLBA())
    test_dataset = LMDBDataset(os.path.join(data_dir, 'test'), transform=GNNTransformLBA())
    
    print('processing train dataset...')
    for i, item in enumerate(tqdm(train_dataset)):
        torch.save(item, os.path.join(save_dir, 'train', f'data_{i}.pt'))
    
    print('processing validation dataset...')
    for i, item in enumerate(tqdm(val_dataset)):
        torch.save(item, os.path.join(save_dir, 'val', f'data_{i}.pt'))
    
    print('processing test dataset...')
    for i, item in enumerate(tqdm(test_dataset)):
        torch.save(item, os.path.join(save_dir, 'test', f'data_{i}.pt'))
