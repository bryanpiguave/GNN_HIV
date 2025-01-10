import pandas as pd 
# Add project directory to path
import pathlib
import sys
import rdkit 
import torch 
import torch_geometric
from rdkit import Chem 
from rdkit.Chem import Draw
import yaml
from torch_geometric.data import Data, Dataset
import numpy as np
from rdkit.Chem import rdmolops
from tqdm import tqdm 
import os 
sys.path.append('../')
main_path = pathlib.Path(__file__).parent.parent

with open(os.path.join(main_path,'config\dataset.yaml'), 'r') as file:
    config = yaml.safe_load(file)


class MoleculeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        """ root is where the dataset will be stored"""
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)   

        @property
        def raw_file_names(self):
            return 'HIV.csv'
        @property
        def processed_file_names(self):
            return 'not_implemented.pt'

        def process(self):
            self.data = pd.read_csv(self.raw_paths[0])
            for index,mol in tqdm(self.data.iterrows(), total= self.data.shape[0]):
                mol_obj = Chem.MolFromSmiles(mol['smiles'])
                node_features = self.get_node_features(mol_obj)
                edge_features = self.get_edge_features(mol_obj)
                edge_index = self.get_adjacency_info(mol_obj)
                #Get labels info 
                label= self._get_label(mol['HIV_active'])
                # Create data object 

                data = Data(x= node_features,
                            edge_index = edge_index,
                            edge_attr = edge_features,
                            y = label,
                            smiles = mol['smiles'])
                torch.save(data,
                           os.path.join(self.processed_dir, f'data_{index}.pt'))

        def download(self):
            """This is to download the dataset, but this is not implemented in this project"""
            pass
        

        def get_node_features(self, mol: Chem.Mol):
            all_node_features = []
            for atom in mol.GetAtoms():
                node_features = []
                node_features.append(atom.GetAtomicNum())
                node_features.append(atom.GetDegree())
                node_features.append(atom.GetFormalCharge())
                node_features.append(atom.GetHybridization())
                node_features.append(atom.GetIsAromatic())
                all_node_features.append(node_features)
            all_node_features = np.asarray(all_node_features, dtype=np.float32)
            return torch.tensor(all_node_features)

        def get_edge_features(self, mol: Chem.Mol):
            all_edge_features = []
            for bond in mol.GetBonds():
                edge_features = []
                edge_features.append(bond.GetBondTypeAsDouble())
                edge_features.append(bond.GetIsConjugated())
                all_edge_features.append(edge_features)
            all_edge_features = np.asarray(all_edge_features, dtype=np.float32)
            return torch.tensor(all_edge_features)
        
        def get_adjacency_info(self, mol: Chem.Mol):
            adj_matrix = rdmolops.GetAdjacencyMatrix(mol)
            row,col = np.where(adj_matrix == 1)
            coo = np.array([row, col])
            coo = np.reshape(coo, (2, -1))
            return torch.tensor(coo, dtype=torch.long)


def main():
    # Load the dataset
    print(main_path)
    data = pd.read_csv(os.path.join(main_path,config['paths']['raw_dataset']))
    print(data.head())

    # Create a dataset object
    dataset = MoleculeDataset(root=r"C:\Users\USER\Documents\Repositories\HIV_Graph_Classification\data\raw")
    return
    
if __name__ == '__main__':
    main()