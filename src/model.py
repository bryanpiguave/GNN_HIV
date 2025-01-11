import torch 
import torch.nn as nn
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch.nn import Linear
import torch.nn.functional as F
torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self,feature_size,model_params):
        super(GNN, self).__init__()
        embedding_size = model_params['embedding_size']
        n_heads = model_params['attention_heads']
        self.n_layers = model_params['n_layers']
        self.dropout = model_params['dropout_rate']
        self.top_k_ratio = model_params['top_k_ratio']
        self.top_k_every_n = model_params['top_k_every_n']
        edge_dim = model_params['edge_dim']
        dense_neurons = model_params["dense_neurons"]

        self.conv_layers = nn.ModuleList([])
        self.transformer_layers = nn.ModuleList([])
        self.pooling_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        # Transformation layers
        self.conv1 = TransformerConv(feature_size, 
                                    embedding_size, 
                                    heads=n_heads, dropout= self.dropout)
        self.transf1 = Linear(embedding_size* n_heads, embedding_size)
        self.bn1 = nn.BatchNorm1d(embedding_size)

        # Other layers
        for i in range(1, self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size, 
                                                    embedding_size, 
                                                    heads=n_heads, dropout= self.dropout))
            self.transformer_layers.append(nn.Linear(embedding_size* n_heads, embedding_size))
            self.bn_layers.append(nn.BatchNorm1d(embedding_size))
        
        # Linear layers
        self.linear1 = Linear(embedding_size*2, dense_neurons)
        self.linear2 = Linear(dense_neurons, int(dense_neurons/2))  
        self.linear3 = Linear(int(dense_neurons/2), 1)  

    def forward(self, x, edge_attr, edge_index, batch_index):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(self.transf1(x))
        x = self.bn1(x)

        #Hold the intermediate embeddings
        global_representation = []
        for i in range(1, self.n_layers):
            x = self.conv_layers[i-1](x, edge_index, edge_attr)
            x = F.relu(self.transformer_layers[i-1](x))
            x = self.bn_layers[i-1](x)
            if i % self.top_k_every_n == 0:
                x, edge_index, edge_attr, batch_index, _ = TopKPooling(ratio=self.top_k_ratio)(x, edge_index, edge_attr, batch_index)
                global_representation.append(torch.cat([global_mean_pool(x, batch_index), global_max_pool(x, batch_index)], dim=1))
        x = torch.sum(global_representation)
        
        #Output block 
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear3(x)
        return x

