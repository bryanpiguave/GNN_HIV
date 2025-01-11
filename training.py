import torch
import os 
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np 
from tqdm import tqdm
try:
    import wandb
except ImportError:
    wandb = None
from src.dataset import MoleculeDataset
from src.model import GNN

# Log wandb 
if wandb is not None:
    wandb.init(project='GNN_HIV', tags=['GNN'])


BEST_PARAMETERS = {
    "batch_size": 128,
    "learning_rate": 0.01,
    "weight_decay": [0.0001],
    "sgd_momentum": [0.8],
    "scheduler_gamma": [0.8],
    "pos_weight": [1.3],
    "embedding_size": [64],
    "attention_heads": [3],
    "n_layers": [4],
    "dropout_rate": [0.2],
    "top_k_ratio": [0.5],
    "top_k_every_n": [1],
    "dense_neurons": [256]
}


def calculate_metrics(preds, labels, epoch, mode):
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    roc_auc = roc_auc_score(labels, preds)
    print(f"Epoch {epoch} {mode} F1: {f1} Acc: {acc} Prec: {prec} Recall: {recall} ROC_AUC: {roc_auc}")
    if wandb is not None:
        wandb.log({f"{mode}_f1": f1, f"{mode}_acc": acc, f"{mode}_prec": prec, f"{mode}_recall": recall, f"{mode}_roc_auc": roc_auc})
    return f1, acc, prec, recall, roc_auc



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred = model(batch.x.float(), 
                                batch.edge_attr.float(),
                                batch.edge_index, 
                                batch.batch) 
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()  
        optimizer.step()  
        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss/step        

def test_one_epoch(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(test_loader)):
        batch.to(device)
        pred = model(batch.x.float(), 
                                batch.edge_attr.float(),
                                batch.edge_index, 
                                batch.batch) 
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return running_loss/step
def main():
    #Load the dataset
    train_dataset = MoleculeDataset(root='data', filename='HIV_train_oversampled.csv')
    edge_dim = train_dataset[0].edge_attr.shape[1]
    BEST_PARAMETERS['edge_dim'] = edge_dim
    test_dataset = MoleculeDataset(root='data', filename='HIV_test.csv', test=True)
    #Create the dataloader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    #Create the model
    model = GNN(feature_size = 75, 
                model_params = BEST_PARAMETERS)
    model.to(device)
    print(f"Model has {count_parameters(model)} parameters")
    #Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #Create the loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    #Train the model
    for epoch in range(10):
        model.train()
        train_loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
        #Log train loss
        if wandb is not None:
            wandb.log({'train_loss': train_loss})
        model.eval()
        test_loss = test_one_epoch(epoch, model, test_loader, loss_fn)
        #Log test loss
        if wandb is not None:
            wandb.log({'test_loss': test_loss})
        
        #Save the model every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join('models', f'gnn_{epoch}.pt'))

    return

if __name__ == '__main__':
    main() 