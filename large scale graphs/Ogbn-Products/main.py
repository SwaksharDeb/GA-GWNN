# Reaches around 0.7945 ± 0.0059 test accuracy.

import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import Linear as Lin
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import NeighborSampler

#from torch_geometric.nn import GATConv
from gat_layer import GATConv


import argparse
parser = argparse.ArgumentParser(description='OGBN-Products (SAGE)')
parser.add_argument('--epochs', type=int, default=200)

parser.add_argument('--test-freq', type=int, default=10)


args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#root = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'products')
dataset = PygNodePropPredDataset('ogbn-products', root='dataset')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0].to(device, 'x', 'y')

train_loader = NeighborLoader(
    data,
    input_nodes=split_idx['train'],
    num_neighbors=[10, 10, 10],
    batch_size=512,
    shuffle=True,
    num_workers=12,
    persistent_workers=True,
)
subgraph_loader = NeighborLoader(
    data,
    input_nodes=None,
    num_neighbors=[-1],
    batch_size=1024,
    num_workers=12,
    shuffle=False,
    persistent_workers=True,
)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_features, hidden_channels,
                                  heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(dataset.num_features, hidden_channels * heads))
        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for skip in self.skips:
            skip.reset_parameters()

    def forward(self, x, edge_index):
        h0 = None
        for i, (conv, skip) in enumerate(zip(self.convs, self.skips)):
            x = conv(x, edge_index, h0) + skip(x)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        h0 = None
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index,h0) + self.skips[i](x)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = F.elu(x)
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            #x = F.dropout(x, p=0.5, training=False)
            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


model = GAT(dataset.num_features, 128, dataset.num_classes, num_layers=3,
            heads=1).to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=split_idx['train'].size(0))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch in train_loader:
        out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        y = batch.y[:batch.batch_size].squeeze()
        #loss = F.cross_entropy(out, y)
        loss = F.nll_loss(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())
        pbar.update(batch.batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / split_idx['train'].size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(data.x)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    
    # import numpy as np
    # #total_correct = int(out.argmax(dim=-1).eq(y_true).sum())
    # acc_list = []
    # for i in range(y_true.shape[1]):
    #     is_labeled = y_true[:,i] == y_true[:,i]
    #     correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
    #     acc_list.append(float(np.sum(correct))/len(correct))

    
    return train_acc, val_acc, test_acc


test_accs = []
for run in range(1):
    np.random.seed(2020)
    torch.manual_seed(2020)
    print(f'\nRun {run:02d}:\n')

    model.reset_parameters()
    criterion = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = final_test_acc = 0
    for epoch in range(1, args.epochs+1):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Train Loss: {loss:.4f}, Train acc: {acc:.4f}')

        if epoch % 10 == 0:
        #if epoch > args.epochs/2 and epoch % args.test_freq == 0 or epoch == args.epochs:
            train_acc, val_acc, test_acc = test()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
            print(f'Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, '
                  f'Test acc: {test_acc:.4f}', 
                  f'Best Test acc: {final_test_acc:.4f}')
    test_accs.append(final_test_acc)

test_acc = torch.tensor(test_accs)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
