#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: main.py
@time: 2022/1/15 15:51
@desc:
'''
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from utils.parameters import get_parser
from utils.dataset import MyBlockChain_TUDataset
from utils.transform import *
from utils.tools import setup_seed, EarlyStopping, data_split

from model.i2bgnn import I2BGNN


# data information
label_abbreviation = {"p": "phish-hack"}

args = get_parser()
setup_seed(args.seed)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

label = label_abbreviation[args.label]  # target account label

# path
data_path = osp.join(osp.dirname(osp.realpath(__file__)), f'{args.root}',
                     '{}/{}/{}hop-{}/{}'.format(args.dataType, label, args.hop, args.topk, args.edge_sample_strategy))


# load dataset
transform = T.Compose([
    MyToUndirected(edge_attr_keys=['edge_attr']) if args.to_undirected else MyAug_Identity(),
    ColumnNormalizeFeatures(['edge_attr']),
    T.NormalizeFeatures()
])

dataset = MyBlockChain_TUDataset(root=data_path, name=args.dataType.upper() + 'G',
                                 use_node_attr=True, use_node_importance=False, use_edge_attr=True,  # feature selection
                                 transform=transform)  # .shuffle()




print('################# dataset information #########################')
print('Type of target accounts:    {}'.format(label))
print('Num. of graphs:    {}'.format(len(dataset)))
print('Ave. num. of nodes:    {}'.format(np.mean([g.x.size(0) for g in dataset])))
print('Ave. num. of edges:    {}'.format(np.mean([g.edge_index.size(1) for g in dataset])))
print('Num. of node features:    {}'.format(dataset.num_node_features))
print('Num. of edge features:    {}'.format(dataset.num_edge_features))
print('Use node attribute:    {}'.format(bool(args.use_node_attribute)))
print('Use node labeling:    {}'.format(bool(args.use_node_labeling)))
print('Use edge attribute:    {}'.format(bool(args.use_edge_attribute)))

train_splits, val_splits, test_splits = data_split(X=np.arange(len(dataset)),
                                                   Y=np.array([dataset[i].y.item() for i in range(len(dataset))]),  # 等官方修复bug
                                                   # Y=np.array([dataset[i].y[0].item() for i in range(len(dataset))]),
                                                   seeds=args.seeds[:args.exp_num], K=args.k_ford)


def train(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct, total_loss = 0, 0
    y_pred_label_list = []
    y_true_label_list = []

    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch, data.edge_attr)
        loss = F.nll_loss(out, data.y)
        total_loss += float(loss) * data.num_graphs

        pred = out.argmax(-1)

        y_pred_label_list.append(pred)
        y_true_label_list.append(data.y)

    y_pred = torch.cat(y_pred_label_list).cpu().detach().numpy()
    y_true = torch.cat(y_true_label_list).cpu().detach().numpy()

    acc = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    return acc, total_loss / len(loader.dataset)


# exp
f1_list = []

i = 0
train_dataset = dataset[train_splits[i]]
val_dataset = dataset[val_splits[i]]
test_dataset = dataset[test_splits[i]]

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)



model = I2BGNN(dataset.num_features, args.hidden_dim, num_layers=args.num_layers, out_channels=dataset.num_classes, pooling=args.pooling,
               which_edge_weight=args.baseline_use_which_edge_weight).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
early_stopping = EarlyStopping(patience=20)

for epoch in range(1, args.epochs):
    loss = train(train_loader)
    train_acc, _ = test(train_loader)
    val_acc, val_loss = test(val_loader)
    test_acc, _ = test(test_loader)

    if args.early_stop:
        early_stopping(val_loss, results=[epoch, loss, val_loss, train_acc, val_acc, test_acc])
        if early_stopping.early_stop:
            print('\n=====final results=====')
            _epoch, _loss, _val_loss, _train_acc, _val_acc, _test_acc = early_stopping.best_results
            f1_list.append(_test_acc)
            print(f'Exp: {i},  Epoch: {_epoch:03d},     '
                  f'Train_Loss: {_loss:.4f}, Val_Loss: {_val_loss:.4f},        '
                  f'Train_Acc: {_train_acc:.4f}, Val_Acc: {_val_acc:.4f},        '
                  f'Test_Acc: {_test_acc:.4f}\n\n')
            break
    else:
        f1_list.append(test_acc)

    print(f'Exp: {i},  Epoch: {epoch:03d},     '
          f'Train_Loss: {loss:.4f}, Val_Loss: {val_loss:.4f},      '
          f'Train_Acc: {train_acc:.4f}, Val_Acc: {val_acc:.4f},      '
          f'Test_Acc: {test_acc:.4f}')

print('Num. of experiments: {}\n'.format(len(train_splits)),
      'Result in terms of f1-score: {} ~ {}\n\n\n'.format(np.mean(f1_list), np.std(f1_list)))

