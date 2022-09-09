#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: baseline.py
@time: 2022/1/16 2:18
@desc:
'''

import torch
from torch.nn import Linear, BatchNorm1d, Dropout

import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool

pooling_dict = {'sum': global_add_pool,
                'mean': global_mean_pool,
                'max': global_max_pool}


class I2BGNN(torch.nn.Module):
    '''
    gcn model, in which the messages are aggregated with the edge weights.
    '''

    def __init__(self, in_channels, dim, out_channels, num_layers, pooling, BN=True, dropout=0.2, which_edge_weight=None):
        super().__init__()
        self.num_layers = num_layers
        self.pooling = pooling
        self.BN = BN
        self.dropout = dropout
        self.which_edge_weight = which_edge_weight

        self.gcs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.drops = torch.nn.ModuleList()
        for i in range(num_layers):
            if i:
                gc = GCNConv(dim, dim)
            else:
                gc = GCNConv(in_channels, dim)
            bn = BatchNorm1d(dim)
            drop = Dropout(p=dropout)

            self.gcs.append(gc)
            self.bns.append(bn)
            self.drops.append(drop)

        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, out_channels)

    def forward(self, x, edge_index, batch, edge_attr):
        if self.which_edge_weight == 'Volume':
            edge_weight = edge_attr[:, 0]
        else:
            edge_weight = edge_attr[:, 1]

        for i in range(self.num_layers):
            x = F.relu(self.gcs[i](x, edge_index, edge_weight))
            if self.BN: x = self.bns[i](x)
            if self.dropout: x = self.drops[i](x)

        x = pooling_dict[self.pooling](x, batch)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)




