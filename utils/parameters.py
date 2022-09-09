#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author: jjzhou012
@contact: jjzhou012@163.com
@file: Parameters.py
@time: 2022/1/15 15:59
@desc:
'''

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Description: Script to run our model.')
    # dataset
    parser.add_argument('--dataType', '-dt', help='eth', default='eth')
    parser.add_argument('--label', '-l', help='', default='i')
    parser.add_argument('--root', help='data', default='data')

    # subgraph config
    parser.add_argument('--hop', type=int, help='order of neighbor nodes', default=2)
    parser.add_argument('--topk', type=int, help='order of neighbor nodes', default=20)
    parser.add_argument('-ess', '--edge_sample_strategy', type=str, help='Volume, Times, averVolume', default='Volume')

    parser.add_argument('--use_node_attribute', '-use_NA', type=int, help='', default=1)
    parser.add_argument('--use_node_labeling', '-use_NL', type=int, help='node labeling', default=0)
    parser.add_argument('--use_edge_attribute', '-use_EA', type=int, help='', default=1)

    parser.add_argument('--num_val', '-val', type=float, help='ratio of val', default=0.2)
    parser.add_argument('--num_test', '-test', type=float, help='ratio of test', default=0.2)
    parser.add_argument('--k_ford', '-KF', type=int, help='', default=3)


    ### transform
    parser.add_argument('--to_undirected', '-undir', type=int, help='', default=0)

    # model setting
    parser.add_argument('--model', type=str, help='', default='i2bgnn')
    parser.add_argument('--hidden_dim', type=int, help='', default=128)
    parser.add_argument('--num_layers', '-layer', type=int, help='', default=2)
    parser.add_argument('--pooling', type=str, help='mean, sum, max', default='max')

    # side information
    parser.add_argument('--use_node_label', '-NL', type=int, help='use node label information', default=0)
    parser.add_argument('-which_ew', '--baseline_use_which_edge_weight', type=str,
                        help='for I2BGNN, which edge weight to use, candidate are Volume, Times', default='Volume')

    # train
    parser.add_argument('--batch_size', '-bs', type=int, help='batch size', default=16)
    parser.add_argument('--epochs', type=int, help='', default=1000)
    parser.add_argument('--lr', type=float, help='Learning rate.', default=0.01)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.)
    parser.add_argument('--gpu', type=str, help='gpu id', default='0')

    # exp setting
    parser.add_argument('--early_stop', type=int, help='', default=1)

    parser.add_argument('--seed', type=int, help='random seed', default=12)
    parser.add_argument('--seeds', type=list, help='random seed', default=[12])
    parser.add_argument('--exp_num', type=int, help='', default=1)

    return parser.parse_args()
