import numpy as np
import torch
import os, time
import os.path as osp
import argparse
import random

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False

def parse_args():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    parser = argparse.ArgumentParser(description='Node CLF')

    parser.add_argument('--mode', type=str,default='dynamic', help='dynamic or static')

    # training
    parser.add_argument('--lr', type=float, default=5e-3,help='[5e-3, 1e-3, 1e-3]')
    parser.add_argument('--re_lr', type=float, default=5e-3,help='[5e-3, 1e-3, 1e-3]')
    parser.add_argument('--epochs', type=int, default=400, help='[400, 300, 300]')
    parser.add_argument('--repochs', type=int, default=300)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--K', type=int, default=256, help='[256 ,256, 512]')
    parser.add_argument('--commitment_weight', type=float, default=0.1, help='[0.1, 0.2, 0.2] ')
    parser.add_argument('--alpha', type=float, default=0.1, help='[0.1, 0.5, 0.5]')
    parser.add_argument('--gamma', type=float, default=0.2, help="[0.2, 0.5, 0.75]")
    parser.add_argument('--beta', type=float, default=1.5, help= "[1.5, 1.5, 1.25]")

    parser.add_argument('--variant', type=str2bool, default=True, help='set to use variant')

    parser.add_argument('--hidden_channels', type=int, default=32, help= "[32, 128, 64")
    parser.add_argument('--path_emb', type=int, default=4)

    parser.add_argument('--datasets', type=str, default='CCTFD', help='datasets')
    parser.add_argument('--device', type=str, default=device)
    parser.add_argument('--dropout', type=float, default=0.2, help='[0.2,0.4,0.3]')
    parser.add_argument('--class_num', type=int, default=2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=321)


    args = parser.parse_args()

    return args