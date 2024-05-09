import argparse
import os
import torch

import random
import numpy as np
from exp.exp_pretrain import UP2ME_exp_pretrain
from utils.tools import string_split

parser = argparse.ArgumentParser(description='UP2ME pretraining')

parser.add_argument('--data_format', type=str, default='csv', help='data format')
parser.add_argument('--data_name', type=str, default='SMD', help='data name')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')  
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2',help='train/val/test split')
parser.add_argument('--valid_prop', type=float, default=0.2, help='proportion of validation set, for numpy data only')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location to store model checkpoints')

parser.add_argument('--position', type=str, default='abs', help='position embedding method')
parser.add_argument('--data_dim', type=int, default=7, help='Number of dimensions of the MTS data (D)')
parser.add_argument('--patch_size', type=int, default=12)
parser.add_argument('--min_patch_num', type=int, default=20, help='minimum number of patches in a sampled series')
parser.add_argument('--max_patch_num', type=int, default=200, help='maximum number of patches in a sampled series')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio of the patch')

parser.add_argument('--d_model', type=int, default=256, help='dimension of hidden states (d_model)')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of MLP in transformer')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers (N)')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers (N)')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')

parser.add_argument('--resample_patch_num', action='store_true', default=False, help='periodically resample the number of patches in a series, for large datasets')
parser.add_argument('--pool_size', type=int, default=10, help='size of the pool for resampling')
parser.add_argument('--resample_freq', type=int, default=5000, help='resampling frequency')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=256, help='batch size of train input data')
parser.add_argument('--train_steps', type=int, default=500000, help='train steps')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--valid_freq', type=int, default=5000, help='validating frequency')
parser.add_argument('--valid_sep_point', type=int, default=10, help='equally sample patch nums for validation')
parser.add_argument('--valid_batches', type=int, default=-1, help='validating batches, -1 means all')
parser.add_argument('--tolerance', type=int, default=10, help='tolerance for early stopping')

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

parser.add_argument('--label', type=str, default='Sliding-Window',help='labels to attach to setting')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    print(args.gpu)

args.data_split = string_split(args.data_split)

print('Args in experiment:')
print(args)

#fix random seed
torch.manual_seed(2023)
random.seed(2023)
np.random.seed(2023)
torch.cuda.manual_seed_all(2023)
torch.backends.cudnn.deterministic = True

exp = UP2ME_exp_pretrain(args)

for ii in range(args.itr):
    setting = 'U2M{}_data{}_dim{}_patch{}_minPatch{}_maxPatch{}_mask{}_dm{}_dff{}_heads{}_eLayer{}_dLayer{}_dropout{}'.format(args.label, args.data_name, 
                args.data_dim, args.patch_size, args.min_patch_num, args.max_patch_num, args.mask_ratio,
                args.d_model, args.d_ff, args.n_heads, args.e_layers, args.d_layers, args.dropout)
    
    print('>>>>>>>start pre-training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.pre_train(setting)