import argparse
import os
import torch
import json
import random
import numpy as np

from exp.exp_forecast import UP2ME_exp_forecast
from utils.tools import string_split

parser = argparse.ArgumentParser(description='UP2ME for forecasting')

parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--IR_mode', action='store_true', help='whether to use immediate reaction mode', default=False)

parser.add_argument('--data_format', type=str, default='csv', help='data format')
parser.add_argument('--data_name', type=str, default='SMD', help='data name')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--valid_prop', type=float, default=0.2, help='proportion of validation set, for numpy data only')
parser.add_argument('--data_split', type=str, default='0.7,0.1,0.2', help='train/val/test split, can be ratio or number')
parser.add_argument('--checkpoints', type=str, default='./checkpoints_forecast/', help='location to store model checkpoints')

parser.add_argument('--pretrained_model_path', type=str, default='./checkpoints/U2M_ETTm1.csv_dim7_patch12_minPatch20_maxPatch200\
                    _mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/epoch-80000.pth', help='location of the pretrained model')
parser.add_argument('--pretrain_args_path', type=str, default='./checkpoints/U2M_ETTm1.csv_dim7_patch12_minPatch20_maxPatch200\
                    _mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json', help='location of the pretrained model parameters')

parser.add_argument('--in_len', type=int, default=720, help='input MTS length (T)')
parser.add_argument('--out_len', type=int, default=96, help='output MTS length (\tau)')

parser.add_argument('--finetune_layers', type=int, default=1, help='forecast layers to finetune')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout ratio for finetune layers')
parser.add_argument('--neighbor_num', type=int, default=10, help='number of neighbors for graph (for high dimensional data)')

parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
parser.add_argument('--lradj', type=str, default='none', help='adjust learning rate')
parser.add_argument('--tolerance', type=int, default=3, help='early stopping tolerance')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--slide_step', type=int, default=10, help='sliding steps for the sliding window of train and valid')

parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS', default=False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser.add_argument('--label', type=str, default='ft', help='labels to attach to setting')

args = parser.parse_args()

args.data_split = string_split(args.data_split)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

args.pretrain_args = json.load(open(args.pretrain_args_path))

# fix random seed
torch.manual_seed(2023)
random.seed(2023)
np.random.seed(2023)
torch.cuda.manual_seed_all(2023)

print('Args in experiment:')
print(args)

for i in range(args.itr):
    setting = 'U2M_forecast_data{}_dim{}_patch{}_dm{}_dff{}_heads{}_eLayer{}_dLayer{}_IRmode{}_ftLayer{}_neighbor{}_inlen{}_outlen{}_itr{}'.format(args.data_name,
                                                                                                                                                           args.pretrain_args['data_dim'], args.pretrain_args['patch_size'], args.pretrain_args['d_model'], args.pretrain_args['d_ff'],
                                                                                                                                                           args.pretrain_args['n_heads'], args.pretrain_args['e_layers'], args.pretrain_args['d_layers'], args.IR_mode, args.finetune_layers, args.neighbor_num,
                                                                                                                                                           args.in_len, args.out_len, i)
    exp = UP2ME_exp_forecast(args)

    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, args.save_pred)