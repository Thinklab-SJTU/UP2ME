import argparse
import os
import torch
import json
from exp.exp_detect import UP2ME_exp_detect

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UP2ME for anomoaly detection')

    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--IR_mode', action='store_true', help='whether to use immediate reaction mode', default=False)

    parser.add_argument('--root_path', type=str, default='./datasets/NIPS_Water/', help='root path of the data file')
    parser.add_argument('--data_name', type=str, default='NIPS_Water', help='data name')  
    parser.add_argument('--valid_prop', type=float, default=0.2,help='valid proportion split from train set')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_detect/', help='location to store model checkpoints')

    parser.add_argument('--pretrained_model_path', type=str, default='pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100\
                        _mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-270000.pth', help='location of the pretrained model')
    parser.add_argument('--pretrain_args_path', type=str, default='pretrain-library/U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100\
                        _mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json', help='location of the pretrained model parameters')

    parser.add_argument('--seg_len', type=int, default=100, help='the non-overlapping segment length')
    parser.add_argument('--finetune_layers', type=int, default=1, help='forecast layers to finetune')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout ratio for finetune layers')
    parser.add_argument('--neighbor_num', type=int, default=10, help='number of neighbors for graph (for high dimensional data)') 

    parser.add_argument('--anomaly_ratio', type=float, default=1, help='anomaly ratio in the dataset (in %)')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
    parser.add_argument('--lradj', type=str, default='none', help='adjust learning rate')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--slide_step', type=int, default=10, help='sliding steps for the sliding window of train and valid')
    parser.add_argument('--tolerance', type=int, default=3, help='tolerance for early stopping')

    parser.add_argument('--save_folder', type=str, default='./detect_result/', help='folder path to save the detection results')
    parser.add_argument('--save_pred', action='store_true', help='whether to save the reconstructed MTS', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

    parser.add_argument('--label', type=str, default='ft',help='labels to attach to setting')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    args.pretrain_args = json.load(open(args.pretrain_args_path))

    print('Args in experiment:')
    print(args)

    for i in range(args.itr):
        setting = 'UP2ME_detect_{}_data{}_seglen{}_IRmode{}_ftlayers{}_neighbor{}_itr{}'.format(args.label, args.data_name, args.seg_len, \
                        args.IR_mode, args.finetune_layers, args.neighbor_num, i)
        exp = UP2ME_exp_detect(args)

        if args.is_training:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, args.save_pred)