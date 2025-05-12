import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import argparse
import os
import torch
import json

from exp.exp_imputation import exp_imputation_U2M
from utils.tools import string_split
import os
from loguru import logger

import numpy as np
import random
import math


# region
PREFIX_PATH = "pretrain-library"

DATASET_DIMS = {
    "weather_Base": 21,
    "ECL_Base": 321,
    "ETTm1_Base": 7,
    "traffic_Base": 862,
    "NIPS_Water_Base": 9,
    "PSM_Base": 25,
    "SWaT_Base": 51,
    "SMD_Base": 38
}


DATASET_MAPPING = {
    "weather.csv": "weather_Base",
    "ECL.csv": "ECL_Base",
    "ETTm1.csv": "ETTm1_Base",
    "traffic.csv": "traffic_Base",
    "NIPS_Water": "NIPS_Water_Base",
    "PSM": "PSM_Base",
    "SWaT": "SWaT_Base",
    "SMD": "SMD_Base",
}


pretrained_models_paths = {
    "NIPS_Water_Base": "U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-500000.pth",
    "traffic_Base": "U2MTraffic-Base_dataTraffic_dim862_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-205000.pth",
    "ECL_Base": "U2MECL-Base_dataElectricity_dim321_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-475000.pth",
    "PSM_Base": "U2MPSM-Base_dataPSM_dim25_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-490000.pth",
    "ETTm1_Base": "U2METTm1-Base_dataETTm1_dim7_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-35000.pth",
    "weather_Base": "U2MWeather-Base_dataweather_dim21_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-60000.pth",
    "SMD_Base": "U2MSMD-Base_dataSMD_dim38_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-140000.pth",
    "SWaT_Base": "U2MSWaT-Base_dataSWaT_dim51_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/bestValid-step-500000.pth",
}

pretrained_models_args_paths = {
    "NIPS_Water_Base": "U2MNIPS_Water-Base_dataNIPS_Water_dim9_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json",
    "traffic_Base": "U2MTraffic-Base_dataTraffic_dim862_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json",
    "ECL_Base": "U2MECL-Base_dataElectricity_dim321_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json",
    "PSM_Base": "U2MPSM-Base_dataPSM_dim25_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json",
    "ETTm1_Base": "U2METTm1-Base_dataETTm1_dim7_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json",
    "weather_Base": "U2MWeather-Base_dataweather_dim21_patch12_minPatch20_maxPatch200_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json",
    "SMD_Base": "U2MSMD-Base_dataSMD_dim38_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json",
    "SWaT_Base": "U2MSWaT-Base_dataSWaT_dim51_patch10_minPatch5_maxPatch100_mask0.5_dm256_dff512_heads4_eLayer4_dLayer1_dropout0.0/args.json",
}

for key, value in pretrained_models_paths.items():
    assert os.path.exists(os.path.join(
        PREFIX_PATH, value)), "The key is %s" % key

for key, value in pretrained_models_args_paths.items():
    assert os.path.exists(os.path.join(
        PREFIX_PATH, value)), "The key is %s" % key

# endregion

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Finetune U2M for forecasting')

    # random seed and test random sed
    parser.add_argument('--random_seed', type=int, default=2023, help='random seed')
    parser.add_argument('--test_random_seed', type=int, default=4069, help='random seedm during test time')

    # Model ID
    parser.add_argument('--model', type=str, default="U2M_imputation", help='Model card.')

    # training or testing
    parser.add_argument('--is_training', action='store_true',
                        help='whether finetuning or perform zero shot imputation', default=False)

    parser.add_argument('--root_path', type=str, default='datasets/', help='root path of the data file')
    parser.add_argument('--data_name', type=str, default='NIPS_Water', help='data file')
    parser.add_argument('--data_split', type=str, default='0.7, 0.1, 0.2', help='train/val/test split, can be ratio or number')
    parser.add_argument('--checkpoints', type=str, default='checkpoints_forecast/weather/', help='location to store model checkpoints')
    parser.add_argument('--type_name', type=str, default='imputation', help='fixed missing gaps in time series')
    parser.add_argument('--data_format', type=str, default='npy', help='data format', choices=["csv", "npy"])
    parser.add_argument('--dropout', type=float, default='0.0', help='dropout rate')
    parser.add_argument('--slide_step', type=int, default=1, help='sliding steps for the sliding window of train and valid')
    parser.add_argument('--valid_prop', type=float, default=0.2, help='proportion of validation set, for numpy data only')

    parser.add_argument('--pretrained_model_path', type=str, default="", help='location of the pretrained model')
    parser.add_argument('--pretrain_args_path', type=str, default="", help='location of the pretrained model parameters')

    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size of train input data')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer initial learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--tolerance', type=int, default=4, help='experiments times')

    # Imputation related coefficient
    parser.add_argument('--min_mask_ratio', type=float, default=0.025, help='the minimum mask ratio')
    parser.add_argument('--max_mask_ratio', type=float, default=0.25, help="the maximum mask ratio")
    parser.add_argument('--in_len', type=int, default=600, help="the imputation time series length")
    parser.add_argument('--out_len', type=int, default=0, help='output MTS length (\tau)')

    # Finetuning options
    parser.add_argument('--neighbor_num', type=int, help='neighbor num')
    parser.add_argument('--imputation_layers', type=int, default=1, help='imputation_layers to finetune')

    parser.add_argument('--save_pred', action='store_true', help='whether to save the predicted future MTS')

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

    # Ablation study
    parser.add_argument('--gnn_type', type=str, default='graph_transformer', help='ablation study', choices=["Channel-Independence", "Full-connection", "Pearson", "DTW", "pretrain_model", "graph_transformer"])
    parser.add_argument('--label', type=str, default='stratch', help='labels to attach to setting')
    # mask generating option
    parser.add_argument('--save_mask', type=bool, default=False, help='save mask for future imputation use.')
    parser.add_argument('--mask_save_path', type=str, default="./data/imputation/mask_generator", help='mask generator result.')

    args = parser.parse_args()

    args.data_split = string_split(args.data_split)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.mask_type = "on_load" if not args.save_mask else "on_save"
    args.dry_run = (args.mask_type == "on_save")
    assert args.mask_type == "on_load"
    # random seed
    fix_seed = args.random_seed
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        logger.warning(f"Using multi-gpu support, the device ids are {args.device_ids}")

    args.data_name = DATASET_MAPPING[args.data_name]
    args.data_dim = DATASET_DIMS[args.data_name]
    args.neighbor_num = min(math.ceil(0.5 * args.data_dim), 10)

    args.pretrained_model_path = os.path.join(
        PREFIX_PATH, pretrained_models_paths[args.data_name])
    args.pretrain_args_path = os.path.join(
        PREFIX_PATH, pretrained_models_args_paths[args.data_name])
    args.pretrain_args = json.load(open(args.pretrain_args_path))

    print('Args in experiment:')
    logger.info(args)

    if args.dry_run:
        exp = exp_imputation_U2M(args)
        test_data, test_loader = exp._get_data(flag='test')
        sys.exit(1)

    if not args.is_training:
        setting = 'U2M_zeroshot_test_name_{}_inlen_{}_patch_size_{}_min_mask_ratio_{}_max_mask_ratio_{}_batch_size_{}'.format(
            args.data_name,
            args.in_len,
            12,
            args.min_mask_ratio,
            args.max_mask_ratio,
            args.batch_size,
        )
        logger.info(setting)
        exp = exp_imputation_U2M(args)
        exp.test(args)
    else:
        for i in range(args.itr):
            setting = '{}_test_name_{}_inlen_{}_patch_size_{}_min_mask_ratio_{}_max_mask_ratio_{}_batch_size_{}'.format(
                args.model,
                args.data_name,
                args.in_len,
                12,
                args.min_mask_ratio,
                args.max_mask_ratio,
                args.batch_size,
            )
            exp = exp_imputation_U2M(args)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            logger.info(setting)
            exp.train(setting=setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(args)
