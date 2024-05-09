import os
import torch
import numpy as np
import random

from models.pretrain_model.UP2ME_model import UP2ME_model
from data_process.pretrain_dataset import Pretrain_Dataset_csv, Pretrain_Dataset_npy
import time

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import json
import pickle

class UP2ME_exp_pretrain(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
    
    def _build_model(self):
        pretrain_model = UP2ME_model
        model = pretrain_model(
            data_dim = self.args.data_dim, 
            patch_size = self.args.patch_size,
            d_model = self.args.d_model, 
            d_ff = self.args.d_ff, 
            n_heads = self.args.n_heads, 
            e_layers = self.args.e_layers, 
            d_layers = self.args.d_layers, 
            dropout = self.args.dropout,
            mask_ratio = self.args.mask_ratio, 
            device=self.device
        ).float()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable parameters: {}'.format(trainable_params))
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag='train', candidate_patch_num=None):
        args = self.args

        if flag == 'val':
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
            #To save time, do not go over all possible window lengths in validation, just evaluate some percentiles from N_min to N_max
            candidate_patch_num = np.arange(args.min_patch_num, self.args.max_patch_num + 1, max(1, (self.args.max_patch_num - self.args.min_patch_num + 1) // self.args.valid_sep_point))
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size;
            if candidate_patch_num is None:
                candidate_patch_num = np.arange(args.min_patch_num, self.args.max_patch_num + 1)

        
        loaders = []
        loader_patch_num = []
        
        #generate one dataloader for each length, this is our original implementation to get results in the paper, but is memory-inefficient
        #we will implement a more efficient version of the sampling process
        for patch_num in candidate_patch_num:
            if args.data_format == 'csv':
                dataset = Pretrain_Dataset_csv(
                    root_path=args.root_path,
                    data_path=args.data_path,
                    flag=flag,
                    ts_len=args.patch_size * patch_num,
                    data_split = args.data_split,
                )
            elif args.data_format == 'npy':
                dataset = Pretrain_Dataset_npy(
                    root_path=args.root_path,
                    data_name=args.data_name,
                    flag=flag,
                    ts_len=args.patch_size * patch_num,
                    valid_prop=args.valid_prop,
                )

            if (dataset.__len__() > 0):
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last)
                loaders.append(loader)
                loader_patch_num.append(patch_num)

        return loaders, loader_patch_num #return a list of data loaders, each loader corresponds to a patch_num
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion
    
    def pretrain_one_batch(self, batch_ts, batch_channel_idx, criterion, optimizer):
        self.model.train()
        optimizer.zero_grad()

        batch_ts = batch_ts.float().to(self.device)
        batch_channel_idx = batch_channel_idx.long().to(self.device)

        masked_reconstruction_patches, masked_target_patches = self.model.pretrain_forward(batch_ts, batch_channel_idx)
        loss = criterion(masked_reconstruction_patches, masked_target_patches)

        loss.backward()
        optimizer.step()

        return loss.item()
    
    def validate_one_batch(self, batch_ts, batch_channel_idx, criterion):
        self.model.eval()
        
        with torch.no_grad():
            batch_ts = batch_ts.float().to(self.device)
            batch_channel_idx = batch_channel_idx.long().to(self.device)

            masked_reconstruction_patches, masked_target_patches = self.model.pretrain_forward(batch_ts, batch_channel_idx)
            loss = criterion(masked_reconstruction_patches, masked_target_patches)

        return loss.item()
    
    def validate(self, val_loaders, criterion):
        valid_loss = []
        epoch_time = time.time()
        
        for val_loader in val_loaders:  #go over each loader(length) in the validation set
            series_loss = []
            for i, (batch_ts, batch_channel_idx) in enumerate(val_loader):
                if self.args.valid_batches > 0 and i >= self.args.valid_batches: #only validate some batches to save time
                    break
                loss = self.validate_one_batch(batch_ts, batch_channel_idx, criterion)
                series_loss.append(loss)
            valid_loss.append(np.mean(series_loss))
        
        avg_loss = np.mean(valid_loss)
        epoch_time = time.time() - epoch_time

        return avg_loss, valid_loss, epoch_time
    
    def pre_train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        candidate_patch_num = np.arange(self.args.min_patch_num, self.args.max_patch_num + 1)
        
        #as maintaining one loader for each length is memory consuming, we sample a small portion from all possible lengths to obtain a pool
        #the pool is updated periodically. 
        if (self.args.resample_patch_num):
            sampled_patch_num = np.random.choice(candidate_patch_num, self.args.pool_size, replace=False)
        else:
            sampled_patch_num = None

        train_loaders, train_patch_nums = self._get_data('train', sampled_patch_num)
        iter_train_loaders= [iter(loader) for loader in train_loaders]
        valid_loaders, val_patch_nums  = self._get_data('val')

        best_valid_loss = np.inf
        non_improve_count = 0

        for step in range(self.args.train_steps):
            #resample some data loaders to update the pool
            if (self.args.resample_patch_num and (step + 1) % self.args.resample_freq == 0):
                sampled_patch_num = np.random.choice(candidate_patch_num, self.args.pool_size, replace=False)
                train_loaders, train_patch_nums = self._get_data('train', sampled_patch_num)
                iter_train_loaders= [iter(loader) for loader in train_loaders]

            # sample one train loader (length), corresponding to Equation (1) in the paper 
            select_loader_idx = random.randint(0, len(train_loaders) - 1)
            select_len = train_patch_nums[select_loader_idx] * self.args.patch_size
            select_iter_loader = iter_train_loaders[select_loader_idx]
            batch_data = next(select_iter_loader, None)
            if batch_data is None:
                iter_train_loaders[select_loader_idx] = iter(train_loaders[select_loader_idx])
                batch_data = next(iter_train_loaders[select_loader_idx], None)
            
            batch_ts, batch_channel_idx = batch_data
            batch_loss = self.pretrain_one_batch(batch_ts, batch_channel_idx, criterion, model_optim)
            
            if (step + 1) % 1000 == 0:
                print('Step {:d}/{:d} | Series Len {:d} | Train Loss {:.6f}'.format(
                    step + 1, self.args.train_steps, select_len, batch_loss))
        
            if (step + 1) % self.args.valid_freq == 0:
                valid_loss, valid_loss_list, valid_time = self.validate(valid_loaders, criterion)
                print('Step {:d}/{:d} | Valid Loss {:.6f} | Valid Time {:.2f}'.format(
                    step + 1, self.args.train_steps, valid_loss, valid_time))
                for i, loss in enumerate(valid_loss_list):
                        print('Series Len {:d} | Valid Loss {:.6f}'.format(
                            val_patch_nums[i] * self.args.patch_size, loss))
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
                    
                    #read all previous saved models and maintain the best three
                    saved_models = []
                    for file in os.listdir(path):
                        if file.startswith('bestValid') and file.endswith('.pth'):
                            saved_models.append(file)
                        saved_models.sort(key=lambda x: int(x.split('.')[0].split('-')[-1]))
                    while len(saved_models) > 3:
                        os.remove(os.path.join(path, saved_models[0]))
                        saved_models.pop(0)
                    torch.save(state_dict, os.path.join(path, 'bestValid-step-{}.pth'.format(step+1)))
                    non_improve_count = 0
                else:
                    non_improve_count += 1
                    if non_improve_count >= self.args.tolerance:
                        print('Early stop at step {:d}.'.format(step+1))
                        break

        
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, os.path.join(path, 'final-step-{}.pth'.format(step+1)))
        
        return self.model