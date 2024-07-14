from models.imputation_model.U2M_imputation import U2M_imputation
from torch.utils.data import DataLoader
from data_process.finetune_dataset import Dataset_Multi_imputation, Dataset_Imputation_npy
from models.imputation_model.mask_generator import generate_mask
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from loguru import logger
from tqdm import tqdm

class exp_imputation_U2M(object):
    def __init__(self, args):
        super(exp_imputation_U2M, self).__init__()
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        model_dict = {
            'U2M_imputation': U2M_imputation
        }
        if self.args.model == 'U2M_imputation':
            model = model_dict[self.args.model](
                pretrained_model_path=self.args.pretrained_model_path,
                pretrain_args=self.args.pretrain_args,
                args=self.args,
                finetune_layers=self.args.imputation_layers).float()
        return model

    def _get_data(self, flag):
        args = self.args

        if self.args.data_name == "NIPS_Water" and self.args.data_format == "npy":
            train_val_step = 1
        else:
            train_val_step = 10

        if flag == 'test':
            shuffle_flag = False
            drop_last = False
            batch_size = args.batch_size
            step = 1
        else:
            shuffle_flag = True
            drop_last = False
            batch_size = args.batch_size
            step = train_val_step

        if args.data_format == 'csv':
            data_set = Dataset_Multi_imputation(
                root_path=args.root_path,
                dataset_name=args.data_name,
                flag=flag,
                size=[args.in_len, args.out_len],
                data_split=args.data_split,
                scale=True,
                scale_statistic=None,
                mask_type=args.mask_type,
                min_mask_ratio=args.min_mask_ratio,
                max_mask_ratio=args.max_mask_ratio,
                mask_path=args.mask_save_path
            )
        elif args.data_format == 'npy':
            assert args.valid_prop == 0.2
            data_set = Dataset_Imputation_npy(
                root_path=os.path.join(args.root_path, args.data_name.replace("_Base", "")),
                dataset_name=args.data_name,
                flag=flag,
                size=[args.in_len, args.out_len],
                step=step,
                valid_prop=args.valid_prop,
                scale=True,
                scale_statistic=None,
                lead_in=1000,
                mask_type=args.mask_type,
                min_mask_ratio=args.min_mask_ratio,
                max_mask_ratio=args.max_mask_ratio,
                mask_path=args.mask_save_path,
            )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval_mode()

        with torch.no_grad():
            for i, (batch_x, _, _) in tqdm(enumerate(vali_loader), desc=f"Conducting training on {self.args.data_name}"):

                batch_x = batch_x.float().to(self.device)
                ori_ts = batch_x.clone()

                masked_ts, mask, _, _ = generate_mask(batch_x, self.args.min_mask_ratio, self.args.max_mask_ratio)

                assert self.args.is_training
                outputs = self.model.forward_finetune(masked_ts, mask, self.args.neighbor_num)

                loss = criterion(outputs[mask == 1], ori_ts[mask == 1])
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train_mode()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.tolerance, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train_mode()
            epoch_time = time.time()
            for i, (batch_x, _, _) in tqdm(enumerate(train_loader), desc=f"Conducting training on {self.args.data_name}"):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                ori_ts = batch_x.clone()

                masked_ts, mask, _, _ = generate_mask(batch_x, self.args.min_mask_ratio, self.args.max_mask_ratio)

                assert self.args.is_training
                outputs = self.model.forward_finetune(masked_ts, mask, self.args.neighbor_num)
                loss = criterion(outputs[mask == 1], ori_ts[mask == 1])

                train_loss.append(loss.item())
                if (i + 1) % 400 == 0:
                    # logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            self.test(save_pred=False) 

            logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss: {3:.7f} | Cost time: {4}".format(epoch + 1, train_steps, train_loss, vali_loss, time.time() - epoch_time))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        logger.info(f'Finish Training on {self.args.data_name.replace("_Base", "")} | min_mask_ratio: {self.args.min_mask_ratio} | max_mask_ratio: {self.args.max_mask_ratio}')
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return early_stopping.best_model

    def test(self, save_pred=False):
        test_data, test_loader = self._get_data(flag='test')

        torch.manual_seed(self.args.test_random_seed)

        self.model.eval_mode()

        preds = []
        trues = []
        metrics_all = []
        instance_num = 0

        with torch.no_grad():
            for i, (batch_x, masked_ts, mask) in tqdm(enumerate(test_loader), desc=f"Conducting test on {self.args.data_name}"):

                batch_x = batch_x.float().to(self.device)
                masked_ts = masked_ts.float().to(self.device)
                mask = mask.bool().to(self.device)

                ori_ts = batch_x.clone()

                if self.args.is_training:
                    outputs = self.model.forward_finetune(masked_ts, mask, self.args.neighbor_num)
                else:
                    outputs = self.model.forward_pretrain(masked_ts, mask)

                batch_size = outputs.shape[0]
                instance_num += batch_size

                pred = outputs[mask == 1]
                true = ori_ts[mask == 1]

                batch_metric = np.array(
                    metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())) * batch_size
                metrics_all.append(batch_metric)

                if (save_pred):
                    preds.append(pred.detach().cpu().numpy())
                    trues.append(true.detach().cpu().numpy())
        metrics_all = np.stack(metrics_all, axis=0)
        metrics_mean = metrics_all.sum(axis=0) / instance_num

        setting = 'U2M_test_name_{}_inlen_{}_min_mask_ratio_{}_max_mask_ratio_{}_batch_size_{}'.format(
            self.args.data_name,
            self.args.in_len,
            self.args.min_mask_ratio,
            self.args.max_mask_ratio,
            self.args.batch_size,
        )

        if self.args.model == 'U2M_imputation':
            setting += f"_patch_size_{self.args.pretrain_args['patch_size']}"

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metrics_mean
        logger.info('Tested on {0}: mse:{1:7f}, mae:{2:7f}'.format(self.args.data_name, mse, mae))

        np.save(folder_path + 'metrics.npy',
                np.array([mae, mse, rmse, mape, mspe]))
        if save_pred:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return
