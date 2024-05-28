import os
import torch
import numpy as np

from models.finetune_model.UP2ME_detector import UP2ME_Detector
from data_process.finetune_dataset import Detection_Dataset_npy
import time
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import json
import pickle

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import segment_adjust, adjusted_precision_recall_curve

class UP2ME_exp_detect(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)
    
    def _build_model(self):
        model = UP2ME_Detector(
            pretrained_model_path = self.args.pretrained_model_path, 
            pretrain_args=self.args.pretrain_args, 
            finetune_flag=(not self.args.IR_mode),
            finetune_layers=self.args.finetune_layers,
            dropout=self.args.dropout
        ).float()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Trainable parameters: {}'.format(trainable_params))
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        args = self.args

        if flag == 'test' or flag == 'threshold':
            shuffle_flag = False; drop_last = False; batch_size = args.batch_size; step=args.seg_len;
        else:
            shuffle_flag = True; drop_last = False; batch_size = args.batch_size; step=args.slide_step;

        data_set = Detection_Dataset_npy(
            root_path=args.root_path,
            data_name=args.data_name, 
            flag=flag, 
            seg_len=args.seg_len,
            step=step,
            valid_prop=args.valid_prop,
            scale=True, 
            scale_statistic=None
        )

        print(flag, len(data_set))

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
        criterion =  nn.MSELoss()
        return criterion
    
    def _process_one_batch(self, dataset_object, batch_x):
        batch_x = batch_x.float().to(self.device)
        
        if self.args.IR_mode:
            outputs = self.model.immediate_detect(batch_x)
        else:
            outputs = self.model(batch_x, self.args.neighbor_num)

        return outputs, batch_x
    
    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval_mode()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,_) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss.detach().item())
        total_loss = np.average(total_loss)

        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, "args.json"), 'w') as f:
            json.dump(vars(self.args), f, indent=True)
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.tolerance, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()
        
        for epoch in range(self.args.train_epochs):
            time_now = time.time()
            iter_count = 0
            train_loss = []
            
            if isinstance(self.model, DataParallel):
                self.model.module.train_mode()
            else:
                self.model.train_mode()

            epoch_time = time.time()
            for i, (batch_x,batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true = self._process_one_batch(train_data, batch_x)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        state_dict = self.model.module.state_dict() if isinstance(self.model, DataParallel) else self.model.state_dict()
        torch.save(state_dict, path+'/'+'checkpoint.pth')
        
        return self.model
    
    def test(self, setting, save_pred = False):
        test_data, test_loader = self._get_data(flag='test')
        threshold_data, threshold_loader = self._get_data(flag='threshold')
        
        self.model.eval_mode()

        # (1) use the threshold loader (train + val) to select the threshold for anomaly annotation
        anomaly_score_threshold = []
        threshold_preds = []
        threshold_trues = []
        self.anomaly_criterion = nn.MSELoss(reduction='none')
        
        self.model.eval_mode()
        with torch.no_grad():
            for i, (batch_x, _) in enumerate(threshold_loader):
                # reconstruction
                pred, true = self._process_one_batch(threshold_data, batch_x)
                threshold_preds.append(pred.detach().cpu().numpy())
                threshold_trues.append(true.detach().cpu().numpy())
                # criterion
                score = torch.mean(self.anomaly_criterion(pred, true), dim=-2) # pred and true in shape [batch_size, ts_len]
                score = score.detach().cpu().numpy()
                anomaly_score_threshold.append(score)

        anomaly_score_threshold = np.concatenate(anomaly_score_threshold, axis=0).reshape(-1) 
        threshold = np.percentile(anomaly_score_threshold, 100 - self.args.anomaly_ratio) #error of both train and val
        print("Threshold :", threshold)

        # (2) calculate the anomaly score on the test set
        test_anomaly_score = []
        test_labels = []
        test_preds = []
        test_trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                pred, true = self._process_one_batch(test_data, batch_x)
                test_preds.append(pred.detach().cpu().numpy())
                test_trues.append(true.detach().cpu().numpy())
                # criterion
                score = torch.mean(self.anomaly_criterion(pred, true), dim=-2)
                score = score.detach().cpu().numpy()
                test_anomaly_score.append(score)
                test_labels.append(batch_y)
        
        test_anomaly_score = np.concatenate(test_anomaly_score, axis=0).reshape(-1)

        # (3) assign a binary label according to the threshold
        anomaly_pred = (test_anomaly_score > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        anomaly_gt = test_labels.astype(int)

        print("pred:   ", anomaly_pred.shape)
        print("gt:     ", anomaly_gt.shape)

        
        # (4) perfrom segment adjustment to measure precision, recall and F1-score
        adjusted_anomaly_pred = segment_adjust(anomaly_gt, anomaly_pred)
        accuracy = accuracy_score(anomaly_gt, adjusted_anomaly_pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(anomaly_gt, adjusted_anomaly_pred, average='binary')

        # (5) evaluate precision-recall curve that is agnostic to the threshold 
        precision_list, recall_list, average_precision = adjusted_precision_recall_curve(anomaly_gt, test_anomaly_score)

        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AP : {:0.4f} ".format(
            accuracy, precision, recall, f_score, average_precision))
        
        folder_path = self.args.save_folder + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open(folder_path + "result.txt", "w")
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, AP : {:0.4f} \n".format(
            accuracy, precision,
            recall, f_score, average_precision))
        f.write('\n')
        f.write('\n')
        f.close()

        if save_pred:
            threshold_preds = np.concatenate(threshold_preds, axis=0)
            threshold_trues = np.concatenate(threshold_trues, axis=0)
            test_preds = np.concatenate(test_preds, axis=0)
            test_trues = np.concatenate(test_trues, axis=0)
            data_save = {
                'threshold_reconstruct': threshold_preds,
                'threshold_trues': threshold_trues,
                'test_reconstruct': test_preds,
                'test_trues': test_trues,
                'test_labels': anomaly_gt
            }
            with open(folder_path + "reconstruction.pkl", 'wb') as f:
                pickle.dump(data_save, f)

        return