import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


#Forecasting dataset for data saved in csv format
class Forecast_Dataset_csv(Dataset):
    def __init__(self, root_path, data_path='ETTm1.csv', flag='train', size=None, 
                  data_split = [0.7, 0.1, 0.2], scale=True, scale_statistic=None):
        
        # size [past_len, pred_len]
        self.in_len = size[0]
        self.out_len = size[1]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path
        self.data_split = data_split
        self.scale_statistic = scale_statistic
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        #int split, e.g. [34560,11520,11520] for ETTm1
        if (self.data_split[0] > 1):
            train_num = self.data_split[0]; 
            val_num = self.data_split[1]; 
            test_num = self.data_split[2];
        #ratio split, e.g. [0.7, 0.1, 0.2] for Weather
        else:
            train_num = int(len(df_raw)*self.data_split[0]); 
            test_num = int(len(df_raw)*self.data_split[2])
            val_num = len(df_raw) - train_num - test_num; 
        
        border1s = [0, train_num - self.in_len, train_num + val_num - self.in_len]
        border2s = [train_num, train_num+val_num, train_num + val_num + test_num]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean = self.scale_statistic['mean'], std = self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
    
    def __getitem__(self, index):
        past_begin = index
        past_end = past_begin + self.in_len
        pred_begin = past_end
        pred_end = pred_begin + self.out_len

        seq_x = self.data_x[past_begin:past_end].transpose() # [ts_d, ts_len]
        seq_y = self.data_y[pred_begin:pred_end].transpose()

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.in_len - self.out_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


#Forecasting dataset for data saved in npy format
class Forecast_Dataset_npy(Dataset):
    '''
    For dataset stored in .npy files (originally for anomaly detection), which is usually saved in '*_train.npy', '*_test.npy' and '*_test_label.npy'
    We split the training set into training and validation set
    We now use this dataset for forecasting task
    '''

    def __init__(self, root_path, data_name='SMD', flag="train", size=None, step=1,
                 valid_prop=0.2, scale=True, scale_statistic=None, lead_in=1000):

        self.in_len = size[0]
        self.out_len = size[1]

        self.root_path = root_path
        self.data_name = data_name
        self.flag = flag
        self.step = step   #like stride in convolution, we may skip multiple steps when using sliding window on large dataset (e.g. SMD has 566,724 timestamps)
        self.valid_prop = valid_prop
        self.scale = scale
        self.scale_statistic = scale_statistic

        '''
        the front part of the test series will be preserved for model input;
        to keep consistant with csv format where some steps in val set are input to the model to predict test set; 
        With lead_in, if pred_len remain unchanged, varying input length will not change y in test set
        '''
        self.lead_in = lead_in  

        self.__read_data__()

    def __read_data__(self):
        train_val_data = np.load(os.path.join(self.root_path, '{}_train.npy'.format(self.data_name)))
        test_data = np.load(os.path.join(self.root_path, '{}_test.npy'.format(self.data_name)))
        self.data_dim = train_val_data.shape[1]
        # we do not need anomaly label for forecasting

        train_num = int(len(train_val_data) * (1 - self.valid_prop))

        if self.scale:
            # use the mean and std of training set
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = train_val_data[:train_num]
                self.scaler.fit(train_data)
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])

            scale_train_val_data = self.scaler.transform(train_val_data)
            scale_test_data = self.scaler.transform(test_data)

            if self.flag == 'train':
                self.data_x = scale_train_val_data[:train_num]
                self.data_y = scale_train_val_data[:train_num]
            elif self.flag == 'val':
                self.data_x = scale_train_val_data[train_num - self.in_len:]
                self.data_y = scale_train_val_data[train_num - self.in_len:]
            elif self.flag == 'test':
                '''
                |------------------|------|----------------------------------------|
                                   ^      ^
                                   |      |
                      lead_in-in_len      lead_in
                '''
                self.data_x = scale_test_data[self.lead_in - self.in_len:] 
                self.data_y = scale_test_data[self.lead_in - self.in_len:]

        else:
            if self.flag == 'train':
                self.data_x = train_val_data[:train_num]
                self.data_y = train_val_data[:train_num]
            elif self.flag == 'val':
                self.data_x = train_val_data[train_num - self.in_len:]
                self.data_y = train_val_data[train_num - self.in_len:]
            elif self.flag == 'test':
                self.data_x = test_data[self.lead_in - self.in_len:]
                self.data_y = test_data[self.lead_in - self.in_len:]

    def __len__(self):
        return (len(self.data_x) - self.in_len - self.out_len) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step

        past_begin = index
        past_end = past_begin + self.in_len
        pred_begin = past_end
        pred_end = pred_begin + self.out_len

        seq_x = self.data_x[past_begin:past_end].transpose()  # [ts_d, ts_len]
        seq_y = self.data_y[pred_begin:pred_end].transpose()

        return seq_x, seq_y

#Anomaly Detection dataset (all saved in npy format)
class Detection_Dataset_npy(Dataset):
    '''
    For dataset stored in .npy files, which is usually saved in '*_train.npy', '*_test.npy' and '*_test_label.npy'
    We split the original training set into training and validation set
    '''

    def __init__(self, root_path, data_name='SMD', flag="train", seg_len=100, step=None,
                 valid_prop=0.2, scale=True, scale_statistic=None):
        self.root_path = root_path
        self.data_name = data_name
        self.flag = flag
        self.seg_len = seg_len # length of time-series segment, usually 100 for all anomaly detection experiments
        self.step = step if step is not None else seg_len #use step to skip some steps when the set is too large
        self.valid_prop = valid_prop
        self.scale = scale
        self.scale_statistic = scale_statistic

        self.__read_data__()

    def __read_data__(self):
        train_val_ts = np.load(os.path.join(self.root_path, '{}_train.npy'.format(self.data_name)))
        test_ts = np.load(os.path.join(self.root_path, '{}_test.npy'.format(self.data_name)))
        test_label = np.load(os.path.join(self.root_path, '{}_test_label.npy'.format(self.data_name)))
        self.data_dim = train_val_ts.shape[1]

        data_len = len(train_val_ts)
        train_ts = train_val_ts[0:int(data_len * (1 - self.valid_prop))]
        val_ts = train_val_ts[int(data_len * (1 - self.valid_prop)):]

        if self.scale:
            if self.scale_statistic is None:
                # use the mean and std of training set
                self.scaler = StandardScaler()
                self.scaler.fit(train_ts)
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
            self.train_ts = self.scaler.transform(train_ts)
            self.val_ts = self.scaler.transform(val_ts)
            self.test_ts = self.scaler.transform(test_ts)

        else:
            self.train_ts = train_ts
            self.val_ts = val_ts
            self.test_ts = test_ts
        self.threshold_ts = np.concatenate([self.train_ts, self.val_ts], axis=0) #use both training and validation set to set threshold
        self.test_label = test_label

    def __len__(self):
        # number of non-overlapping time-series segments
        if self.flag == "train":
            return (self.train_ts.shape[0] - self.seg_len) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val_ts.shape[0] - self.seg_len) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test_ts.shape[0] - self.seg_len) // self.step + 1
        elif (self.flag == 'threshold'):
            return (self.threshold_ts.shape[0] - self.seg_len) // self.step + 1

    def __getitem__(self, index):
        # select data by flag
        if self.flag == "train":
            ts = self.train_ts
        elif (self.flag == 'val'):
            ts = self.val_ts
        elif (self.flag == 'test'):
            ts = self.test_ts
        elif (self.flag == 'threshold'):
            ts = self.threshold_ts

        index = index * self.step

        ts_seg = ts[index:index + self.seg_len, :].transpose()  # [ts_dim, seg_len]
        ts_label = np.zeros(self.seg_len)
        if self.flag == 'test':
            ts_label = self.test_label[index:index + self.seg_len]

        return ts_seg, ts_label