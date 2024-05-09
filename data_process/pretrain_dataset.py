import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

#<---------------------original dataset(loader) to get the main result in the paper, generate one seprate dataset for each window length----------------------->

#for datasets saved in csv format, mostly datasets originally used for forecasting
class Pretrain_Dataset_csv(Dataset):
    def __init__(self, root_path, data_path='ETTh1.csv', flag='train', ts_len=12 * 10,
                 data_split=[0.7, 0.1, 0.2], scale=True, scale_statistic=None):
        self.ts_len = ts_len
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
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
            train_ts_len = self.data_split[0]
            val_ts_len = self.data_split[1]
            test_ts_len = self.data_split[2]
        #ratio split, e.g. [0.7, 0.1, 0.2] for Weather
        else:
            train_ts_len = int(len(df_raw) * self.data_split[0])
            test_ts_len = int(len(df_raw) * self.data_split[2])
            val_ts_len = len(df_raw) - train_ts_len - test_ts_len

        border1s = [0, train_ts_len, train_ts_len + val_ts_len]
        border2s = [train_ts_len, train_ts_len + val_ts_len, train_ts_len + val_ts_len + test_ts_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]  #leave the first column (Timestamp)
        self.data_dim = len(cols_data)

        df_data = df_raw[cols_data]

        if self.scale:
            if self.scale_statistic is None:
                self.scaler = StandardScaler()
                train_data = df_data[border1s[0]:border2s[0]]  #use training set for standardlization
                self.scaler.fit(train_data.values)
            else:
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.window_num = len(self.data_x) - self.ts_len + 1

    def __getitem__(self, index):
        #get one decoupled univariate series and the corresponding channel index
        channel_idx = index // self.window_num
        window_idx = index % self.window_num

        s_begin = window_idx
        s_end = s_begin + self.ts_len

        ts_x = self.data_x[s_begin:s_end, channel_idx]

        return ts_x, channel_idx

    def __len__(self):
        return self.data_dim * self.window_num


#for datasets saved in npy format, mostly datasets originally used for anomaly detection
class Pretrain_Dataset_npy(Dataset):
    '''
    For dataset stored in .npy files, which is usually saved in '*_train.npy', '*_test.npy' and '*_test_label.npy'
    We split the original training set into training and validation set
    '''
    def __init__(self, root_path, data_name='SMD', flag="train", ts_len=10 * 5,
                 valid_prop=0.2, scale=True, scale_statistic=None):
        self.flag = flag
        self.ts_len = ts_len
        self.valid_prop = valid_prop
        self.scale = scale
        self.scale_statistic = scale_statistic
        if flag == 'train' or flag == 'val':
            data_file = os.path.join(root_path, '{}_train.npy'.format(data_name))
            label_file = None
        elif flag == 'test':
            data_file = os.path.join(root_path, '{}_test.npy'.format(data_name))
            label_file = os.path.join(root_path, '{}_test_label.npy'.format(data_name))
        self.__read_data__(data_file, label_file)

    def __read_data__(self, data_file, label_file=None):
        raw_data = np.load(data_file)

        if (self.flag == 'train' or self.flag == 'val'):
            data_len = len(raw_data)
            train_data = raw_data[0:int(data_len * (1 - self.valid_prop))]
            val_data = raw_data[int(data_len * (1 - self.valid_prop)):]
            self.train = train_data
            self.val = val_data
        elif (self.flag == 'test'):
            self.test = raw_data
            self.test_labels = np.load(label_file)

        self.data_dim = raw_data.shape[1]

        if self.scale:
            if self.flag == 'train' or self.flag == 'val':
                self.scaler = StandardScaler()
                self.scaler.fit(self.train)
                self.train = self.scaler.transform(self.train)
                self.val = self.scaler.transform(self.val)

            elif self.flag == 'test':
                # use pre-computed mean and std
                self.scaler = StandardScaler(mean=self.scale_statistic['mean'], std=self.scale_statistic['std'])
                self.test = self.scaler.transform(self.test)

        if self.flag == 'train':
            self.window_num = len(self.train) - self.ts_len + 1
            self.data_x = self.train
        elif self.flag == 'val':
            self.window_num = len(self.val) - self.ts_len + 1
            self.data_x = self.val
        elif self.flag == 'test':
            self.window_num = len(self.test) - self.ts_len + 1
            self.data_x = self.test

    def __len__(self):
        return self.data_dim * self.window_num

    def __getitem__(self, index):
        channel_idx = index // self.window_num
        window_idx = index % self.window_num

        s_begin = window_idx
        s_end = s_begin + self.ts_len

        ts_x = self.data_x[s_begin:s_end, channel_idx]

        return ts_x, channel_idx