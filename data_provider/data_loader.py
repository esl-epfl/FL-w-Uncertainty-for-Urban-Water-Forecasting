import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.timefeatures import time_features
import warnings

# warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.normal = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path[0]))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.normal.fit(train_data.values)
            # print(self.scaler.mean_, self.scaler.var_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date) #, dayfirst=True
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_exo(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size = [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.normal = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path[0]))

        # Move the last column (exo_future) to the end if needed
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.normal.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Exogenous future: last column of data_x, future window after s_end
        exo_future = self.data_x[s_end:s_end + self.pred_len, -2]  # shape: [pred_len], index -2 if the exogenous (rain) is the last column in raw file
        
        if len(exo_future) >= 36:
            exo_future = exo_future[:36]
        elif len(exo_future) == 0:
            exo_future = np.zeros(36, dtype=np.float32)
        else:
            pad = np.full(36 - len(exo_future), exo_future[-1], dtype=np.float32)
            exo_future = np.concatenate([exo_future, pad], axis=0)

        exo_future = exo_future.reshape(36, 1)          # shape: [pred_len, 1]

        return [seq_x, exo_future], seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Custom_real_forecast(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 exo_measured_path='', exo_forecast_path='',
                 target='OT', scale=True, timeenc=0, freq='h'):
        """
        data_path = endo_path: CSV with columns [date, Node1, Node2, ...]
        exo_measured_path: CSV (df_m) with columns [time, value]
        exo_forecast_path: CSV (df_f) with columns [issued_at, forecast_time, value]

        seq_x: history window of endogenous + measured exogenous
        exo_future: forecasted exogenous, issued at end of history (seq end),
                    over the prediction horizon
        """
        # test
        # exo_measured_path = "measure_timeseries_dt_measure.csv"      
        exo_forecast_path = "measure_timeseries_dt_forecast.csv"     

        # size = [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.endo_path = data_path[0]
        # self.exo_measured_path = exo_measured_path
        self.exo_forecast_path = data_path[1]
        # self.root_forecast = "/home/abgo/Data/Hydrique_Lausanne/code/"

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.normal = MinMaxScaler()
        self.exo_forecast_scaler = StandardScaler()

        # --------------------------------------------------
        # 1) Load endogenous series (ensure datetime)
        # --------------------------------------------------
        df_raw_endo = pd.read_csv(os.path.join(self.root_path, self.endo_path))
        df_raw_endo['date'] = pd.to_datetime(df_raw_endo['date'])
        df_raw_endo = df_raw_endo.sort_values('date')

        # --------------------------------------------------
        # 2) Load measured exogenous (df_m: time, value) and merge
        # --------------------------------------------------
        # if self.exo_measured_path:
        #     df_raw_exo_msr = pd.read_csv(os.path.join(self.root_path, self.exo_measured_path))
        #     df_raw_exo_msr['time'] = pd.to_datetime(df_raw_exo_msr['time'])
        #     df_raw_exo_msr = df_raw_exo_msr.rename(columns={'time': 'date', 'value': 'exo_measured'})
        #     df_raw_endo = df_raw_endo.merge(df_raw_exo_msr, on='date', how='left')
        #     df_raw_endo['exo_measured'] = df_raw_endo['exo_measured'].fillna(0.0)
        # else:
        #     df_raw_endo['exo_measured'] = np.nan


        # --------------------------------------------------
        # 4) Reorder columns: date, features..., target last
        # --------------------------------------------------
        cols = list(df_raw_endo.columns)
        cols.remove('date')
        cols.remove(self.target)
        df_raw_endo = df_raw_endo[['date'] + cols + [self.target]]

        self.all_feature_cols = cols + [self.target]

        # --------------------------------------------------
        # 5) Train / Val / Test splits (by index)
        # --------------------------------------------------
        num_train = int(len(df_raw_endo) * 0.7)
        num_test = int(len(df_raw_endo) * 0.2)
        num_vali = len(df_raw_endo) - num_train - num_test

        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw_endo) - num_test - self.seq_len
        ]
        border2s = [
            num_train,
            num_train + num_vali,
            len(df_raw_endo)
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # --------------------------------------------------
        # 6) Choose input features
        # --------------------------------------------------
        if self.features in ['M', 'MS']:
            cols_data = df_raw_endo.columns[1:]  # drop 'date'
            df_data = df_raw_endo[cols_data]
        elif self.features == 'S':
            df_data = df_raw_endo[[self.target]]

        # --------------------------------------------------
        # 7) Scale on training split only
        # --------------------------------------------------
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.normal.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # --------------------------------------------------
        # 8) Time features + keep date_index for each row
        # --------------------------------------------------
        df_stamp = df_raw_endo[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])  # already datetime, explicit

        # actual datetime axis for each row (needed for exo_future lookup)
        self.date_index = df_stamp['date'].to_numpy()

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # --------------------------------------------------
        # 9) Split into:
        #    - data_x: all features (endo + exo_measured)
        #    - data_y: ONLY endogenous series (drop exo_measured)
        # --------------------------------------------------
        self.data_x = data[border1:border2]  # full feature set for encoder

        # df_data_cols = list(df_data.columns)
        # endogenous = all df_data columns except explicit exogenous ones
        # exo_cols = ['exo_measured']  # extend here if you add more exogenous columns later
        # endo_cols = [c for c in df_data_cols if c not in exo_cols]

        # endo_idx = [df_data_cols.index(c) for c in endo_cols]
        # data_endo = data[:, endo_idx]

        # self.data_y = data_endo[border1:border2]  # only endogenous series
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

        # --------------------------------------------------
        # 3) Load forecast exogenous (df_f: issued_at, forecast_time, value)
        #    KEEP ALL FORECASTS; group by issued_at for fast lookup
        # --------------------------------------------------
        self.forecast_by_issue = {}
        if self.exo_forecast_path:
            df_raw_exo_frc = pd.read_csv(os.path.join(self.root_path, self.exo_forecast_path))
            df_raw_exo_frc['issued_at'] = pd.to_datetime(df_raw_exo_frc['issued_at'])
            df_raw_exo_frc['forecast_time'] = pd.to_datetime(df_raw_exo_frc['forecast_time'])

            # optional: avoid forecasts issued after their valid time
            df_raw_exo_frc = df_raw_exo_frc[
                df_raw_exo_frc['issued_at'] <= df_raw_exo_frc['forecast_time']
            ]

            # scale forecast values (so exo_future is on similar scale as seq_x)
            if self.scale:
                self.exo_forecast_scaler.fit(df_raw_exo_frc[border1s[0]:border2s[0]][['value']].values)
                df_raw_exo_frc['value_scaled'] = self.exo_forecast_scaler.transform(
                    df_raw_exo_frc[['value']].values
                )
            else:
                df_raw_exo_frc['value_scaled'] = df_raw_exo_frc['value'].astype(float)

            # group by issued_at, keep each group sorted in forecast_time
            for issued_at, g in df_raw_exo_frc.groupby('issued_at'):
                g = g.sort_values('forecast_time').reset_index(drop=True)
                self.forecast_by_issue[issued_at] = g


    def __getitem__(self, index):

        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        # --------------------------------------------------
        # seq_x, seq_y, time marks
        # --------------------------------------------------
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # --------------------------------------------------
        # Build exo_future
        # --------------------------------------------------
        end_time = pd.Timestamp(self.date_index[s_end - 1])   # datetime at end of history
        
        df_issue = self.forecast_by_issue.get(end_time, None)

        # self.pred_len or fixed 36 steps?
        if df_issue is None:
            vals = np.zeros(36, dtype=np.float32)
        else:
            # forecast_time >= end_time
            df_horizon = df_issue[df_issue['forecast_time'] >= end_time]
            raw_vals = df_horizon['value_scaled'].to_numpy(dtype=np.float32)

            if len(raw_vals) >= 36:
                vals = raw_vals[:36]
            elif len(raw_vals) == 0:
                vals = np.zeros(36, dtype=np.float32)
            else:
                pad = np.full(36 - len(raw_vals), raw_vals[-1], dtype=np.float32)
                vals = np.concatenate([raw_vals, pad], axis=0)
        
        exo_future = vals.reshape(36, 1)

        # --------------------------------------------------
        # FINAL RETURN (ALL TORCH TENSORS)
        # --------------------------------------------------
        return [seq_x, exo_future], seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Custom_add_sin(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        #--- Add Sin Signal ----
        wavelength = self.seq_len
        t_x = np.arange(wavelength)
        t_l_y = np.arange(self.seq_len - self.label_len, self.seq_len - self.label_len + self.label_len + self.pred_len) # len(seq_y) = label_len + pred_len
        t_y = np.arange(self.seq_len, self.seq_len + self.pred_len) # len(seq_y) = label_len + pred_len

        self.sin_x = 0.1 * np.sin(2 * np.pi * t_x / wavelength).reshape(-1, 1)
        self.sin_l_y = 0.1 * np.sin(2 * np.pi * t_l_y / wavelength).reshape(-1, 1)
        self.sin_y = 0.1 * np.sin(2 * np.pi * t_y / wavelength).reshape(-1, 1)
        #--- Take care of Rain data ----
        # self.rain_column = [-3, -2] # [rain, rain_future]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.normal = MinMaxScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            self.normal.fit(train_data.values)
            # print(self.scaler.mean_, self.scaler.var_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date) #, dayfirst=True
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.date = np.array(pd.to_datetime(df_stamp['date'].values))


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        # Added, seq_x_mark is replaced by seq_x_date in the function outputs, the same for seq_y_mark.
        seq_x_date = self.date[s_begin:s_end]
        seq_y_date = self.date[s_end:s_end+self.pred_len]

        #--- Add Sin to features, expept rain and future rain - [-3, -2] > [rain, rain_future] ---- 
        seq_x_org = seq_x.copy()
        seq_y_org = seq_y.copy()
        # seq_x[:,:-3] = seq_x[:,:-3] + self.sin_x 
        # seq_y[:,:-3] = seq_y[:,:-3] + self.sin_l_y
        # seq_x[:,-1:] = seq_x[:,-1:] + self.sin_x 
        # seq_y[:,-1:] = seq_y[:,-1:] + self.sin_l_y
        seq_x = seq_x + self.sin_x 
        seq_y = seq_y + self.sin_l_y

        return seq_x, seq_y, seq_x_mark, seq_y_mark #seq_x_date, seq_y_date

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
