import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.models import Sequential
from keras.optimizers import SGD  # 随机梯度下降法
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from time_util import *

if __name__ == '__main__':
    base_dir = "E:/shell/VMF3_OP/zhd_zwd_period_model/data/"
    pred_residual_dir = 'E:/shell/VMF3_OP/zhd_zwd_period_model/residual/'
    files = os.listdir(base_dir)
    for i in files:
        df_true_value_period_value_all = pd.read_csv(base_dir + i, usecols=['epoch', 'true_ztd', 'c5', 'period_model',
                                                                            'residual_serial'], index_col=None)
        df_residual_value = pd.read_csv(pred_residual_dir + 'predict_vmf/' + i, usecols=['0'])
        start = 17532
        df_true_value_period_value = df_true_value_period_value_all[start:]
        residual = df_residual_value['0'].values
        df_true_value_period_value['residual_pred'] = residual / 100
        df_true_value_period_value['residual_serial'] = df_true_value_period_value['residual_serial'] / 100
        period_add_residual = []
        residual_pred_serial_error = []
        for index, row in df_true_value_period_value.iterrows():
            period_add_residual.append(row['period_model'] + row['residual_pred'])
            residual_pred_serial_error.append((row['residual_serial'] - row['residual_pred']))
        df_true_value_period_value['epoch'] = df_true_value_period_value['epoch'] % 2020
        df_true_value_period_value['epoch'] = df_true_value_period_value['epoch'].map(lambda x: '%.2f' % x)
        df_true_value_period_value['residual_pred_serial_error'] = residual_pred_serial_error
        df_true_value_period_value['period_add_residual'] = period_add_residual
        df_true_value_period_value.to_csv(pred_residual_dir + 'result_data/' + i)

