import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

"""
-0.5_0.5
测试数据均值： 2.631342777777778
预测数据均值： 2.631342706914478
均值之差： 7.086329967265215e-08
bias： 7.086329987742661e-08
方差： 0.0003267868671071026
标准差(std)： 0.018077247221496492
rms： 0.018077247221635385
=================================
-10.5_166.5
测试数据均值： 2.510933611111111
预测数据均值： 2.510933665417565
均值之差： -5.43064540003968e-08
bias： -5.4306454141025046e-08
方差： 0.0006505308363000689
标准差(std)： 0.025505505999686987
rms： 0.025505505999744802
"""


if __name__ == '__main__':
    base_dir = "E:/shell/VMF3_OP/zhd_zwd_lstm/"
    dataframe = pd.read_csv(base_dir + 'result_data/22.5_127.5.csv', index_col='epoch',
                            usecols=['epoch', 'true_ztd', 'period_add_residual'])
    y1, y2 = dataframe['true_ztd'], dataframe['period_add_residual']
    Ai = y1 - y2
    bias = np.average(Ai)
    std = np.sqrt(np.average((Ai - bias) ** 2))
    mae = mean_absolute_error(y1, y2)
    mse = mean_squared_error(y1, y2)
    rmse = np.sqrt(mean_squared_error(y1, y2))
    print(str.format("std:{}, bias:{}, mae: {}, rmse: {}", std, bias, mae, rmse, mse))


# data.to_csv(base_dir + "assess-spring.csv")
