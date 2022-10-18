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


def assess(basedir, file):
    dataframe = pd.read_csv(basedir + 'residual/result_data/' + file + '.csv', index_col='epoch',
                            usecols=['epoch', 'true_ztd', 'period_add_residual'])
    """
    北半球
        春季：3~5月         240:600             60:150
        夏季：6~8月         600:960             150:240
        秋季：9~11月        960~1320            240:330
        冬季：12月、1~2月    :240 U 1320:        :60U330:
    南半球
        春季：9~11月        960~1320             240:330
        夏季：12、1~2月     :240 U 1320:         :60U330:
        秋季：3~5月         240:600              60:150
        冬季：6月8月        600:960              150:240
    """
    if file.startswith("-"):
        # 南半球的点
        season = pd.concat([dataframe[:60], dataframe[330:]])
    else:
        # 北半球的点
        season = dataframe[150:240]
    y1, y2 = season['true_ztd'], season['period_add_residual']
    Ai = y1 - y2
    bias = np.average(Ai)
    std = np.sqrt(np.average((Ai - bias) ** 2))
    mae = mean_absolute_error(y1, y2)
    mse = mean_squared_error(y1, y2)
    rmse = np.sqrt(mean_squared_error(y1, y2))
    print(str.format("std:{}, bias:{}, mae: {}, rmse: {}", std, bias, mae, rmse, mse))
    return std, bias, mae, rmse, mse


if __name__ == '__main__':
    result_data_dir = "E:/shell/VMF3_OP/zhd_zwd_period_model/residual/result_data/"
    base_dir = "E:/shell/VMF3_OP/zhd_zwd_period_model/"
    files = os.listdir(result_data_dir)
    data_set = []
    # j = 0
    for i in files:
        point_str = i[:-4]
        point = point_str.split("_")
        lat = point[0]
        lng = point[1]
        std, bias, mae, rmse, mse = assess(base_dir, point_str)
        data_set.append([lat, lng, std, bias, mae, rmse, mse])
        # j += 1
        # print(j)
    data = pd.DataFrame(data_set)
    data.to_csv(base_dir + "assess-summer_SN.csv")
