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


def assess(y1, y2):
    Ai = y1 - y2
    bias = np.average(Ai)
    std = np.sqrt(np.average((Ai - bias) ** 2))
    mae = mean_absolute_error(y1, y2)
    mse = mean_squared_error(y1, y2)
    rmse = np.sqrt(mean_squared_error(y1, y2))
    # print(str.format("std:{}, bias:{}, mae: {}, rmse: {}", std, bias, mae, rmse))
    return [std * 100, bias * 100, mae * 100, rmse * 100]


if __name__ == "__main__":
    path = "E:/shell/VMF3_OP/zhd_zwd_period_model/SN_area/grid_data/-87.5_12.5.csv"
    dataframe = pd.read_csv(path)
    # 冬季 1200:1460,夏季600:1200
    dataframe_winter = dataframe[:600]
    predict_winter, test_winter = dataframe_winter["period_add_residual"], dataframe_winter["true_ztd"]
    station_accuracy_winter = assess(predict_winter, test_winter)
    print(station_accuracy_winter)
    # 冬季 1200:1460,夏季600:1200
    dataframe_summer = dataframe[600:1200]
    predict_summer, test_summer = dataframe_summer["period_add_residual"], dataframe_summer["true_ztd"]
    station_accuracy_summer = assess(predict_summer, test_summer)
    print(station_accuracy_summer)

