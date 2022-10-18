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
    dataframe = pd.read_csv(basedir+'result_data/'+file+'.csv', index_col='epoch', usecols=['epoch', 'true_ztd', 'period_add_residual'])
    spring = dataframe[60:150]
    # summer = dataframe[90:180]
    # autumn = dataframe[180:270]
    # winter = dataframe[270:]
    y1, y2 = spring['true_ztd'], spring['period_add_residual']
    score = r2_score(y1, y2)
    mae = mean_absolute_error(y1, y2)
    rmse = np.sqrt(mean_squared_error(y1, y2))
    bias = np.average(y1 - y2)
    std = np.sqrt(np.average((y1-y2-bias)**2))
    # plt.plot(y1-y2, color='orange', label='error time serial')
    # plt.title('ZTD predicted error of time serial')
    # plt.xlabel('time')
    # plt.ylabel('vmf predicted error')
    # plt.legend()  # 给图加图例
    # plt.savefig(basedir + 'rms_time_serial/' + file+".png")
    # plt.close()
    # print(str.format("std:{}, bias:{}, mae: {}, rmse: {}", std, bias, mae, rmse))
    print(str.format("std:{}, bias:{}, mae: {}, rmse: {}, score: {}", std, bias, mae, rmse, score))
    return bias, mae, std, rmse


if __name__ == '__main__':
    # pred_vmf_path = r"D:\demo\5_5_predicted\demo\3\_result\predict_vmf\2.5_2.5.txt"
    base_dir = "E:/shell/VMF3_OP/period_model/slurm/12/"
    # test_vmf_path = r"D:\demo\5_5_predicted\demo\3\_result\test_vmf\2.5_2.5.txt"
    files = os.listdir(base_dir+'result_data')
    data_set = []
    j = 0
    for i in files:
        point_str = i[:-4]
        point = point_str.split("_")
        lat = point[0]
        lng = point[1]
        # rms, std, bias, mean_value_offset = assess(base_dir, base_dir+'result_data/' + i)
        bias, mae, std, rmse = assess(base_dir, point_str)
        data_set.append([lat, lng, bias, mae, std, rmse])
        j += 1
        print(j)
    data = pd.DataFrame(data_set)
    # data.to_csv(base_dir + "assess-spring.csv")
