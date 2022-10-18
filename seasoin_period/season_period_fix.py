import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig
import matplotlib.pyplot as plt

from time_util import *

if __name__ == '__main__':
    data_path = "E:/shell/VMF3_OP/zhd_zwd_grid_serial/"
    output_path = "E:/shell/VMF3_OP/zhd_zwd_period_model/"
    files = os.listdir(data_path)
    grid = '-82.5_47.5.csv'
    # 载入数据
    c_unit, c0, c1, c2, c3, c4, c5 = [], [], [], [], [], [], []
    path = data_path + grid
    dataframe = pd.read_csv(path, header=None, names=['epoch', 'true_ztd'])
    for epoch in dataframe['epoch']:
        c_unit.append(1)
        c0.append(epoch)
        c1.append(np.cos(2 * np.pi * float(epoch) / 1000))
        c2.append(np.sin(2 * np.pi * float(epoch) / 1000))
        c3.append(np.cos(4 * np.pi * float(epoch) / 1000))
        c4.append(np.sin(4 * np.pi * float(epoch) / 1000))
        year = int(epoch / 1000)
        doy_and_time = float(epoch) % 1000 / 366.75
        x = year + doy_and_time
        c5.append(x)
    dataframe['c_unit'] = c_unit
    dataframe['c0'] = c0
    dataframe['c1'] = c1
    dataframe['c2'] = c2
    dataframe['c3'] = c3
    dataframe['c4'] = c4
    dataframe['c5'] = c5
    C = np.transpose(np.array([c_unit, c0, c1, c2, c3, c4]))
    CT = np.transpose(C)
    CTC = np.dot(CT, C)
    CTC_n = np.linalg.inv(CTC)
    CTC_n_CT = np.dot(CTC_n, CT)
    A = np.dot(CTC_n_CT, dataframe['true_ztd'])
    period_model = np.dot(A, CT)
    dataframe['period_model'] = period_model
    error_serial, upd_model_serial = [], []
    for i in range(0, dataframe.shape[0]):
        error_ = dataframe['true_ztd'].values[i] - dataframe['period_model'].values[i]
        error_serial.append(error_ * 100)
        upd_model = dataframe['period_model'].values[i] + error_
        upd_model_serial.append(upd_model)
    dataframe['residual_serial'] = error_serial
    dataframe['upd_model_serial'] = upd_model_serial
    dataframe.to_csv(output_path + 'data/' + str(grid))
    period = 12
    n = 1460 * (12 - period)
    grid = grid[:-4]
    plt.plot(dataframe['c5'][n:], dataframe['residual_serial'][n:], color='#ed7d31',
             label="ztd residual at Seasonal model ", linewidth=2)
    plt.title('ztd residual at Seasonal model ' + str(grid))
    plt.xlabel('time serial(year)')
    plt.ylabel('residual(cm)')
    plt.legend()  # 给图加图例
    plt.savefig(output_path + 'fig/residual_serial/' + str(grid) + ".png")
    plt.close()
    plt.plot(dataframe['c5'][n:], dataframe['true_ztd'][n:], color='#0000ee', label="vmf true data")
    plt.plot(dataframe['c5'][n:], dataframe['period_model'][n:], color='#ed7d31',
             label="ztd data at season model without correct residual", linewidth=2)
    plt.title('ztd data at Seasonal model without correct residual' + str(grid))
    plt.xlabel('time serial(year)')
    plt.ylabel('ztd(m)')
    plt.legend()  # 给图加图例
    plt.savefig(output_path + 'fig/period_model/' + str(grid) + ".png")
    plt.cla()
    plt.clf()
    plt.close()
