import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

if __name__ == '__main__':
    base_dir = "E:/shell/VMF3_OP/zhd_zwd_lstm/result_data/"
    dest_path = "E:/shell/VMF3_OP/zhd_zwd_period_model/matlab_code/height/ztd1.csv"
    grids = os.listdir(base_dir)
    data = []
    for grid in grids:
        file_path = base_dir + grid
        grid = grid[:-4]
        pt = grid.split("_")
        dataframe = pd.read_csv(file_path, usecols=['true_ztd', "period_add_residual"])
        true_ztd = np.average(dataframe['true_ztd'])
        pt.append(true_ztd)
        lstm_ztd = np.average(dataframe['period_add_residual'])
        pt.append(lstm_ztd)
        data.append(pt)
    df = pd.DataFrame(data)
    df.to_csv(dest_path)
