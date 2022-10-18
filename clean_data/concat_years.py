import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from time_util import *

if __name__ == '__main__':
    data_path = "E:/shell/VMF3_OP/grid/"
    output_path = "E:/shell/VMF3_OP/grid_serial/"
    years = os.listdir(data_path)
    years.sort()
    for year in years:
        data_path_with_year = data_path + year
        data_with_grids = os.listdir(data_path_with_year)
        print(year)
        for grid_as_fileName in data_with_grids:
            data_path_with_year_grid = data_path_with_year + '/' + grid_as_fileName
            grid_with_ = grid_as_fileName[:-4]
            data = []
            for sample in open(data_path_with_year_grid):
                epoch = int(year) * 1000 + float(sample[0:7].lstrip().rstrip())
                ztd = sample[7:14].lstrip().rstrip()
                data.append(["{:10.2f}".format(epoch), ztd])
            dataframe = pd.DataFrame(data)
            np_array = np.array(data)
            dataframe.to_csv(output_path + grid_with_ + '.csv', mode='a', header=False, index=False)