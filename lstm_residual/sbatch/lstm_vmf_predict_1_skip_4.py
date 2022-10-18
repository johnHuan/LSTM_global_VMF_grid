import time
import datetime
import os
import multiprocessing
import matplotlib
import threading

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig

import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.models import Sequential
from keras.optimizers import SGD 
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from time_util import *
import threading


def plot_predictions(during, dest_path, predict_vmf, test_set, grid_file):
    predict_vmf_no_system_bias = []
    arr_test_mean = np.mean(test_set)
    arr_predict_mean = np.mean(predict_vmf)
    system_error = arr_test_mean - arr_predict_mean
    for i in predict_vmf:
        predict_vmf_no_system_bias.append(i + system_error)
    plt.plot(test_set, color='orange', label='vmf true data')
    plt.plot(predict_vmf_no_system_bias, color='green', label='vmf  predicted data')
    plt.title('lstm model trained ZTD data at ' + grid_file[:-4] + ' for ' + str(during) + ' month')
    plt.xlabel('time')
    plt.ylabel('ZTD')
    plt.legend()  
    img = dest_path + str(during) + '/fig/' + grid_file[:-4] + '.' + str(system_error) + ".png"
    plt.savefig(img)
    plt.cla()
    plt.clf()
    plt.close()
    # plt.show()
    # savefig(img)


def lstm_vmf(during, grid_pos, grid_file, grid_dir, grid_model_pos, dest_path):
    dataset = []
    for sample in open(grid_pos + grid_file):
        current_year = int(sample[5:9])  # 2020
        current_month = int(sample[9:11])  # 9
        current_day = int(sample[11:13])  # 13
        current_doy = get_day_of_year(current_year, current_month, current_day)  # 257
        current_epoch = int(sample[15:17]) / 6 * 0.25
        element_x = current_doy + current_epoch
        element_y = float(sample[18:])
        dataset.append([element_x, element_y])  
    dataset = pd.DataFrame(dataset)
    period = during * 30 * 4
    n = (len(dataset.values) - period)
    train_set = dataset[:n].iloc[:, 1:2].values
    test_set = dataset[n:].iloc[:, 1:2].values

    sc = MinMaxScaler(feature_range=[0, 1])
    sc.fit_transform(train_set)

    model = load_model(grid_model_pos + grid_dir + '-lstm_vmf_model.h5')

    dataset_total = pd.concat((dataset[1][:n], dataset[1][n:]), axis=0)

    gap = 4

    inputs = dataset_total[len(dataset_total) - len(test_set) - gap:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)  # (154, 1)
    x_test = []
    for i in range(gap, inputs.shape[0]):
        x_test.append(inputs[i - gap:i, 0])

    x_test = np.array(x_test)  
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    predict_test = model.predict(x_test)
    predict_vmf = sc.inverse_transform(predict_test)
    save_data(during, dest_path, predict_vmf, test_set, grid_file)
    plot_predictions(during, dest_path, predict_vmf, test_set, grid_file)


def save_data(during, dest_path, predict_vmf, test_set, grid_file):
    predict_vmf_no_system_bias = []
    arr_test_mean = np.mean(test_set)
    arr_predict_mean = np.mean(predict_vmf)
    system_error = arr_test_mean - arr_predict_mean
    for i in predict_vmf:
        predict_vmf_no_system_bias.append(i[0] + system_error)
    pred_dataframe = pd.DataFrame(predict_vmf_no_system_bias)
    pred_dataframe.to_csv(dest_path + str(during) + '/predict_vmf/' + grid_file[:-4] + '.csv')
    test_dataframe = pd.DataFrame(test_set)
    test_dataframe.to_csv(dest_path + str(during) + '/test_vmf/' + grid_file[:-4] + '.csv')


if __name__ == '__main__':
#    base_dir = r"D:/lstm/gobal_648/global_grid_model/"
    base_dir = "/home/zhanghuan/dat01/ztd/global_648/"
    grid_directory_path = base_dir + 'data/'
    dest_path = base_dir + 'result/'
    files = os.listdir(grid_directory_path)
    # pool = multiprocessing.Pool(processes=1)
    try:
        for grid_dir in files:
            grid_pos = grid_directory_path + grid_dir + '/grid/'
            grid_model_pos = grid_directory_path + grid_dir + '/model/'
            for grid_file in os.listdir(grid_pos):
                during = 3
                g = grid_file[:-4]
                csv = g + '.csv'
                dest = dest_path + '3/predict_vmf/' + csv
                if os.path.exists(dest):
                    continue
                else:
                    # pool.apply_async(lstm_vmf, (during, grid_pos, grid_file, grid_dir, grid_model_pos, dest_path))

                    lstm_vmf(during, grid_pos, grid_file, grid_dir, grid_model_pos, dest_path)

                    # process = multiprocessing.Process(target=lstm_vmf,
                    #                               args=(
                    #                                   during, grid_pos, grid_file, grid_dir, grid_model_pos,
                    #                                   dest_path))
                # process.start()
    except Exception as ex:
        print(" lstm model train exception: " + str(ex))
    # pool.close()
    # pool.join()
