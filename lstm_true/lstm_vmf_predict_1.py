import datetime
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model


# 定义显示函数
def plot_predictions(output_path, grid, predict_ztd, test_set):
    """
    test_result: 测试真实值
    predict_result: 预测值
    """
    # 预测值剔除系统差
    predict_ztd_no_system_bias = []
    arr_test_mean = np.mean(test_set)
    arr_predict_mean = np.mean(predict_ztd)
    system_error = arr_test_mean - arr_predict_mean
    for i in predict_ztd:
        predict_ztd_no_system_bias.append(i[0] + system_error)
    plt.plot(test_set, color='orange', label='vmf true data')
    plt.plot(predict_ztd_no_system_bias, color='green', label='vmf  predicted data')
    plt.title('LSTM model trained vmf data at ' + grid)
    plt.xlabel('time')
    plt.ylabel('vmf')
    plt.legend()  # 给图加图例
    img = output_path + 'fig/' + grid + '_' + str(system_error) + ".png"
    plt.savefig(img)
    # savefig(img)
    plt.close()


# 载入数据
# def lstm_vmf(source_path, source_filename, output_path):
def lstm_vmf(source_path, source_filename, output_path, year, month):
    filename = source_path + source_filename
    grid = source_filename[:-4]
    dataset = pd.read_csv(filename, usecols=['c5', 'residual_serial'])
    flag = 0
    for index, row in dataset.iterrows():
        if row['c5'] > year:
            flag = index
            break

    # 训练集合测试集的数据
    # train_period = 3892  # 训练数据用2017001年到2019244
    train_start = flag
    predict_start = 17532  # 第17532行为2020001
    predict_period = month * 30 * 4
    train_set = dataset[train_start:predict_start].iloc[:, 1:2].values
    test_set = dataset[predict_start:predict_start + predict_period].iloc[:, 1:2].values
    # 正则化(归一化): 将每一维的特征映射到指定的区间：【0,1】
    sc = MinMaxScaler(feature_range=[0, 1])
    train_set_scaled = sc.fit_transform(train_set)
    model_path = output_path + str(year) + '_2020/' + str(month) + '/' + '22.5_127.5_lstm_ztd.h5'
    model = load_model(model_path)

    # 第6步 构建数据集, 进行预测
    dataset_total = dataset['residual_serial'][:]
    inputs = dataset_total[train_start:predict_start + predict_period].values
    # 归一化
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)  # (154, 1)
    x_test = []
    for i in range(0, inputs.shape[0]):
        x_test.append(inputs[:, 0])
    # 准备测试集x_test ， 进行预测
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  # 这一步会报错

    predict_test = model.predict(x_test)
    predict_ztd = sc.inverse_transform(predict_test)
    plot_predictions(output_path + str(year) + '_2020/' + str(month) + '/', str(grid), predict_ztd, test_set)
    save_data(output_path + str(year) + '_2020/' + str(month) + '/', str(grid), predict_ztd, test_set)


def save_data(output_path, grid, predict_ztd, test_set):
    predict_ztd_no_system_bias = []
    arr_test_mean = np.mean(test_set)
    arr_predict_mean = np.mean(predict_ztd)
    system_error = arr_test_mean - arr_predict_mean
    for i in predict_ztd:
        predict_ztd_no_system_bias.append(i[0] + system_error)
    dataframe_pred = pd.DataFrame(predict_ztd_no_system_bias)
    dataframe_pred.to_csv(output_path + "predict_vmf/" + grid + '.csv')
    dataframe_test = pd.DataFrame(test_set)
    dataframe_test.to_csv(output_path + "/test_vmf/" + grid + '.csv')


if __name__ == '__main__':
    base_dir = "E:/shell/VMF3_OP/period_model/"
    grid_data_path = base_dir + 'data/'
    dest_path = base_dir + 'true_lstm/'
    files = os.listdir(grid_data_path)
    files.sort()
    for month in range(1, 13):
        try:
            for grid_file in files:
                lstm_vmf(grid_data_path, grid_file, dest_path, 2008, month)
        except Exception as ex:
            print(" lstm model train exception: " + str(ex))