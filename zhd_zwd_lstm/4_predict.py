import datetime
import os

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig
import matplotlib.pyplot as plt

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
    plt.plot(test_set, color='orange', label='Test data of residual')
    plt.plot(predict_ztd_no_system_bias, color='green', label='predicted data of residual')
    plt.title('LSTM model trained ZTD residual data at ' + grid)
    plt.xlabel('epoch')
    plt.ylabel('Residual(cm)')
    plt.legend()  # 给图加图例
    img = output_path + 'fig/' + grid + '_' + str(system_error) + ".png"
    plt.savefig(img)
    # savefig(img)
    plt.close()


def lstm_vmf(source_path, source_filename, output_path):
    filename = source_path + source_filename
    grid = source_filename[:-4]
    dataset = pd.read_csv(filename, usecols=['c5', 'residual_serial'])
    index = 17532
    # 训练集合测试集的数据
    train_set = dataset[0:index].iloc[:, 1:2].values
    test_set = dataset[index:].iloc[:, 1:2].values
    # 正则化(归一化): 将每一维的特征映射到指定的区间：【0,1】
    sc = MinMaxScaler(feature_range=[0, 1])
    sc.fit_transform(train_set)
    model_path = output_path + '22.5_127.5_lstm_ztd.h5'
    model = load_model(model_path)
    # 第6步 构建数据集, 进行预测
    dataset_total = dataset['residual_serial'][:]
    # 获取输入数据
    epoch = 3 * 30 * 4
    # 获取输入数据
    inputs = dataset_total[len(dataset_total) - len(test_set) - epoch:].values
    # 归一化
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    # 准备测试集x_test ， 进行预测
    x_test = []
    for i in range(epoch, inputs.shape[0]):
        x_test.append(inputs[i - epoch:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    predict_test = model.predict(x_test)
    predict_ztd = sc.inverse_transform(predict_test)

    plot_predictions(output_path + '/', grid, predict_ztd, test_set)
    save_data(output_path + '/', grid, predict_ztd, test_set)


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
    base_dir = "E:/shell/VMF3_OP/zhd_zwd_lstm/"
    grid_data_path = base_dir + 'data/'
    dest_path = base_dir
    files = os.listdir(grid_data_path)
    files.sort()
    files.reverse()
    try:
        for grid_file in files:
            dest = dest_path + 'predict_vmf/' + grid_file
            if os.path.exists(dest):
                continue
            lstm_vmf(grid_data_path, grid_file, dest_path)
            print("---" + grid_file)
    except Exception as ex:
        print(" lstm model train exception: " + str(ex))

