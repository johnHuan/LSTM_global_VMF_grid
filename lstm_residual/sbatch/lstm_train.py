import datetime

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


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
'''
start_year: 训练样本起始年份
end_year: 训练样本终止年份
month: 从训练样本终止后，继续往后预测多少个月
'''


def lstm_vmf(source_path, source_filename, output_path, month, start_year, end_year):
    filename = source_path + source_filename
    grid = source_filename[:-4]
    dataset = pd.read_csv(filename, usecols=['c5', 'residual_serial'])
    train_start_index = 0
    for index, row in dataset.iterrows():
        if row['c5'] > start_year:
            train_start_index = index
            break
    train_end_index = 0
    for index, row in dataset.iterrows():
        if row['c5'] > end_year + 1:
            train_end_index = index
            break
    predict_period = month * 30 * 4
    test_end_index = train_end_index + predict_period
    dataset = dataset[train_start_index:test_end_index]

    # 训练集合测试集的数据
    train_set = dataset[0:dataset.shape[0] - predict_period].iloc[:, 1:2].values
    test_set = dataset[dataset.shape[0] - predict_period:].iloc[:, 1:2].values
    # 正则化(归一化): 将每一维的特征映射到指定的区间：【0,1】
    sc = MinMaxScaler(feature_range=[0, 1])
    train_set_scaled = sc.fit_transform(train_set)

    # 创建序列数据集（训练和测试）
    # 训练步长  （=  month * 30 day * 4 epoch） 作为 `时间步` 为一个样本， 1个输出
    step = 3 * 30 * 4
    x_train, y_train = [], []
    for train_time_step in range(step, train_set.shape[0]):  # 训练`时间步`
        x_train.append(train_set_scaled[train_time_step - step:train_time_step, 0])
        y_train.append(train_set_scaled[train_time_step, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)  # np类型变换

    # LSTM的输入：（samples, sequence_length, features）
    y_train = y_train.reshape(-1, 1)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # 搭建LSTM网络模型， 进行训练和预测
    # LSTM 第一层
    model = Sequential()
    model.add(LSTM(step, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.01))  # 丢掉10%
    # LSTM 第二层
    model.add(LSTM(step, return_sequences=True))
    model.add(Dropout(0.01))  # 丢掉10%
    # LSTM 第三次
    model.add(LSTM(step))
    model.add(Dropout(0.01))
    # Dense层
    model.add(Dense(units=1))

    # 模型编译
    model.compile(optimizer='rmsprop', loss='mse')

    # 第6步 构建数据集, 进行预测
    dataset_total = dataset['residual_serial'][:]
    # 获取输入数据
    epoch = step
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
    # 模型训练
    model.fit(x_train, y_train, epochs=10, batch_size=40)
    model.save(output_path + str(month) + '/' + str(grid) + '_lstm_ztd.h5')
    predict_test = model.predict(x_test)
    predict_ztd = sc.inverse_transform(predict_test)
    plot_predictions(output_path + str(month) + '/', grid, predict_ztd, test_set)
    save_data(output_path + str(month) + '/', grid, predict_ztd, test_set)


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
    data_path = "/home/zhanghuan/dat01/ztd/VMF3_OP/period_model/data/"
    output_path = "/home/zhanghuan/dat01/ztd/VMF3_OP/period_model/residual/"
    # files = os.listdir(data_path)
    # for grid in files:
    grid = "22.5_127.5.csv"
    try:
        month = 12
        start_lstm_inner = datetime.datetime.now()
        lstm_vmf(data_path, grid, output_path, month=month, start_year=2008, end_year=2018)
        end = datetime.datetime.now()
        print("耗时： " + str(end - start_lstm_inner))
    except Exception as ex:
        print(grid + " grid lstm model train exception: " + ex)    

