# 载入此项目所需要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 检查你的Python版本
'''
from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7来完成此项目')

'''

# 载入波士顿房屋的数据集
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# 完成
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
######################################################################
# 目标：计算价值的最小值
minimum_price = np.amin(prices)

# 目标：计算价值的最大值
maximum_price = np.amax(prices)

# 目标：计算价值的平均值
mean_price = np.mean(prices)

# 目标：计算价值的中值
median_price = np.median(prices)

# 目标：计算价值的标准差
std_price = np.std(prices)

# 目标：输出计算的结果
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))

LEN, seed = 489, None


def generate_train_and_test(X, y):
    """打乱并分割数据为训练集和测试集"""
    div = np.arange(1, LEN)
    np.random.shuffle(div)
    div_list = div.tolist()
    X_train = features.iloc[div_list[:int(LEN * 0.8)]]
    y_train = prices.iloc[div_list[:int(LEN * 0.8)]]
    X_test = features.iloc[div_list[int(LEN * 0.8):]]
    y_test = prices.iloc[div_list[int(LEN * 0.8):]]
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = generate_train_and_test(features.values, prices.values)

from sklearn.metrics import r2_score


def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""
    score = r2_score(y_true, y_predict)
    return score
