import numpy as np
import math

import pandas as pd


def filter_dict_by_keys(dict_larger, dict_smaller):
    """
       从一个较大的字典中筛选出与较小字典中有相同键的键值对。

       输入:
           larger_dict (dict): 较大的字典，包含多个键值对。
           smaller_dict (dict): 较小的字典，用于提供筛选的键。

       输出:
           dict: 筛选后的字典，只包含与较小字典中键相同的键值对。
       """
    # 创建新字典，只包含与较小字典中键相同的键值对
    filtered_dict = {key: dict_larger[key] for key in dict_smaller if key in dict_larger}

    return filtered_dict


def eval_threshold(coefs, threshold):
    """
    根据系数排序并评估阈值。

    输入:
        coefs (array): 系数数组。
        threshold (int): 阈值索引。

    输出:
        float: 计算得到的阈值。
    """
    sorted_coefs = sorted(coefs, reverse=True)
    return sorted_coefs[min(threshold, len(sorted_coefs) - 1)]


def get_array(my_vars5):
    """
      将字典中的值转换为一个二维numpy数组。

      输入:
          variable_dict (dict): 一个字典，其值是等长度的数组。

      输出:
          ndarray: 从字典的值构成的二维数组。
      """
    list_of_arrays = [value for key, value in my_vars5.items()]
    # 将列表转换为二维数组
    two_dimensional_array = np.array(list_of_arrays)
    return two_dimensional_array


def unbiasedXy1(X6, Y6):
    """
     对特征和目标变量进行去偏和标准化处理。

     输入:
         X (ndarray): 特征数据。
         Y (ndarray): 目标数据。

     输出:
         tuple: 包含去偏和标准化后的特征和目标数据、平均值和标准差。
         :rtype: object
     """
    # 计算均值
    X_means = X6.mean(axis=0, keepdims=True)  # 保留行维度
    Y_means = Y6.mean(axis=0, keepdims=True)

    # 计算标准差,替换为1避免除零
    X_stds = X6.std(axis=0, ddof=1, keepdims=True)
    X_stds[X_stds == 0] = 1
    Y_stds = Y6.std(axis=0, ddof=1, keepdims=True)
    Y_stds[Y_stds == 0] = 1

    # 无偏标准化
    X_unbiased = (X6 - X_means) / X_stds
    Y_unbiased = (Y6 - Y_means) / Y_stds
    X_unbiased[:, 0] = 1
    return (np.squeeze(X_unbiased), np.squeeze(Y_unbiased), np.squeeze(X_means)
            , np.squeeze(X_stds), np.squeeze(Y_means), np.squeeze(Y_stds))


def rebiasCoefs(unbiased_coefs, X_means, X_stds, y_means, y_std):
    """
     将无偏系数重新调整为原始尺度。

     输入:
         unbiased_coefs (ndarray): 无偏系数。
         X_avgs (ndarray): 特征的平均值。
         X_stds (ndarray): 特征的标准差。
         y_avg (float): 目标变量的平均值。
         y_std (float): 目标变量的标准差。

     输出:
         ndarray: 调整后的系数。
     """
    n = len(unbiased_coefs)  # 这里 n 包括截距项和所有斜率项
    coefs5 = np.zeros(n, dtype=float)

    # 计算斜率项
    for j in range(1, n):  # 从1开始因为0是截距
        coefs5[j] = unbiased_coefs[j] * y_std / X_stds[j]

    # # 计算截距项
    coefs5[0] = unbiased_coefs[0] * y_std + y_means - np.sum(unbiased_coefs[1:] * X_means[1:] * (y_std / X_stds[1:]))
    return coefs5


def nrmse(Y, Y_pred):
    """
    计算归一化标准均方根误差（NRMSE）

    参数：
    Y_pred -- 1d array or list of floats -- 预测值
    Y -- 1d array or list of floats -- 真实值

    返回：
    nrmse -- float -- 归一化标准均方根误差
    """
    # 将输入转换为numpy数组
    Y_pred = np.asarray(Y_pred)
    Y = np.asarray(Y)

    # 计算真实值的最小值和最大值
    min_y = np.min(Y)
    max_y = np.max(Y)

    # 计算 y 的范围
    y_range = max_y - min_y

    # 检查 y 的范围是否为 0，如果是，则直接返回 0
    if y_range == 0:
        return 0.0

    # 计算标准均方根误差（RMSE）
    mse = np.mean((Y_pred - Y) ** 2)
    rmse = math.sqrt(mse)

    # 归一化处理
    nrmse1 = rmse / y_range

    return nrmse1


def calculate_mape(Y, Y_pred):
    """
    计算平均绝对百分比误差（MAPE）。

    参数:
    Y (array-like): 实际值的一维数组或列表。
    Y_pred (array-like): 预测值的一维数组或列表。

    返回:
    float: MAPE 值，百分比形式的平均绝对误差。
    """
    Y = np.array(Y)
    Y_pred = np.array(Y_pred)

    # 确保实际值不为零，以避免除以零的情况
    if np.any(Y == 0):
        raise ValueError("实际值不能为零，以避免除以零的情况。")

    # 计算绝对百分比误差
    abs_percentage_error = np.abs((Y - Y_pred) / Y)

    # 计算 MAPE
    mape = np.mean(abs_percentage_error)

    return mape


def construct_matrix(mapping, X):
    """
    根据给定的映射字典和 X 矩阵构造新的矩阵

    Args:
        mapping (dict): 一个字典,将基函数名映射到表达式
        X (ndarray): 原始 X 矩阵

    Returns:
        ndarray: 构造的新矩阵
    """
    num_samples = len(X[0]) # 样本数
    num_features = len(mapping)  # 基函数数量
    new_matrix = np.zeros((num_features,num_samples))

    # 设置常数项为 1
    new_matrix[0] = 1

    # 构造分子项
    for i, key in enumerate(list(mapping.keys())[1:], start=1):
        expr = mapping[key]
        new_matrix[i] = eval(expr)

    return new_matrix


def read_data(x_path,y_path):
    datax_df = pd.read_csv(x_path, header=None)  # 确保没有列名
    datay_df = pd.read_csv(y_path, header=None)  # 确保没有列名
    datax_ndarray = datax_df.values  # 转换为ndarray
    datay_ndarray = datay_df.values  # 转换为ndarray
    datay_ndarray = datay_ndarray.squeeze()
    return datax_ndarray,datay_ndarray


def read_coef_as_ndarray(file_path):
    coef_df = pd.read_csv(file_path, header=None)  # 确保没有列名
    coef_ndarray = coef_df.values  # 转换为ndarray
    coef_ndarray=coef_ndarray.squeeze()
    return coef_ndarray


def read_mapping_as_dict(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 仅分割第一个逗号，避免多余逗号影响
            key_value = line.strip().split(',', 1)
            if len(key_value) == 2:  # 确保行能正确分割
                key, value = key_value
                mapping[key] = value
            else:
                print(f"无法处理的行: {line.strip()}")  # 打印出错行，便于调试
    return mapping

def clean_dataframe(df):
    # 使用applymap将所有值检查并转换
    return df.applymap(lambda x: 0 if (pd.isna(x) or x == '' or not isinstance(x, (float, np.float64))) else x)
