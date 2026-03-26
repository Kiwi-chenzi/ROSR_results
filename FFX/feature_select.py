import math

from sklearn.linear_model import LassoCV
import numpy as np


def dictionary_to_matrix(basis_dict):
    """
    将基函数字典转换为特征矩阵。

    Args:
        basis_dict (dict): 基函数字典，其中键是特征名称，值是对应的特征数组。

    Returns:
        np.ndarray: 形状为 (n_samples, n_features) 的特征矩阵。
        list: 特征名称列表，顺序与矩阵中的列对应。
    """
    # 提取特征名称和对应的数组
    feature_names = list(basis_dict.keys())
    feature_matrix = np.array(list(basis_dict.values())).T

    return feature_matrix, feature_names


def feature_selection_with_lasso_cv(basis_dict, Y, n_splits=5):
    """
    使用5折交叉验证和拉索回归进行特征选择，并返回选择后的特征基函数字典和新的映射字典。

    Args:
        basis_dict (dict): 基函数字典。
        Y (ndarray): 标签数据。
        n_splits (int): 交叉验证的折数，默认是5。

    Returns:
        dict: (选择后的特征基函数字典, 选择后的特征表达式映射字典)。
    """
    # 将基函数字典转换为矩阵
    X_matrix, feature_names = dictionary_to_matrix(basis_dict)

    # 初始化拉索回归模型，使用交叉验证选择正则化参数
    lasso = LassoCV(cv=n_splits, random_state=42)

    # 训练拉索回归模型
    lasso.fit(X_matrix, Y)

    # 获取非零系数对应的特征及其系数的绝对值
    feature_importance = np.abs(lasso.coef_)
    feature_importance_indices = np.argsort(feature_importance)[::-1]  # 按重要性排序

    # 计算需要选择的特征数量：√(basis_dict)
    num_select = math.ceil(2*math.sqrt(len(basis_dict)))

    # 选择前num_select个特征
    selected_indices = feature_importance_indices[:num_select]
    selected_features = [feature_names[i] for i in selected_indices]

    # 创建新的特征基函数字典
    selected_basis = {name: basis_dict[name] for name in selected_features}

    return selected_basis
