from .feature_select import feature_selection_with_lasso_cv
from .get_basis import get_double_basis
from .model_training import train_model2
from .utils import get_array


def linearmodel(X_dict, mapping, Y,l1_ratio):
    """
    使用线性模型进行特征选择、扩展和训练。

    参数:
        X_dict (dict): 输入的特征字典，其中键是样本编号，值是特征列表。
        mapping (dict): 特征映射字典，用于将基函数名映射到表达式。
        Y (numpy.ndarray): 目标变量数组。

    返回:
        model: 训练好的线性模型对象。
    """
    lenx=len(X_dict)
    kk=5
    if lenx<5:
        kk=2
    # 使用 LASSO 交叉验证进行特征选择，返回选择后的特征字典
    X_con = feature_selection_with_lasso_cv(X_dict, Y, n_splits=kk)
    print("LASSO选出的单变量基函数如下：")
    for key, value in X_con.items():
        print(f"{key}: shape = {value.shape}, values (前5个) = {value[:5]}")

    # 使用选择的特征字典扩展为双基函数，并更新映射字典
    all_basis, mapping = get_double_basis(X_con, X_dict, mapping)

    # 将特征字典转换为数组形式，准备训练数据
    X_data = get_array(all_basis).T
    # # 计算均值和标准差
    # mean_Y = np.mean(Y)
    # std_Y = np.std(Y, ddof=1)  # 使用样本标准差，ddof=1

    # 计算归一标准差
    # normalized_std = std_Y / mean_Y if mean_Y != 0 else np.nan

    # if normalized_std<0.8:
    #     k1=1.1
    #     k2=1.21
    # else:
    #     k1=1.2
    #     k2=1.44

    # # 训练第一个线性模型并计算 MAPE
    # coef1, mape = train_model1(X_data, Y, l1_ratio)

    # 训练第二个线性模型并计算 NRMSE
    coef2, nrmse = train_model2(X_data, Y, l1_ratio)



    # # 根据给定条件进行选择
    # if mape < 0.05:
    #     return coef1, mapping  # 采用 MAPE 生成的模型
    # elif 0.05 <= mape < 0.1:
    #     if nrmse < mape:
    #         return coef2, mapping  # 采用 NRMSE 生成的模型
    #     else:
    #         return coef1, mapping  # 采用 MAPE 生成的模型
    # elif 0.1 <= mape < 0.2:
    #     if nrmse < k1 * mape:
    #         return coef2, mapping  # 采用 NRMSE 生成的模型
    #     else:
    #         return coef1, mapping  # 采用 MAPE 生成的模型
    # elif 0.2 <= mape < 0.3:
    #     if nrmse < k2 * mape:
    #         return coef2, mapping  # 采用 NRMSE 生成的模型
    #     else:
    #         return coef1, mapping  # 采用 MAPE 生成的模型
    # else:  # mape >= 0.3
    #     return coef2, mapping  # 采用 NRMSE 生成的模型
    return coef2, mapping