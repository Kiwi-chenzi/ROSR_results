import numpy as np


def get_single_basis(X):
    """为给定数据生成单变量和双变量特征。

    Args:
        X (ndarray): 原始特征数据。

    Returns:
        get_single_basis: 包含新特征的字典。
    """
    if X.ndim == 1:
        X = X[np.newaxis,:]  # 将其扩展为1行n列的格式
    m = len(X)
    n = len(X[0])
    one_array = np.ones(n)
    single_basis = {}  # 创建一个空字典
    expr_mapping = {'x00': 1}  # 创建一个新字典来存储基函数表达式与原始输入变量的映射关系
    single_basis["x00"] = one_array  # 加入常数项
    # expr_mapping["x00"] = "1"  # 常数项没有对应的原始输入变量
    for i1 in range(0, m):
        var_name = f"X{i1}"
        single_basis[f"{var_name}_1"] = X[i1]
        expr_mapping[f"{var_name}_1"] = f"X[{i1}]"
        basis_value = single_basis[f"{var_name}_1"]
        basis_mapping = expr_mapping[f"{var_name}_1"]

        single_basis[f"{var_name}_0.5"] = np.sqrt(np.abs(np.maximum(0, basis_value)))
        single_basis[f"{var_name}_2"] = basis_value ** 2
        single_basis[f"{var_name}_3"] = basis_value ** 3
        single_basis[f"{var_name}_reciprocal"] = np.where(basis_value==0, 0, 1 / (basis_value+1e-50))
        single_basis[f"{var_name}_log"] = np.log(np.where(basis_value > 0, basis_value, 1))
        # single_basis[f"{var_name}_max"] = np.maximum(0, basis_value)
        # single_basis[f"{var_name}_min"] = np.minimum(0, basis_value)
        # single_basis[f"{var_name}_exp0.5x"] = np.exp(np.clip(0.5 * basis_value, -50, 50))
        # single_basis[f"{var_name}_sin"] = np.sin(basis_value)
        # single_basis[f"{var_name}_cos"] = np.cos(basis_value)
        # single_basis[f"{var_name}_exp0.2x"] = np.exp(np.clip(0.2 * basis_value, -50, 50))
        # single_basis[f"{var_name}_exp_neg_sq"] = np.exp(-basis_value ** 2)
        # single_basis[f"{var_name}_exp_neg_abs"] = np.exp(-np.abs(basis_value))


        # 更新表达式映射
        expr_mapping[f"{var_name}_0.5"] = f"np.sqrt(np.abs(np.maximum(0, {basis_mapping})))"
        expr_mapping[f"{var_name}_2"] = f"({basis_mapping})**2"
        expr_mapping[f"{var_name}_3"] = f"({basis_mapping})**3"
        expr_mapping[f"{var_name}_reciprocal"] = f"np.where({basis_mapping} ==0, 0, 1 /( {basis_mapping}+1e-50))"
        expr_mapping[f"{var_name}_log"] = f"np.log(np.where({basis_mapping} > 0, {basis_mapping}, 1))"
        # expr_mapping[f"{var_name}_max"] = f"np.maximum(0, {basis_mapping})"
        # expr_mapping[f"{var_name}_min"] = f"np.minimum(0, {basis_mapping})"
        #expr_mapping[f"{var_name}_exp0.5x"] = f"np.exp(np.clip(0.5 * {basis_mapping}, -50, 50))"
        #expr_mapping[f"{var_name}_sin"] = f"np.sin({basis_mapping})"
        # expr_mapping[f"{var_name}_cos"] = f"np.cos({basis_mapping})"
        # expr_mapping[f"{var_name}_exp0.2x"] = f"np.exp(np.clip(0.2 * {basis_mapping}, -50, 50))"
        # expr_mapping[f"{var_name}_exp_neg_sq"] = f"np.exp(-({basis_mapping})**2)"
        # expr_mapping[f"{var_name}_exp_neg_abs"] = f"np.exp(-np.abs({basis_mapping}))"

    return single_basis, expr_mapping


def get_double_basis(single_basis, cut_single_basis, expr_mapping=None):
    """为给定的特征生成双变量交互特征。

    Args:
        single_basis (dict): 候选单变量基函数字典。
        cut_single_basis:单变量基函数字典
        expr_mapping:函数映射字典

    Returns:
        dict: 包含单变量和双变量交互特征的字典。

    """
    Q2 = cut_single_basis.copy()  # 创建一个字典来存储双变量基函数
    keys = list(single_basis.keys())  # 获取所有单变量基函数的键
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            # 创建新的双变量基函数键
            new_key = f"{keys[i]}*{keys[j]}"
            # 计算新的双变量基函数值
            new_value = single_basis[keys[i]] * single_basis[keys[j]]
            # 将新的键值对添加到字典Q中
            Q2.update({new_key: new_value})
            # 更新映射字典
            expr1 = expr_mapping.get(keys[i], None)
            expr2 = expr_mapping.get(keys[j], None)
            if expr1 is not None and expr2 is not None:
                expr_mapping[new_key] = f"({expr1})*({expr2})"
    return Q2, expr_mapping


def get_denominator(all_basis, y, expr_mapping):
    """引入分母项。

    Args:
        all_basis (dict): 目前的基函数字典
        y (any): y的值
        expr_mapping (dict): 一个字典，映射表达式的字符串表示。

    Returns:
        tuple: 包含更新后的基函数字典和表达式映射字典。
    """
    denominator = {}  # 用于存放分母项
    items = list(all_basis.items())[1:]  # 将items转换为列表并取除第一个元素之外的所有元素
    for key, value in items:
        new_key = key + "d"
        denominator[new_key] = value * y * -1
        # 更新映射字典
        expr = expr_mapping.get(key, None)
        if expr is not None:
            expr_mapping[new_key] = f"({expr})"
    all_basis.update(denominator)
    return all_basis, expr_mapping


def get_allbasis(X):
    single_basis, expr_mapping = get_single_basis(X)
    return single_basis, expr_mapping
