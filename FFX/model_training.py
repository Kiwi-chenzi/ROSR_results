import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from FFX import utils


def precompute_matrices(X):
    """
    预计算特征矩阵的转置与自身的点积矩阵，以便加速模型训练中的计算。

    参数:
        X (np.ndarray): 特征矩阵。

    返回:
        np.ndarray: 预计算的矩阵 (X^T * X)。
    """
    return np.dot(X.T, X)


def compute_alpha_max(X_unbiased, y_unbiased, l1_ratio):
    """
    计算 alpha_max，这是 ElasticNet 正则化参数的初始最大值，用于控制模型的复杂度。

    参数:
        X_unbiased (np.ndarray): 无偏特征矩阵。
        y_unbiased (np.ndarray): 无偏目标值向量。
        l1_ratio (float): L1 正则化的比例。

    返回:
        float: 计算得到的 alpha_max 值。
    """
    n_samples = X_unbiased.shape[0]
    vals = np.dot(X_unbiased.T, y_unbiased)
    vals = vals[~np.isnan(vals)]
    if vals.size > 0:
        alpha_max = np.abs(np.max(vals) / (n_samples * l1_ratio))
    else:
        alpha_max = 1.0  # 备用值，当没有有效计算结果时使用

    if np.isnan(alpha_max) or alpha_max <= 0:
        alpha_max = 1.0  # 设置默认值

    return alpha_max


def compute_eps():
    """
    计算机器精度的最小浮点数 eps，以避免在计算 log10(0) 时出现问题。

    返回:
        float: 计算得到的 eps 值。
    """
    return np.finfo(float).eps


def parse_feature_name(name):
    """
    解析特征名称，区分一般特征和特殊操作特征，并返回原始特征名称及操作类型。

    参数:
        name (str): 特征名称字符串。

    返回:
        tuple: 包含三个元素的元组:
        - original_feature (str): 原始特征名称。
        - operation (str): 应用于特征的操作类型，如 'sqrt', 'log', 'reciprocal', 'none' 等。
        - power (float): 应用于特征的幂次，默认为 1。
    """
    parts = name.split('_')

    if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
        # 一般特征 (如平方根、幂次)
        original_feature = f"{parts[0]}_1"
        operation = 'sqrt' if '.' in parts[1] else 'none'
        power = float(parts[1])
    elif len(parts) == 2 and parts[1] in ['log', 'reciprocal', 'max', 'min']:
        # 特殊操作 (如对数、倒数、最大值、最小值)
        original_feature = f"{parts[0]}_1"
        operation = parts[1]
        power = 1
    else:
        # 原始特征
        original_feature = name
        operation = 'none'
        power = 1

    return original_feature, operation, power


def generate_alphas(alpha_max, num_alphas, eps, split_ratio=0.25):
    """
    生成一组 alpha 值，用于 ElasticNet 正则化参数的路径搜索。在较小的区域更密集采样，初始部分较稀疏采样。

    参数:
        alpha_max (float): 最大的 alpha 值。
        num_alphas (int): 要生成的 alpha 值数量。
        eps (float): 一个小数值，用于避免 log10(0)。
        split_ratio (float): 用于密集采样的 alpha 范围比例。

    返回:
        np.ndarray: 生成的 alpha 值数组，从大到小排列。
    """
    if np.isnan(alpha_max) or alpha_max <= 0:
        raise ValueError(f"Invalid alpha_max: {alpha_max}. Check input data and calculations.")

    # 计算 alpha 的对数空间起点和终点
    st, fin = np.log10(alpha_max * eps), np.log10(alpha_max)

    # 确定不同区间的比例
    num_alphas_sparse = int(num_alphas * (1 - split_ratio))
    num_alphas_dense = num_alphas - num_alphas_sparse

    # 生成稀疏区间的 alpha 值
    alphas_sparse = np.logspace(st, fin, num=num_alphas_sparse)

    # 生成密集区间的 alpha 值
    dense_st = st + split_ratio * (fin - st)
    alphas_dense = np.logspace(dense_st, fin, num=num_alphas_dense)

    # 合并两个 alpha 数组并去重排序，从大到小排列
    alphas = np.unique(np.concatenate([alphas_sparse, alphas_dense]))[::-1]
    return alphas


def train_model1(X_1, Y_1, l1_ratio=0.22, max_num_bases=200):
    """
    使用 ElasticNet 进行路径学习，选择最佳 alpha 并训练模型。

    参数:
        X_2 (np.ndarray): 特征矩阵。
        y (np.ndarray): 目标变量。
        l1_ratio (float): L1 正则化的比例。
        max_num_bases (int): 模型中最大允许的基数数量。

    返回:
        sklearn.linear_model.ElasticNet: 训练好的模型。
    """
    X_2, Y, X_mean, X_std, Y_mean, Y_std = utils.unbiasedXy1(X_1, Y_1)
    alpha_max = compute_alpha_max(X_2, Y, l1_ratio)
    num_alphas = 666
    alphas = generate_alphas(alpha_max, num_alphas, 1e-10)
    model = None
    best_alpha = None
    best_mape = np.inf
    best_coefs = None
    early_stop_patience_counter = 0
    kf = KFold(n_splits=3)
    # print(f"Pathwise learn: begin. max_num_bases={max_num_bases}")

    nrmse_improvement_threshold = 0.0001  # NRMSE 改善阈值
    avg_nrmse_history = []  # 用于存储历史平均 NRMSE

    # 遍历所有生成的 alpha 值
    for i, alpha in enumerate(alphas):
        # if i==200:
        #     pass
        nrmse_scores = []
        num_bases_list = []

        # 进行 KFold 交叉验证
        for fold, (train_index, val_index) in enumerate(kf.split(X_2)):
            X_train, X_val = X_2[train_index], X_2[val_index]
            y_train, y_val = Y[train_index], Y[val_index]
            precomputed_matrix = precompute_matrices(X_train)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, precompute=precomputed_matrix,fit_intercept=False,
                               max_iter=8000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            nrmse = utils.calculate_mape(y_val, y_pred)
            nrmse_scores.append(nrmse)
            num_bases_list.append(np.count_nonzero(model.coef_))

        avg_mape = np.mean(nrmse_scores)
        avg_num_bases = np.mean(num_bases_list)

        # 根据已经训练的 alpha 数量调整 patience
        if i < 50:
            patience = 20
        elif i < 100:
            patience = 15
        else:
            patience = 10

        # 检查当前模型是否是最佳模型
        if avg_mape < best_mape:
            best_mape = avg_mape
            best_alpha = alpha
            best_coefs = model.coef_
            early_stop_patience_counter = 0
        else:
            early_stop_patience_counter += 1

        # 打印当前状态信息
        if (i + 1) % 5 == 0 or (i + 1) == len(alphas):
            print(f"alpha {i + 1}/{len(alphas)} ({alpha:.6e}): num_bases={avg_num_bases}, nrmse={avg_mape:.6f}")
            # if best_coefs is not None:
            #     # print(
            #     #     f"Current best alpha: {best_alpha}, best_num_bases={np.count_nonzero(best_coefs)}, "
            #     #     f"best_mape={best_mape:.6f}")

            # 每5步检查并更新历史 NRMSE
            avg_nrmse_history.append(avg_mape)
            if len(avg_nrmse_history) > 1:
                if abs(avg_nrmse_history[-1] - avg_nrmse_history[-2]) < nrmse_improvement_threshold:
                    # print(
                    #     f"Pathwise learn: Early stop because average NRMSE change is less than"
                    #     f" {nrmse_improvement_threshold} for two consecutive iterations")
                    break

        # 提前停止条件
        if early_stop_patience_counter >= patience:
            # print(f"Pathwise learn: Early stop because no improvement for {patience} consecutive iterations")
            break
        if avg_num_bases > max_num_bases:
            # print(f"Pathwise learn: Early stop because num bases > {max_num_bases}")
            break
        # if avg_mape <= 0.01 and avg_num_bases>0:
        #     print("Pathwise learn: Early stop because NRMSE reached the target value of 0.01")
        #     break

    # print(f"Best alpha: {best_alpha}, Best mape: {best_mape}")
    # if best_coefs is not None:
    #     # print(f"Num basis: {np.count_nonzero(best_coefs)}")

    best_coefs = utils.rebiasCoefs(best_coefs, X_mean, X_std, Y_mean, Y_std)
    return best_coefs,best_mape

def train_model2(X_1, Y_1, l1_ratio=0.22, max_num_bases=200):
    """
    使用 ElasticNet 进行路径学习，选择最佳 alpha 并训练模型。

    参数:
        X_2 (np.ndarray): 特征矩阵。
        y (np.ndarray): 目标变量。
        l1_ratio (float): L1 正则化的比例。
        max_num_bases (int): 模型中最大允许的基数数量。

    返回:
        sklearn.linear_model.ElasticNet: 训练好的模型。
    """
    X_2, Y, X_mean, X_std, Y_mean, Y_std = utils.unbiasedXy1(X_1, Y_1)
    alpha_max = compute_alpha_max(X_2, Y, l1_ratio)
    num_alphas = 666
    alphas = generate_alphas(alpha_max, num_alphas, 1e-10)
    model = None
    best_alpha = None
    best_nrmse = np.inf
    best_coefs = None
    early_stop_patience_counter = 0
    kf = KFold(n_splits=5)
    # print(f"Pathwise learn: begin. max_num_bases={max_num_bases}")

    nrmse_improvement_threshold = 0.0001  # NRMSE 改善阈值
    avg_nrmse_history = []  # 用于存储历史平均 NRMSE

    # 遍历所有生成的 alpha 值
    for i, alpha in enumerate(alphas):
        # if i==200:
        #     pass
        nrmse_scores = []
        num_bases_list = []

        # 进行 KFold 交叉验证
        for fold, (train_index, val_index) in enumerate(kf.split(X_2)):
            X_train, X_val = X_2[train_index], X_2[val_index]
            y_train, y_val = Y[train_index], Y[val_index]
            precomputed_matrix = precompute_matrices(X_train)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, precompute=precomputed_matrix,fit_intercept=False,
                               max_iter=8000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            nrmse = utils.nrmse(y_val, y_pred)
            nrmse_scores.append(nrmse)
            num_bases_list.append(np.count_nonzero(model.coef_))

        avg_nrmse = np.mean(nrmse_scores)
        avg_num_bases = np.mean(num_bases_list)

        # 根据已经训练的 alpha 数量调整 patience
        if i < 50:
            patience = 20
        elif i < 100:
            patience = 15
        else:
            patience = 10

        # 检查当前模型是否是最佳模型
        if avg_nrmse < best_nrmse:
            best_nrmse = avg_nrmse
            best_alpha = alpha
            best_coefs = model.coef_
            early_stop_patience_counter = 0
        else:
            early_stop_patience_counter += 1

        # 打印当前状态信息
        if (i + 1) % 5 == 0 or (i + 1) == len(alphas):
            # print(f"alpha {i + 1}/{len(alphas)} ({alpha:.6e}): num_bases={avg_num_bases}, nrmse={avg_nrmse:.6f}")
            # if best_coefs is not None:
            #     # print(
            #     #     f"Current best alpha: {best_alpha}, best_num_bases={np.count_nonzero(best_coefs)}, "
            #     #     f"best_nrmse={best_nrmse:.6f}")

            # 每5步检查并更新历史 NRMSE
            avg_nrmse_history.append(avg_nrmse)
            if len(avg_nrmse_history) > 1:
                if abs(avg_nrmse_history[-1] - avg_nrmse_history[-2]) < nrmse_improvement_threshold:
                    print(
                        f"Pathwise learn: Early stop because average NRMSE change is less than"
                        f" {nrmse_improvement_threshold} for two consecutive iterations")
                    break

        # 提前停止条件
        if early_stop_patience_counter >= patience:
            print(f"Pathwise learn: Early stop because no improvement for {patience} consecutive iterations")
            break
        if avg_num_bases > max_num_bases:
            print(f"Pathwise learn: Early stop because num bases > {max_num_bases}")
            break
        if avg_nrmse <= 0.01 and avg_num_bases>0:
            print("Pathwise learn: Early stop because NRMSE reached the target value of 0.01")
            break

    print(f"Best lambda: {best_alpha}, Best NRMSE: {best_nrmse}")

    best_coefs = utils.rebiasCoefs(best_coefs, X_mean, X_std, Y_mean, Y_std)
    if best_coefs is not None:
        print(f"Num basis: {np.count_nonzero(best_coefs)}")
    return best_coefs,best_nrmse



def get_consingle(w_pre, targets):
    """
    根据给定的预测值和目标值计算并筛选特征，用于构造线性回归模型。

    参数:
        w_pre (dict): 一个字典，包含特征的权重或基函数。
        targets (np.ndarray): 目标值向量。

    返回:
        dict: 筛选后的特征字典。
    """
    # 转换权重矩阵并进行转置
    X_consingle = utils.get_array(w_pre).transpose()

    # 对特征矩阵和目标值进行去偏处理
    X_consingle, y_unbiased, X_avgs, X_stds, y_avg, y_std = utils.unbiasedXy1(X_consingle, targets)

    # 使用线性回归拟合模型
    model_l1 = LinearRegression(fit_intercept=False)
    model_l1.fit(X_consingle, y_unbiased)

    # 预测目标值并计算 R2 分数
    y_pred = model_l1.predict(X_consingle)
    r21 = r2_score(y_pred, y_unbiased)
    # print(r21)

    # 计算 L1 正则化系数的绝对值
    l1_coefficients = np.abs(model_l1.coef_)

    # 根据三分之一的特征数量设定阈值
    threshold2 = int(len(w_pre) / 3)
    # print(f'STEP1D:number of con single basis: {threshold2}')

    # 评估特征选择的阈值
    cutoff2 = utils.eval_threshold(l1_coefficients, threshold2)

    # 根据阈值筛选出重要的特征
    composite_importances = {name: coeff for name, coeff in zip(w_pre.keys(), l1_coefficients)}
    filtered_composite_variables = {name: coeff for name, coeff in composite_importances.items() if coeff >= cutoff2}

    # 根据筛选出的特征返回新的字典
    dict2 = utils.filter_dict_by_keys(w_pre, filtered_composite_variables)
    return dict2
