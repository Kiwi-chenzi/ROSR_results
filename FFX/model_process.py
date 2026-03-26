from .get_basis import get_single_basis
from .linear import linearmodel


def model_process(X_1, Y_1, l1_ratio=0.222):
    """
    对输入的数据进行处理，并根据选择的模型类型进行线性或非线性模型的训练。

    参数:
        X_1 : 输入的特征数据。X的维度是M*N，M是样本数，N是特征数
        Y_1 : 输入的目标变量数据。
        l1_ratio=0.222:正则化率

    返回:
        tuple: 返回以下两个元素的元组
        - model (object): 训练好的模型对象，可能是线性或非线性模型。
        - mapping (dict): 特征映射的字典，用于将原始特征映射到新的特征空间。
    """

    X_dict, mapping = get_single_basis(X_1.T)  # 获取特征的单一基函数表示及其映射
    coef, mapping = linearmodel(X_dict, mapping, Y_1,l1_ratio)

    return coef, mapping  # 返回训练好的模型、特征映射字典
