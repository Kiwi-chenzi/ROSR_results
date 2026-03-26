import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from FFX import model_process, predict

np.random.seed(42)

# 生成100个样本，每个样本有一个原始特征
n_samples = 100
X = np.random.rand(n_samples, 1) * 10

# 生成二次多项式特征，包括X、X²（或更高次）
degree = 2
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# 定义真实系数和生成Y
true_coeff = np.array([5, 2, 3])  # 截距、X系数、X²系数
noise = np.random.randn(n_samples, 1) * 0.5  # 添加噪声
Y = np.dot(X_poly, true_coeff).reshape(-1, 1) + noise

# 将数据分成训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)

# 训练模型并预测
coef, mapping = model_process(X_train, Y_train, 0.95)
y_pred = predict(coef, mapping, X_test)
print(f"Y_pred:", y_pred)
print(f"Y_test:", Y_test)
