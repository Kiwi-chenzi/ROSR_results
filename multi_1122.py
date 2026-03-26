#
# 主程序 四种参数 加输出
#
import numpy as np
import h5py
import warnings
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import os
import time
import tables
from FFX import model_process, predict
warnings.filterwarnings("ignore", category=UserWarning)
from multiprocessing import Pool
import tables


# 用户配置
# 可选目标 ["position", "velocity", "pressure", "density"]
targets = ["position"]
#targets = ["position", "velocity"]

# 参数
Np = 1055812
angle = np.arange(0, 61)
angle_test = [20.2,30.8,43.6,53.25]
hh_test = np.array(angle_test)
# angle_train = np.setdiff1d(angle, angle_test)
angle_train = np.arange(0, 61)
time_steps = range(10,15)  # 总共41个时刻
regression_method = "FFX"  # 选择 "GPR" 或 "FFX"

output_folder =regression_method + "_predict_plot0717" # 输出文件夹名
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"文件夹 '{output_folder}' 已创建。")
else:
    print(f"文件夹 '{output_folder}' 已存在。")

T_predict = 0
T_train = 0
T_PCA = 0
T_DATA = 0

# def calculate_ek(original, reconstructed, eps=1e-8):
#     """计算能量保持率 E_k"""
#     original = np.array(original)
#     reconstructed = np.array(reconstructed)
#
#     numerator = np.sum((original - reconstructed) ** 2, axis=(0))
#     denominator = np.sum(original ** 2, axis=(0)) + eps
#
#     return np.mean(1 - (numerator / denominator))


# 主循环
for t in time_steps:
    start_time = time.time()
    print(f"处理时刻 {t}...")

    with tables.open_file('data_61x1055812x8.h5', 'r') as f:
        data_read_start_time = time.time()
        dataset_name = f"t{t:02d}"  # 动态生成类似 't00', 't01' 的名称
        dataset = getattr(f.root, dataset_name).read().astype(np.float32)  # 动态访问节点
        print(f"  加载数据集 {dataset_name} 完成。")

        # 数据拆分
        positions = dataset[..., :3]  # x, y, z
        # velocities = dataset[..., 3:6]  # vx, vy, vz
        # pressures = dataset[..., 6]  # pressure
        # densities = dataset[..., 7]  # density

        data_read_end_time = time.time()
        data_read_duration = data_read_end_time - data_read_start_time
        print(f"  数据读取耗时: {data_read_duration:.4f}秒")
        T_DATA += data_read_duration

        predictions = {}

        for target in targets:
            target_start_time = time.time()  # 记录目标处理开始时间
            print(f"  处理目标: {target}...")
            if target == "position":
                data_train = np.concatenate([positions[angle] for angle in angle_train],
                                            axis=0).reshape(len(angle_train), -1, 3).transpose(0, 1, 2).reshape(len(angle_train), -1)
                # data_test = np.concatenate([positions[angle] for angle in angle_test],
                #                            axis=0).reshape(len(angle_test), -1, 3).transpose(0, 1, 2).reshape(len(angle_test), -1)
            # elif target == "velocity":
            #     data_train = np.concatenate([velocities[angle] for angle in angle_train],
            #                                 axis=0).reshape(len(angle_train), -1, 3).transpose(0, 1, 2).reshape(len(angle_train), -1)
            #     data_test = np.concatenate([velocities[angle] for angle in angle_test],
            #                                axis=0).reshape(len(angle_test), -1, 3).transpose(0, 1, 2).reshape(len(angle_test), -1)
            # elif target == "pressure":
            #     data_train = pressures[angle_train]
            #     data_test = pressures[angle_test]
            # elif target == "density":
            #     data_train = densities[angle_train]
            #     data_test = densities[angle_test]
            # else:
            #     continue

            # PCA降维
            pca_start_time = time.time()
            print(f"    执行 PCA...")
            # 先全部训练，得到所有成分和对应方差比例
            pca_full = PCA(n_components=None)
            pca_full.fit(data_train)
            explained_var_ratio_cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            # 找出累计能量保持率超过99%的最小主成分数
            n_components_99 = np.searchsorted(explained_var_ratio_cumsum, 0.99) + 1
            print(f"    选取主成分数以保证能量保持率≥99%: {n_components_99}")
            # 用选出的主成分数重新训练PCA
            pca = PCA(n_components=n_components_99)
            score = pca.fit_transform(data_train)
            mu = pca.mean_
            coeff = pca.components_
            pca_end_time = time.time()
            T_PCA += pca_end_time - pca_start_time
            weights = {f"PC{i + 1}": pca.explained_variance_ratio_[i] for i in
                       range(len(pca.explained_variance_ratio_))}
            print(weights)
            print(f"    PCA 完成，耗时 {pca_end_time - pca_start_time:.4f} 秒")
            print(f"    PCA 能量保持率（explained_variance_ratio_）: {np.sum(pca.explained_variance_ratio_):.6f}")

            # # 数据进行SVD分解
            # U, S, Vt = np.linalg.svd(data_train, full_matrices=False)
            #
            # # 选择前 k 个奇异值和对应的奇异向量进行降维
            # k = 30  # 选择前 k 个主成分
            # U_k = U[:, :k]
            # S_k = np.diag(S[:k])
            # Vt_k = Vt[:k, :]
            #
            # # Step 4: 降维后的数据（得分矩阵）
            # score = np.dot(U_k, S_k)
            #
            # # Step 5: 重构数据
            # reconstructed = np.dot(score, Vt_k)
            #
            # # 计算能量保持率 E_k
            # ek = calculate_ek(data_train, reconstructed)
            # print("降维维度", k)
            # print(f"    能量保持率 E_k: {ek:.6f}")



            if regression_method == "GPR":
                # 高斯过程回归
                print(f"    执行高斯过程回归...")
                GPR_time = 0
                newY = np.zeros((len(angle_test), score.shape[1]))
                for jj in range(score.shape[1]):
                    gpr = GaussianProcessRegressor(
                        kernel=C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)))
                    gpr.fit(angle_train.reshape(-1, 1), score[:, jj])
                    GPR_start_time = time.time()
                    newY[:, jj] = gpr.predict(hh_test.reshape(-1, 1))
                    GPR_end_time = time.time()
                    GPR_time += GPR_end_time - GPR_start_time
                T_predict += GPR_time
                print(f"    高斯过程预测时间，耗时 {GPR_time:.4f} 秒")

            elif regression_method == "FFX":
                #print(f"    执行 FFX 过程回归...")
                ffx_time = 0
                ffx_time_train= 0
                newY = np.zeros((len(angle_test), score.shape[1]))
                for jj in range(score.shape[1]):
                    ffx_start_train_time = time.time()
                    coef, mapping = model_process(angle_train.reshape(-1, 1), score[:, jj], 0.2)
                    #print("coef系数是:",coef)
                    for key, value in mapping.items():
                        print(f"{key}: {value}")
                    print("非零项如下：")
                    for key, c in zip(mapping.keys(), coef):
                        if np.abs(c) > 1e-12:
                         print(f"{c:.6f} × {mapping[key]}")
                    print("#################################################################################"
                          "#################################################################################")
                    ffx_start_time = time.time()
                    newY[:, jj] = predict(coef, mapping, hh_test.reshape(-1, 1))
                    ffx_end_time = time.time()
                    ffx_time += ffx_end_time - ffx_start_time
                    ffx_time_train +=ffx_start_time - ffx_start_train_time
                T_predict += ffx_time
                T_train += ffx_time_train
                print(f"    FFX 过程预测时间，耗时 {ffx_time:.4f} 秒")
                print(f"    FFX 过程训练时间，耗时 {ffx_time_train:.4f} 秒")


            # 重构预测数据
            Data_rctY = mu + newY @ coeff
            # Data_rctY = newY @ Vt_k
            predictions[target] = Data_rctY.reshape(len(angle_test), -1, 3 if target in ["position", "velocity"] else 1)
            target_end_time = time.time()  # 记录目标处理结束时间
            print(f"    处理目标 {target} 完成，耗时 {target_end_time - target_start_time:.2f} 秒")

        # 输出 VTK 文件
        vtk_start_time = time.time()  # 记录VTK输出开始时间
        print(f"  输出 VTK 文件...")
        for angle_idx, angle in enumerate(angle_test):
            vtk_filename = os.path.join(output_folder, f"FFX_predict_{angle:.2f}_{t:02d}.vtk")
            with open(vtk_filename, "w") as vtk_file:
                # 写入 VTK 文件头
                vtk_file.write("# vtk DataFile Version 4.0\n")
                vtk_file.write(f"timeStep= {t} \ttime= {t * 0.00000005:.6f}\n")  # time步和时间信息
                vtk_file.write("ASCII\n")
                vtk_file.write("DATASET POLYDATA\n")

                # 输出粒子位置
                vtk_file.write(f"POINTS {Np} float\n")
                position_data = predictions["position"][angle_idx]  # 使用预测数据
                position_data = position_data.astype(np.float32)
                for i in range(Np):
                    vtk_file.write(f"{position_data[i, 0]} {position_data[i, 1]} {position_data[i, 2]}\n")

                # 输出速度信息
                if "velocity" in predictions:
                    vtk_file.write(f"POINT_DATA {Np}\n")
                    vtk_file.write("SCALARS velocity float 3\n")
                    vtk_file.write("LOOKUP_TABLE default\n")
                    velocity_data = predictions["velocity"][angle_idx]
                    velocity_data = velocity_data.astype(np.float32)
                    for i in range(Np):
                        vtk_file.write(f"{velocity_data[i, 0]} {velocity_data[i, 1]} {velocity_data[i, 2]}\n")

                # 输出密度信息
                if "density" in predictions:
                    vtk_file.write("SCALARS rho float 1\n")
                    vtk_file.write("LOOKUP_TABLE default\n")
                    density_data = predictions["density"][angle_idx]
                    density_data = density_data.astype(np.float32)
                    for i in range(Np):
                        vtk_file.write(f"{density_data[i, 0]}\n")

                # 输出压力信息
                if "pressure" in predictions:
                    vtk_file.write("SCALARS pressure float 1\n")
                    vtk_file.write("LOOKUP_TABLE default\n")
                    pressure_data = predictions["pressure"][angle_idx]
                    pressure_data = pressure_data.astype(np.float32)
                    for i in range(Np):
                        vtk_file.write(f"{pressure_data[i, 0]}\n")

        vtk_end_time = time.time()  # 记录VTK输出结束时间
        print(f"    VTK 输出 完成，耗时 {vtk_end_time - vtk_start_time:.2f} 秒")

    end_time = time.time()  # 记录整个时刻处理结束时间
    print(f"时刻 {t} 的处理完成，耗时 {end_time - start_time:.2f} 秒")
    print(f"预测耗时{T_predict}秒")

print("###########################################")
print("所有时刻的处理完成。")
print(f"数据读取时间{T_DATA}秒")
print(f"PCA耗时{T_PCA}秒")
print(f"总训练耗时{T_predict}秒")
print(f"总预测耗时{T_predict}秒")

