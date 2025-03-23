EEG数据预处理与EA变换
项目概述
本项目主要用于对EEG（脑电图）数据进行预处理，包括数据重塑、数据合并与拆分，并对数据进行EA变换。最终将处理后的数据保存为.npy文件格式。

项目结构
EEG_EA.py
DiffCopyInsert
EEG_EA.py：包含所有的功能函数和主程序逻辑。
主要功能
数据重塑：将输入的数据从形状 (num_samples, num_time_samples, num_channels) 转换为 (num_samples, num_channels, num_time_samples)，并在需要时转换回原形状。
数据合并与拆分：合并多个特征数据和标签数据数组，并在后续步骤中将它们拆分回原始形状。
数据下采样：将输入的时间样本数量统一为指定的新值，保证数据的一致性。
EA变换：计算加权协方差矩阵，确保其为正定矩阵，然后计算协方差矩阵的逆平方根，对输入数据进行EA变换。
数据保存：将处理后的数据保存为.npy文件格式。
使用说明
环境依赖
Python 3.x
numpy
scipy
运行代码
确保您的数据文件位于prepro文件夹中，文件命名格式为prepro_subject_{i}.mat，其中{i}为subject id。
运行EEG_EA.py文件。
python EEG_EA.py
DiffCopyInsert
数据格式
输入数据文件为.mat格式，其中必须包含X（特征数据）和y（标签数据）两个变量。
特征数据X的形状应为(num_samples, num_time_samples, num_channels)。
标签数据y的形状应为(1, num_samples)，且值应为0或1。
注意事项
代码中假设每个样本的通道数为18。如果您的数据通道数不同，请修改reshape_data和EA函数中的num_channels参数。
请确保输入的.mat文件中的X和y变量的形状与代码中的假设一致，否则可能会引发异常。
代码示例
# 读取MAT文件
mat_data = sio.loadmat(mat_file_path)

# 获取标签数据
y = mat_data.get('y', None)

# 获取特征数据
X = mat_data.get('X', None)

# 对X数据进行下采样处理
X = downsample_data(X, new_num_time_samples=2000)

# 将特征数据和标签数据加入列表
X_data_arrays.append(X)
y_label_arrays.append(y)

# 合并X数据
merged_X, sample_counts = merge_data_arrays(X_data_arrays)
reshaped_X = reshape_data(merged_X, num_channels=18)

# 合并y数据
merged_y = merge_label_arrays(y_label_arrays)

# EA处理
XEA = EA(reshaped_X, merged_y)

# 转换回原形状并拆分
reshaped_XEA_back = reshape_data_back(XEA)
split_XEA = split_data_array(reshaped_XEA_back, sample_counts)
split_yEA = split_label_array(merged_y, sample_counts)

# 保存处理后的数据
for i in range(start_id, end_id + 1):
    y_name = f'y_subject{i}.npy'
    X_name = f'X_subject{i}.npy'
    y_path = os.path.join(target_folder, y_name)
    X_path = os.path.join(target_folder, X_name)
    np.save(y_path, split_yEA[i - start_id])
    np.save(X_path, split_XEA[i - start_id])
DiffCopyInsert
贡献说明
欢迎各位开发者对本项目进行改进和贡献。请确保代码风格与现有代码一致，并添加相应的文档。

联系方式
如有任何问题或建议，请通过以下方式联系：

邮箱：your_email@example.com
GitHub：https://github.com/your_github_username/your_repository
请根据实际情况调整README中的内容，以适应您的项目需求。
