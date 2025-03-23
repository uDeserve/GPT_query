# @File    : EEG_EA.py
import os
import numpy as np
import scipy.io as sio
from scipy.linalg import fractional_matrix_power
from scipy.signal import resample

# 定义reshape_data函数
def reshape_data(data, num_channels):
    """
    将数据从形状 (num_samples, num_time_samples, num_channels) 转换为
    (num_samples, num_channels, num_time_samples)

    Parameters
    ----------
    data : numpy array
        数据，形状为 (num_samples, num_time_samples, num_channels)
    num_channels : int
        数据中每个样本的通道数

    Returns
    -------
    reshaped_data : numpy array
        重塑后的数据，形状为 (num_samples, num_channels, num_time_samples)
    """
    # 获取数据的三个维度
    num_samples, num_time_samples, original_num_channels = data.shape

    # 确保提供的 num_channels 与数据的原始通道数一致
    if num_channels != original_num_channels:
        raise ValueError("提供的通道数与数据的原始通道数不一致")

    # 重塑数据
    reshaped_data = data.transpose(0, 2, 1)

    return reshaped_data
# 定义reshape_data_back函数
def reshape_data_back(data):
    """
    将数据从形状 (num_samples, num_channels, num_time_samples) 转换回
    (num_samples, num_time_samples , num_channels)

    Parameters
    ----------
    data : numpy array
        数据，形状为 (num_samples, num_channels, num_time_samples)

    Returns
    -------
    reshaped_data : numpy array
        重塑后的数据，形状为 (num_samples, num_time_samples, num_channels)
    """
    # 确保输入数据的形状是三维的
    if len(data.shape) != 3:
        raise ValueError("输入数据必须是三维数组，形状为 (num_samples, num_channels, num_time_samples)")

    # 获取原始数据的形状
    num_samples, num_channels, num_time_samples = data.shape

    # 重塑数据
    reshaped_data = data.transpose(0, 2, 1)
    return reshaped_data
# 定义merge_data_arrays函数
def merge_data_arrays(data_arrays):
    """
    合并多个形状为 (num_samples, num_channels, num_time_samples) 的数组，
    使他们按照第一个维度顺序向后排列。

    Parameters
    ----------
    data_arrays : list of numpy arrays
        要合并的数组列表，每个数组的形状必须是 (num_samples, num_channels, num_time_samples)

    Returns
    -------
    merged_array : numpy array
        合并后的数组，形状为 (total_num_samples, num_channels, num_time_samples)
    sample_counts : list of int
        每个子数组的 num_samples 计数，用于后续的拆分
    """
    # 检查输入是否为空
    if not data_arrays:
        raise ValueError("输入的数据数组列表不能为空。")

    # 获取第一个数组的 num_channels 和 num_time_samples
    num_channels = data_arrays[0].shape[1]
    num_time_samples = data_arrays[0].shape[2]

    # 检查所有数组的 num_channels 和 num_time_samples 是否一致
    for index, data_array in enumerate(data_arrays):
        if data_array.shape[1] != num_channels or data_array.shape[2] != num_time_samples:
            raise ValueError(f"数组索引 {index} 的形状为 {data_array.shape}，与第一个数组的 num_channels ({num_channels}) 和 num_time_samples ({num_time_samples}) 不一致。")

    # 使用 np.concatenate 合并数组
    merged_array = np.concatenate(data_arrays, axis=0)

    # 记录每个子数组的 num_samples
    sample_counts = [data_array.shape[0] for data_array in data_arrays]

    return merged_array, sample_counts


# 定义split_data_array函数
def split_data_array(merged_array, sample_counts):
    """
    将合并后的数组拆分回原来的多个数组。

    Parameters
    ----------
    merged_array : numpy array
        合并后的数组，形状为 (total_num_samples, num_channels, num_time_samples)
    sample_counts : list of int
        每个子数组的 num_samples 计数，用于拆分

    Returns
    -------
    split_arrays : list of numpy arrays
        拆分后的数组列表，每个数组的形状为 (num_samples, num_channels, num_time_samples)
    """
    split_arrays = []
    start_index = 0
    for count in sample_counts:
        end_index = start_index + count
        split_arrays.append(merged_array[start_index:end_index])
        start_index = end_index
    return split_arrays

# 定义merge_label_arrays函数
def merge_label_arrays(label_arrays):
    """
    合并多个形状为 (1, num_samples) 的标签数组，
    使他们按照第二个维度顺序向后排列。

    Parameters
    ----------
    label_arrays : list of numpy arrays
        要合并的标签数组列表，每个数组的形状必须是 (1, num_samples)

    Returns
    -------
    merged_labels : numpy array
        合并后的标签数组，形状为 (1, total_num_samples)
    """
    # 检查输入是否为空
    if not label_arrays:
        raise ValueError("输入的标签数组列表不能为空。")

    # 使用 np.concatenate 合并标签数组
    merged_labels = np.concatenate(label_arrays, axis=1)
    return merged_labels

# 定义split_label_array函数
def split_label_array(merged_labels, sample_counts):
    """
    将合并后的标签数组拆分回原来的多个数组。

    Parameters
    ----------
    merged_labels : numpy array
        合并后的标签数组，形状为 (1, total_num_samples)
    sample_counts : list of int
        每个子数组的 num_samples 计数，用于拆分

    Returns
    -------
    split_labels : list of numpy arrays
        拆分后的标签数组列表，每个数组的形状为 (1, num_samples)
    """
    split_labels = []
    start_index = 0
    for count in sample_counts:
        end_index = start_index + count
        split_labels.append(merged_labels[:, start_index:end_index])
        start_index = end_index
    return split_labels

# 定义EA函数
def EA(x, y):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    y : numpy array
        labels of shape (1, num_samples), values are 0 or 1

    Returns
    -------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    # 获取多数类和少数类的样本数量
    num_samples = x.shape[0]
    num_majority = np.sum(y == 0)
    num_minority = np.sum(y == 1)

    # 计算权重
    weight_majority = num_majority / num_samples
    weight_minority = num_minority / num_samples
    if num_minority != 0:
        weight_minority = num_majority / num_minority

    # 计算加权协方差矩阵
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        print(f"Processing sample {i}: x[i].shape={x[i].shape}")
        if y[0, i] == 0:
            cov[i] = np.cov(x[i]) * weight_majority
        else:
            cov[i] = np.cov(x[i]) * weight_minority

    refEA = np.mean(cov, axis=0)
    # Ensure refEA is a positive definite matrix
    refEA += np.eye(refEA.shape[0]) * 1e-6
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    print("refEA:\n", refEA)
    print("sqrtRefEA:\n", sqrtRefEA)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        # Ensure x[i] is transposed if necessary
        if x[i].shape[0] != 18:
            x[i] = x[i].T
        XEA[i] = np.dot(sqrtRefEA, x[i])
        print(f"XEA[i].shape={XEA[i].shape}")
    return XEA


# 定义downsample_data函数
import numpy as np
from scipy.signal import resample

def downsample_data(X, new_num_time_samples=2000):
    """
    对数据进行下采样处理，使得 num_time_samples 统一为指定的新值。

    Parameters
    ----------
    X : numpy array
        原始数据，形状为 (num_samples, num_time_samples, num_channels)
    new_num_time_samples : int, optional
        新的 num_time_samples 大小，默认为 2000

    Returns
    -------
    downsampled_X : numpy array
        下采样后的数据，形状为 (num_samples, new_num_time_samples, num_channels)
    """
    num_samples, num_time_samples, num_channels = X.shape
    if num_time_samples > new_num_time_samples:
        downsampled_X = np.zeros((num_samples, new_num_time_samples, num_channels))
        for i in range(num_samples):
            for j in range(num_channels):
                downsampled_X[i, :, j] = resample(X[i, :, j], new_num_time_samples)
    else:
        downsampled_X = X
    return downsampled_X

# 主函数
def main():
    # 定义源文件夹和目标文件夹的路径
    source_folder = r'prepro'
    target_folder = r'CHS_EA'

    # 确保目标文件夹存在，如果不存在则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 定义起始和结束的subject id
    start_id = 1
    end_id = 27

    # 存储X和y数据的列表
    X_data_arrays = []
    y_label_arrays = []

    # 遍历每个subject id
    for i in range(start_id, end_id + 1):
        # 构建源MAT文件的完整路径
        mat_file_path = os.path.join(source_folder, f'prepro_subject_{i}.mat')

        # 检查文件是否存在
        if not os.path.isfile(mat_file_path):
            print(f"文件 {mat_file_path} 不存在，跳过该文件。")
            continue

        # 读取MAT文件
        mat_data = sio.loadmat(mat_file_path)

        # 获取标签数据
        y = mat_data.get('y', None)
        if y is None:
            print(f"文件 {mat_file_path} 中没有找到 'y' 变量，跳过该文件。")
            continue

        # 获取特征数据
        X = mat_data.get('X', None)
        if X is None:
            print(f"文件 {mat_file_path} 中没有找到 'X' 变量，跳过该文件。")
            continue

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

    print("转换完成，所有的数据已保存为npy文件。")

# 运行主函数
if __name__ == '__main__':
    main()
