def EA_classweighted(x, y):
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

    y = y.reshape(-1,)

    # 获取多数类和少数类的样本数量
    num_majority = np.sum(y == 0)
    num_minority = np.sum(y == 1)

    # 计算类特定的协方差矩阵
    cov_majority = np.zeros((num_majority, x.shape[1], x.shape[1]))
    cov_minority = np.zeros((num_minority, x.shape[1], x.shape[1]))
    cnt_majority = 0
    cnt_minority = 0
    for i in range(x.shape[0]):
        if y[i] == 0:
            cov_majority[cnt_majority] = np.cov(x[i])
            cnt_majority += 1
        else:
            cov_minority[cnt_minority] = np.cov(x[i])
            cnt_minority += 1

    # 计算类平均的平均协方差矩阵
    stacked_class_mean = np.stack([np.mean(cov_majority, axis=0), np.mean(cov_minority, axis=0)])
    print('stacked_class_mean.shape', stacked_class_mean.shape)
    refEA = np.mean(stacked_class_mean, axis=0)
    print('refEA.shape', refEA.shape)

    # 计算协方差矩阵的逆平方根
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)

    # 计算EA处理后的数据
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA
