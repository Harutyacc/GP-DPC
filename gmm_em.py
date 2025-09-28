import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def gmm_em(X, mu_init, cov_init, max_iter=100, tol=1e-6):
    """
    自实现的高斯混合模型（GMM）使用EM算法，增强数值稳定性
    增加了迭代过程的可视化
    """
    def _plot_current_state(X, mu_history, iteration):
        """绘制当前的聚类状态"""
        plt.clf()
        # 绘制数据点
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Data points')
        
        # 绘制当前聚类中心
        current_mu = mu_history[-1]
        plt.scatter(current_mu[:, 0], current_mu[:, 1],
                   c='red', marker='*', s=200, label='Current centers')
        
        # 如果有历史记录，绘制轨迹
        if len(mu_history) > 1:
            means_history = np.array(mu_history)
            for k in range(len(current_mu)):
                plt.plot(means_history[:, k, 0], means_history[:, k, 1],
                        'r--', alpha=0.3)
        
        plt.title(f'Iteration {iteration}')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.legend()
        plt.grid(True)
        
        clear_output(wait=True)
        plt.pause(0.1)

    n_samples, n_features = X.shape
    n_components = len(mu_init)
    
    # 初始化参数
    mu = mu_init.copy()
    
    # 确保cov_init是一个包含每个聚类的协方差矩阵的列表
    if isinstance(cov_init, list):
        cov = [c.copy() for c in cov_init]  # 为每个聚类复制独立的协方差矩阵
    else:
        # 如果cov_init只是一个单一的协方差矩阵，可以将它复制到每个聚类
        cov = [cov_init.copy() for _ in range(n_components)]
        
    weights = np.ones(n_components) / n_components
    
    # 存储均值历史
    mu_history = [mu.copy()]
    
    # 创建图像窗口
    plt.figure(figsize=(10, 8))
    _plot_current_state(X, mu_history, 0)

    def log_gaussian_pdf(X, mu, cov):
        """计算多维高斯分布的对数概率密度，避免数值溢出"""
        n = X.shape[1]
        diff = X - mu
        
        # 确保协方差矩阵正定
        cov_reg = cov + np.eye(n) * 1e-6
        
        try:
            # 使用Cholesky分解计算行列式和逆矩阵，提高数值稳定性
            chol = np.linalg.cholesky(cov_reg)
            log_det = 2 * np.sum(np.log(np.diag(chol)))
            inv = np.linalg.solve(cov_reg, np.eye(n))
        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，使用传统方法
            log_det = np.log(np.linalg.det(cov_reg))
            inv = np.linalg.inv(cov_reg)
        
        # 计算指数项
        exponent = -0.5 * np.sum(diff.dot(inv) * diff, axis=1)
        
        # 计算对数概率密度
        log_norm_const = -0.5 * (n * np.log(2 * np.pi) + log_det)
        
        return log_norm_const + exponent

    def special_logsumexp(x, axis=None):
        """数值稳定的logsumexp实现"""
        x_max = np.max(x, axis=axis, keepdims=True)
        result = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
        return result.squeeze()

    for iteration in range(max_iter):
        # E步：计算责任度（使用对数空间计算）
        log_responsibilities = np.zeros((n_samples, n_components))
        
        # 计算每个组分的对数概率
        for k in range(n_components):
            log_responsibilities[:, k] = np.log(weights[k] + 1e-300) + \
                                       log_gaussian_pdf(X, mu[k], cov[k])
        
        # 使用log-sum-exp技巧来避免数值溢出
        log_sum = special_logsumexp(log_responsibilities, axis=1)
        log_responsibilities = log_responsibilities - log_sum[:, np.newaxis]
        responsibilities = np.exp(log_responsibilities)
        
        # 检查数值稳定性
        responsibilities = np.nan_to_num(responsibilities, 0)
        responsibilities = responsibilities / (responsibilities.sum(axis=1, keepdims=True) + 1e-10)
        
        # M步：更新参数
        N_k = responsibilities.sum(axis=0)
        weights = (N_k + 1e-10) / (n_samples + n_components * 1e-10)
        
        # 更新均值
        for k in range(n_components):
            if N_k[k] > 1e-10:
                mu[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
        
        # 更新协方差
        for k in range(n_components):
            diff = X - mu[k]
            if N_k[k] > 1e-10:
                cov[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k]
                # 添加正则化项
                cov[k] += np.eye(n_features) * 1e-6
        
        # 存储当前迭代的均值
        mu_history.append(mu.copy())
        
        # 更新可视化
        _plot_current_state(X, mu_history, iteration + 1)
        
        # 计算对数似然度
        if iteration > 0:
            old_ll = curr_ll
        curr_ll = np.sum(log_sum)
        
        # 检查收敛性
        if iteration > 0:
            if abs(curr_ll - old_ll) < tol * abs(curr_ll):
                break
    
    # 计算最终的聚类标签
    labels = np.argmax(responsibilities, axis=1)
    
    plt.close()  # 关闭最后的图像窗口
    
    mu = np.array(mu)
    cov = np.array(cov)
    
    return mu, cov, weights, labels, responsibilities, iteration+1