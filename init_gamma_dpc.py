import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# 设置全局字体参数
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',  # 使用 STIX 字体设置，更接近 Times New Roman
    'font.size': 28,          # 基础字体大小
    'font.weight': 'normal',  # 正常字重，不加粗
    'axes.labelsize': 32,     # 轴标签字体大小
    'axes.titlesize': 32,     # 标题字体大小
    'xtick.labelsize': 28,    # x轴刻度字体大小
    'ytick.labelsize': 28,    # y轴刻度字体大小
    'legend.fontsize': 28,    # 图例字体大小
    'legend.title_fontsize': 26,  # 图例标题字体大小
    'axes.unicode_minus': False,  # 使用 Times New Roman 的负号
})

class DensityPeakClustering:
    def __init__(self, gamma_threshold=100):
        self.n_components_candidates = [4, 8, 16]
        self.n_components = None
        self._cached_distances = None
        self._cached_rho = None
        self._cached_delta = None
        self.means = None
        self.covariance = None
        self.means_history = []
        self.gamma_threshold = gamma_threshold  # 设置gamma的阈值
    
    def _get_distances(self, X):
        """获取或计算距离矩阵"""
        if self._cached_distances is None:
            self._cached_distances = cdist(X, X)
        return self._cached_distances
    
    def _calculate_local_density(self, X, dc=None):
        """计算每个点的局部密度ρ"""
        if self._cached_rho is not None:
            return self._cached_rho
        
        n_samples = X.shape[0]
        if dc is None:
            distances = self._get_distances(X)
            dc = np.percentile(distances.flatten(), 2)
        
        distances = self._get_distances(X)
        self._cached_rho = np.zeros(n_samples)
        for i in range(n_samples):
            self._cached_rho[i] = np.sum(distances[i] < dc) - 1
        
        return self._cached_rho
    
    # def _calculate_delta(self, X, rho):
    #     """计算每个点到密度更高的点的最小距离δ"""
    #     if self._cached_delta is not None:
    #         return self._cached_delta
        
    #     n_samples = X.shape[0]
    #     distances = self._get_distances(X)
    #     self._cached_delta = np.zeros(n_samples)
        
    #     max_rho_idx = np.argmax(rho)
    #     self._cached_delta[max_rho_idx] = np.max(distances[max_rho_idx])
        
    #     for i in range(n_samples):
    #         if i != max_rho_idx:
    #             higher_density_pts = np.where(rho > rho[i])[0]
    #             if len(higher_density_pts) > 0:
    #                 self._cached_delta[i] = np.min(distances[i][higher_density_pts])
    #             else:
    #                 self._cached_delta[i] = distances[i][max_rho_idx]
        
    #     return self._cached_delta

    def _calculate_delta(self, X, rho):
        """计算每个点到密度更高的点的最小距离δ"""
        if self._cached_delta is not None:
            return self._cached_delta
        
        n_samples = X.shape[0]
        distances = self._get_distances(X)
        self._cached_delta = np.zeros(n_samples)
        
        max_rho_idx = np.argmax(rho)
        self._cached_delta[max_rho_idx] = np.max(distances[max_rho_idx])
        
        for i in range(n_samples):
            if i != max_rho_idx:
                higher_or_equal = np.where(rho >= rho[i])[0]
                # 排除自身
                higher_or_equal = higher_or_equal[higher_or_equal != i]
                
                if len(higher_or_equal) == 0:
                    self._cached_delta[i] = np.max(distances[i])
                else:
                    self._cached_delta[i] = np.min(distances[i][higher_or_equal])
        
        return self._cached_delta
    
    def compute_kl_divergence(self, mean1, mean2, cov):
        """计算两个高斯分布之间的KL散度"""
        diff = mean2 - mean1
        inv_cov = np.linalg.inv(cov)
        kl = 0.5 * (diff.T @ inv_cov @ diff)
        return kl

    def verify_centers(self, centers, covariance):
        """验证聚类中心是否存在冗余（基于KLD）"""
        n_centers = len(centers)
        kld_matrix = np.zeros((n_centers, n_centers))
        
        # 计算所有中心点对之间的KLD
        for i in range(n_centers):
            for j in range(n_centers):
                if i != j:
                    kld_matrix[i,j] = self.compute_kl_divergence(centers[i], centers[j], covariance)
                else:
                    kld_matrix[i,j] = np.inf
        
        # 找出每个中心点的最小KLD
        min_klds = np.min(kld_matrix, axis=1)
        
        min_klds_cv = np.std(min_klds) / np.mean(min_klds)
        
        # 如果任何一对中心点的KLD小于1，则认为存在冗余
        has_redundancy = np.any(min_klds < 1)
        if has_redundancy == False:
            if min_klds_cv > 1:
              has_redundancy = True
        
        return has_redundancy, min_klds
    
    def _initialize_parameters(self, X):
        """使用密度峰值方法初始化参数"""
        rho = self._calculate_local_density(X)
        delta = self._calculate_delta(X, rho)
        gamma = rho * delta
        
        n_samples = X.shape[0]
        
        # 直接选择gamma大于阈值的点作为聚类中心
        selected_centers = np.where(gamma > self.gamma_threshold)[0]

        centers_points = X[selected_centers]
        
        # 初始化协方差矩阵
        n_samples_per_component = n_samples // len(selected_centers)
        covariance = np.zeros((X.shape[1], X.shape[1]))
        
        for k in range(len(selected_centers)):
            distances = np.linalg.norm(X - centers_points[k], axis=1)
            nearest_indices = np.argsort(distances)[:n_samples_per_component]
            nearest_points = X[nearest_indices]
            
            diff = nearest_points - centers_points[k]
            component_covariance = np.dot(diff.T, diff) / len(nearest_points)
            covariance += component_covariance
        
        covariance /= len(selected_centers)
        min_covar = 1e-6
        covariance += np.eye(X.shape[1]) * min_covar
        covariance = (covariance + covariance.T) / 2
        
        """绘制决策图"""
        plt.figure(figsize=(10, 8))
        plt.scatter(rho, delta, alpha=0.5)
        plt.scatter(rho[selected_centers], delta[selected_centers], s=200, c='red', marker='x')
        plt.xlabel('Local Density (ρ)')
        plt.ylabel('Distance to Higher Density Point (δ)')
        plt.title('Decision Graph')
        plt.grid()
        plt.show()
        
        """绘制当前的聚类状态"""
        plt.figure(figsize=(10, 8))
        # 绘制数据点
        plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Data points')
        
        plt.scatter(X[selected_centers, 0], X[selected_centers, 1],
                c='red', marker='*', s=200, label='Current centers')

        plt.title(f'DPC Centers')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.legend()
        plt.grid(True)
        
        # 将选中的中心点作为聚类中心
        self.means = centers_points
        self.means_history.append(self.means.copy())
        self.n_components = len(selected_centers)
        self.covariance = covariance
        return

    
    def fit(self, X):
        """初始化聚类中心和协方差矩阵"""
        self._initialize_parameters(X)
        return self.means, self.covariance