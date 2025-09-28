import numpy as np 
import matplotlib.pyplot as plt
import data_generator_from_k
import init_gp_dpc as gpdpc
import init_p_dpc as gpdpc_no_greedy
import init_gamma_dpc as gtdpc
from scipy.optimize import linear_sum_assignment
import gmm_em as gmm
from sklearn.metrics import mean_squared_error

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
    'legend.fontsize': 32,    # 图例字体大小
    'legend.title_fontsize': 26,  # 图例标题字体大小
    'axes.unicode_minus': False,  # 使用 Times New Roman 的负号
})

class ModulationAnalyzer:
    def __init__(self, N_samples=10000, scale_factor=600, modulation_type=None, k_val=None):
        self.N_samples = N_samples
        self.scale_factor = scale_factor
        self.modulation_type = modulation_type
        self.true_k = k_val
        self.true_clusters = np.zeros(1, dtype=int)
        self.estimated_k = np.zeros(1)
        self.estimated_clusters = np.zeros(1, dtype=int)
        self.means = {}
        self.ovariance = {}
        self.gp_iteration = {}
        self.p_iteration = {}
        self.gamma_iteration = {}
        self.gp_componets = {}
        self.p_componets = {}
        self.gt_componets = {}

        # 定义不同调制方式的功率归一化系数，这里所有调制方式都是1，是因为生成数据的时候已经进行了归一化
        self.power_normalization = {
            'QPSK': 1.0,
            '8PSK': 1.0,
            '16QAM': 1.0
        }

    def determine_modulation_type(self):
        """根据k值确定真实的调制方式（聚类数）"""
        if self.true_k <= 8:
            self.modulation_type = 'QPSK'
            self.true_clusters = 4
        elif self.true_k <= 14:
            self.modulation_type = '8PSK'
            self.true_clusters = 8
        else:
            self.modulation_type = '16QAM'
            self.true_clusters = 16
    
    def estimate_channel_parameters(self):
        """从GMM参数估计莱斯信道参数"""
        means = self.means / self.scale_factor
        covariance = self.covariance / (self.scale_factor ** 2)
        rx_power = np.mean(np.sum(means**2, axis=1))
        noise_variance = np.trace(covariance, axis1=1, axis2=2).mean() / 2
        if self.modulation_type in self.power_normalization:
            normalization_factor = self.power_normalization[self.modulation_type]
            signal_power = rx_power / normalization_factor
        else:
            signal_power = rx_power
        k_factor = signal_power / (2 * noise_variance)
        return k_factor
      
    def verify_centers_with_data(self):
        """验证计算的聚类中心与实际数据分布的关系"""
        dataMod, _ = data_generator_from_k.data_generator_from_k(self.N_samples, np.pi/8, self.true_k, self.modulation_type)
        
        # DPC初始化方法1
        print(f"\nGP-DPC Now...")
        rdpc_init = gpdpc.DensityPeakClustering()
        dpc_centers, dpc_covariance = rdpc_init.fit(dataMod)
        # 运行GMM-EM
        mu, cov, _, _, _, iteration = gmm.gmm_em(dataMod, dpc_centers, dpc_covariance, max_iter=100, tol=1e-6)
        self.means = mu
        self.covariance = cov
        self.gp_iteration = iteration
        self.gp_componets = len(mu)
        estimated_k_dpc = self.estimate_channel_parameters()

        # DPC无贪心初始化方法2
        print(f"\nNoGreedy DPC Now...")
        rdpc_init_no_greedy = gpdpc_no_greedy.DensityPeakClustering()
        dpc_centers, dpc_covariance = rdpc_init_no_greedy.fit(dataMod)
        mu, cov, _, _, _, iteration = gmm.gmm_em(dataMod, dpc_centers, dpc_covariance, max_iter=100, tol=1e-6)
        self.means = mu
        self.covariance = cov
        self.p_iteration = iteration
        self.p_componets = len(mu)
        estimated_k_dpc_no_greedy = self.estimate_channel_parameters()

        # gamma-DPC初始化方法3
        print(f"\nClassic DPC Now...")
        rdpc_init_gt = gtdpc.DensityPeakClustering()
        dpc_centers, dpc_covariance = rdpc_init_gt.fit(dataMod)
        mu, cov, _, _, _, iteration = gmm.gmm_em(dataMod, dpc_centers, dpc_covariance, max_iter=100, tol=1e-6)
        self.means = mu
        self.covariance = cov
        self.gt_iteration = iteration
        self.gt_componets = len(mu)
        estimated_k_gt_dpc = self.estimate_channel_parameters()

        return estimated_k_dpc, estimated_k_dpc_no_greedy, estimated_k_gt_dpc
    
k_values = np.linspace(4, 20, 19)
N_frames = 50

# 存储每个k值的估算K值和RMSE结果
rmse_results = {}
avg_iterations = {}
correct_components = {}

for k in k_values:
    print(f"\nAnalyzing Rice-k: {k}...")

    modulation_analyzer = ModulationAnalyzer(N_samples=10000, scale_factor=600, k_val=k)
    modulation_analyzer.true_k = k
    modulation_analyzer.determine_modulation_type()  # 根据k值确定调制方式

    estimated_k_list_dpc = []
    estimated_k_list_dpc_no_greedy = []
    estimated_k_list_gt_dpc = []
    
    estimated_iteration_list_dpc = []
    estimated_iteration_list_dpc_no_greedy = []
    estimated_iteration_list_gt_dpc = []
    
    correct_components_dpc = []
    correct_components_dpc_no_greedy = []
    correct_components_gt_dpc = []

    for i in range(N_frames):
        print(f"\nAnalyzing Frame: {i}...")
        # 获取三种初始化方法的K值估计
        estimated_k_dpc, estimated_k_dpc_no_greedy, estimated_k_gt_dpc = modulation_analyzer.verify_centers_with_data()
        estimated_k_list_dpc.append(estimated_k_dpc)
        estimated_k_list_dpc_no_greedy.append(estimated_k_dpc_no_greedy)
        estimated_k_list_gt_dpc.append(estimated_k_gt_dpc)
        
        estimated_iteration_list_dpc.append(modulation_analyzer.gp_iteration)
        estimated_iteration_list_dpc_no_greedy.append(modulation_analyzer.p_iteration)
        estimated_iteration_list_gt_dpc.append(modulation_analyzer.gt_iteration)
        
        # 记录components是否正确
        correct_dpc = int(modulation_analyzer.gp_componets == modulation_analyzer.true_clusters)
        correct_dpc_no = int(modulation_analyzer.p_componets == modulation_analyzer.true_clusters)
        correct_gt = int(modulation_analyzer.gt_componets == modulation_analyzer.true_clusters)
        
        correct_components_dpc.append(correct_dpc)
        correct_components_dpc_no_greedy.append(correct_dpc_no)
        correct_components_gt_dpc.append(correct_gt)

    # 计算每个k值下估算K值与真实K值之间的RMSE
    rmse_dpc = np.sqrt(mean_squared_error([k] * N_frames, estimated_k_list_dpc))
    rmse_dpc_no_greedy = np.sqrt(mean_squared_error([k] * N_frames, estimated_k_list_dpc_no_greedy))
    rmse_gt_dpc = np.sqrt(mean_squared_error([k] * N_frames, estimated_k_list_gt_dpc))
    
    # 平均迭代次数统计
    avg_iterations[k] = {
        'DPC': np.mean(estimated_iteration_list_dpc),
        'DPC_no_greedy': np.mean(estimated_iteration_list_dpc_no_greedy),
        'GT-DPC': np.mean(estimated_iteration_list_gt_dpc)
    }
    
    # 正确聚类数目统计
    correct_components[k] = {
        'DPC': np.sum(correct_components_dpc),
        'DPC_no_greedy': np.sum(correct_components_dpc_no_greedy),
        'GT-DPC': np.sum(correct_components_gt_dpc)
    }

    rmse_results[k] = {
        'DPC': rmse_dpc,
        'DPC_no_greedy': rmse_dpc_no_greedy,
        'GT-DPC': rmse_gt_dpc
    }

    print(f"RMSE for Rice-k={k} (DPC): {rmse_dpc:.4f}")
    print(f"RMSE for Rice-k={k} (DPC_no_greedy): {rmse_dpc_no_greedy:.4f}")
    print(f"RMSE for Rice-k={k} (GT-DPC): {rmse_gt_dpc:.4f}")
    
    print(f"Average iterations for k={k}:")
    print(f"DPC: {avg_iterations[k]['DPC']:.1f}, NoGreedy: {avg_iterations[k]['DPC_no_greedy']:.1f}, GT-DPC: {avg_iterations[k]['GT-DPC']:.1f}")
    
    print(f"Correct components estimated for k={k}:")
    print(f"DPC: {correct_components[k]['DPC']}, NoGreedy: {correct_components[k]['DPC_no_greedy']}, GT-DPC: {correct_components[k]['GT-DPC']}")

# 保存RMSE结果
np.save('rmse_result.npy', rmse_results)
np.save('iteration_results.npy', avg_iterations)
np.save('components_results.npy', correct_components)

# 可视化RMSE结果
plt.figure(figsize=(12, 9))
for method in ['DPC', 'DPC_no_greedy', 'GT-DPC']:
    plt.plot(k_values, [rmse_results[k][method] for k in k_values], label=f"RMSE of {method}", marker='o')
plt.xlabel('Rice-k Values')
plt.ylabel('RMSE')
plt.title('RMSE vs Rice-k')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
