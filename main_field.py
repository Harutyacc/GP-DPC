import numpy as np
import matplotlib.pyplot as plt
from data_generator import data_generator
from data_trans import mat_to_numpy
import gmm_em as gmm

import init_gamma_dpc as gamma_dpc
import init_gp_dpc as gp_dpc
import init_p_dpc as p_dpc

import os
import logging
from datetime import datetime

class ModulationAnalyzer:
    def __init__(self, file_name, init_method, N_samples=10000, N_frames=600, scale_factor=600):
        self.file_name = file_name
        self.init_method = init_method
        self.N_samples = N_samples
        self.N_frames = N_frames
        self.scale_factor = scale_factor
        
        # 创建结果存储数组
        self.results = {
            'estimated_k': np.zeros(N_frames),
            'true_k': np.zeros(N_frames),
            'estimated_clusters': np.zeros(N_frames, dtype=int),
            'true_clusters': np.zeros(N_frames, dtype=int),
            'gmm_iterations': np.zeros(N_frames, dtype=int),
            'error_frames': []
        }
        
        # 功率归一化系，这里所有调制方式都是1，是因为生成数据的时候已经进行了归一化
        self.power_norm_factor = {
          'QPSK': 1.0,
          '8PSK': 1.0,
          '16APSK': 1.0,
        }
        
        # 设置日志和错误帧保存路径
        self.setup_logging_and_paths()
        
    def get_scenario_type(self):
        """获取场景类型"""
        if 'Vehicle' in self.file_name:
            return 'Vehicle'
        elif 'Person' in self.file_name:
            return 'Person'
        elif 'Direc' in self.file_name:
            return 'Direc'
        else:
            return 'unknown'
          
    def get_init_method(self):
        """获取初始化方法"""
        if self.init_method == 'Gamma-DPC':
            return gamma_dpc.DensityPeakClustering(gamma_threshold=100)
        elif self.init_method == 'GP-DPC':
            return gp_dpc.DensityPeakClustering()
        elif self.init_method == 'P-DPC':
            return p_dpc.DensityPeakClustering()
        else:  
            raise ValueError(f'初始化方法 {self.init_method} 不支持')
        
    def setup_logging_and_paths(self):
        """设置日志和错误帧保存路径"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        self.scenario_type = self.get_scenario_type()
        # self.error_frames_dir = os.path.join('error_frames', f'{timestamp}_{self.scenario_type}_{self.init_method}')
        
        # if not os.path.exists(self.error_frames_dir):
        #     os.makedirs(self.error_frames_dir)
        
        logging.basicConfig(
            filename=os.path.join('logs', f'{timestamp}.log'), 
            level=logging.INFO, 
            encoding='utf-8',
            format='%(asctime)s - %(message)s')
        
    def determin_true_clusters(self, k_val):
        """确定真实的聚类数量"""
        if k_val <= 8:
            return 4  # QPSK
        elif k_val <= 14:
            return 8  # 8PSK
        else:
            return 16  # 16QAM
        
    def get_modulation_type(self, n_clusters):
        """获取调制方式"""
        if n_clusters == 4:
            return 'QPSK'
        elif n_clusters == 8:
            return '8PSK'
        elif n_clusters == 16:
            return '16QAM'
        else:
            return 'unknown'
          
    def estimate_channel_parameters(self, means, covariance):
        """从GMM模型估计信道参数"""
        # 1.去除缩放因子的影响
        means /= self.scale_factor
        covariance /= self.scale_factor**2
        
        # 2.计算平均接收信号功率
        rx_power = np.mean(np.sum(means**2, axis=1))
        
        # 3.计算噪声方差（求迹平均后除以2），协方差矩阵的迹表示总噪声功率，除以2表示每个维度的噪声方差
        noise_variance = np.trace(covariance, axis1=1, axis2=2).mean() / 2
        
        # 4.获取调制方式并进行功率归一化
        modulation_type = self.get_modulation_type(means.shape[0])
        if modulation_type in self.power_norm_factor:
            normalization_factor = self.power_norm_factor[modulation_type]
            signal_power = rx_power / normalization_factor
          
        else:
            signal_power = rx_power
            
        # 5.计算信道参数
        s = np.sqrt(signal_power)  # 直射分量幅度
        sigma = np.sqrt(noise_variance)  # 散射分量标准差
        k_factor = signal_power / (2 * noise_variance)  # 莱斯因子
        
        return k_factor, s, sigma
        
    # def save_error_frame_plot(self, frame_idx, Symbol_Data, mu, labels):
    #     """保存错误帧的星座图"""
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(Symbol_Data[:, 0], Symbol_Data[:, 1], 
    #                c=labels, cmap='viridis', s= 10, alpha=0.5)
    #     plt.scatter(mu[:, 0], mu[:, 1], 
    #                color='red', marker='*', s=50, label='GMM Centers')
    #     plt.title(f'Error Frame {frame_idx} (Est:{self.results["estimated_clusters"][frame_idx]} vs True:{self.results["true_clusters"][frame_idx]})')
    #     plt.xlabel('Real')
    #     plt.ylabel('Imaginary')
    #     plt.legend()
    #     plt.savefig(f'{self.error_frames_dir}/frame_{frame_idx}.png')
    #     plt.close()
        
    def analyze_frame(self, frame_idx, s_val, sigma_val, k_val, Symbol_Data):
        """分析单帧数据"""   
        # 获取初始化方法
        dpc_init = self.get_init_method()
        
        # 完成初始化
        mu_init, cov_init = dpc_init.fit(Symbol_Data)
        
        # GMM模型估计
        mu, cov, _, labels, _, iteration = gmm.gmm_em(
            Symbol_Data, 
            mu_init, 
            cov_init,
            max_iter=100,
            tol=1e-6
        )
        
        # 估计信道参数
        k_est, s_est, sigma_est = self.estimate_channel_parameters(mu, cov)
        
        # 确定真实的聚类数量
        true_clusters = self.determin_true_clusters(k_val[0])
        estimated_clusters = len(mu)
        
        # 保存结果
        self.results['estimated_k'][frame_idx] = k_est
        self.results['true_k'][frame_idx] = k_val[0]
        self.results['estimated_clusters'][frame_idx] = estimated_clusters
        self.results['true_clusters'][frame_idx] = true_clusters
        self.results['gmm_iterations'][frame_idx] = iteration
        
        # 输出当前帧的结果
        print(f"\n帧 {frame_idx + 1}:")
        print(f"K值 - 估计: {k_est:.4f}, 真实: {k_val[0]:.4f}")
        print(f"调制方式 - 估计: {self.get_modulation_type(estimated_clusters)}, 真实: {self.get_modulation_type(true_clusters)}")
        print(f"GMM迭代次数: {iteration}")
        print(f"信道参数估计 - S: {s_est:.4f}, σ: {sigma_est:.4f}")
        
        # 检查是否为错误帧
        if estimated_clusters != true_clusters:
            self.results['error_frames'].append(frame_idx)
            error_msg = f"错误帧 {frame_idx+1}: 估计聚类数={estimated_clusters}, 真实聚类数={true_clusters}"
            print(f"错误: {error_msg}")
            log_error_msg = f"场景：{self.scenario_type}，初始化方法：{self.init_method}，错误帧 {frame_idx+1}: 估计聚类数={estimated_clusters}, 真实聚类数={true_clusters}"
            logging.error(log_error_msg)
            # self.save_error_frame_plot(frame_idx, Symbol_Data, mu, labels)
                
    def analyze_all_frames(self):
        """分析所有帧数据"""
        structs_data = mat_to_numpy(self.file_name)
        
        print(f'开始分析 {self.file_name} 的 {self.N_frames} 帧数据...')
        
        for structs_name, fields in structs_data.items():
            for frame_idx, (s_val, sigma_val, k_val) in enumerate(
                zip(
                    fields['s'][:self.N_frames],
                    fields['sigma'][:self.N_frames],
                    fields['k'][:self.N_frames])):
              
                Symbol_Data, _ = data_generator(self.N_samples, Phi=np.pi/8, S=s_val, Sigma=sigma_val, Rice_K=k_val)
                Symbol_Data = Symbol_Data * self.scale_factor
                
                try:
                    # 分析当前帧
                    self.analyze_frame(frame_idx, s_val, sigma_val, k_val, Symbol_Data)
                    
                    if (frame_idx + 1) % 100 == 0:
                        print(f'已完成 {frame_idx+1} 帧数据分析')
                        
                except Exception as e:
                    logging.error(f'分析帧 {frame_idx} 出错：{e}, 非法数据')
                    self.results['error_frames'].append(frame_idx)
                    
        # 输出统计
        total_errors = len(self.results['error_frames'])
        error_rate = total_errors / self.N_frames * 100
        
        print(f"\n分析完成:")
        print(f"总帧数: {self.N_frames}")
        print(f"错误帧数: {total_errors}")
        print(f"错误率: {error_rate:.4f}%")
        
        print(f"\n错误帧详细信息已保存到日志文件")
        # print(f"错误帧星座图已保存到 {self.error_frames_dir} 文件夹")
        
        return self.results
            
            
def run_analysis(file_name, init_method, N_samples=10000, N_frames=600, scale_factor=600):
    """运行分析"""
    analyzer = ModulationAnalyzer(file_name, init_method, N_samples, N_frames, scale_factor)
    results = analyzer.analyze_all_frames()
      
    def get_scenario_type(file_name):
        """获取场景类型"""
        if 'Vehicle' in file_name:
            return 'Vehicle'
        elif 'Person' in file_name:
            return 'Person'
        elif 'Direc' in file_name:
            return 'Direc'
        else:
            return 'unknown'
      
    scenario_type = get_scenario_type(file_name)
    
    # 保存结果
    if not os.path.exists('results/field_experienment'):
        os.makedirs('results/field_experienment')
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'results/field_experienment/{scenario_type}_{init_method}_{timestamp}'
    
    # 保存每个数组到单独的npy文件
    for key, value in results.items():
        if key != 'error_frames':  # error_frames单独处理
            np.save(f'{save_path}_{key}.npy', value)
    
    # 保存错误帧列表到txt文件
    with open(f'{save_path}_error_frames.txt', 'w') as f:
        for frame_idx in results['error_frames']:
            f.write(f'{frame_idx+1}\n')

    print(f"\n分析结果已保存到 {save_path}_*.npy 文件")
    return results
                    
if __name__ == "__main__":
    # 三种场景数据
    file_names = [
        'RawData/Vehicle915Test.mat',
        'RawData/Person915Test.mat',
        'RawData/Direc915Test.mat'
    ]
    
    # 三种不同方法
    methods = [
        'Gamma-DPC',
        'GP-DPC',
        'P-DPC'
    ]
    
    for file_name in file_names:
        print(f"\n{'='*50}")
        print(f"开始分析文件: {file_name}")
        print(f"{'='*50}")
        for init_method in methods:
            print(f"\n{'-'*25}")
            print(f"开始分析方法: {init_method}")
            print(f"{'-'*25}")
            results = run_analysis(file_name, init_method)    
            