import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 设置全局字体参数（保持与参考代码一致）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 28,
    'font.weight': 'normal',
    'axes.labelsize': 32,
    'axes.titlesize': 32,
    'xtick.labelsize': 28,
    'ytick.labelsize': 28,
    'legend.fontsize': 24,
    'legend.title_fontsize': 26,
    'axes.unicode_minus': False,
})

k_values = np.linspace(2, 20, 19)
N_frames = 50

# 加载数据
rmse_results = np.load('results/differ_k_evaluation/rmse_result.npy', allow_pickle=True).item()

# 初始化画布
plt.figure(figsize=(15, 10))

# 创建背景色区域
modulation_regions = [
    (2, 8, 'QPSK', '#B3E5FC', r'$K \in [2,8]$'),
    (8, 14, '8PSK', '#C8E6C9', r'$K \in (8,14]$'),
    (14, 20.1, '16QAM', '#FFCDD2', r'$K \in (14,20]$')
]

ax = plt.gca()

# 绘制背景区域
for xmin, xmax, label, color, text in modulation_regions:
    ax.axvspan(xmin, xmax, facecolor=color, alpha=0.3)
    ax.text((xmin+xmax)/2, ax.get_ylim()[1]*16, 
            f"{text}\n{label}", 
            ha='center', va='bottom',
            fontsize=24, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# 绘制RMSE曲线
colors = ['#003366', '#006400', '#990000']
labels = ['GP-DPC(this paper)', 'P-DPC(no greedy)', r'$\gamma$-DPC($\gamma$ threshold)']
line_styles = ['-', '--', '-.']
markers = ['o', 's', '^']

for method, label, color, line_style, marker in zip(['DPC', 'DPC_no_greedy', 'GT-DPC'], labels, colors, line_styles, markers):
    plt.plot(k_values, [rmse_results[k][method] for k in k_values], 
             label=label, 
             color=color,
             linestyle=line_style,
             marker=marker,
             markersize=10,
             linewidth=2)

# 坐标轴设置
plt.xlabel(r'Rice Factor K', fontsize=32)
plt.ylabel('RMSE', fontsize=32)
plt.xticks(np.arange(2, 21, 2))
plt.xlim(2, 20)

# 网格设置
plt.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')
plt.gca().set_axisbelow(True)

# 背景色设置
plt.gca().set_facecolor('#F5F5F5')  # 浅灰色背景

# 边框优化
for spine in plt.gca().spines.values():
    spine.set_color('#444444')
    spine.set_linewidth(2)

# 图例优化
legend = plt.legend(
    ncol=3,
    bbox_to_anchor=(0.5, 1.18),  # 调整垂直位置
    loc='upper center',
    frameon=True,
    framealpha=0.9,
    edgecolor='#444444',
    fontsize=26
)

# 布局调整
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整上边距

# 保存设置
plt.savefig('RMSE_with_modulation_regions_optimized_v3.pdf', 
           dpi=300, 
           bbox_inches='tight')

plt.show()