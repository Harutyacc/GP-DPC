import numpy as np
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
    'legend.fontsize': 24,    # 图例字体大小
    'legend.title_fontsize': 26,  # 图例标题字体大小
    'axes.unicode_minus': False,  # 使用 Times New Roman 的负号
})

# 加载迭代次数数据
avg_iterations = np.load('results/differ_k_evaluation/iteration_results.npy', allow_pickle=True).item()

# 按调制方式分组
modulation_groups = {
    'QPSK': {'DPC': [], 'DPC_no_greedy': [], 'GT-DPC': []},
    '8PSK': {'DPC': [], 'DPC_no_greedy': [], 'GT-DPC': []},
    '16QAM': {'DPC': [], 'DPC_no_greedy': [], 'GT-DPC': []}
}

# 遍历所有K值进行分类
for k in avg_iterations:
    # 确定调制类型
    if k <= 8:
        mod_type = 'QPSK'
    elif 8 < k <= 14:
        mod_type = '8PSK'
    else:
        mod_type = '16QAM'
        
    # 收集各方法迭代次数
    for method in ['DPC', 'DPC_no_greedy', 'GT-DPC']:
        modulation_groups[mod_type][method].append(avg_iterations[k][method])
        
# 计算各组平均值
averages = {
    mod: [np.mean(values['DPC']),
          np.mean(values['DPC_no_greedy']),
          np.mean(values['GT-DPC'])]
    for mod, values in modulation_groups.items()
}

# 可视化设置
modulations = ['QPSK', '8PSK', '16QAM']
methods = ['DPC', 'DPC_no_greedy', 'GT-DPC']
bar_width = 0.25
x = np.arange(len(modulations))
colors = ['#80B1D3',  '#6EC6CA', '#F7CAC9']
patterns = ['/', '\\', '|']  # 使用不同的填充图案来区分不同的方法
labels = ['GP-DPC(this paper)', 'P-DPC(no greedy)', r'$\gamma$-DPC($\gamma$ threshold)']  # 使用原始字符串处理数学符号

plt.figure(figsize=(15, 10))
# plt.rc('text', usetex=True)  # 启用LaTeX渲染

# 绘制分组柱状图
for i, (method, label) in enumerate(zip(methods, labels)):
    bars = plt.bar(x + i*bar_width, 
            [averages[mod][i] for mod in modulations],
            width=bar_width,
            color=colors[i],
            hatch=patterns[i],
            edgecolor='black',  # 添加边框
            linewidth=1,       # 边框宽度
            label=label)
    
    # 添加数值标注
    for bar, value in zip(bars, [averages[mod][i] for mod in modulations]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + 0.5,  # 数值显示在柱顶上方
                f'{value:.1f}', 
                ha='center', 
                va='bottom',
                color='black',  # 白色文字
                fontsize=22,    # 适当调大字号
                fontweight='bold')

# 图表装饰优化
plt.xticks(x + bar_width, modulations, fontsize=28)
plt.yticks(fontsize=28)
plt.ylabel('Average Iterations', fontsize=32, labelpad=15)
plt.legend(ncol=3, 
          bbox_to_anchor=(0.5, 1.18), 
          loc='upper center',
          frameon=True,
          framealpha=0.9,
          edgecolor='#444444',
          fontsize=26)

# 网格和背景优化
plt.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')
plt.gca().set_axisbelow(True)  # 网格线在图层下方
# plt.gca().set_facecolor('#F5F5F5')  # 浅灰色背景

# 边框优化
for spine in plt.gca().spines.values():
    spine.set_color('#444444')
    spine.set_linewidth(2)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整上边距

# 显示图表
plt.show()