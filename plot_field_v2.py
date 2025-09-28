import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 20,
    'font.weight': 'normal',
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 21,
    'legend.title_fontsize': 30,
    'axes.unicode_minus': False,
})

# 读取数据
gp_dpc_k = np.load(r'results\field_experienment\Direc_GP-DPC_20250410_110535_estimated_k.npy')
true_k = np.load(r'results\field_experienment\Direc_Gamma-DPC_20250410_101029_true_k.npy')
p_dpc_k = np.load(r'results\field_experienment\Direc_P-DPC_20250410_131225_estimated_k.npy')
gamma_dpc_k = np.load(r'results\field_experienment\Direc_Gamma-DPC_20250410_101029_estimated_k.npy')


# gp_dpc_k = np.load(r'results\field_experienment\Person_GP-DPC_20250410_064234_estimated_k.npy')
# true_k = np.load(r'results\field_experienment\Person_Gamma-DPC_20250410_060238_true_k.npy')
# p_dpc_k = np.load(r'results\field_experienment\Person_P-DPC_20250410_074707_estimated_k.npy')
# gamma_dpc_k = np.load(r'results\field_experienment\Person_Gamma-DPC_20250410_060238_estimated_k.npy')

# gp_dpc_k = np.load(r'results\field_experienment\Vehicle_GP-DPC_20250410_040222_estimated_k.npy')
# true_k = np.load(r'results\field_experienment\Vehicle_Gamma-DPC_20250410_033436_true_k.npy')
# p_dpc_k = np.load(r'results\field_experienment\Vehicle_P-DPC_20250410_045904_estimated_k.npy')
# gamma_dpc_k = np.load(r'results\field_experienment\Vehicle_Gamma-DPC_20250410_033436_estimated_k.npy')

frames = np.arange(600)

# 创建图形布局
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)

# 计算主图的y轴范围
y_min = min(
    np.min(true_k[:600]),
    np.min(gp_dpc_k[:600]),
    np.min(p_dpc_k[:600]),
    np.min(gamma_dpc_k[:600])
)
y_max = max(
    np.max(true_k[:600]),
    np.max(gp_dpc_k[:600]),
    np.max(p_dpc_k[:600]),
    np.max(gamma_dpc_k[:600])
)
y_margin = (y_max - y_min) * 0.05

# 上半部分：完整时间序列
ax1 = fig.add_subplot(gs[0])
ax1.plot(frames, true_k[:600], color='#B8860B', linestyle='-', label='Ground Truth', linewidth=1.5)
ax1.plot(frames, gp_dpc_k[:600], color='#3366CC', linestyle='--', label='GP-DPC(this paper)', linewidth=1.5)
ax1.plot(frames, p_dpc_k[:600], color='#355E3B', linestyle='-.', label='P-DPC', linewidth=1.5)
ax1.plot(frames, gamma_dpc_k[:600], color='#990033', linestyle=':', label='$\gamma$-DPC', linewidth=1.5)
# ax1.plot(frames, gp_dpc_k[:600], color='#3366CC', linestyle='--', label='GP-DPC(this paper)', linewidth=1.5)
# ax1.plot(frames, p_dpc_k[:600], color='#6EC6CA', linestyle='-.', label='P-DPC', linewidth=1.5)
# ax1.plot(frames, gamma_dpc_k[:600], color='#990010', linestyle=':', label='$\gamma$-DPC', linewidth=1.5)

ax1.grid(True, linestyle='--', alpha=0.3)
ax1.set_ylabel('K-factor')
ax1.set_ylim(y_min - y_margin, y_max + y_margin)

# 图例优化
legend = plt.legend(
    ncol=4,
    bbox_to_anchor=(0.5, 1.18),  # 调整垂直位置
    loc='upper center',
    frameon=True,
    framealpha=0.9,
    edgecolor='#444444',
    fontsize=26
)


# 下半部分：放大显示
zoom_start, zoom_end = 400, 600

# 使用虚线框标识放大区域
# rect = patches.Rectangle((zoom_start, y_min - y_margin), zoom_end - zoom_start, y_max - y_min + 2 * y_margin,
#                          linewidth=2, edgecolor='black', linestyle='-', fill=True, hatch='//', facecolor='none')
rect = patches.Rectangle((zoom_start, y_min - y_margin), zoom_end - zoom_start, y_max - y_min + 2 * y_margin,
                         linewidth=3, edgecolor='red', linestyle='--', facecolor='none')
ax1.add_patch(rect)

# 标记放大区域为 "II"，其他区域为 "I"
ax1.text(200, y_max - y_margin*1.5, 'I', fontsize=26, ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
ax1.text(500, y_max - y_margin*1.5, 'II', fontsize=26, ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))

zoom_y_min = min(
    np.min(true_k[zoom_start:zoom_end]),
    np.min(gp_dpc_k[zoom_start:zoom_end]),
    np.min(p_dpc_k[zoom_start:zoom_end]),
    np.min(gamma_dpc_k[zoom_start:zoom_end])
)
zoom_y_max = max(
    np.max(true_k[zoom_start:zoom_end]),
    np.max(gp_dpc_k[zoom_start:zoom_end]),
    np.max(p_dpc_k[zoom_start:zoom_end]),
    np.max(gamma_dpc_k[zoom_start:zoom_end])
)
zoom_y_margin = (zoom_y_max - zoom_y_min) * 0.05

ax2 = fig.add_subplot(gs[1])
ax2.plot(frames[zoom_start:zoom_end], true_k[zoom_start:zoom_end], color='#B8860B', linestyle='-', linewidth=1.5)
ax2.plot(frames[zoom_start:zoom_end], gp_dpc_k[zoom_start:zoom_end], color='#3366CC', linestyle='--', linewidth=1.5)
ax2.plot(frames[zoom_start:zoom_end], p_dpc_k[zoom_start:zoom_end], color='#355E3B', linestyle='-.', linewidth=1.5)
ax2.plot(frames[zoom_start:zoom_end], gamma_dpc_k[zoom_start:zoom_end], color='#990033', linestyle=':', linewidth=1.5)

ax2.grid(True, linestyle='--', alpha=0.3)
ax2.set_xlabel('Frame Index')
ax2.set_ylabel('K-factor')
ax2.set_ylim(zoom_y_min - zoom_y_margin, zoom_y_max + zoom_y_margin)

# 在放大区域的图表中标记 "II"
ax2.text(500, zoom_y_max + zoom_y_margin*1.6, 'Enlarged Area II', fontsize=26, ha='center', va='bottom', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))

# 计算和添加RMSE
rmse_gp_dpc = np.sqrt(np.mean((true_k[:600] - gp_dpc_k[:600])**2))
rmse_p_dpc = np.sqrt(np.mean((true_k[:600] - p_dpc_k[:600])**2))
rmse_gamma_dpc = np.sqrt(np.mean((true_k[:600] - gamma_dpc_k[:600])**2))

stats_text = f'RMSE Analysis:\nGP-DPC: {rmse_gp_dpc:.4f}\nP-DPC: {rmse_p_dpc:.4f}\n$\gamma$-DPC: {rmse_gamma_dpc:.4f}'
plt.figtext(0.01, 0.01, stats_text, fontsize=18, family='serif')


plt.tight_layout()
plt.show()