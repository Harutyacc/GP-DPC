import numpy as np

# 加载数据
components_results = np.load('results/differ_k_evaluation/components_results.npy', allow_pickle=True).item()

# 初始化统计字典
correct_counts = {
    'QPSK': {'DPC':0, 'DPC_no_greedy':0, 'GT-DPC':0},
    '8PSK': {'DPC':0, 'DPC_no_greedy':0, 'GT-DPC':0},
    '16QAM': {'DPC':0, 'DPC_no_greedy':0, 'GT-DPC':0}
}

# 遍历所有k值
for k in components_results:
    # 确定调制类型
    if k <= 8:
        mod_type = 'QPSK'
    elif k <= 14:
        mod_type = '8PSK'
    else:
        mod_type = '16QAM'
    
    # 累加正确计数
    for method in ['DPC', 'DPC_no_greedy', 'GT-DPC']:
        correct_counts[mod_type][method] += components_results[k][method]

# 计算正确率
total_frames = {
    'QPSK': 350,
    '8PSK': 300,
    '16QAM': 300
}

correct_rates = {}
for mod_type in correct_counts:
    correct_rates[mod_type] = {
        method: f"{count/total_frames[mod_type]*100:.2f}%"
        for method, count in correct_counts[mod_type].items()
    }

# 打印结果
print("调制类型 | DPC正确率 | DPC无贪心正确率 | GT-DPC正确率")
print("--------|-----------|----------------|-------------")
for mod_type in ['QPSK', '8PSK', '16QAM']:
    rates = correct_rates[mod_type]
    print(f"{mod_type:7} | {rates['DPC']:9} | {rates['DPC_no_greedy']:14} | {rates['GT-DPC']:10}")
