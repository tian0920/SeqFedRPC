import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib

# 设置全局字体和样式
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Times New Roman'
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams['font.weight'] = 'bold'
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.it'] = 'STIXGeneral:italic'
matplotlib.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['axes.grid'] = True

# 设置线型和 marker
line_styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

# 数据根目录
root_dir = 'D://SeqFedRPC//out//scalability'
join_ratios = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))], key=float)

# 收集数据：dataset -> {join_ratio -> accuracy}
dataset_accuracy = defaultdict(dict)

for jr in join_ratios:
    jr_path = os.path.join(root_dir, jr)
    for file in os.listdir(jr_path):
        if file.endswith('.csv') and '_acc_metrics' in file:
            dataset = file.split('_acc_metrics')[0]
            file_path = os.path.join(jr_path, file)
            df = pd.read_csv(file_path)
            if 'test_before' in df.columns and len(df) >= 20:
                acc = round(df['test_before'].iloc[-20:].mean(), 3)
                dataset_accuracy[dataset][float(jr)] = acc

# 统一 join_ratio 排序
join_ratio_float = sorted([float(jr) for jr in join_ratios])

# 创建图像
plt.figure(figsize=(10, 6))

# 绘制每个数据集的曲线及最大点标注
for idx, (dataset, acc_dict) in enumerate(sorted(dataset_accuracy.items())):
    acc_values = [acc_dict.get(jr, None) for jr in join_ratio_float]
    plt.plot(
        join_ratio_float,
        acc_values,
        linestyle=line_styles[idx % len(line_styles)],
        marker=markers[idx % len(markers)],
        label=dataset.upper(),
        linewidth=2
    )

    # 找出最大值及其位置
    acc_series = pd.Series(acc_values, index=join_ratio_float)
    max_jr = acc_series.idxmax()
    max_val = acc_series.max()

    # 标注最大值（cifar100 偏移更大）
    offset = 2 if dataset.lower() == 'cifar100' else 0.2
    plt.annotate(
        f'{max_val:.3f}',
        xy=(max_jr, max_val),
        xytext=(max_jr, max_val + offset),
        fontsize=12,
        ha='center'
    )

# 图像标签和图例
plt.xlabel('Join Ratio', fontsize=16, fontweight='bold')
plt.ylabel('Accuracy', fontsize=16, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig('scalability.pdf')
plt.show()
