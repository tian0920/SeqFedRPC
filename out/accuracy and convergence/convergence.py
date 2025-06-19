import os
import pandas as pd
import matplotlib.pyplot as plt
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

matplotlib.rcParams['axes.linewidth'] = 1  # 线型粗细为 1
matplotlib.rcParams['axes.grid'] = True  # 启用网格

# 根目录路径，包含13个算法文件夹
# root_dir = 'D:\\0.1\\out\\accuracy and convergence\\alpha=1.0'  # 检查并创建输出目录
root_dir = 'alpha=1.0'  # 检查并创建输出目录


# 获取所有算法文件夹的列表
algorithm_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# 创建一个字典，用于存储相同数据集名称的 "test_before" 数据
dataset_dict = {}

# 设置窗口大小
window_size = 10  # 可以根据需要调整窗口大小

for algorithm in algorithm_folders:
    algorithm_dir = os.path.join(root_dir, algorithm)

    # 获取当前算法文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(algorithm_dir) if f.endswith('.csv')]

    for csv_file in csv_files:
        csv_path = os.path.join(algorithm_dir, csv_file)

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        dataset_name = csv_file[:csv_file.index("_acc_metrics")]

        # 检查CSV文件是否包含 'test_before' 列
        if dataset_name not in dataset_dict:
            dataset_dict[dataset_name] = {}

        # 平滑数据（使用滚动窗口）
        # 只使用前100个数据点再进行平滑处理
        smoothed_data = df['test_before'].iloc[:100].rolling(window=window_size, min_periods=1).mean()
        dataset_dict[dataset_name][algorithm] = smoothed_data

# 颜色列表，确保足够多的颜色
colors = [
    '#FFD700',  # 金色 (Gold)
    '#00FF00',  # 绿色 (Green)
    '#0000FF',  # 蓝色 (Blue)
    '#32CD32',  # 新绿色 (Lime Green)
    '#FF00FF',  # 洋红 (Magenta)
    '#00FFFF',  # 青色 (Cyan)
    '#800080',  # 紫色 (Purple)
    '#FFA500',  # 橙色 (Orange)
    '#A52A2A',  # 棕色 (Brown)
    '#008000',  # 深绿色 (Dark Green)
    '#FF1493',  # 深粉色 (Deep Pink)
    '#808080',  # 灰色 (Gray)
    '#4682B4'   # 钢蓝色 (Steel Blue)
]

# 修改为2行5列的布局，确保每个子图宽高比为3:4
fig, axes = plt.subplots(4, 2, figsize=(8, 10.5))
axes = axes.flatten()  # 将axes从二维数组平铺成一维数组

# 确保绘制的图数量不超过10张
for i, (dataset_name, data) in enumerate(dataset_dict.items()):
    if i >= 8:  # 因为2*5=10个子图
        break

    ax = axes[i]

    # 为每个算法分配不同颜色
    for j, (algorithm, values) in enumerate(data.items()):
        if algorithm == '0.1':
            ax.plot(values, label='0.1', linewidth=1.8, color='#FF0000')
        else:
            ax.plot(values, label=algorithm, color=colors[j % len(colors)], linewidth=1)  # 使用颜色列表

    # 设置标题
    label_fontsize = 9
    legend_fontsize = 7

    cleaned_name = dataset_name.split('_acc_metrics')[0]
    ax.set_title(f'{cleaned_name.upper()}', fontsize=label_fontsize)

    # 设置y轴从40开始（仅针对指定的数据集）
    if dataset_name in ['emnist', 'fmnist', ]:
        ax.set_ylim(bottom=50)
    if dataset_name in ['mnist']:
        ax.set_ylim(bottom=70)
    if dataset_name in ['svhn', 'cinic10', 'cifar10']:
        ax.set_ylim(bottom=40)
    if dataset_name in ['cifar100']:
        ax.set_ylim(bottom=0)

    ax.set_xlabel('Communication Epochs', fontsize=label_fontsize)
    ax.set_ylabel('Accuracy', fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=legend_fontsize)  # 调整坐标轴字体大小
    ax.grid(True, color='lightgray', linestyle='--', linewidth=0.3)
    ax.legend(fontsize=legend_fontsize)

# 隐藏未使用的子图（若图数量少于10个）
for i in range(len(dataset_dict), 8):
    fig.delaxes(axes[i])

# 保存图像
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.32)  # 适当调节数字，wspace越小间距越小
plt.savefig('convergence_vgg11_1.0.pdf', dpi=300)
plt.show()
plt.close()

print("Combined plots have been generated and saved.")
