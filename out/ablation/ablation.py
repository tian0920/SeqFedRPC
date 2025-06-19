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

# 文件夹路径
folder_paths = [
    'D:\\0.1\\out\\ablation\\FedAvg',
    'D:\\0.1\\out\\ablation\\Local-only',
    'D:\\0.1\\out\\ablation\\SFL',
    'D:\\0.1\\out\\ablation\\0.1',
    'D:\\0.1\\out\\ablation\\SeqFedRPC_L0',
    'D:\\0.1\\out\\ablation\\SeqFedRPC_L1',
    'D:\\0.1\\out\\ablation\\SeqFedRPC_L2',]

# 创建一个空的字典来存储每个数据集的DataFrame
dataset_data = {}

# 滑动窗口大小
window_size = 8

# 遍历每个文件夹
for folder_path in folder_paths:
    # 获取文件夹中所有CSV文件的路径
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # 遍历每个CSV文件
    for file in csv_files:
        # 提取数据集名称
        dataset_name = file[:file.index("_acc_metrics")]

        # 读取CSV文件的第一列数据，只取前200行数据
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df = df['test_before'].iloc[:100].round(3)  # 保留三位小数

        # 对数据进行平滑处理
        df_smoothed = df.rolling(window=window_size, min_periods=1).mean()

        # 将平滑数据添加到相应的数据集中
        if dataset_name not in dataset_data:
            dataset_data[dataset_name] = [df_smoothed]
        else:
            dataset_data[dataset_name].append(df_smoothed)

# 计算需要绘制的子图数量
num_plots = len(dataset_data)

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 10))

# 绘制折线图
for i, (dataset_name, dfs) in enumerate(dataset_data.items()):
    row = i // 2
    col = i % 2

    for j, df in enumerate(dfs):
        if j == 0:
            label = 'FedAvg'
        elif j == 1:
            label = 'Local-Only'
        elif j == 2:
            label = 'SFL'
        elif j == 3:
            label = '0.1'
        elif j == 4:
            label = 'SeqFedRPC_$L_0$'
        elif j == 5:
            label = 'SeqFedRPC_$L_1$'
        elif j == 6:
            label = 'SeqFedRPC_$L_2$'

        if label == '0.1':
            axes[row, col].plot(df.index, df, label=label, linewidth=2, color='red')
        else:
            axes[row, col].plot(df.index, df, label=label, linewidth=1.1)


    # 设置当前子图的标题和标签
    if dataset_name in ['emnist', 'fmnist']:
        axes[row, col].set_ylim(40, 100)  # 这里设定 y 轴范围
    elif dataset_name in ['cinic10']:
        axes[row, col].set_ylim(20, 100)
    elif dataset_name in ['mnist']:
        axes[row, col].set_ylim(60, 100)

    axes[row, col].set_title(dataset_name.upper(), fontsize=10)
    axes[row, col].set_xlabel('Communication Epochs', fontsize=10)
    axes[row, col].set_ylabel('Accuracy', fontsize=10)
    # axes[row, col].set_ylim(0, 100)  # 设置 y 轴范围为 0 到 100
    axes[row, col].legend(fontsize=9)  # 添加图例
    axes[row, col].grid(True, color='lightgray', linestyle='--', linewidth=0.3)

# 调整子图间的间距
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # 适当调节数字，wspace越小间距越小
# 保存图片为PDF格式
plt.savefig('ablation.pdf')
# 显示图表
plt.show()
