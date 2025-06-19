import subprocess
import sys
from pathlib import Path


def run_command(command, cwd=None):
    """
    直接运行命令，将输出打印到控制台即可，不保存日志文件。
    """
    process = subprocess.Popen(command, cwd=cwd)
    process.wait()
    if process.returncode != 0:
        print(f"命令失败: {' '.join(command)}")
    else:
        print(f"命令成功完成")


def main():
    ## baselines
    datasets_name = ['cifar10', 'cifar100', 'cinic10', 'emnist', ]
    # datasets_name = ['tiny_imagenet', ]

    # methods = ['fedfomo', 'pfedsim', 'fedrep', ]
    # methods = ['pfedsim', 'feddyn', 'lgfedavg', ]
    methods = ['pspfl', ]
    # methods = ['fedfomo', 'fedper', 'fedrep', 'lgfedavg', 'fedavg', 'local', 'cfl', 'feddyn', 'pfedsim']


    # ## 0.1
    # datasets_name = ['mnist', 'tiny_imagenet']
    # datasets_name = ['cifar10', 'cifar100', 'mnist', 'fmnist', 'svhn', 'cinic10', 'emnist', 'tiny_imagenet']
    # datasets_name = ['cifar10', 'cifar100', 'mnist', 'fmnist', ]
    # methods = ['0.1']

    # 自动识别当前脚本所在的目录，并切换到其下的 server 文件夹
    base_dir = Path(__file__).resolve().parent / 'src' / 'server'

    for method in methods:
        for dataset in datasets_name:
            if dataset == 'svhn' or dataset == 'emnist' or dataset == 'cifar10':
                command = [
                    sys.executable,
                    str(base_dir / f'{method}.py'),
                    '-d', dataset,
                    '--global_epoch', '100',
                    '--join_ratio', '0.1',
                    '--local_lr', '5e-4'
                ]
            else:
                command = [
                    sys.executable,
                    str(base_dir / f'{method}.py'),
                    '-d', dataset,
                    '--global_epoch', '100',
                    '--join_ratio', '0.1',
                    '--local_lr', '1e-3'
                ]

            print(f"运行命令: {' '.join(command)}")
            run_command(command, cwd=str(base_dir))

    print("\n所有实验已完成。")


if __name__ == '__main__':
    main()

