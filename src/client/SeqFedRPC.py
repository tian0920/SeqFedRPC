import pickle
from argparse import Namespace
from copy import deepcopy
from typing import Dict, List, Tuple, Union
from pathlib import Path

import torch, csv
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.cluster import KMeans

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

from src.config.utils import trainable_params, evalutate_model, Logger, evalutate_model_fedfew
from src.config.models import DecoupledModel
from data.utils.constants import MEAN, STD
from data.utils.datasets import DATASETS
import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import json


class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        args: Namespace,
        logger: Logger,
        device: torch.device,
    ):
        self.args = args
        self.device = device
        self.client_id: int = None
        self.assignment_mask = None  # 用于记录掩码
        self.fixed_epoch = 1
        self.sample_ratio = 0
        self.grad_means = []  # 用于存储每轮梯度均值

        # load dataset and clients' data indices
        try:
            partition_path = PROJECT_DIR / "data" / self.args.dataset / "partition.pkl"
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {args.dataset} first.")

        self.data_indices: List[List[int]] = partition["data_indices"]

        # --------- you can define your own data transformation strategy here ------------
        general_data_transform = transforms.Compose(
            [transforms.Normalize(MEAN[self.args.dataset], STD[self.args.dataset])]
        )
        general_target_transform = transforms.Compose([])
        train_data_transform = transforms.Compose([])
        train_target_transform = transforms.Compose([])
        # --------------------------------------------------------------------------------

        self.dataset = DATASETS[self.args.dataset](
            root=PROJECT_DIR / "data" / args.dataset,
            args=args.dataset_args,
            general_data_transform=general_data_transform,
            general_target_transform=general_target_transform,
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
        )

        self.trainloader: DataLoader = None
        self.testloader: DataLoader = None
        self.trainset: Subset = Subset(self.dataset, indices=[])
        self.testset: Subset = Subset(self.dataset, indices=[])
        self.global_testset: Subset = None
        if self.args.global_testset:
            all_testdata_indices = []
            for indices in self.data_indices:
                all_testdata_indices.extend(indices["test"])
            self.global_testset = Subset(self.dataset, all_testdata_indices)

        self.model = model.to(self.device)
        self.local_epoch = self.args.local_epoch
        # self.criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)


        self.logger = logger
        self.personal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.personal_params_name: List[str] = []
        self.init_personal_params_dict: Dict[str, torch.Tensor] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if not param.requires_grad
        }
        self.updated_parameters = {}
        self.opt_state_dict = {}
        self.updated_params = {}

        self.optimizer = torch.optim.SGD(
            params=trainable_params(self.model),
            lr=self.args.local_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.init_opt_state_dict = deepcopy(self.optimizer.state_dict())

        # 设置分布统计结果的保存路径
        self.distribution_save_path = "distribution_results"
        os.makedirs(self.distribution_save_path, exist_ok=True)

        self.best_assignment_mask = None
        self.best_cluster_quality = float('inf')  # Initial quality set to infinity (lower is better)
        self.update_counter = 0
        self.has_switched_to_personalized = False  # 标志是否切换到个性化训练
        self.clustering_done = False

    def load_dataset(self):
        """This function is for loading data indices for No.`self.client_id` client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.testset.indices = self.data_indices[self.client_id]["test"]
        self.trainloader = DataLoader(self.trainset, self.args.batch_size)
        if self.args.global_testset:
            self.testloader = DataLoader(self.global_testset, self.args.batch_size)
        else:
            self.testloader = DataLoader(self.testset, self.args.batch_size)


    def train_and_log(self, verbose=False) -> Dict[str, Dict[str, float]]:
        """This function includes the local training and logging process.

        Args:
            verbose (bool, optional): Set to `True` for print logging info onto the stdout (Controled by the server by default). Defaults to False.

        Returns:
            Dict[str, Dict[str, float]]: The logging info, which contains metric stats.
        """
        before = {
            "train_loss": 0,
            "test_loss": 0,
            "train_correct": 0,
            "test_correct": 0,
            "train_size": 1,
            "test_size": 1,
        }
        after = deepcopy(before)
        before = self.evaluate()
        if self.local_epoch > 0:
            self.fit()
            self.save_state()
            after = self.evaluate()
        if verbose:
            if len(self.trainset) > 0 and self.args.eval_train:
                self.logger.log(
                    "client [{}] (train)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["train_loss"] / before["train_size"],
                        after["train_loss"] / after["train_size"],
                        before["train_correct"] / before["train_size"] * 100.0,
                        after["train_correct"] / after["train_size"] * 100.0,
                    )
                )
            if len(self.testset) > 0 and self.args.eval_test:
                self.logger.log(
                    "client [{}] (test)  [bold red]loss: {:.4f} -> {:.4f}   [bold blue]acc: {:.2f}% -> {:.2f}%".format(
                        self.client_id,
                        before["test_loss"] / before["test_size"],
                        after["test_loss"] / after["test_size"],
                        before["test_correct"] / before["test_size"] * 100.0,
                        after["test_correct"] / after["test_size"] * 100.0,
                    )
                )

        eval_stats = {"before": before, "after": after}
        return eval_stats


    def save_state(self):
        """Save client model personal parameters and the state of optimizer at the end of local training."""
        self.personal_params_dict[self.client_id] = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items()
            if (not param.requires_grad) or (key in self.personal_params_name)
        }
        self.opt_state_dict[self.client_id] = deepcopy(self.optimizer.state_dict())


    def train(
        self,
        client_id: int,
        local_epoch: int,
        current_epoch: int,
        # lower_quantile: float,
        # upper_quantile:float,
        new_parameters: OrderedDict[str, torch.Tensor],
        return_diff=False,
        verbose=False,
    ) -> Tuple[Union[OrderedDict[str, torch.Tensor], List[torch.Tensor]], int, Dict]:
        # self.lower_quantile = lower_quantile
        # self.upper_quantile = upper_quantile
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.current_epoch = current_epoch
        self.last_epoch = -1
        self.new_parameters = new_parameters
        self.load_dataset()
        self.set_parameters(self.new_parameters)
        eval_stats = self.train_and_log(verbose=verbose)

        if return_diff:
            delta = OrderedDict()
            for (name, p0), p1 in zip(
                    new_parameters.items(), trainable_params(self.model)
            ):
                delta[name] = p0 - p1

            return delta, len(self.trainset), eval_stats
        else:
            return (
                trainable_params(self.model, detach=True),
                len(self.trainset),
                eval_stats,
            )

    def fit(self):
        self.model.train()

        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                # 前向传播
                if self.model.name == "DecoupledModel":
                    logit = self.model(x)
                else:
                    logit_vae_list = self.model(x)
                    logit = logit_vae_list[-1]
                if self.current_epoch < self.fixed_epoch:
                    self.criterion = CustomLoss(lambda_=0.01, beta=0.1,
                                                theta_global=list(self.new_parameters.values())).to(self.device)
                    loss = self.criterion(logit, y, self.old_parameters.values())
                else:
                    self.criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(self.device)
                    loss = self.criterion(logit, y)

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()


    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        """本地模型接收全局参数，并应用掩码逻辑更新本地模型"""
        self.old_parameters = {
            key: param.clone().detach()
            for key, param in self.model.state_dict(keep_vars=True).items() if key in new_parameters
        }
        self.optimizer.load_state_dict(
            self.opt_state_dict.get(self.client_id, self.init_opt_state_dict)
        )

        updated_parameters = self.update_parameters(self.old_parameters, new_parameters)
        self.model.load_state_dict(updated_parameters, strict=False)

    ###################################################  SeqFedRPC: 一次聚类为三类  + K-means ########################################################
    def update_parameters(self, old_parameters, new_parameters):
        # 创建一个字典来存储更新后的参数
        updated_parameters = OrderedDict()

        # 如果是第一次更新，只计算梯度并判断是否需要个性化训练
        if not self.has_switched_to_personalized:
            # 计算每轮梯度均值
            all_grads = []
            all_diffs = []
            for (name_old, old_param), (name_new, new_param) in zip(old_parameters.items(), new_parameters.items()):
                grad = new_param - old_param
                grad_mean = torch.mean(torch.abs(grad)).item()
                all_grads.append(grad_mean)

            epoch_grad_mean = sum(all_grads) / len(all_grads)
            self.grad_means.append(epoch_grad_mean)

            for (name_old, old_param), (name_new, new_param) in zip(old_parameters.items(), new_parameters.items()):
                assert name_old == name_new, "Parameter names do not match"
                # Calculate the difference and flatten to 1D
                diff = torch.abs(old_param - new_param)
                all_diffs.append(diff.view(-1))  # Flatten tensor to 1D

            # all_diffs_combined = torch.cat(all_diffs).cpu()  # Uncomment for histogram plotting
            # self._analyze_and_plot_distribution(all_diffs_combined)  # Optional plotting

            print(f"Update {self.update_counter}: Mean Gradient Value = {epoch_grad_mean:.6f}")

            # 判断是否需要切换到个性化训练
            if len(self.grad_means) > 1:
                prev_grad_mean = self.grad_means[-2]
                if prev_grad_mean > 0:
                    print("Significant gradient change detected. Switching to personalized training mode.")
                    self.has_switched_to_personalized = True
                else:
                    print("Performing standard training...")
                    self.has_switched_to_personalized = False

            # 如果没有切换到个性化训练，直接返回旧参数
            if not self.has_switched_to_personalized:
                return old_parameters

        # 聚类与掩码生成逻辑（仅执行一次）
        if not self.clustering_done:
            all_params_combined = []
            for name, param in old_parameters.items():
                all_params_combined.append(param.view(-1))

            # 合并所有参数
            all_params_combined = torch.cat(all_params_combined).cpu().numpy()

            # K-means 聚类
            kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(abs(all_params_combined).reshape(-1, 1))
            labels = kmeans.labels_
            self.clustering_done = True  # 标记聚类已完成

            # 确定最小和最大值的簇
            cluster_centers = kmeans.cluster_centers_.squeeze()
            min_cluster, max_cluster = cluster_centers.argmin(), cluster_centers.argmax()

            ################################################# 统计簇参数比例 ##################################################
            # unique_labels, counts = np.unique(labels, return_counts=True)
            # total_params = len(labels)
            # ratios = counts / total_params
            #
            # # 获取 K-means 簇中心，并按升序排序，重新映射类标签
            # cluster_centers = kmeans.cluster_centers_.squeeze()
            # sorted_indices = np.argsort(cluster_centers)  # 按簇中心升序排序
            #
            # # 确定各类对应的比例
            # small_class_ratio = ratios[sorted_indices[0]]  # 最小簇
            # large_class_ratio = ratios[sorted_indices[-1]]  # 最大簇
            # remaining_class_ratio = ratios[sorted_indices[1]]  # 中间簇
            #
            # filename = "params_50_0.2.csv"
            # file_exists = os.path.isfile(filename)
            # with open(filename, "a", newline='') as f:  # 使用 "a" 模式表示追加
            #     writer = csv.writer(f)
            #     # 写入表头（仅在文件不存在时写入一次）
            #     if not file_exists:
            #         writer.writerow(["Background", "Personalized", "Global"])
            #     # 写入数据行
            #     writer.writerow([f"{small_class_ratio:.3f}", f"{large_class_ratio:.3f}",
            #                      f"{remaining_class_ratio:.3f}"])
            # print("Cluster ratios appended to", filename)
            # ###############################################################################################################

            # 生成掩码
            self.assignment_mask = {}
            start_idx = 0

            for name, param in old_parameters.items():
                param_flat = param.view(-1)
                num_elements = param_flat.numel()

                # 根据聚类标签生成掩码
                param_labels = labels[start_idx: start_idx + num_elements]
                start_idx += num_elements
                mask = (param_labels == min_cluster) | (param_labels == max_cluster)
                self.assignment_mask[name] = torch.tensor(mask, dtype=torch.bool, device=param.device).view(param.shape)

            print("Clustering completed. Assignment mask generated.")

        # 使用掩码更新参数
        for name, (old_param, new_param) in zip(old_parameters.keys(),
                                                zip(old_parameters.values(), new_parameters.values())):
            mask = self.assignment_mask[name]
            updated_param = torch.where(mask, old_param, new_param)
            updated_parameters[name] = updated_param

        self.update_counter += 1  # 更新计数器
        return updated_parameters


    # ################################################## 两次聚类为2类 + 2类 + K-means  #######################################################
    # def update_parameters(self, old_parameters, new_parameters):
    #     # 创建一个字典来存储更新后的参数
    #     updated_parameters = OrderedDict()
    #     # 如果是第一次更新，只计算梯度并判断是否需要个性化训练
    #     if not self.has_switched_to_personalized:
    #         # 计算每轮梯度均值
    #         all_grads = []
    #         all_diffs = []
    #         for (name_old, old_param), (name_new, new_param) in zip(old_parameters.items(), new_parameters.items()):
    #             grad = new_param - old_param
    #             grad_mean = torch.mean(torch.abs(grad)).item()
    #             all_grads.append(grad_mean)
    #
    #         epoch_grad_mean = sum(all_grads) / len(all_grads)
    #         self.grad_means.append(epoch_grad_mean)
    #
    #         for (name_old, old_param), (name_new, new_param) in zip(old_parameters.items(), new_parameters.items()):
    #             assert name_old == name_new, "Parameter names do not match"
    #             # Calculate the difference and flatten to 1D
    #             ############# 1. 梯度 ############
    #             # diff = torch.abs(old_param - new_param)
    #             # all_diffs.append(diff.view(-1))  # Flatten tensor to 1D
    #
    #             ############# 2. fisher值 ############
    #             # diff = torch.abs(old_param - new_param)
    #             # modified_diff = diff ** 2  # fisher值
    #             # all_diffs.append(modified_diff.view(-1))  # Flatten tensor to 1D
    #
    #             ############# 3. importance 指标  ############
    #             theta = torch.abs(old_param)
    #             diff = torch.abs(old_param - new_param)
    #             modified_diff = diff * theta  # importance 指标
    #             all_diffs.append(modified_diff.view(-1))  # Flatten tensor to 1D
    #
    #         all_diffs_combined = torch.cat(all_diffs).cpu()  # Uncomment for histogram plotting
    #         self._analyze_and_plot_distribution(all_diffs_combined)  # Optional plotting
    #
    #         print(f"Update {self.update_counter}: Mean Gradient Value = {epoch_grad_mean:.6f}")
    #
    #         # 判断是否需要切换到个性化训练
    #         if len(self.grad_means) > 1:
    #             prev_grad_mean = self.grad_means[-2]
    #             # 判断梯度变化
    #             # if prev_grad_mean > 0 and abs(epoch_grad_mean / prev_grad_mean) > 3 or epoch_grad_mean > 0.001:
    #             if prev_grad_mean > 0:
    #                 print("Significant gradient change detected. Switching to personalized training mode.")
    #                 self.has_switched_to_personalized = True
    #             else:
    #                 print("Performing standard training...")
    #                 self.has_switched_to_personalized = False
    #
    #         # 如果没有切换到个性化训练，直接返回旧参数
    #         if not self.has_switched_to_personalized:
    #             return old_parameters
    #
    #     # 聚类与掩码生成逻辑（仅执行一次）
    #     if not self.clustering_done:
    #         self.clustering_done = True  # 标记聚类已完成
    #         all_params_combined = []
    #         for name, param in old_parameters.items():
    #             all_params_combined.append(param.view(-1))
    #
    #         # 合并所有参数
    #         all_params_combined = torch.cat(all_params_combined).cpu().numpy()
    #
    #         ### 第一次聚类 ###
    #         # K-means 聚类为 2 类（小 和 大）
    #         kmeans_1 = KMeans(n_clusters=2, random_state=0, n_init=10).fit(abs(all_params_combined).reshape(-1, 1))
    #         labels_1 = kmeans_1.labels_
    #
    #         # 确定第一次聚类中“小的类”
    #         cluster_centers_1 = kmeans_1.cluster_centers_.squeeze()
    #         min_cluster_1 = cluster_centers_1.argmin()  # 第一次聚类中最小的簇
    #
    #         # 第一次掩码：小的类被掩码覆盖
    #         first_mask_indices = (labels_1 == min_cluster_1)
    #
    #         ### 第二次聚类 ###
    #         # 从第一次聚类的“大类”中筛选参数
    #         large_param_indices = (labels_1 != min_cluster_1)  # 非“小类”的索引
    #         large_params = all_params_combined[large_param_indices]
    #
    #         # 再次进行 K-means 聚类为 2 类（小 和 大）
    #         kmeans_2 = KMeans(n_clusters=2, random_state=0, n_init=10).fit(abs(large_params).reshape(-1, 1))
    #         labels_2 = kmeans_2.labels_
    #
    #         # 确定第二次聚类中“大类”
    #         cluster_centers_2 = kmeans_2.cluster_centers_.squeeze()
    #         max_cluster_2 = cluster_centers_2.argmax()  # 第二次聚类中最大的簇
    #
    #         # 第二次掩码：在第一次“大类”中，标记出第二次聚类的“大类”
    #         second_mask_indices = large_param_indices.copy()  # 基于第一次结果的索引
    #         second_mask_indices[large_param_indices] = (labels_2 == max_cluster_2)  # 大类掩码
    #
    #         ### 合并两次掩码 ###
    #         final_mask_indices = first_mask_indices | second_mask_indices  # 两次掩码取 OR
    #
    #         ################################################# 统计簇参数比例 ##################################################
    #         # 统计三个类别的参数数量
    #         total_params = len(all_params_combined)
    #         small_class_count = np.sum(first_mask_indices)  # 第一次聚类的小类数量
    #         large_class_count = np.sum(second_mask_indices)  # 第二次聚类的大类数量
    #         remaining_class_count = total_params - small_class_count - large_class_count  # 剩余的类数量
    #
    #         # 计算比例
    #         small_class_ratio = small_class_count / total_params
    #         large_class_ratio = large_class_count / total_params
    #         remaining_class_ratio = remaining_class_count / total_params
    #
    #         file_exists = os.path.isfile("cluster_params_ratios.csv")
    #         with open("cluster_params_ratios.csv", "a", newline='') as f:  # 使用 "a" 模式表示追加
    #             writer = csv.writer(f)
    #             # 写入表头（仅在文件不存在时写入一次）
    #             if not file_exists:
    #                 writer.writerow(["Dataset", "Background", "Personalized", "Global"])
    #             # 写入数据行
    #             writer.writerow([self.dataset.targets, f"{small_class_ratio:.4f}", f"{large_class_ratio:.4f}",
    #                              f"{remaining_class_ratio:.4f}"])
    #         print("Cluster ratios appended to", "cluster_params_ratios.csv")
    #         ###############################################################################################################
    #
    #         # 生成掩码字典
    #         self.assignment_mask = {}
    #         start_idx = 0
    #         for name, param in old_parameters.items():
    #             param_flat = param.view(-1)
    #             num_elements = param_flat.numel()
    #             # 提取当前参数的掩码
    #             mask = final_mask_indices[start_idx: start_idx + num_elements]
    #             start_idx += num_elements
    #             # 生成掩码并保存到字典中
    #             self.assignment_mask[name] = torch.tensor(mask, dtype=torch.bool, device=param.device).view(param.shape)
    #         print("Two-step clustering completed. Assignment mask generated.")
    #
    #     # 使用掩码更新参数
    #     for name, (old_param, new_param) in zip(old_parameters.keys(),
    #                                             zip(old_parameters.values(), new_parameters.values())):
    #         mask = self.assignment_mask[name]
    #         updated_param = torch.where(mask, old_param, new_param)
    #         updated_parameters[name] = updated_param
    #
    #     self.update_counter += 1  # 更新计数器
    #     return updated_parameters


    def _analyze_and_plot_distribution(self, all_diffs_combined):
        """计算最大值和最小值，等间隔划分区间，绘制分布图，并将结果保存至文件"""

        # 过滤掉等于0的差值
        non_zero_diffs = all_diffs_combined[all_diffs_combined != 0]

        # 如果数据点数少于1000，直接跳过
        if len(non_zero_diffs) <= 1000:
            return

        # 计算最大值和最小值
        max_value = torch.max(non_zero_diffs).item()
        min_value = torch.min(non_zero_diffs).item()

        # 划分区间并统计分布
        num_bins = 100  # 设置区间数
        bins = torch.linspace(min_value, max_value, num_bins + 1)
        hist, _ = torch.histogram(non_zero_diffs, bins=bins)

        # 绘制分布图
        plt.figure(figsize=(14, 8))
        plt.hist(non_zero_diffs.numpy(), bins=bins.numpy(), color="skyblue", edgecolor="black")
        plt.yscale("log")  # 设置 y 轴为对数刻度

        # 设置坐标轴标签字体大小
        plt.xlabel("Gradient Difference", fontsize=22)  # 设置 x 轴字体大小
        plt.ylabel("Proportion (Log Scale)", fontsize=22)  # 设置 y 轴字体大小

        # 添加网格
        plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7, alpha=0.7)

        # 设置坐标轴刻度字体大小
        plt.xticks(fontsize=18)  # 设置 x 轴刻度字体大小
        plt.yticks(fontsize=18)  # 设置 y 轴刻度字体大小

        # 绘制标题
        # plt.title(f"Distribution of all_diffs_combined at Epoch {self.current_epoch}", fontsize=18)

        # 显示图像
        plt.show()

        # 检查保存路径是否存在，不存在则创建
        os.makedirs(self.distribution_save_path, exist_ok=True)

        # 保存图像
        plot_path = os.path.join(self.distribution_save_path, f"distribution_epoch_{self.current_epoch}.png")
        plt.savefig(plot_path)
        plt.close()

        # 保存分布结果至文件
        distribution_data = {
            "epoch": self.current_epoch,
            "min_value": min_value,
            "max_value": max_value,
            "histogram": hist.tolist(),
            "bins": bins.tolist()
        }
        json_path = os.path.join(self.distribution_save_path, f"distribution_data_epoch_{self.current_epoch}.json")
        with open(json_path, "w") as f:
            json.dump(distribution_data, f, indent=4)


    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> Dict[str, float]:
        """The evaluation function. Would be activated before and after local training if `eval_test = True` or `eval_train = True`.

        Args:
            model (torch.nn.Module, optional): The target model needed evaluation (set to `None` for using `self.model`). Defaults to None.

        Returns:
            Dict[str, float]: The evaluation metric stats.
        """
        # disable train data transform while evaluating
        self.dataset.enable_train_transform = False

        eval_model = self.model if model is None else model
        eval_model.eval()
        train_loss, test_loss = 0, 0
        train_correct, test_correct = 0, 0
        train_sample_num, test_sample_num = 0, 0
        if self.current_epoch < self.fixed_epoch:
            criterion = CustomLoss(lambda_=0.01, beta=-0.1, theta_global=list(self.new_parameters.values())).to(
                self.device)
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction="sum")


        if len(self.testset) > 0 and self.args.eval_test:
            test_loss, test_correct, test_sample_num = evalutate_model_fedfew(
                current_epoch=self.current_epoch,
                fixed_epoch=self.fixed_epoch,
                old_parameters=self.old_parameters,
                model=eval_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if len(self.trainset) > 0 and self.args.eval_train:
            train_loss, train_correct, train_sample_num = evalutate_model_fedfew(
                model=eval_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )

        self.dataset.enable_train_transform = True

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_correct": train_correct,
            "test_correct": test_correct,
            "train_size": float(max(1, train_sample_num)),
            "test_size": float(max(1, test_sample_num)),
        }

    def test(
        self, client_id: int, new_parameters: OrderedDict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """Test function. Only be activated while in FL test round.

        Args:
            client_id (int): The ID of client.
            new_parameters (OrderedDict[str, torch.Tensor]): The FL model parameters.

        Returns:
            Dict[str, Dict[str, float]]: the evalutaion metrics stats.
        """
        self.client_id = client_id
        self.load_dataset()
        self.set_parameters(new_parameters)

        before = {
            "train_loss": 0,
            "train_correct": 0,
            "train_size": 1.0,
            "test_loss": 0,
            "test_correct": 0,
            "test_size": 1.0,
        }
        after = deepcopy(before)

        before = self.evaluate()
        if self.args.finetune_epoch > 0:
            self.finetune()
            after = self.evaluate()
        return {"before": before, "after": after}

    def finetune(self):
        """
        The fine-tune function. If your method has different fine-tuning opeation, consider to override this.
        This function will only be activated while in FL test round.
        """
        self.model.train()
        for _ in range(self.args.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                # logit = self.model(x)
                if self.model.name == "DecoupledModel":
                    logit = self.model(x)
                else:
                    logit_vae_list = self.model(x)
                    logit = logit_vae_list[-1]
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

class CustomLoss(torch.nn.Module):
    def __init__(self, lambda_, beta, theta_global):
        super(CustomLoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.lambda_ = lambda_
        self.beta = beta
        self.theta_global = theta_global

    def forward(self, outputs, targets, model_params):
        # 基础交叉熵损失
        ce_loss = self.cross_entropy(outputs, targets)

        # 稀疏化项: L1 正则化
        l1_regularization = self.lambda_ * sum(torch.sum(torch.abs(param)) for param in model_params)

        # 差值项: 个性化差值双侧拉伸
        difference_loss = 0
        for param_local, param_global in zip(model_params, self.theta_global):
            diff = param_local - param_global
            difference_loss -= torch.sum(self.beta * torch.abs(diff)**2)  # 这里使用平方作为差值拉伸

        # 总损失
        total_loss = ce_loss + l1_regularization + difference_loss
        return total_loss
