from fedavg import FedAvgServer
from copy import deepcopy
from argparse import Namespace
from typing import List, OrderedDict, Tuple, Dict
import torch
from src.client.fedseq import FedSeqClient
import random

class FedSeqServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedSeq_random",
        args: Namespace = None,
        unique_model=False,
        default_trainer=False,
    ):
        super().__init__(algo, args, unique_model, default_trainer)
        # 确保每一轮都有所有客户端参与
        self.args.num_clients_per_round = len(self.train_clients)

        # 初始化 selected_clients 为空列表
        self.selected_clients: List[int] = []

        # 初始化 client_stats 字典，存储每个客户端每轮的评估统计
        if not hasattr(self, 'client_stats'):
            self.client_stats: Dict[int, Dict[int, Dict[str, float]]] = {c: {} for c in self.train_clients}

    def train(self):
        """
        重写 train 方法以实现顺序训练。
        """
        for E in self.train_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log("-" * 26, f"TRAINING EPOCH: {E + 1}", "-" * 26)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.train_one_round()
            self.log_info()

    def train_one_round(self):
        """
        实现随机选择客户端进行顺序训练。
        """
        # 初始化模型参数
        model_parameters = deepcopy(self.global_params_dict)

        # 随机选择一部分客户端进行训练
        self.selected_clients = random.sample(self.train_clients, self.args.num_clients_per_round)

        # 逐个客户端顺序迭代
        for idx, client_id in enumerate(self.selected_clients):
            self.logger.log(
                f"Training on client {idx + 1}/{len(self.selected_clients)} (Client ID: {client_id})"
            )

            # 客户端训练并返回更新的模型参数和评估统计
            updated_parameters, eval_stats = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                model_parameters=model_parameters,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            # 更新模型参数供下一个客户端使用
            model_parameters = updated_parameters

            # 收集评估统计信息
            if client_id not in self.client_stats:
                self.client_stats[client_id] = {}
            self.client_stats[client_id][self.current_epoch] = eval_stats

        # 使用最终的参数更新服务器的全局模型
        self.global_params_dict = model_parameters
        self.model.load_state_dict(self.global_params_dict, strict=False)

    def test(self):
        """使用父类的 test 方法进行评估。"""
        super().test()

    def log_info(self):
        """This function is for logging each selected client's training info."""
        for label in ["train", "test"]:
            # In the `user` split, there is no test data held by train clients, so plotting is unnecessary.
            if (label == "train" and self.args.eval_train) or (
                    label == "test"
                    and self.args.eval_test
                    and self.args.dataset_args["split"] != "user"
            ):
                correct_before = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["before"]["correct"]
                        for c in self.selected_clients
                    ]
                )
                correct_after = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["after"]["correct"]
                        for c in self.selected_clients
                    ]
                )
                num_samples = torch.tensor(
                    [
                        self.client_stats[c][self.current_epoch]["before"]["sample_num"]
                        for c in self.selected_clients
                    ]
                )

                acc_before = (
                        correct_before.sum(dim=-1, keepdim=True) / num_samples.sum() * 100.0
                ).item()
                acc_after = (
                        correct_after.sum(dim=-1, keepdim=True) / num_samples.sum() * 100.0
                ).item()

                if self.args.visible:
                    self.viz.line(
                        [acc_before],
                        [self.current_epoch],
                        win=self.viz_win_name,
                        update="append",
                        name=f"{label}_acc(before)",
                        opts=dict(
                            title=self.viz_win_name,
                            xlabel="Communication Rounds",
                            ylabel="Accuracy",
                        ),
                    )
                    self.viz.line(
                        [acc_after],
                        [self.current_epoch],
                        win=self.viz_win_name,
                        update="append",
                        name=f"{label}_acc(after)",
                    )


if __name__ == "__main__":
    server = FedSeqServer(default_trainer=False)
    server.trainer = FedSeqClient(
        deepcopy(server.model), server.args, server.logger, server.device
    )
    server.run()