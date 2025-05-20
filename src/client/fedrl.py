from collections import OrderedDict
from typing import Dict, Set
from pathlib import Path
from copy import deepcopy

import torch

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

from fedavg import FedAvgClient


class FedrlClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)

        # 存储个性化层的名称
        self.personalized_layer_names = set()
        self.personal_params = {}

    def set_personalized_layers(self, layer_names: Set[str]):
        """接收服务器端传递的个性化层名称"""
        self.personalized_layer_names = layer_names

    def fit(self):
        """客户端本地训练"""
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        # 保存个性化参数
        self.save_personalized_parameters()

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
                loss = self.criterion(logit, y)

                # 反向传播并优化
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

                total_loss += loss.sum().item()
                total_samples += y.size(0)

        # 计算平均损失
        avg_loss = total_loss / total_samples

    def save_personalized_parameters(self):
        """保存个性化层的参数"""
        model = deepcopy(self.model)
        personalized_parameters = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if name in self.personalized_layer_names
        }
        self.personal_params = personalized_parameters

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        """加载全局参数，并根据个性化层的掩码应用个性化的参数。"""
        # 加载全局参数
        self.model.load_state_dict(new_parameters, strict=False)

        for name, mask in self.personal_params.items():
            self.model.state_dict()[name] = mask.clone().detach()