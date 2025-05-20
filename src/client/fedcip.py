# from collections import OrderedDict
# from pathlib import Path
# import torch
#
# PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
#
# from fedavg import FedAvgClient
#
#
# class FedCIPClient(FedAvgClient):
#     def __init__(self, model, args, logger, device):
#         super().__init__(model, args, logger, device)
#         # 添加阈值 'a' 和 'b' 的属性
#         self.gradient_threshold_a = getattr(self.args, 'gradient_threshold_a', 0.01)  # 梯度阈值 'a'
#         self.delta_threshold_b = getattr(self.args, 'delta_threshold_b', 0.01)        # 参数差值阈值 'b'
#
#     def fit(self):
#         self.model.train()
#
#         # 保存训练前的初始参数
#         initial_params = {
#             name: param.clone().detach()
#             for name, param in self.model.named_parameters()
#         }
#
#         total_loss = 0.0
#         total_samples = 0
#
#         for _ in range(self.local_epoch):
#             for x, y in self.trainloader:
#                 if len(x) <= 1:
#                     continue
#                 x, y = x.to(self.device), y.to(self.device)
#
#                 # 前向传播
#                 if self.model.name == "DecoupledModel":
#                     logit = self.model(x)
#                 else:
#                     logit_vae_list = self.model(x)
#                     logit = logit_vae_list[-1]
#                 loss = self.criterion(logit, y)
#
#                 # 反向传播和优化
#                 self.optimizer.zero_grad()
#                 loss.mean().backward()
#                 # 在优化器更新前获取梯度
#                 gradients = {
#                     name: param.grad.clone().detach()
#                     for name, param in self.model.named_parameters()
#                     if param.grad is not None
#                 }
#                 self.optimizer.step()
#
#                 total_loss += loss.sum().item()
#                 total_samples += y.size(0)
#
#         # 计算平均损失
#         avg_loss = total_loss / total_samples
#
#         # 保存训练后的更新参数
#         updated_params = {
#             name: param.clone().detach()
#             for name, param in self.model.named_parameters()
#         }
#
#         # 计算参数更新的差值
#         delta_params = {}
#         for name in updated_params:
#             delta_params[name] = updated_params[name] - initial_params[name]
#
#         # 根据梯度阈值 'a' 生成掩码张量
#         a = self.gradient_threshold_a
#         mask_params = {}
#         for name in delta_params:
#             mask_params[name] = (delta_params[name].abs() > a).float()
#
#         # 将掩码应用于参数更新，得到稀疏参数
#         masked_params = {}
#         for name in updated_params:
#             masked_params[name] = updated_params[name] * mask_params[name]
#
#         # 存储稀疏参数和掩码，以便上传到服务器
#         self.masked_params = masked_params
#         self.mask_params = mask_params
#
#         # 也存储更新后的参数，以便在 set_parameters() 中使用
#         self.updated_params = updated_params
#
#     def get_parameters(self):
#         # 返回要发送到服务器的稀疏参数
#         return self.masked_params
#
#     def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
#         # 如果 updated_params 不存在，则初始化为 new_parameters
#         if not hasattr(self, 'updated_params'):
#             self.updated_params = {
#                 name: param.clone().detach()
#                 for name, param in new_parameters.items()
#             }
#             # 初次调用时，直接加载新参数到模型中
#             self.model.load_state_dict(new_parameters, strict=False)
#             # 初始化优化器
#             optimizer_parameters = self.model.parameters()
#             self.optimizer = torch.optim.SGD(
#                 params=optimizer_parameters,
#                 lr=self.args.local_lr,
#                 momentum=self.args.momentum,
#                 weight_decay=self.args.weight_decay,
#             )
#             # 重置优化器状态
#             self.optimizer.state = {}
#             return  # 初次调用时，不需要执行后续操作
#
#         # 以下是原有的参数更新逻辑
#         # 计算接收到的参数与本地更新参数的差值
#         delta_received = {}
#         for name in new_parameters:
#             delta_received[name] = new_parameters[name] - self.updated_params[name]
#
#         # 参数差值阈值 'b'
#         b = self.delta_threshold_b
#
#         # 准备模型的状态字典
#         state_dict = self.model.state_dict()
#
#         # 根据阈值 'b' 更新参数
#         for name in new_parameters:
#             # 创建掩码，当差值超过阈值 'b' 时，掩码位置设为1
#             mask = (delta_received[name].abs() > b).float()
#             # 用本地参数覆盖掩码为1的位置
#             state_dict[name] = new_parameters[name] * (1 - mask) + self.updated_params[name] * mask
#
#         # 加载更新后的参数到模型中
#         self.model.load_state_dict(state_dict)
#
#         # 重新初始化优化器
#         optimizer_parameters = self.model.parameters()
#         self.optimizer = torch.optim.SGD(
#             params=optimizer_parameters,
#             lr=self.args.local_lr,
#             momentum=self.args.momentum,
#             weight_decay=self.args.weight_decay,
#         )
#         # 重置优化器状态
#         self.optimizer.state = {}

#####################################################
#
# from collections import OrderedDict
# import torch
# from typing import Dict, List, Tuple, Union
# from .fedavg_V3 import FedAvgClient
#
# class FedCIPClient(FedAvgClient):
#     def __init__(self, model, args, logger, device):
#         super().__init__(model, args, logger, device)
#         # 添加梯度阈值 'a' 和参数差值阈值 'b'
#         self.gradient_threshold_a = getattr(self.args, 'gradient_threshold_a', 1e-2)  # 梯度阈值 'a'
#         # self.delta_threshold_b = getattr(self.args, 'delta_threshold_b', 1e-1)        # 参数差值阈值 'b'
#
#
#     def fit(self):
#         """本地训练，并生成稀疏的参数更新"""
#         self.model.train()
#
#         # 保存训练前的初始参数
#         initial_params = {
#             name: param.clone().detach()
#             for name, param in self.model.named_parameters()
#         }
#
#         # total_loss = 0.0
#         # total_samples = 0
#
#         for _ in range(self.local_epoch):
#             for x, y in self.trainloader:
#                 if len(x) <= 1:
#                     continue
#                 x, y = x.to(self.device), y.to(self.device)
#
#                 # 前向传播
#                 if self.model.name == "DecoupledModel":
#                     logit = self.model(x)
#                 else:
#                     logit_vae_list = self.model(x)
#                     logit = logit_vae_list[-1]
#                 loss = self.criterion(logit, y)
#
#                 # 反向传播和优化
#                 self.optimizer.zero_grad()
#                 loss.mean().backward()
#                 self.optimizer.step()
#
#                 # total_loss += loss.sum().item()
#                 # total_samples += y.size(0)
#
#
#         # 保存训练后的更新参数
#         updated_params = {
#             name: param.clone().detach()
#             for name, param in self.model.named_parameters()
#         }
#         # print(updated_params)
#
#         # 计算参数更新的差值
#         delta_params = {}
#         for name in updated_params:
#             delta_params[name] = updated_params[name] - initial_params[name]
#
#         # 根据梯度阈值 'a' 生成掩码张量
#         a = self.gradient_threshold_a
#         mask_params = {}
#         for name in delta_params:
#             mask_params[name] = (delta_params[name].abs() > a).float()
#
#         # 将掩码应用于参数更新，得到稀疏参数
#         masked_params = {}
#         for name in updated_params:
#             masked_params[name] = updated_params[name] * mask_params[name]
#
#         # 存储稀疏参数和掩码
#         self.masked_params = masked_params
#         self.updated_params = updated_params  # 更新的全量参数保存起来
#         self.mask_params = mask_params        # 掩码保存起来
#
#     def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
#         """本地模型接收全局参数，并应用掩码逻辑更新本地模型"""
#         result_params = {}
#
#         # Type check for each parameter
#         for name in self.updated_params:
#             new_param = new_parameters[name]
#             updated_param = self.updated_params[name]
#             mask = self.mask_params[name]
#
#             # Ensure all are tensors
#             if not isinstance(new_param, torch.Tensor):
#                 print(f"new_param is not a tensor for {name}, it is {type(new_param)}")
#             if not isinstance(updated_param, torch.Tensor):
#                 print(f"updated_param is not a tensor for {name}, it is {type(updated_param)}")
#             if not isinstance(mask, torch.Tensor):
#                 print(f"mask is not a tensor for {name}, it is {type(mask)}")
#
#             # Safeguard if some are not tensors (avoid crash)
#             if isinstance(new_param, torch.Tensor) and isinstance(updated_param, torch.Tensor) and isinstance(mask,
#                                                                                                               torch.Tensor):
#                 # Use torch.where for tensor masking logic
#                 result_param = torch.where(mask != 0, new_param, updated_param)
#                 result_params[name] = result_param
#             else:
#                 # Handle the case where types are incorrect (add your own error handling or logging here)
#                 print(f"Skipping parameter {name} due to type mismatch.")
#
#         # Proceed with loading the parameters as usual
#         if not hasattr(self, 'updated_params'):
#             self.updated_params = {
#                 name: param.clone().detach()
#                 for name, param in new_parameters.items()
#             }
#             self.model.load_state_dict(result_params, strict=False)
#             optimizer_parameters = self.model.parameters()
#             self.optimizer = torch.optim.SGD(
#                 params=optimizer_parameters,
#                 lr=self.args.local_lr,
#                 momentum=self.args.momentum,
#                 weight_decay=self.args.weight_decay,
#             )
#             self.optimizer.state = {}
#             return
#
#         self.model.load_state_dict(result_params, strict=False)
#
#         # Reinitialize optimizer
#         optimizer_parameters = self.model.parameters()
#         self.optimizer = torch.optim.SGD(
#             params=optimizer_parameters,
#             lr=self.args.local_lr,
#             momentum=self.args.momentum,
#             weight_decay=self.args.weight_decay,
#         )
#         self.optimizer.state = {}


from collections import OrderedDict
import torch
from .fedavg_V3 import FedAvgClient

class FedCIPClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)
        # 添加梯度阈值 'a' 和计数器的超参数
        self.gradient_threshold_a = getattr(self.args, 'gradient_threshold_a', 1e-6)  # 梯度阈值 'a'
        self.mask_update_threshold = getattr(self.args, 'mask_update_threshold', 0)  # 更新 mask 的阈值
        self.iteration_count = 0  # 初始化计数器

    def fit(self):
        """本地训练，并生成稀疏的参数更新"""
        self.model.train()

        # 保存训练前的初始参数
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

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

                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # 增加计数器
        self.iteration_count += 1

        # 如果计数器未达到阈值，继续计算新的 masked_params
        if self.iteration_count <= self.mask_update_threshold:
            # 保存训练后的更新参数
            updated_params = {
                name: param.clone().detach()
                for name, param in self.model.named_parameters()
            }

            # 计算参数更新的差值
            delta_params = {}
            for name in updated_params:
                delta_params[name] = updated_params[name] - initial_params[name]

            # 根据梯度阈值 'a' 生成掩码张量
            a = self.gradient_threshold_a
            mask_params = {}
            for name in delta_params:
                mask_params[name] = (delta_params[name].abs() > a).float()

            # 将掩码应用于参数更新，得到稀疏参数
            masked_params = {}
            for name in updated_params:
                masked_params[name] = updated_params[name] * mask_params[name]

            # 存储稀疏参数和掩码
            self.masked_params = masked_params
            self.updated_params = updated_params  # 更新的全量参数保存起来
            self.mask_params = mask_params        # 掩码保存起来

        else:
            # 如果计数器达到了阈值，保持之前的 masked_params，不再更新
            print(f"Iteration {self.iteration_count}: Mask update threshold reached. Using previous masks.")
            # 保持 self.masked_params, self.updated_params, self.mask_params 不变

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        """本地模型接收全局参数，并应用掩码逻辑更新本地模型"""
        result_params = {}

        for name in self.updated_params:
            new_param = new_parameters[name]
            updated_param = self.updated_params[name]
            mask = self.mask_params[name]

            # 使用 torch.where 根据掩码更新参数
            result_param = torch.where(mask != 0, new_param, updated_param)
            result_params[name] = result_param

        # 加载更新后的参数
        self.model.load_state_dict(result_params, strict=False)

        # 重新初始化优化器
        optimizer_parameters = self.model.parameters()
        self.optimizer = torch.optim.SGD(
            params=optimizer_parameters,
            lr=self.args.local_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.optimizer.state = {}
