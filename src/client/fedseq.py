from collections import OrderedDict
from typing import Dict, Set, Tuple
import torch

from fedavg import FedAvgClient
from torch.utils.data import DataLoader
import json


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion=torch.nn.CrossEntropyLoss(reduction="sum"),
    device=torch.device("cpu"),
    type_flag=0  # 避免使用 'type' 作为参数名
) -> Dict[str, float]:
    """Evaluate the `model` over `dataloader` and return the result calculated by `criterion`.

    Args:
        model (torch.nn.Module): Target model.
        dataloader (DataLoader): Target dataloader.
        criterion (optional): The metric criterion. Defaults to torch.nn.CrossEntropyLoss(reduction="sum").
        device (torch.device, optional): The device that holds the computation. Defaults to torch.device("cpu").
        type_flag (int, optional): If set to 1, compute per-class accuracy. Defaults to 0.

    Returns:
        Dict[str, float]: [metric (loss), correct predictions, sample num]
    """
    model.eval()
    correct = 0
    loss = 0
    sample_num = 0
    class_correct_dict = {}

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if hasattr(model, 'name') and model.name == "DecoupledModel":
                logit = model(x)
            else:
                logit_vae_list = model(x)
                logit = logit_vae_list[-1] if isinstance(logit_vae_list, (list, tuple)) else logit_vae_list

            loss += criterion(logit, y).item()
            pred = torch.argmax(logit, dim=-1)

            if type_flag:
                # 初始化 class_correct_dict
                correct_preds = (pred == y)
                for label in y:
                    label = label.item()
                    if label not in class_correct_dict:
                        class_correct_dict[label] = {"correct": 0, "total": 0}
                    class_correct_dict[label]["total"] += 1

                for label, is_correct in zip(y, correct_preds):
                    label = label.item()
                    if is_correct.item():
                        class_correct_dict[label]["correct"] += 1
                print(json.dumps(class_correct_dict))

            correct += (pred == y).sum().item()
            sample_num += y.size(0)

    return {
        "loss": loss / sample_num,
        "correct": correct,
        "sample_num": sample_num
    }


class FedSeqClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)

        # 初始化个性化参数
        self.personal_params_dict: Dict[str, torch.Tensor] = {}
        self.personal_params_name_set: Set[str] = set()

        # 初始化个性化比例和动态调整参数
        self.personalization_proportion = getattr(args, 'personalization_proportion', 0.0)
        self.personalization_proportion_max = 0.0  # 最大比例
        self.personalization_proportion_min = 0.0  # 最小比例

        # 控制是否保留个性化参数
        self.retain_personalized = getattr(args, 'retain_personalized', False)

        # 正则化系数
        self.reg_lambda = getattr(args, 'reg_lambda', 0.01)

        # 初始化初始损失字典
        self.loss_initial_dict: Dict[int, float] = {}

    def set_parameters(self, new_parameters: OrderedDict[str, torch.Tensor]):
        # 加载模型参数
        self.model.load_state_dict(new_parameters, strict=False)
        # 如果保留个性化参数，则加载它们
        if self.retain_personalized and self.personal_params_dict:
            self.model.load_state_dict(self.personal_params_dict, strict=False)

        # 初始化优化器，排除个性化参数
        personal_param_names = self.personal_params_name_set
        optimizer_parameters = [
            param for name, param in self.model.named_parameters()
            if name not in personal_param_names and param.requires_grad
        ]

        # 检查优化器参数是否为空
        if not optimizer_parameters:
            # 如果为空，包含所有需要梯度的参数以避免错误
            optimizer_parameters = [
                param for name, param in self.model.named_parameters()
                if param.requires_grad
            ]
            self.logger.log("Warning: All parameters are personalized or have requires_grad=False. Including all trainable parameters in optimizer.")
            self.personal_params_name_set.clear()  # 清除个性化参数集合

        self.optimizer = torch.optim.SGD(
            params=optimizer_parameters,
            lr=self.args.local_lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        self.optimizer.state = {}

    def save_state(self):
        # 保存个性化参数
        self.personal_params_dict = {
            name: param.clone().detach()
            for name, param in self.model.state_dict().items()
            if name in self.personal_params_name_set
        }

    def fit(self) -> Dict[str, float]:
        self.model.train()

        # 保存初始参数
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        # 本地训练前进行评估（before）
        eval_before = self.evaluate_model_wrapper()
        # 'before' 评估统计

        # 本地训练
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)

                # 前向传播
                output = self.model(x)
                # 如果输出是列表，则取最后一个元素
                if isinstance(output, list):
                    output = output[-1]
                loss = self.criterion(output, y)

                # 添加正则化项
                reg_loss = self.reg_lambda * self.personalization_proportion
                total_loss = loss.mean() + reg_loss

                # 反向传播和优化
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # 本地训练后进行评估（after）
        eval_after = self.evaluate_model_wrapper()
        # 'after' 评估统计

        # 动态调整 personalization_proportion
        avg_loss = eval_after["loss"]  # 使用测试损失作为指标
        loss_initial = self.loss_initial_dict.get(self.client_id, avg_loss)

        if loss_initial > 0:
            ratio = avg_loss / loss_initial
            # 在 loss 很小时，比例增加；loss 很大时，比例减小
            new_proportion = self.personalization_proportion_max * max(
                self.personalization_proportion_min,
                min(1.0, 1 - ratio)
            )
            # 确保 personalization_proportion 在 [0, 0.2] 之间
            self.personalization_proportion = max(
                self.personalization_proportion_min,
                min(self.personalization_proportion_max, new_proportion)
            )
            # 更新初始损失
            self.loss_initial_dict[self.client_id] = avg_loss
        else:
            # 如果初始损失为 0，设置为最大比例
            self.personalization_proportion = self.personalization_proportion_max

        # 识别个性化参数
        updated_params = {
            name: param for name, param in self.model.named_parameters()
        }

        # 计算参数变化的幅度
        param_deltas = {}
        for name in updated_params:
            delta = (updated_params[name] - initial_params[name]).abs().mean().item()
            param_deltas[name] = delta

        # 确定要个性化的参数数量
        num_total_params = len(param_deltas)
        num_personalized = int(num_total_params * self.personalization_proportion)

        # 确保不会个性化所有参数
        if num_personalized >= num_total_params:
            num_personalized = num_total_params - 1

        # 选择变化最大的参数作为个性化参数
        if num_personalized > 0:
            sorted_params = sorted(param_deltas.items(), key=lambda item: item[1], reverse=True)
            top_param_names = {name for name, _ in sorted_params[:num_personalized]}
            self.personal_params_name_set.update(top_param_names)

        # 保存个性化参数
        self.save_state()

        # 返回评估统计
        return {
            "before": eval_before,
            "after": eval_after
        }

    def train(
        self,
        client_id: int,
        local_epoch: int,
        model_parameters: OrderedDict[str, torch.Tensor],
        verbose=False,
    ) -> Tuple[OrderedDict[str, torch.Tensor], Dict]:
        """
        执行本地训练并返回更新后的模型参数和评估统计。
        """
        self.client_id = client_id
        self.local_epoch = local_epoch
        self.load_dataset()
        self.set_parameters(model_parameters)
        eval_stats = self.fit()

        # 准备要传递给下一个客户端的模型参数
        updated_parameters = self.model.state_dict()

        # 如果保留个性化参数，则在发送之前移除它们
        if self.retain_personalized:
            for name in self.personal_params_name_set:
                if name in updated_parameters:
                    del updated_parameters[name]

        return updated_parameters, eval_stats

    def evaluate_model_wrapper(self) -> Dict[str, float]:
        """
        包装 evaluate_model 函数，提供模型、数据加载器、损失函数和设备。
        """
        # 假设您要在测试集上评估
        return evaluate_model(
            model=self.model,
            dataloader=self.testloader,
            criterion=self.criterion,
            device=self.device,
            type_flag=0  # 设置为 1 以获取类别级别统计
        )
