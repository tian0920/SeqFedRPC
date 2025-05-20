import torch
import numpy as np
from argparse import ArgumentParser, Namespace
from typing import List
from src.config.utils import trainable_params
from fedavg import FedAvgServer, get_fedavg_argparser  # 使用 FedAvgClient 作为客户端
from rich.progress import track

def get_fedrl_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("-wr", "--warmup_round", type=float, default=0)
    return parser

class QLearningAgent:
    def __init__(self, num_layers, lr=0.5, gamma=0.9):
        self.num_layers = num_layers  # 可选择的层数
        self.lr = lr  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = 0.1  # 探索率
        self.current_epoch = 0
        self.explore_epoch = 100

        # 定义状态空间大小（划分为 20 个区间，每个区间为 5%）
        self.num_states = 50  # 状态值范围为 0 到 19
        self.q_table = np.zeros((self.num_states, num_layers))

    def choose_action(self, state):
        if self.current_epoch < self.explore_epoch:
            return np.random.choice(self.num_layers)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_layers)
        else:
            return np.argmax(self.q_table[state])
        self.epsilon = max(0.01, self.epsilon * 0.99)


    def update_q(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        q_value = self.q_table[state, action]
        q_value_next = self.q_table[next_state, best_next_action]
        self.q_table[state, action] = q_value + self.lr * (reward + self.gamma * q_value_next - q_value)


class FedrlServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "FedRL-50",
        args: Namespace = None,
        unique_model=True,
        default_trainer=True,
    ):
        super().__init__(algo="FedRL", args=args, unique_model=unique_model, default_trainer=default_trainer)

        if args is None:
            args = get_fedrl_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)

        self.warmup_round = 0
        if 0 <= self.args.warmup_round <= 1:
            self.warmup_round = int(self.args.global_epoch * self.args.warmup_round)
        elif 1 < self.args.warmup_round < self.args.global_epoch:
            self.warmup_round = int(self.args.warmup_round)
        else:
            raise RuntimeError(
                "args.warmup_round need to be set in the range of [0, 1) or [1, args.global_epoch)."
            )

        # 过滤掉 bias 层，只考虑 weight 层
        # self.weight_layer_names = [None] + [name for name, param in self.model.named_parameters() if 'weight' in name]
        self.weight_layer_names = [name for name, param in self.model.named_parameters() if 'weight' in name]
        num_layers = len(self.weight_layer_names)  # 只考虑 weight 层作为动作空间
        self.agent = QLearningAgent(num_layers=num_layers)  # 强化学习智能体


    def train(self):
        # Warm-up Phase
        self.train_progress_bar = track(
            range(self.warmup_round),
            f"[bold cyan]Warming-up...",
            console=self.logger.stdout,
        )
        super().train()

        # Personalization Phase
        self.unique_model = True
        pfedsim_progress_bar = track(
            range(self.warmup_round, self.args.global_epoch),
            "[bold green]Personalizing...",
            console=self.logger.stdout,
        )
        self.client_trainable_params = [
            trainable_params(self.global_params_dict, detach=True)
            for _ in self.train_clients
        ]

        for E in pfedsim_progress_bar:
            self.current_epoch = E

            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            client_params_cache = []
            actions = {}
            client_accuracies_before = {}
            client_accuracies_after = {}

            for client_id in self.selected_clients:

                # 训练前选择个性化层
                local_accuracy_before, _ = self.get_client_accuracy(client_id)
                state = int((local_accuracy_before * 100) // 5)
                state = min(state, self.agent.num_states - 1)  # 确保状态不超过最大值
                action = self.agent.choose_action(state)

                actions[client_id] = action

                # 获取选中的个性化层名称，并传递给客户端
                layer_name = self.weight_layer_names[action]
                self.trainer.personal_params_name = [layer_name]  # 设置个性化层，确保该层不会被全局更新
                print("客户端" + str(client_id) + "在第" + str(self.current_epoch) + "轮的个性化层为：" + str(self.trainer.personal_params_name))


                # print(f"Client {client_id}: Personalizing layer {layer_name}")

                client_pers_params = self.generate_client_params(client_id)
                (
                    client_params,
                    _,
                    self.client_stats[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    new_parameters=client_pers_params,
                    return_diff=False,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                client_params_cache.append(client_params)

                # 获取个性化后的准确率
                _, local_accuracy_after = self.get_client_accuracy(client_id)
                client_accuracies_before[client_id] = local_accuracy_before
                client_accuracies_after[client_id] = local_accuracy_after

            global_accuracy = self.get_global_accuracy()

            # 更新强化学习奖励
            # self.personalize(client_accuracies, global_accuracy, actions)
            self.personalize(client_accuracies_before, client_accuracies_after, global_accuracy, actions)

            self.update_client_params(client_params_cache)  # 结果相同的原因是这行代码
            self.log_info()

    # def personalize(self, client_accuracies: List[float], global_accuracy: float, actions):
    #     """使用强化学习选择个性化层"""
    #     for client_id, local_accuracy in enumerate(client_accuracies):
    #         action = actions[client_id]
    #         # 将准确率按 5% 区间划分，得到状态值（范围 0 到 9）
    #         state = int((local_accuracy * 100) // 5)
    #         state = min(state, self.agent.num_states - 1)  # 确保状态不超过最大值
    #
    #         # 更新 RL 奖励
    #         reward = self.calculate_reward(global_accuracy, local_accuracy)
    #
    #         # print(f"Client {client_id}: Reward {reward}")
    #         # 计算下一个状态（假设全局准确率为下一状态的准确率）
    #         next_state = int((global_accuracy * 100) // 5)
    #         next_state = min(next_state, self.agent.num_states - 1)
    #
    #         self.agent.update_q(state, action, reward, next_state)

    def personalize(self, client_accuracies_before, client_accuracies_after, global_accuracy, actions):
        """使用强化学习选择个性化层"""
        for client_id in actions.keys():
            action = actions[client_id]
            # print(action, actions.values(), len(actions))
            state = int((client_accuracies_before[client_id] * 100) // 2)

            state = min(state, self.agent.num_states - 1)
            print(client_accuracies_before[client_id], client_accuracies_after[client_id])
            # 更新 RL 奖励
            reward = self.calculate_reward(global_accuracy, client_accuracies_before[client_id], client_accuracies_after[client_id])
            print(f"Client {client_id}: Reward {reward}")
            next_state = int((global_accuracy * 100) // 2)
            next_state = min(next_state, self.agent.num_states - 1)

            self.agent.update_q(state, action, reward, next_state)

    def calculate_reward(self, global_accuracy, local_accuracy_before, local_accuracy_after):
        """基于准确率计算奖励"""
        return local_accuracy_after - local_accuracy_before + 0.1 * global_accuracy

    # def get_client_accuracy(self, client_id):
    #     """获取单个客户端的准确率，用于选择个性化层"""
    #     client_local_params = self.generate_client_params(client_id)
    #     stats = self.trainer.test(client_id, client_local_params)
    #
    #     return stats["after"]["test_correct"] / stats["after"]["test_size"]

    def get_client_accuracy(self, client_id):
        """获取单个客户端个性化前后的准确率"""
        client_local_params = self.generate_client_params(client_id)
        stats = self.trainer.test_1(client_id, client_local_params)
        before_acc = stats["before"]["test_correct"] / stats["before"]["test_size"]
        after_acc = stats["after"]["test_correct"] / stats["after"]["test_size"]
        return before_acc, after_acc

    def get_global_accuracy(self):
        """获取全局模型在测试集上的准确率"""
        self.test_flag = True
        correct, num_samples = [], []

        for client_id in self.test_clients:
            client_local_params = self.generate_client_params(client_id)
            stats = self.trainer.test_1(client_id, client_local_params)
            correct.append(stats["after"]["test_correct"])
            num_samples.append(stats["after"]["test_size"])

        self.test_flag = False

        correct = torch.tensor(correct)
        num_samples = torch.tensor(num_samples)

        # 计算全局准确率
        global_accuracy = (correct.sum() / num_samples.sum()).item()
        return global_accuracy


if __name__ == "__main__":
    # 使用 FedAvgClient 作为客户端
    server = FedrlServer()
    server.run()
