from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from copy import deepcopy

import torch
from rich.progress import track

from fedavg import FedAvgServer, get_fedavg_argparser
from src.config.utils import trainable_params


def get_pfedsim_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    parser.add_argument("--x_threshold", type=float, default=0.01)
    return parser


class CustomServer(FedAvgServer):
    def __init__(
        self,
        algo: str = "CustomFL",
        args: Namespace = None,
        unique_model=False,
        default_trainer=True,
    ):
        if args is None:
            args = get_pfedsim_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        delta_cache = []
        weight_cache = []
        for client_id in self.selected_clients:
            client_local_params = self.generate_client_params(client_id)
            (
                delta,
                weight,
                self.client_stats[client_id][self.current_epoch],
            ) = self.trainer.train(
                client_id=client_id,
                local_epoch=self.clients_local_epoch[client_id],
                new_parameters=client_local_params,
                verbose=((self.current_epoch + 1) % self.args.verbose_gap) == 0,
            )

            delta_cache.append(delta)
            weight_cache.append(weight)

        self.aggregate(delta_cache, weight_cache)

        # After aggregation, send the global model to the selected clients for personalization
        for client_id in self.selected_clients:
            client_global_params = self.generate_client_params(client_id)
            self.trainer.personalize(client_id, client_global_params)


if __name__ == "__main__":
    server = CustomServer()
    server.run()
