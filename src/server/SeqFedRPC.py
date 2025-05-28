from fedavg_SeqFedRPC import FedAvgServer, get_fedavg_argparser
from argparse import ArgumentParser, Namespace

from src.config.utils import trainable_params
from rich.progress import track


def get_fedcip_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    return parser

class FedCIPServer(FedAvgServer):
    def __init__(
            self,
            algo: str = "SeqFedRPC_0.5",
            args: Namespace = None,
            unique_model=False,
            default_trainer=True,
    ):

        if args is None:
            args = get_fedcip_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.test_flag = False


    def train(self):
        self.unique_model = True
        fedcip_progress_bar = track(
            range(self.args.global_epoch),
            "[bold green]Personalizing...",
            console=self.logger.stdout,
        )
        self.client_trainable_params = [
            trainable_params(self.global_params_dict, detach=True)
            for _ in self.train_clients
        ]
        for E in fedcip_progress_bar:
            self.current_epoch = E
            if (E + 1) % self.args.verbose_gap == 0:
                self.logger.log(" " * 30, f"TRAINING EPOCH: {E + 1}", " " * 30)

            if (E + 1) % self.args.test_gap == 0:
                self.test()

            self.selected_clients = self.client_sample_stream[E]
            client_params_cache = []
            for client_id in self.selected_clients:
                client_pers_params = self.generate_client_params(client_id)
                (
                    client_params,
                    weight,
                    self.client_stats[client_id][E],
                ) = self.trainer.train(
                    client_id=client_id,
                    local_epoch=self.clients_local_epoch[client_id],
                    current_epoch=self.current_epoch,
                    new_parameters=client_pers_params,
                    return_diff=False,
                    verbose=((E + 1) % self.args.verbose_gap) == 0,
                )
                client_params_cache.append(client_params)

            self.update_client_params(client_params_cache)
            self.log_info()


if __name__ == "__main__":
    server = FedCIPServer(default_trainer=True)
    server.run()

