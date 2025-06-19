from fedavg_sfl import FedAvgServer, get_fedavg_argparser
from argparse import ArgumentParser, Namespace

from src.client.fedavg_sfl import FedAvgClient
from copy import deepcopy


def get_sfl_argparser() -> ArgumentParser:
    parser = get_fedavg_argparser()
    return parser

class SFLServer(FedAvgServer):
    def __init__(
            self,
            algo: str = "SFL",
            args: Namespace = None,
            unique_model=False,
            default_trainer=False,
    ):

        if args is None:
            args = get_sfl_argparser().parse_args()
        super().__init__(algo, args, unique_model, default_trainer)
        self.trainer = FedAvgClient(
            deepcopy(self.model), self.args, self.logger, self.device
        )


if __name__ == "__main__":
    server = SFLServer()
    server.run()