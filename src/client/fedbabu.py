from fedavg import FedAvgClient


class FedBabuClient(FedAvgClient):
    def __init__(self, model, args, logger, device):
        super().__init__(model, args, logger, device)

    def fit(self):
        self.model.train()
        for _ in range(self.local_epoch):
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
                # fix head(classifier)
                for param in self.model.classifier.parameters():
                    param.grad.zero_()
                self.optimizer.step()

    # def fit(self):
    #     """
    #     The function for specifying operations in local training phase.
    #     If you wanna implement your method and your method has different local training operations to FedAvg, this method has to be overrided.
    #     """
    #
    #     self.model.train()
    #
    #     for _ in range(self.local_epoch):
    #         for x, y in self.trainloader:
    #             # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
    #             # So the latent size 1 data batches are discarded.
    #
    #             if len(x) <= 1:
    #                 continue
    #
    #             x, y = x.to(self.device), y.to(self.device)
    #             logit = self.model(x)
    #             # if self.model.name == "DecoupledModel":
    #             #     logit = self.model(x)
    #             # else:
    #             #     logit_vae_list = self.model(x)
    #             #     logit = logit_vae_list[-1]
    #
    #             # max_contrib_layer = self.model.determine_max_contribution_layer(logit_vae_list, x, y)
    #             # message = f"{self.client_id} Layer with minimum contribution to output y: {max_contrib_layer}"
    #             # self.write_output(message)
    #             loss = self.criterion(logit, y)
    #
    #             self.optimizer.zero_grad()
    #             loss.backward()
    #             self.optimizer.step()