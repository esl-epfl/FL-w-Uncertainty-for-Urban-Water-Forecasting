"""Dense Flower client extracted from FederatedMonarch."""

import time

import flwr as fl

from federated_learning.train import train_aquacast, test_aquacast
from federated_learning.utils import get_parameters, get_server_model, set_parameters


# Keep the original worker-local optimizer cache for the paper experiments.
# A client assigned to a different Ray worker starts with that worker's cache.
_OPT_STATES = {}


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, valloader, args):
        self.cid = cid
        self.args = args
        self.net, self.optimizer = get_server_model(args)
        self.net = self.net.to(args.device)
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        fit_t0 = time.perf_counter()
        set_parameters(self.net, parameters)
        set_params_dt = time.perf_counter() - fit_t0
        if self.args.preserve_optimizer_state and self.cid in _OPT_STATES:
            self.optimizer.load_state_dict(_OPT_STATES[self.cid])
        for group in self.optimizer.param_groups:
            group['lr'] = self.args.lr

        train_t0 = time.perf_counter()
        train_aquacast(self.net, self.optimizer, self.trainloader,
                       epochs=self.args.local_epochs, args=self.args)
        train_dt = time.perf_counter() - train_t0
        if self.args.preserve_optimizer_state:
            _OPT_STATES[self.cid] = self.optimizer.state_dict()

        get_params_t0 = time.perf_counter()
        parameters = get_parameters(self.net)
        get_params_dt = time.perf_counter() - get_params_t0
        num_examples = len(self.trainloader.dataset)
        print(
            f"[TIMING] round={config['round']} stage=client_fit cid={self.cid} "
            f"epochs={self.args.local_epochs} num_train_samples={num_examples} "
            f"set_params_seconds={set_params_dt:.4f} train_seconds={train_dt:.4f} "
            f"get_params_seconds={get_params_dt:.4f} "
            f"fit_total_seconds={time.perf_counter() - fit_t0:.4f}", flush=True,
        )
        return parameters, num_examples, {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, _ = test_aquacast(self.net, self.valloader, self.args)
        return float(loss), len(self.valloader.dataset), {'mse': float(loss)}


def gen_client_fn(trainloaders, valloaders, args):
    def client_fn(cid):
        return FlowerClient(cid, trainloaders[int(cid)], valloaders[int(cid)], args).to_client()
    return client_fn
